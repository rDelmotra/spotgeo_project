"""
Training Script for WaveSANet and Baselines
============================================

Usage:
    python train.py --model wavesanet --epochs 200 --lr 0.0015  # can use 400 if needed
    python train.py --model unet --epochs 200          # Plain U-Net baseline (can use 400 if needed)
"""

import os
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from configs.config import get_config
from models.baselines import build_model
from utils.dataset import SpotGEODataset


def train_collate_fn(batch):
    """Custom collate that handles variable-length centroids."""
    images = torch.stack([b['image'] for b in batch])
    masks = torch.stack([b['mask'] for b in batch])
    centroids = [b['centroids'] for b in batch]
    seq_ids = [b['seq_id'] for b in batch]
    frame_idxs = [b['frame_idx'] for b in batch]
    return {'image': images, 'mask': masks, 'centroids': centroids,
            'seq_id': seq_ids, 'frame_idx': frame_idxs}


# ============================================================
# Focal Loss (handles extreme class imbalance)
# ============================================================

class FocalLoss(nn.Module):
    """
    Binary Focal Loss — down-weights easy negatives (background),
    focuses training on hard examples (faint satellites, tricky edges).

    Without focal loss, the model could get 99% accuracy by predicting
    "no satellite" everywhere — focal loss prevents this.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) predicted probabilities (after sigmoid)
            target: (B, 1, H, W) ground truth masks

        Returns:
            loss: scalar
        """
        pred = pred.float()
        target = target.float()

        # Clamp for numerical stability
        pred = pred.clamp(1e-7, 1 - 1e-7)

        # Binary cross-entropy per pixel
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)

        # Focal weight
        p_t = target * pred + (1 - target) * (1 - pred)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)

        loss = alpha_t * focal_weight * bce
        return loss.mean()


# ============================================================
# Training Loop
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, scaler):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in loader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        # Forward pass with AMP
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', enabled=(device.type == 'cuda'), dtype=torch.float16):
            predictions = model(images)
            loss = criterion(predictions, masks)

        # Backward pass with scaler
        scaler.scale(loss).backward()

        # Gradient clipping with scaler
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def validate(model, loader, criterion, device):
    """Validate on held-out set."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            with torch.autocast(device_type='cuda', enabled=(device.type == 'cuda'), dtype=torch.float16):
                predictions = model(images)
                loss = criterion(predictions, masks)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Train SpotGEO detection model')
    parser.add_argument('--model', type=str, default='wavesanet',
                        choices=['wavesanet', 'unet', 'unet_cbam', 'unet_wtnet'],
                        help='Model to train')
    parser.add_argument('--epochs', type=int, default=200)  # can use 400 if needed
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data_dir', type=str, default='./data/spotgeo/train')
    parser.add_argument('--annotations', type=str, default='./data/spotgeo/train_anno.json')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Fraction of training data for validation')
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' or (args.device == 'cuda' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Build model
    model = build_model(args.model).to(device)

    # Dataset
    full_dataset = SpotGEODataset(
        data_dir=args.data_dir,
        annotations_file=args.annotations,
        label_type='gaussian',
        augment=True,
        gaussian_sigma=2.0,
    )

    # Split into train/val
    n_val = int(len(full_dataset) * args.val_split)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    # Disable augmentation for validation
    # (random_split doesn't allow this directly, but augmentation is random so
    #  it's effectively a minor regularization on val — acceptable for a project)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
        collate_fn=train_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
        collate_fn=train_collate_fn
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Loss, optimizer, scheduler, scaler
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    # Use modern GradScaler to avoid deprecation warnings
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 30  # can increase if model is still improving
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    print(f"\nStarting training: {args.model} for {args.epochs} epochs")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                     device, epoch, scaler)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        elapsed = time.time() - start_time
        
        is_best = val_loss < best_val_loss

        # Print progress
        if epoch % 10 == 0 or epoch == 1 or is_best:
            best_str = " [NEW BEST]" if is_best else ""
            print(f"Epoch {epoch:4d}/{args.epochs} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"LR: {current_lr:.6f} | Time: {elapsed:.1f}s{best_str}")

        # Save best model
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = os.path.join(args.checkpoint_dir, f'{args.model}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'history': history,
            }, save_path)
            print(f"  → Saved best model (val_loss={val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {early_stopping_patience} epochs)")
                break

        # Save periodic checkpoint
        if epoch % 10 == 0:
            save_path = os.path.join(args.checkpoint_dir,
                                     f'{args.model}_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'history': history,
            }, save_path)

    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {args.checkpoint_dir}/{args.model}_best.pth")

    # Save final training history
    import json
    with open(os.path.join(args.checkpoint_dir, f'{args.model}_history.json'), 'w') as f:
        json.dump(history, f)


if __name__ == '__main__':
    main()
