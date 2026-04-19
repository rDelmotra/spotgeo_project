"""
Full Pipeline Evaluation Script
================================

Runs the complete detection pipeline:
1. Load trained model
2. Pre-process each sequence (median subtraction + denoising)
3. Single-frame detection (model inference)
4. Post-processing (Hungarian matching + trajectory completion)
5. Evaluate using official SpotGEO metrics

Usage:
    python evaluate.py --model wavesanet --checkpoint checkpoints/wavesanet_best.pth
    python evaluate.py --model wavesanet --checkpoint checkpoints/wavesanet_best.pth --no-postprocess
"""

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

from models.baselines import build_model
from utils.dataset import SpotGEOSequenceDataset
from utils.preprocessing import preprocess_sequence
from utils.postprocessing import postprocess_sequence, heatmap_to_centroids
from utils.evaluation import evaluate_dataset, print_results


def run_evaluation(model, dataset, device, use_preprocess=True,
                   use_postprocess=True, threshold=0.5):
    """
    Run full pipeline evaluation on all sequences.

    Args:
        model: trained detection model
        dataset: SpotGEOSequenceDataset
        device: torch device
        use_preprocess: apply median subtraction + wavelet denoising
        use_postprocess: apply trajectory completion
        threshold: detection confidence threshold

    Returns:
        all_predictions, all_groundtruths: for evaluation
    """
    model.eval()
    all_predictions = []
    all_groundtruths = []

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Evaluating"):
            sample = dataset[idx]
            frames_tensor = sample['frames']  # (5, 1, H, W)
            gt_centroids = sample['gt_centroids']  # list of 5 lists

            # Pre-process
            if use_preprocess:
                frames_np = [f.squeeze().numpy() for f in frames_tensor]
                frames_np = preprocess_sequence(frames_np)
                frames_tensor = torch.stack([
                    torch.from_numpy(f).unsqueeze(0) for f in frames_np
                ])

            # Single-frame detection
            heatmaps = []
            for f_idx in range(5):
                frame = frames_tensor[f_idx].unsqueeze(0).to(device)  # (1, 1, H, W)
                heatmap = model(frame)  # (1, 1, H, W)
                heatmaps.append(heatmap.squeeze().cpu().numpy())

            # Post-processing
            if use_postprocess:
                pred_centroids = postprocess_sequence(heatmaps, threshold=threshold)
            else:
                pred_centroids = [
                    heatmap_to_centroids(hm, threshold=threshold)
                    for hm in heatmaps
                ]

            all_predictions.append(pred_centroids)
            all_groundtruths.append(gt_centroids)

    return all_predictions, all_groundtruths


def main():
    parser = argparse.ArgumentParser(description='Evaluate SpotGEO detection pipeline')
    parser.add_argument('--model', type=str, default='wavesanet',
                        choices=['wavesanet', 'unet', 'unet_cbam', 'unet_wtnet'])
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data/spotgeo/train')
    parser.add_argument('--annotations', type=str,
                        default='./data/spotgeo/train_anno.json')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--no-preprocess', action='store_true',
                        help='Skip pre-processing')
    parser.add_argument('--no-postprocess', action='store_true',
                        help='Skip trajectory post-processing')
    parser.add_argument('--tau', type=float, default=5.0,
                        help='Matching distance threshold')
    parser.add_argument('--epsilon', type=float, default=2.0,
                        help='Tolerance distance')
    args = parser.parse_args()

    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' or (args.device == 'cuda' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Build and load model
    model = build_model(args.model).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    print(f"Checkpoint val_loss: {checkpoint.get('val_loss', '?')}")

    # Load dataset
    dataset = SpotGEOSequenceDataset(
        data_dir=args.data_dir,
        annotations_file=args.annotations,
    )
    print(f"Evaluating on {len(dataset)} sequences")

    # Run evaluation
    all_preds, all_gts = run_evaluation(
        model, dataset, device,
        use_preprocess=not args.no_preprocess,
        use_postprocess=not args.no_postprocess,
        threshold=args.threshold,
    )

    # Compute metrics
    results = evaluate_dataset(all_preds, all_gts, tau=args.tau, epsilon=args.epsilon)

    # Print results
    suffix = ""
    if args.no_preprocess:
        suffix += " (no pre-process)"
    if args.no_postprocess:
        suffix += " (no post-process)"
    print_results(results, f"{args.model}{suffix}")

    # ---- Ablation: also evaluate without post-processing for comparison ----
    if not args.no_postprocess:
        print("\n--- Ablation: Without post-processing ---")
        all_preds_raw, _ = run_evaluation(
            model, dataset, device,
            use_preprocess=not args.no_preprocess,
            use_postprocess=False,
            threshold=args.threshold,
        )
        results_raw = evaluate_dataset(all_preds_raw, all_gts,
                                       tau=args.tau, epsilon=args.epsilon)
        print_results(results_raw, f"{args.model} (single-frame only)")

        # Show improvement from post-processing
        f1_gain = results['f1'] - results_raw['f1']
        mse_reduction = results_raw['mse'] - results['mse']
        print(f"Post-processing improvement:")
        print(f"  F1: +{f1_gain:.4f}")
        print(f"  MSE: -{mse_reduction:.2f}")


if __name__ == '__main__':
    main()
