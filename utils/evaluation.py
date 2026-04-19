"""
SpotGEO Official Evaluation Metrics
====================================

Implements the exact F1 and MSE metrics from the SpotGEO challenge paper
(Chen et al., CVPRW 2021).

The evaluation uses:
- Hungarian matching to associate predictions with ground truth
- Truncated distance for matching (threshold τ)
- Tolerance distance ε for labeling inaccuracy
- F1 score (primary ranking metric)
- MSE (tie-breaker)

These are implemented to match the official eval toolkit on Zenodo.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def evaluate_frame(predictions, groundtruth, tau=5.0, epsilon=2.0):
    """
    Evaluate predictions for a single frame.

    Args:
        predictions: list of (x, y) predicted object locations
        groundtruth: list of (x, y) ground truth locations
        tau: matching distance threshold
        epsilon: tolerance distance

    Returns:
        tp: number of true positives
        fn: number of false negatives
        fp: number of false positives
        sse: sum of squared errors
    """
    M = len(predictions)  # number of predictions
    N = len(groundtruth)  # number of ground truth objects

    # Handle empty cases
    if M == 0 and N == 0:
        return 0, 0, 0, 0.0
    if M == 0:
        return 0, N, 0, N * tau ** 2
    if N == 0:
        return 0, 0, M, M * tau ** 2

    # Build cost matrix with truncated distance
    # If M <= N: each prediction matched to one GT
    # If M > N: swap roles
    swapped = False
    if M > N:
        predictions, groundtruth = groundtruth, predictions
        M, N = N, M
        swapped = True

    # Cost matrix: M x N
    cost = np.zeros((M, N))
    distances = np.zeros((M, N))
    diag_length = np.sqrt(640 ** 2 + 480 ** 2)  # sufficiently large number

    for i in range(M):
        for j in range(N):
            px, py = predictions[i]
            gx, gy = groundtruth[j]
            d = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)
            distances[i, j] = d
            # Truncated distance
            cost[i, j] = d if d <= tau else diag_length

    # Solve minimum weight assignment
    row_ind, col_ind = linear_sum_assignment(cost)

    # Count TP, FN, FP
    tp = 0
    fp_set = set(range(M))
    fn_set = set(range(N))
    sse = 0.0

    matched_pairs = []

    for i, j in zip(row_ind, col_ind):
        d = distances[i, j]
        if d <= tau:
            tp += 1
            fp_set.discard(i)
            fn_set.discard(j)
            matched_pairs.append((i, j, d))

    fn = len(fn_set)
    fp = len(fp_set)

    # Compute SSE
    for i, j, d in matched_pairs:
        if d <= epsilon:
            sse += 0.0  # Within tolerance
        else:
            sse += d ** 2

    # Add penalty for misses and false detections
    sse += fn * tau ** 2
    sse += fp * tau ** 2

    # Swap back if we swapped
    if swapped:
        fp, fn = fn, fp

    return tp, fn, fp, sse


def evaluate_sequence(pred_per_frame, gt_per_frame, tau=5.0, epsilon=2.0):
    """
    Evaluate predictions for a full 5-frame sequence.

    Args:
        pred_per_frame: list of 5 lists of (x, y) predictions
        gt_per_frame: list of 5 lists of (x, y) ground truth

    Returns:
        dict with tp, fn, fp, sse, f1, mse for this sequence
    """
    total_tp = 0
    total_fn = 0
    total_fp = 0
    total_sse = 0.0

    for f in range(5):
        preds = pred_per_frame[f] if f < len(pred_per_frame) else []
        gts = gt_per_frame[f] if f < len(gt_per_frame) else []

        tp, fn, fp, sse = evaluate_frame(preds, gts, tau, epsilon)
        total_tp += tp
        total_fn += fn
        total_fp += fp
        total_sse += sse

    # Compute F1
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Compute MSE
    total_count = total_tp + total_fn + total_fp
    mse = total_sse / total_count if total_count > 0 else 0.0

    return {
        'tp': total_tp,
        'fn': total_fn,
        'fp': total_fp,
        'sse': total_sse,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mse': mse,
    }


def evaluate_dataset(all_predictions, all_groundtruths, tau=5.0, epsilon=2.0):
    """
    Evaluate over the entire test set (all sequences).

    Args:
        all_predictions: list of K sequences, each containing 5 frame predictions
        all_groundtruths: list of K sequences, each containing 5 frame GT

    Returns:
        dict with overall F1, MSE, precision, recall
    """
    total_tp = 0
    total_fn = 0
    total_fp = 0
    total_sse = 0

    for pred_seq, gt_seq in zip(all_predictions, all_groundtruths):
        result = evaluate_sequence(pred_seq, gt_seq, tau, epsilon)
        total_tp += result['tp']
        total_fn += result['fn']
        total_fp += result['fp']
        total_sse += result['sse']

    # Overall metrics (micro-averaged across all sequences)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    total_count = total_tp + total_fn + total_fp
    mse = total_sse / total_count if total_count > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        '1-f1': 1 - f1,  # This is what's used for ranking
        'mse': mse,
        'tp': total_tp,
        'fn': total_fn,
        'fp': total_fp,
    }


def print_results(results, model_name="Model"):
    """Pretty-print evaluation results."""
    print(f"\n{'=' * 50}")
    print(f"  Results for: {model_name}")
    print(f"{'=' * 50}")
    print(f"  F1 Score:    {results['f1']:.4f}  (1-F1 = {results['1-f1']:.4f})")
    print(f"  Precision:   {results['precision']:.4f}")
    print(f"  Recall:      {results['recall']:.4f}")
    print(f"  MSE:         {results['mse']:.2f}")
    print(f"  TP: {results['tp']}, FN: {results['fn']}, FP: {results['fp']}")
    print(f"{'=' * 50}\n")
