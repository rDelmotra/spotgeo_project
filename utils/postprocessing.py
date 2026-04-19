"""
Post-Processing: Hungarian Matching + Trajectory Completion
============================================================

After single-frame detection, we exploit the fact that satellites
move in straight lines across the 5 frames. This lets us:
1. Match detections across frames (Hungarian algorithm)
2. Fill in missed detections via interpolation
3. Remove false positives that don't fit any trajectory

Grandma version: If you see a dot in frames 1, 2, and 4 all moving
in a straight line, you can guess where it should be in frames 3 and 5.
That's trajectory completion. And if a dot appears in only one frame
and doesn't fit any line, it's probably noise — throw it out.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import label as connected_components


def heatmap_to_centroids(heatmap, threshold=0.5, min_area=2):
    """
    Convert detection heatmap to centroid coordinates.

    1. Threshold heatmap → binary
    2. Find connected components (blobs)
    3. Filter by minimum area
    4. Return centroid of each blob

    Args:
        heatmap: (H, W) float32 probability map
        threshold: detection confidence threshold
        min_area: minimum blob area in pixels

    Returns:
        centroids: list of (x, y) coordinates
    """
    binary = (heatmap > threshold).astype(np.uint8)

    # Find connected components
    labeled, num_features = connected_components(binary)

    centroids = []
    for i in range(1, num_features + 1):
        component = (labeled == i)
        area = component.sum()

        if area < min_area:
            continue

        # Centroid = weighted average of pixel positions by heatmap value
        ys, xs = np.where(component)
        weights = heatmap[ys, xs]
        weight_sum = weights.sum()

        if weight_sum > 0:
            cx = np.average(xs.astype(float), weights=weights)
            cy = np.average(ys.astype(float), weights=weights)
        else:
            cx = xs.mean()
            cy = ys.mean()

        centroids.append((cx, cy))

    return centroids


def hungarian_match(points_a, points_b, max_distance=50.0):
    """
    Optimal one-to-one matching between two sets of points
    using the Hungarian algorithm.

    Args:
        points_a: list of (x, y) from frame A
        points_b: list of (x, y) from frame B
        max_distance: maximum allowed matching distance

    Returns:
        matches: list of (idx_a, idx_b) matched pairs
        unmatched_a: list of unmatched indices in A
        unmatched_b: list of unmatched indices in B
    """
    if len(points_a) == 0 or len(points_b) == 0:
        return [], list(range(len(points_a))), list(range(len(points_b)))

    # Build cost matrix
    m, n = len(points_a), len(points_b)
    cost = np.zeros((m, n))
    for i, (ax, ay) in enumerate(points_a):
        for j, (bx, by) in enumerate(points_b):
            cost[i, j] = np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

    # Solve assignment
    row_ind, col_ind = linear_sum_assignment(cost)

    # Filter by max distance
    matches = []
    matched_a = set()
    matched_b = set()

    for i, j in zip(row_ind, col_ind):
        if cost[i, j] <= max_distance:
            matches.append((i, j))
            matched_a.add(i)
            matched_b.add(j)

    unmatched_a = [i for i in range(m) if i not in matched_a]
    unmatched_b = [j for j in range(n) if j not in matched_b]

    return matches, unmatched_a, unmatched_b


def interpolate_position(p1, p2, f1, f2, target_f):
    """
    Linear interpolation of position between two frames.

    Args:
        p1: (x, y) position in frame f1
        p2: (x, y) position in frame f2
        f1, f2: frame indices
        target_f: frame index to interpolate at

    Returns:
        (x, y) interpolated position
    """
    alpha = (target_f - f1) / (f2 - f1)
    x = p1[0] + alpha * (p2[0] - p1[0])
    y = p1[1] + alpha * (p2[1] - p1[1])
    return (x, y)


def extrapolate_position(ref_point, velocity, delta_frames):
    """
    Extrapolate position using estimated velocity.

    Args:
        ref_point: (x, y) reference position
        velocity: (vx, vy) pixels per frame
        delta_frames: number of frames to extrapolate

    Returns:
        (x, y) extrapolated position
    """
    x = ref_point[0] + velocity[0] * delta_frames
    y = ref_point[1] + velocity[1] * delta_frames
    return (x, y)


def temporal_support_filter(detections_per_frame, window=2,
                            distance_threshold=30.0, min_support_ratio=0.3):
    """
    Remove isolated detections that lack temporal support.

    For each detection in each frame, check how many neighboring frames
    have a detection nearby. If too few → it's noise.

    Args:
        detections_per_frame: list of 5 lists of (x, y) centroids
        window: temporal window radius
        distance_threshold: max distance to count as "support"
        min_support_ratio: minimum fraction of supporting frames

    Returns:
        filtered: list of 5 lists of (x, y) — noise removed
    """
    num_frames = len(detections_per_frame)
    filtered = [[] for _ in range(num_frames)]

    for f in range(num_frames):
        for point in detections_per_frame[f]:
            # Count supporting frames
            support_count = 0
            valid_frames = 0

            for f2 in range(max(0, f - window), min(num_frames, f + window + 1)):
                if f2 == f:
                    continue
                valid_frames += 1

                # Check if any detection in f2 is close enough
                for p2 in detections_per_frame[f2]:
                    dist = np.sqrt((point[0] - p2[0]) ** 2 + (point[1] - p2[1]) ** 2)
                    if dist <= distance_threshold:
                        support_count += 1
                        break

            # Keep if sufficient support (or if sequence is very short)
            if valid_frames == 0:
                ratio = 1.0
            else:
                ratio = support_count / valid_frames

            if ratio >= min_support_ratio or num_frames <= 3:
                filtered[f].append(point)

    return filtered


def trajectory_completion(detections_per_frame, max_match_distance=50.0,
                          max_interp_gap=2, distance_threshold=30.0,
                          min_support_ratio=0.3):
    """
    Full multi-frame trajectory completion pipeline.

    Steps:
    1. Match detections across consecutive frames (Hungarian)
    2. Interpolate missing detections in gaps
    3. Filter noise via temporal support
    4. Progressive refinement for remaining gaps

    Args:
        detections_per_frame: list of 5 lists of (x, y) centroids
            (output from heatmap_to_centroids for each frame)

    Returns:
        completed: list of 5 lists of (x, y) — refined detections
    """
    num_frames = len(detections_per_frame)

    # Make mutable copies
    working = [list(d) for d in detections_per_frame]

    # ---- Step 1: Interpolation-based completion ----
    # For each pair of frames with detections, interpolate missing frames
    for f1 in range(num_frames):
        for f2 in range(f1 + 2, min(f1 + max_interp_gap + 2, num_frames)):
            if len(working[f1]) == 0 or len(working[f2]) == 0:
                continue

            # Match points between f1 and f2
            matches, _, _ = hungarian_match(
                working[f1], working[f2], max_match_distance
            )

            # Interpolate for missing intermediate frames
            for idx_a, idx_b in matches:
                p1 = working[f1][idx_a]
                p2 = working[f2][idx_b]

                for target_f in range(f1 + 1, f2):
                    interp = interpolate_position(p1, p2, f1, f2, target_f)

                    # Only add if no existing detection is nearby
                    is_duplicate = any(
                        np.sqrt((interp[0] - ex[0]) ** 2 + (interp[1] - ex[1]) ** 2)
                        < distance_threshold
                        for ex in working[target_f]
                    )
                    if not is_duplicate:
                        working[target_f].append(interp)

    # ---- Step 2: Temporal support filtering ----
    working = temporal_support_filter(
        working,
        window=2,
        distance_threshold=distance_threshold,
        min_support_ratio=min_support_ratio
    )

    # ---- Step 3: Progressive refinement (extrapolation) ----
    # Use velocity estimates from matched frames to fill remaining gaps
    for f in range(num_frames):
        if len(working[f]) > 0:
            continue  # Frame already has detections

        # Try to extrapolate from the nearest frames with detections
        # Look backward
        for ref_f in range(f - 1, -1, -1):
            if len(working[ref_f]) == 0:
                continue

            # Need at least 2 reference frames to estimate velocity
            for ref_f2 in range(ref_f - 1, -1, -1):
                if len(working[ref_f2]) == 0:
                    continue

                matches, _, _ = hungarian_match(
                    working[ref_f2], working[ref_f], max_match_distance
                )

                for idx_a, idx_b in matches:
                    p_early = working[ref_f2][idx_a]
                    p_later = working[ref_f][idx_b]
                    dt = ref_f - ref_f2

                    velocity = (
                        (p_later[0] - p_early[0]) / dt,
                        (p_later[1] - p_early[1]) / dt
                    )

                    extrap = extrapolate_position(p_later, velocity, f - ref_f)

                    # Bounds check
                    if 0 <= extrap[0] <= 639.5 and 0 <= extrap[1] <= 479.5:
                        is_dup = any(
                            np.sqrt((extrap[0] - ex[0]) ** 2 + (extrap[1] - ex[1]) ** 2)
                            < distance_threshold
                            for ex in working[f]
                        )
                        if not is_dup:
                            working[f].append(extrap)

                break  # Only use closest pair
            break

    return working


def postprocess_sequence(heatmaps, threshold=0.5, min_area=2,
                         max_match_distance=50.0):
    """
    Full post-processing: heatmaps → refined centroids.

    Args:
        heatmaps: list of 5 (H, W) probability maps
        threshold: detection threshold
        min_area: minimum blob area
        max_match_distance: for Hungarian matching

    Returns:
        detections: list of 5 lists of (x, y) coordinates
    """
    # Step 1: Heatmap → raw centroids per frame
    raw_detections = [
        heatmap_to_centroids(hm, threshold=threshold, min_area=min_area)
        for hm in heatmaps
    ]

    # Step 2: Trajectory completion
    refined = trajectory_completion(
        raw_detections,
        max_match_distance=max_match_distance
    )

    return refined
