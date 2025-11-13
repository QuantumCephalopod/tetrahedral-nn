"""
IMPROVED OPTICAL FLOW COMPUTATION
==================================

Improvements over original:
1. DIS optical flow (faster, better for large displacement)
2. Biological saccades (less frequent, Gaussian distribution)
3. Padding instead of stretching (preserves aspect ratio)
4. Better tuned parameters

Author: Philipp Remy Bartholomäus & Claude
Date: November 12, 2025
"""

import numpy as np
import cv2
import torch


def pad_to_square(frame, fill_value=0):
    """
    Pad rectangular frame to square WITHOUT stretching.

    Atari: 210×160 → 210×210 (add 25 pixels on left and right)
    """
    if len(frame.shape) == 2:
        h, w = frame.shape
        c = None
    else:
        h, w, c = frame.shape

    size = max(h, w)
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2

    if c is not None:
        padded = np.full((size, size, c), fill_value, dtype=frame.dtype)
        padded[pad_h:pad_h+h, pad_w:pad_w+w, :] = frame
    else:
        padded = np.full((size, size), fill_value, dtype=frame.dtype)
        padded[pad_h:pad_h+h, pad_w:pad_w+w] = frame

    return padded


def add_biological_saccade(frame, saccade_rate=0.2, saccade_std=0.7):
    """
    Add biological microsaccade with correct frequency and distribution.

    Real microsaccades:
    - Frequency: 1-2 Hz (not every frame!)
    - Distribution: Mostly small, few large (Gaussian, not uniform)
    - Magnitude: <1% of visual field

    Args:
        frame: Input frame (numpy uint8)
        saccade_rate: Probability of saccade (0.2 = ~1-2 Hz at 6-10 Hz sampling)
        saccade_std: Standard deviation in pixels (0.7 = most saccades <1 pixel)

    Returns:
        Frame with or without saccade applied
    """
    if np.random.random() > saccade_rate:
        return frame  # No saccade this frame (80% of time)

    # Gaussian distribution (most saccades are small)
    dx = int(np.random.normal(0, saccade_std))
    dy = int(np.random.normal(0, saccade_std))

    # Clip to reasonable range (biological limit)
    dx = np.clip(dx, -2, 2)
    dy = np.clip(dy, -2, 2)

    # Apply shift
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

    return shifted


def compute_optical_flow_improved(frame1, frame2, target_size=None,
                                  add_saccade=False, method='auto'):
    """
    Compute optical flow with IMPROVED settings.

    Improvements:
    1. Tries DIS first (faster, better for large displacement)
    2. Falls back to Farneback if DIS unavailable
    3. Biological saccades (less frequent, Gaussian)
    4. Padding instead of stretching

    Args:
        frame1, frame2: Numpy arrays (H, W) grayscale, values in [0, 1]
        target_size: Square size to pad to (e.g., 210)
        add_saccade: Whether to add artificial microsaccades
        method: 'auto' (try DIS), 'DIS', or 'farneback'

    Returns:
        flow: torch.Tensor (2, H, W) where H=W=target_size
    """
    # Convert to uint8
    f1 = (frame1 * 255).astype(np.uint8)
    f2 = (frame2 * 255).astype(np.uint8)

    # Pad to square BEFORE flow (preserves aspect ratio!)
    if target_size is not None:
        f1 = pad_to_square(f1, fill_value=0)
        f2 = pad_to_square(f2, fill_value=0)

        # Resize if needed (square to square, no distortion)
        if f1.shape[0] != target_size:
            f1 = cv2.resize(f1, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            f2 = cv2.resize(f2, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    # Add biological saccades (AFTER padding, BEFORE flow)
    if add_saccade:
        f2 = add_biological_saccade(f2, saccade_rate=0.2, saccade_std=0.7)

    # Compute optical flow
    flow = None

    # Try DIS first (better for large displacement!)
    if method in ['auto', 'DIS']:
        try:
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            flow = dis.calc(f1, f2, None)
            if method == 'auto':
                # Only print once
                if not hasattr(compute_optical_flow_improved, '_dis_available'):
                    print("✅ Using DIS optical flow (fast, accurate for large displacement)")
                    compute_optical_flow_improved._dis_available = True
        except (AttributeError, cv2.error):
            if method == 'DIS':
                raise RuntimeError("DIS optical flow not available in this OpenCV build")
            # Fallback to Farneback
            if not hasattr(compute_optical_flow_improved, '_dis_unavailable'):
                print("⚠️  DIS unavailable, using Farneback (slower, less accurate for large motion)")
                compute_optical_flow_improved._dis_unavailable = True

    # Farneback fallback
    if flow is None:
        flow = cv2.calcOpticalFlowFarneback(
            f1, f2,
            None,
            pyr_scale=0.5,      # Pyramid scale
            levels=3,           # Pyramid levels
            winsize=15,         # Window size
            iterations=3,       # Iterations per level
            poly_n=5,           # Polynomial neighborhood size
            poly_sigma=1.2,     # Gaussian sigma for poly
            flags=0
        )

    # Convert to tensor (2, H, W)
    flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).float()

    return flow_tensor


# ============================================================================
# COMPARISON TOOL
# ============================================================================

def compare_flow_methods(env_name='ALE/Pong-v5', frameskip=10, target_size=210):
    """
    Compare different flow methods visually.

    Shows:
    1. Farneback (old)
    2. DIS (new)
    3. Difference between them
    """
    import gymnasium as gym
    import matplotlib.pyplot as plt
    from EXPERIMENTS.ACTIVE_INFERENCE.FLOW_INVERSE_MODEL import flow_to_rgb

    env = gym.make(env_name, frameskip=frameskip)

    # Get two frames with action
    frame1_raw, _ = env.reset()
    action = 2  # UP action in Pong
    for _ in range(frameskip):
        frame2_raw, _, _, _, _ = env.step(action)

    # Preprocess
    def to_gray(frame):
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
        return gray.astype(np.float32) / 255.0

    frame1 = to_gray(frame1_raw)
    frame2 = to_gray(frame2_raw)

    # Compute flow with both methods
    flow_farneback = compute_optical_flow_improved(
        frame1, frame2,
        target_size=target_size,
        add_saccade=False,
        method='farneback'
    )

    try:
        flow_dis = compute_optical_flow_improved(
            frame1, frame2,
            target_size=target_size,
            add_saccade=False,
            method='DIS'
        )
        dis_available = True
    except RuntimeError:
        dis_available = False
        flow_dis = flow_farneback  # Fallback

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Padded frames
    f1_padded = pad_to_square(frame1, fill_value=0)
    if f1_padded.shape[0] != target_size:
        f1_padded = cv2.resize(f1_padded, (target_size, target_size))

    axes[0, 0].imshow(f1_padded, cmap='gray')
    axes[0, 0].set_title(f'Frame (padded to {target_size}×{target_size})')
    axes[0, 0].axis('off')

    # Farneback
    axes[0, 1].imshow(flow_to_rgb(flow_farneback))
    axes[0, 1].set_title('Farneback (OLD)')
    axes[0, 1].axis('off')

    # DIS
    axes[0, 2].imshow(flow_to_rgb(flow_dis))
    axes[0, 2].set_title('DIS (NEW)' if dis_available else 'DIS (Unavailable)')
    axes[0, 2].axis('off')

    # Magnitude comparison
    def flow_magnitude(flow):
        return torch.sqrt((flow**2).sum(dim=0)).numpy()

    mag_farneback = flow_magnitude(flow_farneback)
    mag_dis = flow_magnitude(flow_dis)

    axes[1, 0].imshow(mag_farneback, cmap='jet')
    axes[1, 0].set_title('Farneback Magnitude')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(mag_dis, cmap='jet')
    axes[1, 1].set_title('DIS Magnitude')
    axes[1, 1].axis('off')

    if dis_available:
        diff = np.abs(mag_dis - mag_farneback)
        axes[1, 2].imshow(diff, cmap='hot')
        axes[1, 2].set_title('Difference (DIS - Farneback)')
        axes[1, 2].axis('off')
    else:
        axes[1, 2].text(0.5, 0.5, 'DIS not available\nin this OpenCV build',
                       ha='center', va='center', fontsize=12)
        axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

    # Statistics
    print("\n" + "="*70)
    print("FLOW COMPARISON")
    print("="*70)
    print(f"Farneback - max magnitude: {mag_farneback.max():.3f}")
    print(f"Farneback - mean magnitude: {mag_farneback.mean():.3f}")
    if dis_available:
        print(f"DIS - max magnitude: {mag_dis.max():.3f}")
        print(f"DIS - mean magnitude: {mag_dis.mean():.3f}")
        print(f"Difference - mean: {diff.mean():.3f} (lower = more similar)")
    print("="*70)

    env.close()


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("""
IMPROVED OPTICAL FLOW
=====================

Key improvements:
1. DIS optical flow (2-3× faster, better for large displacement)
2. Biological saccades (20% frequency, Gaussian distribution)
3. Padding preserves aspect ratio

To use in your code:
--------------------
from EXPERIMENTS.ACTIVE_INFERENCE.FLOW_COMPUTATION_IMPROVED import (
    compute_optical_flow_improved as compute_optical_flow
)

# Then use exactly like before:
flow = compute_optical_flow(frame1, frame2, target_size=210, add_saccade=True)

To compare methods:
-------------------
from EXPERIMENTS.ACTIVE_INFERENCE.FLOW_COMPUTATION_IMPROVED import compare_flow_methods
compare_flow_methods()
""")
