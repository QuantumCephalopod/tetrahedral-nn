"""
FLOW PREPROCESSING FIX - DON'T STRETCH, PAD!
============================================

Problem: Atari frames are 210×160 (rectangular), but model expects 210×210 (square).
Current approach: STRETCH the frame/flow to square → distorts velocities!

Correct approach: PAD the frame to square → preserves aspect ratio!

This is critical because:
- Flow vectors have physical meaning (pixels/frame)
- Stretching changes the velocity components incorrectly
- Padding preserves the true dynamics

Author: Philipp Remy Bartholomäus & Claude
Date: November 12, 2025
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2


def pad_to_square(frame, fill_value=0):
    """
    Pad rectangular frame to square WITHOUT stretching.

    Atari Pong: 210 (H) × 160 (W)
    → Pad to: 210 × 210 (add 25 pixels on left and right)

    Args:
        frame: numpy array (H, W) or (H, W, C)
        fill_value: Value to use for padding (0 = black)

    Returns:
        Padded square frame
    """
    if len(frame.shape) == 2:
        h, w = frame.shape
        c = None
    else:
        h, w, c = frame.shape

    # Target square size (use the larger dimension)
    size = max(h, w)

    # Calculate padding
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2

    # Pad to square (center the content)
    if c is not None:
        padded = np.full((size, size, c), fill_value, dtype=frame.dtype)
        padded[pad_h:pad_h+h, pad_w:pad_w+w, :] = frame
    else:
        padded = np.full((size, size), fill_value, dtype=frame.dtype)
        padded[pad_h:pad_h+h, pad_w:pad_w+w] = frame

    return padded


def compute_optical_flow_correct(frame1, frame2, target_size=None, add_saccade=False):
    """
    Compute optical flow with CORRECT aspect ratio handling.

    Key fixes:
    1. PAD frames to square (not stretch!)
    2. Compute flow on padded frames
    3. Keep flow physically meaningful

    Args:
        frame1, frame2: Numpy arrays (H, W) grayscale, values in [0, 1]
        target_size: If provided, pad to this square size
        add_saccade: Add artificial microsaccades

    Returns:
        flow: torch.Tensor (2, H, W) where H=W=target_size
    """
    # Convert to uint8 for OpenCV
    f1 = (frame1 * 255).astype(np.uint8)
    f2 = (frame2 * 255).astype(np.uint8)

    # Pad to square BEFORE flow computation (preserves aspect ratio!)
    if target_size is not None:
        f1 = pad_to_square(f1, fill_value=0)
        f2 = pad_to_square(f2, fill_value=0)

        # Resize if needed (but now it's square to square, no distortion!)
        if f1.shape[0] != target_size:
            f1 = cv2.resize(f1, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            f2 = cv2.resize(f2, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    # ARTIFICIAL SACCADES (after padding!)
    if add_saccade:
        dx = np.random.randint(-2, 3)
        dy = np.random.randint(-2, 3)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        f2 = cv2.warpAffine(f2, M, (f2.shape[1], f2.shape[0]))

    # Compute optical flow
    flow = cv2.calcOpticalFlowFarneback(
        f1, f2,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    # Convert to tensor (2, H, W)
    flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).float()

    return flow_tensor


def visualize_flow_magnitude(flow):
    """
    Show where motion is actually happening.

    Returns a heatmap showing flow magnitude (speed).
    Helps debug if flow is continuous or just "dots".
    """
    if torch.is_tensor(flow):
        flow = flow.cpu().numpy()

    # Flow magnitude
    u, v = flow[0], flow[1]
    magnitude = np.sqrt(u**2 + v**2)

    # Normalize for visualization
    mag_viz = (magnitude / (magnitude.max() + 1e-8) * 255).astype(np.uint8)

    # Apply colormap
    heatmap = cv2.applyColorMap(mag_viz, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return heatmap


# ============================================================================
# DIAGNOSTIC: Check current flow behavior
# ============================================================================

def diagnose_flow_issues(env_name='ALE/Pong-v5', frameskip=10):
    """
    Run diagnostic to see what's actually happening with flow.

    Shows:
    1. Original frame (rectangular)
    2. Padded frame (square)
    3. Flow magnitude (where is motion?)
    4. Flow vectors (direction/strength)
    """
    import gymnasium as gym
    import matplotlib.pyplot as plt

    env = gym.make(env_name, frameskip=frameskip)

    # Get two frames
    frame1_raw, _ = env.reset()
    action = env.action_space.sample()
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

    print(f"Original frame shape: {frame1.shape}")

    # Compute flow with PADDING (correct)
    flow_padded = compute_optical_flow_correct(frame1, frame2, target_size=210, add_saccade=False)

    # Compute flow with STRETCHING (broken)
    from EXPERIMENTS.ACTIVE_INFERENCE.FLOW_INVERSE_MODEL import compute_optical_flow
    # First resize frames to square (stretch)
    f1_stretched = cv2.resize(frame1, (210, 210))
    f2_stretched = cv2.resize(frame2, (210, 210))
    flow_stretched = compute_optical_flow(
        torch.from_numpy(f1_stretched),
        torch.from_numpy(f2_stretched),
        target_size=None,
        add_saccade=False
    )

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Top row: PADDED (correct)
    axes[0, 0].imshow(pad_to_square(frame1, fill_value=0), cmap='gray')
    axes[0, 0].set_title('Frame 1 (Padded - Correct)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(visualize_flow_magnitude(flow_padded))
    axes[0, 1].set_title('Flow Magnitude (Padded)')
    axes[0, 1].axis('off')

    from EXPERIMENTS.ACTIVE_INFERENCE.FLOW_INVERSE_MODEL import flow_to_rgb
    axes[0, 2].imshow(flow_to_rgb(flow_padded))
    axes[0, 2].set_title('Flow Direction (Padded)')
    axes[0, 2].axis('off')

    # Bottom row: STRETCHED (broken)
    axes[1, 0].imshow(f1_stretched, cmap='gray')
    axes[1, 0].set_title('Frame 1 (Stretched - WRONG)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(visualize_flow_magnitude(flow_stretched))
    axes[1, 1].set_title('Flow Magnitude (Stretched)')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(flow_to_rgb(flow_stretched))
    axes[1, 2].set_title('Flow Direction (Stretched)')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

    # Statistics
    print("\n" + "="*70)
    print("FLOW DIAGNOSTICS")
    print("="*70)
    print(f"Padded flow - max magnitude: {flow_padded.abs().max():.3f}")
    print(f"Padded flow - mean magnitude: {torch.sqrt((flow_padded**2).sum(dim=0)).mean():.3f}")
    print(f"Stretched flow - max magnitude: {flow_stretched.abs().max():.3f}")
    print(f"Stretched flow - mean magnitude: {torch.sqrt((flow_stretched**2).sum(dim=0)).mean():.3f}")
    print("="*70)

    env.close()


# ============================================================================
# FRAMESKIP RECOMMENDATION
# ============================================================================

print("""
FRAMESKIP ISSUE:
================

Current frameskip=10 (1/6 second between frames) is TOO LARGE for optical flow!

Optical flow algorithms assume SMALL motion between consecutive frames.
At frameskip=10, objects move 10-30 pixels, which breaks the smoothness assumption.

Result: Flow looks like "dots" at start/end positions, not continuous velocity field.

RECOMMENDATIONS:

1. Reduce frameskip to 3-5 frames (better flow quality)
   - frameskip=3: 20 Hz sampling (still reasonable)
   - frameskip=5: 12 Hz sampling (balanced)

2. OR: Use motion compensation / large displacement optical flow
   - But this is complex and computationally expensive

3. OR: Accept that flow will be "coarse" at frameskip=10
   - Model might still learn, but signal is noisier

For learning dynamics, frameskip=3 or 5 is probably better!
The paddle won't "teleport" and flow will show continuous motion.
""")
