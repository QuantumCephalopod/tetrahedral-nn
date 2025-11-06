#!/usr/bin/env python3
"""
DEMO: BIOLOGICAL VISION SYSTEM
===============================

Just run this file and WATCH IT GO!

No setup needed - it creates a test video and learns from it.

WHAT YOU'LL SEE:
- Model watching video with biological time-space coupling
- Time is LINEAR (frame by frame)
- Space is NON-LINEAR (saccadic eye movements)
- Real-time learning progress

USAGE:
    python DEMO_BIOLOGICAL_VISION.py

That's it!
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import os
from pathlib import Path

# Import our biological vision system
from BIOLOGICAL_VISION_SYSTEM import create_biological_vision_system


# ============================================================================
# STEP 1: CREATE A TEST VIDEO (if you don't have one)
# ============================================================================

def create_test_video(output_path: str = "test_video.mp4", duration_seconds: int = 5):
    """
    Create a simple test video with moving shapes.

    You can skip this if you already have a video!
    """
    print("=" * 70)
    print("📹 CREATING TEST VIDEO")
    print("=" * 70)

    # Video settings
    fps = 30
    width = 640
    height = 480
    total_frames = fps * duration_seconds

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Creating {duration_seconds}s video at {fps}fps ({total_frames} frames)")
    print(f"Size: {width}×{height}")

    for frame_idx in range(total_frames):
        # Create blank frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Moving circle (bouncing ball)
        circle_x = int(width/2 + 200 * np.sin(frame_idx * 0.1))
        circle_y = int(height/2 + 150 * np.cos(frame_idx * 0.15))
        cv2.circle(frame, (circle_x, circle_y), 30, (0, 255, 255), -1)

        # Moving rectangle
        rect_x = int((frame_idx * 5) % width)
        rect_y = int(height/2 + 100 * np.sin(frame_idx * 0.05))
        cv2.rectangle(frame,
                     (rect_x, rect_y),
                     (rect_x + 50, rect_y + 50),
                     (255, 0, 255), -1)

        # Rotating line
        angle = frame_idx * 0.1
        center = (width//2, height//2)
        end_x = int(center[0] + 150 * np.cos(angle))
        end_y = int(center[1] + 150 * np.sin(angle))
        cv2.line(frame, center, (end_x, end_y), (255, 255, 0), 3)

        # Add frame number
        cv2.putText(frame, f"Frame {frame_idx}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(frame)

        if (frame_idx + 1) % fps == 0:
            print(f"  Written {frame_idx + 1}/{total_frames} frames...")

    out.release()
    print(f"✓ Test video created: {output_path}")
    print()
    return output_path


# ============================================================================
# STEP 2: CREATE BIOLOGICAL VISION SYSTEM
# ============================================================================

def setup_biological_system():
    """
    Create the biological vision system.
    """
    print("=" * 70)
    print("🧠 CREATING BIOLOGICAL VISION SYSTEM")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    learner, attention = create_biological_vision_system(
        window_size=128,        # Size of attention window
        center_bias=0.7,        # 70% center-focused, 30% exploratory
        learning_rate=0.0001,   # Learning rate
        device=device
    )

    print(f"✓ System created")
    print(f"  Device: {device}")
    print(f"  Window: 128×128 pixels")
    print(f"  Center bias: 70% (like a fovea)")
    print(f"  Model params: {sum(p.numel() for p in learner.model.parameters()):,}")
    print()

    return learner, attention


# ============================================================================
# STEP 3: WATCH VIDEO AND LEARN
# ============================================================================

def watch_video_with_biological_vision(learner, attention, video_path: str):
    """
    Watch video with biological vision system.
    """
    print("=" * 70)
    print("👁️  WATCHING VIDEO WITH BIOLOGICAL VISION")
    print("=" * 70)
    print()
    print("Time is LINEAR → Space is NON-LINEAR")
    print("This is not machine learning. This is BIOLOGY.")
    print()

    # Watch the video
    learner.watch_video_biological(
        video_path=video_path,
        attention=attention,
        report_every=30  # Report progress every 30 frames
    )


# ============================================================================
# STEP 4: SAVE RESULTS
# ============================================================================

def save_model(learner, checkpoint_path: str = "biological_vision_checkpoint.pt"):
    """
    Save the trained model.
    """
    print()
    print("=" * 70)
    print("💾 SAVING MODEL")
    print("=" * 70)

    learner.save_checkpoint(checkpoint_path)
    print(f"✓ Model saved: {checkpoint_path}")
    print()


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    """
    Run the complete demo.
    """
    print("\n" * 2)
    print("=" * 70)
    print("🧠 BIOLOGICAL VISION SYSTEM DEMO")
    print("=" * 70)
    print()
    print("This demo will:")
    print("  1. Create a test video (or use existing one)")
    print("  2. Create biological vision system")
    print("  3. Watch video with LINEAR time + NON-LINEAR space")
    print("  4. Save the trained model")
    print()
    print("=" * 70)
    print()

    # Check if we have a video already
    video_path = "test_video.mp4"

    if os.path.exists(video_path):
        print(f"✓ Found existing video: {video_path}")
        print()
    else:
        print("No test video found, creating one...")
        video_path = create_test_video(video_path, duration_seconds=5)

    # Create biological vision system
    learner, attention = setup_biological_system()

    # Watch video
    watch_video_with_biological_vision(learner, attention, video_path)

    # Save model
    save_model(learner)

    # Done!
    print("=" * 70)
    print("✅ DEMO COMPLETE!")
    print("=" * 70)
    print()
    print("WHAT JUST HAPPENED:")
    print("  - Model watched video with biological vision")
    print("  - Time: LINEAR (frame by frame, tick-tock)")
    print("  - Space: NON-LINEAR (saccades, center-biased)")
    print("  - Each time step = ONE frame + ONE attention point")
    print()
    print("The model learned:")
    print(f"  - Final loss: {learner.current_loss:.6f}")
    print(f"  - Total updates: {learner.total_updates}")
    print(f"  - Spatial diversity: {learner.spatial_diversity:.1f}")
    print()
    print("Model saved to: biological_vision_checkpoint.pt")
    print()
    print("🧠 This is how biology learns from vision!")
    print("=" * 70)


if __name__ == "__main__":
    main()
