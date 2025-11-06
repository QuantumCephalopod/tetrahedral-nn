#!/usr/bin/env python3
"""
DEMO: WATCH BIOLOGICAL VISION WITH VISUALIZATION
=================================================

This shows you WHAT THE MODEL SEES as it learns!

Creates a visualization showing:
- Current attention window (where it's looking)
- Attention position on full frame
- Learning statistics
- Saccadic eye movements

USAGE:
    python DEMO_WATCH_BIOLOGICAL_VISION.py

Press 'q' to quit the visualization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import os
from collections import deque

from BIOLOGICAL_VISION_SYSTEM import (
    BiologicalAttention,
    BiologicalVisionProcessor,
    BiologicalVisionLearner,
    create_biological_vision_system
)


# ============================================================================
# CREATE TEST VIDEO
# ============================================================================

def create_test_video(output_path: str = "test_video.mp4", duration_seconds: int = 10):
    """Create a test video with interesting motion."""
    print("📹 Creating test video...")

    fps = 30
    width = 640
    height = 480
    total_frames = fps * duration_seconds

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_idx in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Multiple moving objects
        # Bouncing ball
        circle_x = int(width/2 + 200 * np.sin(frame_idx * 0.1))
        circle_y = int(height/2 + 150 * np.cos(frame_idx * 0.15))
        cv2.circle(frame, (circle_x, circle_y), 30, (0, 255, 255), -1)

        # Sliding rectangle
        rect_x = int((frame_idx * 5) % width)
        rect_y = int(height/3)
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + 50, rect_y + 50), (255, 0, 255), -1)

        # Rotating line
        angle = frame_idx * 0.1
        center = (width//2, height//2)
        end_x = int(center[0] + 150 * np.cos(angle))
        end_y = int(center[1] + 150 * np.sin(angle))
        cv2.line(frame, center, (end_x, end_y), (255, 255, 0), 3)

        out.write(frame)

    out.release()
    print(f"✓ Created: {output_path}")
    return output_path


# ============================================================================
# VISUAL DEMO
# ============================================================================

def visualize_biological_vision(video_path: str, max_frames: int = 300):
    """
    Show biological vision system in action with visualization.
    """
    print("\n" + "=" * 70)
    print("👁️  BIOLOGICAL VISION VISUALIZATION")
    print("=" * 70)
    print()
    print("WHAT YOU'LL SEE:")
    print("  - Full frame with attention window highlighted")
    print("  - Current attention window (what model sees)")
    print("  - Eye movement trail (saccadic path)")
    print("  - Learning statistics")
    print()
    print("Press 'q' to quit")
    print("=" * 70)
    print()

    # Create biological vision system
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    learner, attention = create_biological_vision_system(
        window_size=128,
        center_bias=0.7,
        learning_rate=0.0001,
        device=device
    )

    # Setup processor
    processor = BiologicalVisionProcessor(attention, grayscale=False, normalize=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return

    # Get video info
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    attention.set_frame_size((frame_width, frame_height))

    print(f"📹 Video: {frame_width}×{frame_height} @ {fps:.1f}fps")
    print(f"🧠 Window: {attention.window_size}")
    print()

    # Track eye movements
    position_history = deque(maxlen=50)
    frame_idx = 0
    learning_count = 0

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert for learning (grayscale, normalized)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray_norm = frame_gray.astype(np.float32) / 255.0

        # Get attention position (non-linear)
        position = attention.next_position()
        position_history.append(position)

        # Extract window
        window = attention.extract_window(frame_gray_norm, position)

        # Learn from window (if valid size)
        if window.shape[:2] == attention.window_size[::-1]:
            # Convert to 2D array for learning
            loss = learner.observe_frame(window)
            if loss is not None:
                learning_count += 1

        # ========================================
        # CREATE VISUALIZATION
        # ========================================

        # Make display frame (copy original)
        display_frame = frame.copy()

        # Draw eye movement trail
        if len(position_history) > 1:
            points = list(position_history)
            for i in range(len(points) - 1):
                p1 = (points[i][0] + attention.window_size[0]//2,
                      points[i][1] + attention.window_size[1]//2)
                p2 = (points[i+1][0] + attention.window_size[0]//2,
                      points[i+1][1] + attention.window_size[1]//2)
                alpha = i / len(points)  # Fade trail
                color = (int(255 * alpha), int(100 * alpha), int(50 * alpha))
                cv2.line(display_frame, p1, p2, color, 2)

        # Draw current attention window
        x, y = position
        win_w, win_h = attention.window_size
        cv2.rectangle(display_frame, (x, y), (x + win_w, y + win_h), (0, 255, 0), 3)

        # Draw center crosshair (fovea)
        center_x = (frame_width - win_w) // 2 + win_w // 2
        center_y = (frame_height - win_h) // 2 + win_h // 2
        cv2.drawMarker(display_frame, (center_x, center_y),
                      (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

        # Add info text
        cv2.putText(display_frame, f"Frame: {frame_idx}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Position: {position}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Updates: {learning_count}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if loss is not None:
            cv2.putText(display_frame, f"Loss: {loss:.6f}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Create window view (zoomed attention window)
        window_display = (window * 255).astype(np.uint8)
        window_display = cv2.cvtColor(window_display, cv2.COLOR_GRAY2BGR)
        window_display = cv2.resize(window_display, (256, 256))

        # Add label to window
        cv2.putText(window_display, "Attention Window", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Combine displays side by side
        # Resize full frame to fit
        display_height = 480
        display_width = int(display_height * frame_width / frame_height)
        display_frame = cv2.resize(display_frame, (display_width, display_height))

        # Create combined view
        combined = np.zeros((display_height, display_width + 256 + 20, 3), dtype=np.uint8)
        combined[:display_height, :display_width] = display_frame
        combined[:256, display_width + 20:] = window_display

        # Add title
        title_height = 50
        full_display = np.zeros((title_height + display_height, combined.shape[1], 3), dtype=np.uint8)
        full_display[title_height:] = combined

        cv2.putText(full_display, "BIOLOGICAL VISION: Time=Linear, Space=Non-Linear",
                   (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Show
        cv2.imshow('Biological Vision System', full_display)

        # Handle keyboard
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            print("\n⏹️  Stopped by user")
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 70)
    print("✅ VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"Frames processed: {frame_idx}")
    print(f"Learning updates: {learning_count}")
    if learning_count > 0:
        print(f"Final loss: {learner.current_loss:.6f}")
    print(f"Spatial diversity: {learner.spatial_diversity:.1f}")
    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" * 2)
    print("=" * 70)
    print("👁️  BIOLOGICAL VISION VISUALIZATION DEMO")
    print("=" * 70)
    print()

    # Check for video
    video_path = "test_video.mp4"
    if not os.path.exists(video_path):
        print("Creating test video...")
        video_path = create_test_video(video_path, duration_seconds=10)
        print()

    # Run visualization
    visualize_biological_vision(video_path, max_frames=300)

    print()
    print("🧠 This is how biology sees the world!")
    print("   Time flows linearly (frame by frame)")
    print("   Space is explored non-linearly (saccades)")
    print()


if __name__ == "__main__":
    main()
