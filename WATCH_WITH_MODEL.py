"""
WATCH WITH THE MODEL - See What It Sees
========================================

Visualize the model's learning in real-time.
See through its eyes as it scans and learns.
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
import time
from typing import Optional, Tuple

from SCANNING_EYE_SYSTEM import ScanningLearner, ScanningEye, ScanningVideoProcessor


# ============================================================================
# LIVE VISUALIZATION
# ============================================================================

class ModelVisionVisualizer:
    """
    Visualize what the model sees and predicts in real-time.

    Shows:
    - Current attention window (what it's looking at)
    - Model prediction (what it thinks comes next)
    - Ground truth (what actually happens)
    - Error map (where it's wrong)
    """
    def __init__(self, learner: ScanningLearner):
        self.learner = learner

        # For Colab display
        self.fig = None
        self.axes = None

    def watch_with_model_notebook(
        self,
        video_path: str,
        eye: ScanningEye,
        max_frames: int = 100,
        update_every: int = 10
    ):
        """
        Watch video with model visualization in Jupyter/Colab.

        Args:
            video_path: Path to video
            eye: ScanningEye instance
            max_frames: Max frames to process
            update_every: Update visualization every N windows
        """
        from IPython.display import clear_output

        processor = ScanningVideoProcessor(eye, grayscale=True, normalize=True)

        print("👁️  WATCHING WITH MODEL...")
        print("=" * 70)

        window_count = 0
        vis_count = 0

        for window, position in processor.stream_windows_from_video(video_path):
            loss = self.learner.observe_window(window, position)
            window_count += 1

            # Visualize periodically
            if loss is not None and window_count % update_every == 0:
                vis_count += 1

                # Get prediction
                with torch.no_grad():
                    # Need 3 frames in buffer to predict
                    if len(self.learner.frame_buffer) >= 4:
                        input_frames = list(self.learner.frame_buffer)[:3]
                        target_frame = self.learner.frame_buffer[3]

                        # Flatten and predict
                        input_flat = np.concatenate([f.flatten() for f in input_frames])
                        input_tensor = torch.tensor(input_flat, dtype=torch.float32).unsqueeze(0)

                        pred_tensor = self.learner.model(input_tensor)
                        pred_frame = pred_tensor.view(eye.window_size[1], eye.window_size[0]).numpy()

                        # Clear and show
                        clear_output(wait=True)

                        # Create visualization
                        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

                        # Row 1: Current window, Prediction, Ground truth
                        axes[0, 0].imshow(window, cmap='gray', vmin=0, vmax=1)
                        axes[0, 0].set_title(f'Looking At\nPos: {position}', fontsize=12)
                        axes[0, 0].axis('off')

                        axes[0, 1].imshow(pred_frame, cmap='gray', vmin=0, vmax=1)
                        axes[0, 1].set_title(f'Model Predicts\nLoss: {loss:.6f}', fontsize=12)
                        axes[0, 1].axis('off')

                        axes[0, 2].imshow(target_frame, cmap='gray', vmin=0, vmax=1)
                        axes[0, 2].set_title('Ground Truth', fontsize=12)
                        axes[0, 2].axis('off')

                        # Row 2: Previous frames, Error map, Stats
                        if len(input_frames) >= 2:
                            axes[1, 0].imshow(input_frames[-2], cmap='gray', vmin=0, vmax=1)
                            axes[1, 0].set_title('Previous Frame', fontsize=12)
                            axes[1, 0].axis('off')

                        # Error map
                        error_map = np.abs(pred_frame - target_frame)
                        axes[1, 1].imshow(error_map, cmap='hot', vmin=0, vmax=1)
                        axes[1, 1].set_title(f'Error Map\nMax: {error_map.max():.4f}', fontsize=12)
                        axes[1, 1].axis('off')

                        # Stats
                        axes[1, 2].axis('off')
                        stats_text = f"""
LEARNING STATS

Windows Seen: {window_count}
Updates: {self.learner.total_updates}
Current Loss: {loss:.6f}
Avg Loss (last 100): {np.mean(self.learner.loss_history[-100:]):.6f}

Position: {position}
Scan Pattern: {eye.scan_pattern}
Window Size: {eye.window_size}

Learning Rate: {self.learner.optimizer.param_groups[0]['lr']:.6f}
                        """
                        axes[1, 2].text(0.1, 0.5, stats_text.strip(),
                                       fontsize=10, family='monospace',
                                       verticalalignment='center')

                        plt.tight_layout()
                        plt.show()

                        print(f"Window {window_count} | Loss: {loss:.6f} | Pos: {position}")

            # Stop after max_frames
            if window_count >= max_frames:
                break

        print("\n✓ Visualization complete!")

    def create_side_by_side_video(
        self,
        video_path: str,
        eye: ScanningEye,
        output_path: str = "model_vision.mp4",
        max_windows: int = 500,
        fps: int = 10
    ):
        """
        Create a video showing model's vision side-by-side with predictions.

        Args:
            video_path: Input video
            eye: ScanningEye
            output_path: Where to save output video
            max_windows: Max windows to process
            fps: Output video FPS
        """
        processor = ScanningVideoProcessor(eye, grayscale=True, normalize=True)

        print("🎬 Creating model vision video...")

        # Setup video writer
        frame_size = (eye.window_size[0] * 3, eye.window_size[1] * 2)  # 3 cols, 2 rows
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        window_count = 0

        for window, position in processor.stream_windows_from_video(video_path):
            loss = self.learner.observe_window(window, position)

            if loss is not None and len(self.learner.frame_buffer) >= 4:
                # Get frames
                input_frames = list(self.learner.frame_buffer)[:3]
                target_frame = self.learner.frame_buffer[3]

                # Get prediction
                with torch.no_grad():
                    input_flat = np.concatenate([f.flatten() for f in input_frames])
                    input_tensor = torch.tensor(input_flat, dtype=torch.float32).unsqueeze(0)
                    pred_tensor = self.learner.model(input_tensor)
                    pred_frame = pred_tensor.view(eye.window_size[1], eye.window_size[0]).numpy()

                # Create visualization frame
                vis_frame = self._create_vis_frame(
                    window, pred_frame, target_frame,
                    input_frames[-2] if len(input_frames) >= 2 else window,
                    loss, position
                )

                # Write frame
                out.write(vis_frame)

                window_count += 1

                if window_count % 50 == 0:
                    print(f"  Processed {window_count} windows...")

                if window_count >= max_windows:
                    break

        out.release()
        print(f"✓ Saved model vision video: {output_path}")

    def _create_vis_frame(
        self,
        current_window: np.ndarray,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        previous_window: np.ndarray,
        loss: float,
        position: Tuple[int, int]
    ) -> np.ndarray:
        """Create a single visualization frame."""
        h, w = current_window.shape

        # Create canvas (2 rows, 3 cols)
        canvas = np.zeros((h * 2, w * 3), dtype=np.uint8)

        # Convert to uint8 for opencv
        def to_uint8(img):
            return (np.clip(img, 0, 1) * 255).astype(np.uint8)

        # Row 1
        canvas[0:h, 0:w] = to_uint8(current_window)
        canvas[0:h, w:2*w] = to_uint8(prediction)
        canvas[0:h, 2*w:3*w] = to_uint8(ground_truth)

        # Row 2
        canvas[h:2*h, 0:w] = to_uint8(previous_window)

        # Error map
        error_map = np.abs(prediction - ground_truth)
        error_colored = cv2.applyColorMap(to_uint8(error_map), cv2.COLORMAP_HOT)
        canvas_color = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        canvas_color[h:2*h, w:2*w] = error_colored

        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas_color, f"Looking At {position}", (10, 30),
                   font, 0.5, (255, 255, 255), 1)
        cv2.putText(canvas_color, f"Prediction L:{loss:.4f}", (w+10, 30),
                   font, 0.5, (255, 255, 255), 1)
        cv2.putText(canvas_color, "Ground Truth", (2*w+10, 30),
                   font, 0.5, (255, 255, 255), 1)
        cv2.putText(canvas_color, "Previous", (10, h+30),
                   font, 0.5, (255, 255, 255), 1)
        cv2.putText(canvas_color, "Error Map", (w+10, h+30),
                   font, 0.5, (255, 255, 255), 1)

        return canvas_color


# ============================================================================
# EASY USAGE FUNCTIONS
# ============================================================================

def watch_with_model_live(
    learner: ScanningLearner,
    eye: ScanningEye,
    video_path: str,
    max_frames: int = 100,
    update_every: int = 10
):
    """
    Watch video WITH the model in Colab (live visualization).

    Args:
        learner: ScanningLearner instance
        eye: ScanningEye instance
        video_path: Video to watch
        max_frames: Max windows to process
        update_every: Update visualization every N windows
    """
    visualizer = ModelVisionVisualizer(learner)
    visualizer.watch_with_model_notebook(video_path, eye, max_frames, update_every)


def create_model_vision_video(
    learner: ScanningLearner,
    eye: ScanningEye,
    video_path: str,
    output_path: str = "model_vision.mp4",
    max_windows: int = 500
):
    """
    Create a video showing what the model sees.

    Args:
        learner: ScanningLearner instance
        eye: ScanningEye instance
        video_path: Input video
        output_path: Where to save output
        max_windows: Max windows to process
    """
    visualizer = ModelVisionVisualizer(learner)
    visualizer.create_side_by_side_video(video_path, eye, output_path, max_windows)


# ============================================================================
# SUMMARY
# ============================================================================

"""
WATCH_WITH_MODEL.py - See Through The Model's Eyes

THE COOLEST FEATURE:
Watch videos THROUGH the model as it learns.
See what it sees, what it predicts, where it's wrong.

USAGE IN COLAB:

# After creating learner and eye:
watch_with_model_live(
    learner=learner,
    eye=eye,
    video_path="your_video.mp4",
    max_frames=100,
    update_every=10  # Update viz every 10 windows
)

WHAT YOU SEE:
┌─────────────┬─────────────┬─────────────┐
│ Looking At  │ Prediction  │ Ground Truth│
│  (current)  │ (what model │ (what really│
│             │   thinks)   │   happens)  │
├─────────────┼─────────────┼─────────────┤
│ Previous    │ Error Map   │   Stats     │
│  (memory)   │ (mistakes)  │ (progress)  │
└─────────────┴─────────────┴─────────────┘

SUPER INTUITIVE:
- Red in error map = model was very wrong
- Dark error map = model predicting well
- Watch loss go down as it learns
- See spatial patterns emerge

You literally SEE the model learning!
"""
