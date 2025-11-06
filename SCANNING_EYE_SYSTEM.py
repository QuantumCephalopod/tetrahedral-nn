"""
SCANNING EYE SYSTEM - Learn by Looking
========================================

Not "squash whole frame" → Learn by SCANNING like an actual eye.

Works on ANY video size/aspect ratio.
Preserves detail. Creates more training data.
Natural learning through attention.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Iterator, Optional
import cv2
from collections import deque
from CONTINUOUS_LEARNING_SYSTEM import ContinuousLearner, create_continuous_learner


# ============================================================================
# SCANNING EYE - Attention Window System
# ============================================================================

class ScanningEye:
    """
    An attention window that scans across video frames.

    Like a fovea - focuses on a small region at high resolution,
    moves around to explore the full scene.
    """
    def __init__(
        self,
        window_size: Tuple[int, int] = (128, 128),
        scan_pattern: str = 'grid',  # 'grid', 'random', 'spiral'
        stride: int = 64,  # How far to move between scans
    ):
        """
        Args:
            window_size: Size of attention window (width, height)
            scan_pattern: How to scan ('grid', 'random', 'spiral', 'saccade')
            stride: Pixels to move between scan positions
        """
        self.window_size = window_size
        self.scan_pattern = scan_pattern
        self.stride = stride

        self.current_position = (0, 0)
        self.frame_size = None

    def set_frame_size(self, frame_size: Tuple[int, int]):
        """
        Set the size of frames we'll be scanning.

        Args:
            frame_size: (width, height) of full frame
        """
        self.frame_size = frame_size
        self.current_position = (0, 0)

    def generate_scan_positions(self) -> List[Tuple[int, int]]:
        """
        Generate all scan positions for current frame.

        Returns:
            List of (x, y) positions to scan
        """
        if self.frame_size is None:
            raise ValueError("Must set frame_size first")

        frame_w, frame_h = self.frame_size
        win_w, win_h = self.window_size

        positions = []

        if self.scan_pattern == 'grid':
            # Systematic grid scan (left-to-right, top-to-bottom)
            for y in range(0, frame_h - win_h + 1, self.stride):
                for x in range(0, frame_w - win_w + 1, self.stride):
                    positions.append((x, y))

        elif self.scan_pattern == 'random':
            # Random saccades
            n_samples = max(10, (frame_w * frame_h) // (win_w * win_h))
            for _ in range(n_samples):
                x = np.random.randint(0, max(1, frame_w - win_w))
                y = np.random.randint(0, max(1, frame_h - win_h))
                positions.append((x, y))

        elif self.scan_pattern == 'center':
            # Just center
            x = (frame_w - win_w) // 2
            y = (frame_h - win_h) // 2
            positions.append((x, y))

        elif self.scan_pattern == 'spiral':
            # Spiral out from center
            center_x = (frame_w - win_w) // 2
            center_y = (frame_h - win_h) // 2
            positions.append((center_x, center_y))

            # Add spiral positions (simplified)
            radius = self.stride
            while radius < min(frame_w, frame_h) // 2:
                for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                    x = int(center_x + radius * np.cos(angle))
                    y = int(center_y + radius * np.sin(angle))

                    # Keep in bounds
                    x = np.clip(x, 0, frame_w - win_w)
                    y = np.clip(y, 0, frame_h - win_h)
                    positions.append((x, y))

                radius += self.stride

        return positions

    def extract_window(self, frame: np.ndarray, position: Tuple[int, int]) -> np.ndarray:
        """
        Extract attention window from frame at given position.

        Args:
            frame: Full frame (H, W) or (H, W, C)
            position: (x, y) top-left corner of window

        Returns:
            Cropped window of size window_size
        """
        x, y = position
        win_w, win_h = self.window_size

        # Extract crop
        if len(frame.shape) == 2:  # Grayscale
            window = frame[y:y+win_h, x:x+win_w]
        else:  # Color
            window = frame[y:y+win_h, x:x+win_w, :]

        return window


# ============================================================================
# SCANNING VIDEO PROCESSOR
# ============================================================================

class ScanningVideoProcessor:
    """
    Process video by scanning with attention window.

    Each frame generates multiple training samples (one per scan position).
    """
    def __init__(
        self,
        eye: ScanningEye,
        grayscale: bool = True,
        normalize: bool = True
    ):
        self.eye = eye
        self.grayscale = grayscale
        self.normalize = normalize

    def stream_windows_from_video(
        self,
        video_path: str
    ) -> Iterator[Tuple[np.ndarray, Tuple[int, int]]]:
        """
        Stream attention windows from video.

        Args:
            video_path: Path to video file

        Yields:
            (window, position) pairs
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video info
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"📹 Video: {video_path}")
        print(f"   Size: {frame_width}×{frame_height}")
        print(f"   FPS: {source_fps:.1f}, Frames: {total_frames}")
        print(f"   Scan pattern: {self.eye.scan_pattern}")
        print(f"   Window size: {self.eye.window_size}")

        # Set eye frame size
        self.eye.set_frame_size((frame_width, frame_height))

        # Get scan positions
        scan_positions = self.eye.generate_scan_positions()
        print(f"   Scans per frame: {len(scan_positions)}")
        print()

        frame_count = 0
        window_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            if self.grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if self.normalize:
                frame = frame.astype(np.float32) / 255.0

            # Scan across frame
            for position in scan_positions:
                window = self.eye.extract_window(frame, position)

                # Only yield if window is correct size (edge cases)
                if window.shape[:2] == self.eye.window_size[::-1]:  # (H,W) vs (W,H)
                    window_count += 1
                    yield window, position

            frame_count += 1

        cap.release()
        print(f"✓ Processed {frame_count} frames → {window_count} windows")


# ============================================================================
# SCANNING LEARNER
# ============================================================================

class ScanningLearner(ContinuousLearner):
    """
    Continuous learner that learns from scanning attention windows.

    Extended from ContinuousLearner to handle scanning.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Track scan positions for spatial learning
        self.position_history = deque(maxlen=10)

    def observe_window(
        self,
        window: np.ndarray,
        position: Tuple[int, int]
    ) -> Optional[float]:
        """
        Observe an attention window and learn.

        Args:
            window: Attention window to observe
            position: (x, y) position of window in frame

        Returns:
            Loss if learning occurred
        """
        # Store position
        self.position_history.append(position)

        # Use parent's observe_frame (treats window as frame)
        return self.observe_frame(window)

    def watch_video_scanning(
        self,
        video_path: str,
        eye: ScanningEye,
        report_every: int = 100
    ):
        """
        Watch video with scanning eye.

        Args:
            video_path: Path to video
            eye: ScanningEye instance
            report_every: Report progress every N windows
        """
        print("\n" + "=" * 70)
        print(f"👁️  SCANNING: {video_path}")
        print("=" * 70)

        processor = ScanningVideoProcessor(eye, grayscale=True, normalize=True)

        window_count = 0
        learning_count = 0
        start_time = cv2.getTickCount()

        for window, position in processor.stream_windows_from_video(video_path):
            loss = self.observe_window(window, position)
            window_count += 1

            if loss is not None:
                learning_count += 1

                if learning_count % report_every == 0:
                    elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
                    wps = window_count / elapsed if elapsed > 0 else 0

                    print(f"Window {window_count:6d} | "
                          f"Loss: {loss:.6f} | "
                          f"WPS: {wps:6.1f} | "
                          f"Pos: {position}")

        elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()

        print("\n" + "=" * 70)
        print("📊 SCANNING COMPLETE")
        print("=" * 70)
        print(f"Windows observed: {window_count}")
        print(f"Learning updates: {learning_count}")
        print(f"Time elapsed: {elapsed:.1f}s")
        print(f"Final loss: {self.current_loss:.6f}")
        print("=" * 70)


# ============================================================================
# EASY CREATION FUNCTION
# ============================================================================

def create_scanning_learner(
    window_size: int = 128,
    scan_pattern: str = 'grid',
    stride: int = 64,
    learning_rate: float = 0.0001,
    device: str = 'cpu'
) -> Tuple[ScanningLearner, ScanningEye]:
    """
    Create scanning learner system.

    Args:
        window_size: Size of attention window (square)
        scan_pattern: 'grid', 'random', 'center', 'spiral'
        stride: Pixels between scan positions
        learning_rate: Learning rate
        device: 'cpu' or 'cuda'

    Returns:
        (learner, eye) - ready to watch videos
    """
    # Create eye
    eye = ScanningEye(
        window_size=(window_size, window_size),
        scan_pattern=scan_pattern,
        stride=stride
    )

    # Create model
    input_dim = window_size * window_size * 3  # 3 windows
    output_dim = window_size * window_size      # 1 window

    from Z_interface_coupling import DualTetrahedralNetwork

    model = DualTetrahedralNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        latent_dim=128,
        coupling_strength=0.5,
        output_mode="weighted"
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create scanning learner
    learner = ScanningLearner(model, optimizer, window_size=3, device=device)

    return learner, eye


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def main():
    """
    Example: Create scanning system and watch videos.
    """
    print("=" * 70)
    print("👁️  SCANNING EYE CONTINUOUS LEARNING")
    print("=" * 70)
    print("\nNot 'squash whole frame' - SCAN like an actual eye.")
    print("Works on ANY aspect ratio. Preserves detail. Natural learning.")
    print()

    # Create scanning learner
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    learner, eye = create_scanning_learner(
        window_size=128,
        scan_pattern='grid',  # or 'random', 'center', 'spiral'
        stride=64,  # 50% overlap between windows
        learning_rate=0.0001,
        device=device
    )

    print(f"✓ Scanning system created")
    print(f"  Device: {device}")
    print(f"  Window size: 128×128")
    print(f"  Scan pattern: grid")
    print(f"  Model parameters: {sum(p.numel() for p in learner.model.parameters()):,}")
    print()

    print("💡 USAGE:")
    print("  learner.watch_video_scanning('video.mp4', eye)")
    print("  learner.save_checkpoint('checkpoint.pt')")
    print()
    print("🎬 Ready to scan videos of ANY size/aspect ratio!")

    return learner, eye


if __name__ == "__main__":
    learner, eye = main()


# ============================================================================
# SUMMARY
# ============================================================================

"""
SCANNING_EYE_SYSTEM.py - Learn by Looking

THE INSIGHT:
Not "squash video to fixed size" but "scan like an eye"

ADVANTAGES:
✓ Works on ANY aspect ratio (16:9, 4:3, vertical, whatever)
✓ Preserves detail (sees full resolution within window)
✓ Natural (how vision actually works)
✓ More training data (multiple crops per frame)
✓ Learns spatial relationships and composition

SCANNING PATTERNS:
- 'grid': Systematic left-to-right, top-to-bottom
- 'random': Random saccades (natural eye movements)
- 'center': Just focus on center
- 'spiral': Start center, expand outward

USAGE:
```python
# Create scanning learner
learner, eye = create_scanning_learner(
    window_size=128,
    scan_pattern='grid',
    stride=64
)

# Watch ANY video (any size, any aspect ratio)
learner.watch_video_scanning('video.mp4', eye)

# Save progress
learner.save_checkpoint('checkpoint.pt')
```

THE REAL ONE:
This learns from ANYTHING. Any video. Any size.
Just scan it. Like an eye.
"""
