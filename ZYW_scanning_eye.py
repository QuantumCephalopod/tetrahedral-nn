"""
ZYW - SCANNING EYE SYSTEM (Geometry subdivision of temporal)
============================================================

NOT "squash whole frame" - SCAN like an actual eye.
Works on ANY video size/aspect ratio.

This is ZY (temporal) + W (geometry) = geometric attention patterns.

GEOMETRIC PATTERNS:
- Grid scanning (systematic exploration)
- Random saccades (exploratory)
- Spiral (center-outward)
- Center-focus (foveal)

Each pattern = different geometric topology of attention.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Iterator, Optional, Dict
import cv2
from collections import deque

# Import base components
from Z_interface_coupling import DualTetrahedralNetwork


# ============================================================================
# CONTINUOUS LEARNER BASE CLASS (embedded for Colab self-containment)
# ============================================================================

class ContinuousLearner:
    """
    Base class for continuous learning from frame streams.

    Handles:
    - Frame buffering (sliding window)
    - Learning from sequences
    - Checkpointing
    - Learning rate scheduling
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        window_size: int = 3,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.window_size = window_size
        self.device = device

        # Frame buffer (sliding window)
        self.frame_buffer = deque(maxlen=window_size + 1)

        # Learning statistics
        self.total_updates = 0
        self.current_loss = 0.0
        self.loss_history = []

        # Learning rate scheduling
        self.initial_lr = optimizer.param_groups[0]['lr']

    def observe_frame(self, frame: np.ndarray) -> Optional[float]:
        """Observe a single frame and learn if possible."""
        self.frame_buffer.append(frame)

        # Need full window + target to learn
        if len(self.frame_buffer) < self.window_size + 1:
            return None

        # Create input window and target
        input_frames = list(self.frame_buffer)[:self.window_size]
        target_frame = self.frame_buffer[-1]

        # Flatten and concatenate
        input_flat = np.concatenate([f.flatten() for f in input_frames])
        target_flat = target_frame.flatten()

        # Convert to tensors
        input_tensor = torch.tensor(input_flat, dtype=torch.float32).unsqueeze(0).to(self.device)
        target_tensor = torch.tensor(target_flat, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Learn from this observation
        return self._learn_step(input_tensor, target_tensor)

    def _learn_step(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> float:
        """Single learning step."""
        self.model.train()

        # Forward
        output = self.model(input_tensor)
        loss = nn.MSELoss()(output, target_tensor)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update statistics
        self.total_updates += 1
        self.current_loss = loss.item()
        self.loss_history.append(self.current_loss)

        # Adaptive learning rate
        if self.total_updates % 1000 == 0:
            self._adjust_learning_rate()

        return self.current_loss

    def _adjust_learning_rate(self):
        """Adjust learning rate based on progress."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.95

    def save_checkpoint(self, path: str):
        """Save current state."""
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'total_updates': self.total_updates,
            'loss_history': self.loss_history
        }
        torch.save(checkpoint, path)
        print(f"✓ Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load previous state."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.total_updates = checkpoint['total_updates']
        self.loss_history = checkpoint['loss_history']
        print(f"✓ Checkpoint loaded: {path}")
        print(f"   Resuming from update {self.total_updates}")


# ============================================================================
# SCANNING EYE - Geometric Attention Patterns
# ============================================================================

class ScanningEye:
    """
    Geometric attention window that scans across frames.

    Like a fovea - focuses on small region at high resolution,
    moves around to explore the full scene.

    GEOMETRIC PATTERNS:
    - 'grid': Systematic left-to-right, top-to-bottom
    - 'random': Random saccades across frame
    - 'spiral': Center-outward spiral pattern
    - 'center': Just focus on center
    """

    def __init__(
        self,
        window_size: Tuple[int, int] = (128, 128),
        scan_pattern: str = 'grid',
        stride: int = 64,
    ):
        """
        Args:
            window_size: Size of attention window (width, height)
            scan_pattern: Geometric pattern ('grid', 'random', 'spiral', 'center')
            stride: Pixels to move between scan positions
        """
        self.window_size = window_size
        self.scan_pattern = scan_pattern
        self.stride = stride

        self.current_position = (0, 0)
        self.frame_size = None

    def set_frame_size(self, frame_size: Tuple[int, int]):
        """Set the size of frames to scan."""
        self.frame_size = frame_size
        self.current_position = (0, 0)

    def generate_scan_positions(self) -> List[Tuple[int, int]]:
        """
        Generate all scan positions using geometric pattern.

        Returns:
            List of (x, y) positions
        """
        if self.frame_size is None:
            raise ValueError("Must set frame_size first")

        frame_w, frame_h = self.frame_size
        win_w, win_h = self.window_size

        positions = []

        if self.scan_pattern == 'grid':
            # Systematic grid scan (geometric regularity)
            for y in range(0, frame_h - win_h + 1, self.stride):
                for x in range(0, frame_w - win_w + 1, self.stride):
                    positions.append((x, y))

        elif self.scan_pattern == 'random':
            # Random saccades (exploration geometry)
            n_samples = max(10, (frame_w * frame_h) // (win_w * win_h))
            for _ in range(n_samples):
                x = np.random.randint(0, max(1, frame_w - win_w))
                y = np.random.randint(0, max(1, frame_h - win_h))
                positions.append((x, y))

        elif self.scan_pattern == 'center':
            # Center focus (foveal geometry)
            x = (frame_w - win_w) // 2
            y = (frame_h - win_h) // 2
            positions.append((x, y))

        elif self.scan_pattern == 'spiral':
            # Spiral geometry (center-outward)
            center_x = (frame_w - win_w) // 2
            center_y = (frame_h - win_h) // 2
            positions.append((center_x, center_y))

            radius = self.stride
            while radius < min(frame_w, frame_h) // 2:
                for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                    x = int(center_x + radius * np.cos(angle))
                    y = int(center_y + radius * np.sin(angle))

                    x = np.clip(x, 0, frame_w - win_w)
                    y = np.clip(y, 0, frame_h - win_h)
                    positions.append((x, y))

                radius += self.stride

        return positions

    def extract_window(self, frame: np.ndarray, position: Tuple[int, int]) -> np.ndarray:
        """Extract attention window from frame at position."""
        x, y = position
        win_w, win_h = self.window_size

        if len(frame.shape) == 2:  # Grayscale
            return frame[y:y+win_h, x:x+win_w]
        else:  # Color
            return frame[y:y+win_h, x:x+win_w, :]


# ============================================================================
# SCANNING VIDEO PROCESSOR
# ============================================================================

class ScanningVideoProcessor:
    """
    Process video by scanning with geometric attention patterns.

    Each frame → multiple training samples (one per scan position).
    Learns spatial structure through geometric exploration.
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

            # Scan across frame using geometric pattern
            for position in scan_positions:
                window = self.eye.extract_window(frame, position)

                if window.shape[:2] == self.eye.window_size[::-1]:
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
    Continuous learner with geometric scanning attention.

    Learns from scanning windows - spatial structure emerges
    from geometric exploration patterns.
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
            window: Attention window
            position: (x, y) position in frame

        Returns:
            Loss if learning occurred
        """
        self.position_history.append(position)
        return self.observe_frame(window)

    def watch_video_scanning(
        self,
        video_path: str,
        eye: ScanningEye,
        report_every: int = 100
    ):
        """
        Watch video with geometric scanning.

        Args:
            video_path: Path to video
            eye: ScanningEye with geometric pattern
            report_every: Report every N windows
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
# CREATION FUNCTION
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
        scan_pattern: Geometric pattern ('grid', 'random', 'center', 'spiral')
        stride: Pixels between scan positions
        learning_rate: Learning rate
        device: 'cpu' or 'cuda'

    Returns:
        (learner, eye) ready to watch videos
    """
    # Create scanning eye with geometric pattern
    eye = ScanningEye(
        window_size=(window_size, window_size),
        scan_pattern=scan_pattern,
        stride=stride
    )

    # Create model
    input_dim = window_size * window_size * 3
    output_dim = window_size * window_size

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
# SUMMARY
# ============================================================================

"""
ZYW_scanning_eye.py - Geometric Attention for Temporal Learning

FRACTAL STRUCTURE:
ZY = temporal adapter
W = geometry dimension
ZYW = temporal learning with geometric attention patterns

KEY INSIGHT:
Not "squash frame to fixed size" but "scan with geometric patterns"

GEOMETRIC PATTERNS:
- Grid: Systematic, complete coverage
- Random: Exploratory, diverse sampling
- Spiral: Center-outward expansion
- Center: Foveal focus

ADVANTAGES:
✓ Works on ANY aspect ratio
✓ Preserves detail (full resolution within window)
✓ Natural (how vision actually works)
✓ More training data (multiple crops per frame)
✓ Learns spatial relationships through geometry

USAGE:
```python
learner, eye = create_scanning_learner(
    window_size=128,
    scan_pattern='grid',  # or 'random', 'spiral', 'center'
    stride=64
)

learner.watch_video_scanning('video.mp4', eye)
learner.save_checkpoint('checkpoint.pt')
```

This is ZY + W:
Temporal learning through geometric spatial exploration.
"""
