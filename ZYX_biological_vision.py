"""
ZYX - BIOLOGICAL VISION SYSTEM (Linear subdivision of temporal)
===============================================================

Time is LINEAR. Space is NON-LINEAR.
This is not machine learning. This is BIOLOGY.

This is ZY (temporal) + X (linear) = linear time progression.

ARCHITECTURE:
- Time flows linearly: tick → tock → tick → tock (LINEAR)
- Space is non-linear: can look ANYWHERE, starts at CENTER (fovea)
- Each time step = ONE frame + ONE spatial location
- Model learns temporal dynamics + spatial attention

Like an actual eye watching the world unfold.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Optional, Dict
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
# BIOLOGICAL ATTENTION - Non-linear Spatial Selector
# ============================================================================

class BiologicalAttention:
    """
    Non-linear spatial attention system.

    Like biological vision:
    - Starts at CENTER (fovea)
    - Makes saccadic movements (jumps around)
    - Center-biased (prefers middle, but can look anywhere)
    - Exploratory (random walks, not uniformly random)

    This is NON-LINEAR spatial selection.
    """

    def __init__(
        self,
        window_size: Tuple[int, int] = (128, 128),
        center_bias: float = 0.7,
        exploration_radius: int = 200,
        saccade_scale: int = 100
    ):
        """
        Args:
            window_size: Size of attention window (W, H)
            center_bias: Probability of center-biased attention (0-1)
            exploration_radius: Max distance from center
            saccade_scale: Typical distance of saccadic jumps
        """
        self.window_size = window_size
        self.center_bias = center_bias
        self.exploration_radius = exploration_radius
        self.saccade_scale = saccade_scale

        self.frame_size = None
        self.current_position = None

    def set_frame_size(self, frame_size: Tuple[int, int]):
        """Set frame size and reset to center (fovea)."""
        self.frame_size = frame_size
        center_x = (frame_size[0] - self.window_size[0]) // 2
        center_y = (frame_size[1] - self.window_size[1]) // 2
        self.current_position = (center_x, center_y)

    def next_position(self) -> Tuple[int, int]:
        """
        Generate next attention position (NON-LINEAR saccade).

        Returns:
            (x, y) position for next window
        """
        if self.frame_size is None:
            raise ValueError("Must set frame_size first")

        frame_w, frame_h = self.frame_size
        win_w, win_h = self.window_size

        # Center position
        center_x = (frame_w - win_w) // 2
        center_y = (frame_h - win_h) // 2

        # Center-biased or exploratory? (non-linear decision)
        if np.random.random() < self.center_bias:
            # Center-biased: Gaussian around center (foveal preference)
            x = int(np.random.normal(center_x, self.exploration_radius / 3))
            y = int(np.random.normal(center_y, self.exploration_radius / 3))
        else:
            # Exploratory saccade: Random walk from current position
            if self.current_position is not None:
                dx = np.random.normal(0, self.saccade_scale)
                dy = np.random.normal(0, self.saccade_scale)
                x = int(self.current_position[0] + dx)
                y = int(self.current_position[1] + dy)
            else:
                x = np.random.randint(0, max(1, frame_w - win_w))
                y = np.random.randint(0, max(1, frame_h - win_h))

        # Keep in bounds
        x = np.clip(x, 0, frame_w - win_w)
        y = np.clip(y, 0, frame_h - win_h)

        self.current_position = (x, y)
        return (x, y)

    def extract_window(self, frame: np.ndarray, position: Tuple[int, int]) -> np.ndarray:
        """Extract attention window from frame."""
        x, y = position
        win_w, win_h = self.window_size

        if len(frame.shape) == 2:  # Grayscale
            return frame[y:y+win_h, x:x+win_w]
        else:  # Color
            return frame[y:y+win_h, x:x+win_w, :]


# ============================================================================
# BIOLOGICAL VISION PROCESSOR - Linear Time + Non-linear Space
# ============================================================================

class BiologicalVisionProcessor:
    """
    Process video with biological time-space coupling.

    TIME: Linear progression (frame by frame) - X dimension
    SPACE: Non-linear attention (saccades, center-biased)

    Each time step = ONE frame + ONE spatial location.
    """

    def __init__(
        self,
        attention: BiologicalAttention,
        grayscale: bool = True,
        normalize: bool = True
    ):
        self.attention = attention
        self.grayscale = grayscale
        self.normalize = normalize

    def stream_biological_vision(
        self,
        video_path: str
    ):
        """
        Stream windows with biological time-space coupling.

        LINEAR time: frame 0 → frame 1 → frame 2 → ...
        NON-LINEAR space: position changes with biological attention

        Yields:
            (window, position, frame_idx) tuples
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
        print(f"   🧠 BIOLOGICAL MODE:")
        print(f"      - Time: LINEAR (frame by frame)")
        print(f"      - Space: NON-LINEAR (center-biased saccades)")
        print(f"      - Window size: {self.attention.window_size}")
        print()

        # Set attention frame size
        self.attention.set_frame_size((frame_width, frame_height))

        frame_idx = 0

        # LINEAR temporal progression
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            if self.grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if self.normalize:
                frame = frame.astype(np.float32) / 255.0

            # NON-LINEAR spatial attention: pick ONE position
            position = self.attention.next_position()
            window = self.attention.extract_window(frame, position)

            # Yield ONE window per time step
            if window.shape[:2] == self.attention.window_size[::-1]:
                yield window, position, frame_idx

            frame_idx += 1

        cap.release()
        print(f"✓ Processed {frame_idx} time steps (frames)")


# ============================================================================
# BIOLOGICAL LEARNER
# ============================================================================

class BiologicalVisionLearner(ContinuousLearner):
    """
    Learner with biological time-space coupling.

    LINEAR time + NON-LINEAR space.
    Model sees windows from different times AND different places.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Track spatial attention history
        self.position_history = deque(maxlen=100)
        self.spatial_diversity = 0.0

    def observe_window(
        self,
        window: np.ndarray,
        position: Tuple[int, int],
        frame_idx: int
    ) -> Optional[float]:
        """
        Observe window at this time-space point.

        Args:
            window: Attention window
            position: Spatial position (non-linear)
            frame_idx: Temporal frame index (linear)

        Returns:
            Loss if learning occurred
        """
        # Track spatial exploration
        self.position_history.append(position)

        # Calculate spatial diversity
        if len(self.position_history) >= 10:
            positions_array = np.array(list(self.position_history))
            self.spatial_diversity = np.std(positions_array)

        return self.observe_frame(window)

    def watch_video_biological(
        self,
        video_path: str,
        attention: BiologicalAttention,
        report_every: int = 100
    ):
        """
        Watch video with biological vision system.

        Args:
            video_path: Path to video
            attention: BiologicalAttention instance
            report_every: Report every N time steps
        """
        print("\n" + "=" * 70)
        print(f"🧠 BIOLOGICAL VISION: {video_path}")
        print("=" * 70)
        print("Time is LINEAR → Space is NON-LINEAR")
        print("This is not machine learning. This is BIOLOGY.")
        print("=" * 70)

        processor = BiologicalVisionProcessor(
            attention,
            grayscale=True,
            normalize=True
        )

        time_step = 0
        learning_count = 0
        start_time = cv2.getTickCount()

        for window, position, frame_idx in processor.stream_biological_vision(video_path):
            loss = self.observe_window(window, position, frame_idx)
            time_step += 1

            if loss is not None:
                learning_count += 1

                if learning_count % report_every == 0:
                    elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
                    fps = time_step / elapsed if elapsed > 0 else 0

                    print(f"t={time_step:6d} | "
                          f"Frame {frame_idx:4d} | "
                          f"Pos {position} | "
                          f"Loss: {loss:.6f} | "
                          f"FPS: {fps:5.1f} | "
                          f"Spatial diversity: {self.spatial_diversity:.1f}")

        elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()

        print("\n" + "=" * 70)
        print("🧠 BIOLOGICAL LEARNING COMPLETE")
        print("=" * 70)
        print(f"Time steps: {time_step}")
        print(f"Learning updates: {learning_count}")
        print(f"Time elapsed: {elapsed:.1f}s")
        print(f"Final loss: {self.current_loss:.6f}")
        print(f"Spatial diversity: {self.spatial_diversity:.1f}")
        print("=" * 70)


# ============================================================================
# CREATION FUNCTION
# ============================================================================

def create_biological_vision_system(
    window_size: int = 128,
    center_bias: float = 0.7,
    learning_rate: float = 0.0001,
    device: str = 'cpu'
) -> Tuple[BiologicalVisionLearner, BiologicalAttention]:
    """
    Create biological vision system.

    Args:
        window_size: Size of attention window (square)
        center_bias: How much to prefer center (0-1, foveal bias)
        learning_rate: Learning rate
        device: 'cpu' or 'cuda'

    Returns:
        (learner, attention) system ready to watch
    """
    # Create biological attention
    attention = BiologicalAttention(
        window_size=(window_size, window_size),
        center_bias=center_bias,
        exploration_radius=200,
        saccade_scale=100
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

    # Create biological learner
    learner = BiologicalVisionLearner(
        model,
        optimizer,
        window_size=3,
        device=device
    )

    return learner, attention


# ============================================================================
# SUMMARY
# ============================================================================

"""
ZYX_biological_vision.py - Linear Time + Non-linear Space

FRACTAL STRUCTURE:
ZY = temporal adapter
X = linear dimension
ZYX = temporal learning with linear time progression

THE CORE INSIGHT:
Time flows LINEARLY: tick → tock → tick → tock
Space is NON-LINEAR: can look ANYWHERE, starts at CENTER (fovea)

This is NOT machine learning statistics.
This is BIOLOGY.

ARCHITECTURE:
Each time step = ONE frame + ONE spatial location

Time t=0: Frame 0, Position (center)
Time t=1: Frame 1, Position (saccade somewhere)
Time t=2: Frame 2, Position (saccade elsewhere)
...

LINEAR: Temporal progression (frame sequence)
NON-LINEAR: Spatial attention (center-biased saccades)

MODEL LEARNS:
Window(t-2, pos_a) → Window(t-1, pos_b) → Window(t, pos_c) → PREDICT

Couples:
✓ Temporal dynamics (motion, change over time) - LINEAR
✓ Spatial attention (where to look, exploratory) - NON-LINEAR

SPATIAL ATTENTION:
- CENTER-BIASED (70% looks near center, like fovea)
- EXPLORATORY (30% makes saccadic jumps)
- NON-LINEAR (not grid-based, organic movement)

USAGE:
```python
learner, attention = create_biological_vision_system(
    window_size=128,
    center_bias=0.7,  # 70% center, 30% exploration
    learning_rate=0.0001
)

learner.watch_video_biological('video.mp4', attention)
learner.save_checkpoint('biological_vision.pt')
```

THIS IS HOW EYES WORK.
Linear time. Non-linear space.
Tick. Tock. Look here. Look there.
Biology.
"""
