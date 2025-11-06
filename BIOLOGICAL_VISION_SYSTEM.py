"""
BIOLOGICAL VISION SYSTEM - Time is Linear, Space is Not
=========================================================

THE INSIGHT:
Time flows linearly: tick → tock → tick → tock
Space is non-linear: can look ANYWHERE, start at CENTER (fovea)

This is NOT machine learning. This is BIOLOGY.

ARCHITECTURE:
- Each time step = ONE frame + ONE spatial location
- Temporal progression is LINEAR (frame by frame)
- Spatial attention is NON-LINEAR (saccades, center-biased)
- Model learns BOTH temporal dynamics AND spatial attention

Like an actual eye watching the world unfold.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Optional
import cv2
from collections import deque

from CONTINUOUS_LEARNING_SYSTEM import ContinuousLearner
from Z_interface_coupling import DualTetrahedralNetwork


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
    - Exploratory (random walks, but not uniformly random)
    """

    def __init__(
        self,
        window_size: Tuple[int, int] = (128, 128),
        center_bias: float = 0.7,  # Probability of looking near center
        exploration_radius: int = 200,  # How far from center to explore
        saccade_scale: int = 100  # How far to jump each saccade
    ):
        """
        Args:
            window_size: Size of attention window (W, H)
            center_bias: Probability of center-biased attention (0-1)
            exploration_radius: Max distance from center to explore
            saccade_scale: Typical distance of saccadic jumps
        """
        self.window_size = window_size
        self.center_bias = center_bias
        self.exploration_radius = exploration_radius
        self.saccade_scale = saccade_scale

        self.frame_size = None
        self.current_position = None

    def set_frame_size(self, frame_size: Tuple[int, int]):
        """Set frame size and reset to center."""
        self.frame_size = frame_size
        # Start at center (fovea)
        center_x = (frame_size[0] - self.window_size[0]) // 2
        center_y = (frame_size[1] - self.window_size[1]) // 2
        self.current_position = (center_x, center_y)

    def next_position(self) -> Tuple[int, int]:
        """
        Generate next attention position (non-linear saccade).

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

        # Decide: center-biased or exploratory?
        if np.random.random() < self.center_bias:
            # Center-biased: Gaussian around center
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
                # First saccade: random
                x = np.random.randint(0, max(1, frame_w - win_w))
                y = np.random.randint(0, max(1, frame_h - win_h))

        # Keep in bounds
        x = np.clip(x, 0, frame_w - win_w)
        y = np.clip(y, 0, frame_h - win_h)

        self.current_position = (x, y)
        return (x, y)

    def extract_window(self, frame: np.ndarray, position: Tuple[int, int]) -> np.ndarray:
        """
        Extract attention window from frame.

        Args:
            frame: Full frame (H, W) or (H, W, C)
            position: (x, y) top-left corner

        Returns:
            Cropped window
        """
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

    TIME: Linear progression (frame by frame)
    SPACE: Non-linear attention (saccades, center-biased)

    Each time step = ONE frame + ONE spatial location
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

        Yields ONE window per frame (linear time, non-linear space).

        Args:
            video_path: Path to video

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

        # Linear temporal progression
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            if self.grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if self.normalize:
                frame = frame.astype(np.float32) / 255.0

            # Non-linear spatial attention: pick ONE position for this time step
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

    Learns from linear temporal stream with non-linear spatial attention.
    Model sees windows from DIFFERENT times and DIFFERENT places.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Track spatial attention history
        self.position_history = deque(maxlen=100)
        self.spatial_diversity = 0.0  # Measure of exploration

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
            position: Spatial position in frame
            frame_idx: Temporal frame index

        Returns:
            Loss if learning occurred
        """
        # Track spatial exploration
        self.position_history.append(position)

        # Calculate spatial diversity (standard deviation of positions)
        if len(self.position_history) >= 10:
            positions_array = np.array(list(self.position_history))
            self.spatial_diversity = np.std(positions_array)

        # Learn using parent's method
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
        center_bias: How much to prefer center (0-1)
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
    input_dim = window_size * window_size * 3  # 3 windows
    output_dim = window_size * window_size      # 1 window

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
# USAGE EXAMPLE
# ============================================================================

def main():
    """
    Example: Create biological vision system.
    """
    print("=" * 70)
    print("🧠 BIOLOGICAL VISION SYSTEM")
    print("=" * 70)
    print("\nTime is LINEAR (tick → tock)")
    print("Space is NON-LINEAR (can look anywhere, starts at center)")
    print("\nThis is not machine learning. This is BIOLOGY.")
    print("=" * 70)
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    learner, attention = create_biological_vision_system(
        window_size=128,
        center_bias=0.7,  # 70% center-biased, 30% exploratory
        learning_rate=0.0001,
        device=device
    )

    print(f"✓ Biological vision system created")
    print(f"  Device: {device}")
    print(f"  Window size: 128×128")
    print(f"  Center bias: 70% (foveal preference)")
    print(f"  Model parameters: {sum(p.numel() for p in learner.model.parameters()):,}")
    print()

    print("💡 USAGE:")
    print("  learner.watch_video_biological('video.mp4', attention)")
    print("  learner.save_checkpoint('checkpoint.pt')")
    print()
    print("🧠 This is how biology learns from vision!")

    return learner, attention


if __name__ == "__main__":
    learner, attention = main()


# ============================================================================
# SUMMARY
# ============================================================================

"""
BIOLOGICAL_VISION_SYSTEM.py - Time is Linear, Space is Not

THE CORE INSIGHT:
Time flows linearly: tick → tock → tick → tock
Space is non-linear: can look ANYWHERE, starts at CENTER (fovea)

This is NOT machine learning statistics.
This is BIOLOGY.

ARCHITECTURE:
Each time step = ONE frame + ONE spatial location

Time t=0: Frame 0, Position (center)
Time t=1: Frame 1, Position (saccade somewhere)
Time t=2: Frame 2, Position (saccade elsewhere)
Time t=3: Frame 3, Position (maybe back to center)
...

MODEL LEARNS:
Window(t-2, pos_a) → Window(t-1, pos_b) → Window(t, pos_c) → PREDICT

This couples:
✓ Temporal dynamics (motion, change over time)
✓ Spatial attention (where to look, exploratory)

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
