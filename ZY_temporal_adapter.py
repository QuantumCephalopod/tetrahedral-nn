"""
ZY - TEMPORAL PREDICTION ADAPTER (Video Subdivision of Z)
==========================================================

Learn the STRUCTURE of motion and change.
Like arithmetic learned addition, rotation learned transformation,
this learns TEMPORAL TOPOLOGY - what comes next.

BASELINE: Moving shapes predict next frame
FUTURE: Continuous learning from video streams

Architecture designed for continual learning from the start.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Import dual-tetrahedral components
from Z_interface_coupling import DualTetrahedralNetwork, DualTetrahedralTrainer


# ============================================================================
# MOVING SHAPE GENERATOR
# ============================================================================

class MovingShapeGenerator:
    """
    Generate sequences of moving shapes for temporal learning.

    Shapes move with constant velocity (baseline), but architecture
    is designed to handle acceleration, bouncing, and complex motions.
    """
    def __init__(self, img_size: int = 32):
        self.img_size = img_size

    def create_shape(self, shape_type: str, center: Tuple[float, float],
                    size: int) -> np.ndarray:
        """
        Create a single shape at given position.

        Args:
            shape_type: 'circle', 'square', 'triangle'
            center: (x, y) float coordinates (can be fractional)
            size: Shape size (radius or half-width)

        Returns:
            Image with shape drawn
        """
        img = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        cx, cy = int(center[0]), int(center[1])

        if shape_type == 'circle':
            y, x = np.ogrid[:self.img_size, :self.img_size]
            mask = (x - cx)**2 + (y - cy)**2 <= size**2
            img[mask] = 1.0

        elif shape_type == 'square':
            x0 = max(0, cx - size)
            y0 = max(0, cy - size)
            x1 = min(self.img_size, cx + size)
            y1 = min(self.img_size, cy + size)
            img[y0:y1, x0:x1] = 1.0

        elif shape_type == 'triangle':
            # Simple filled triangle
            for y in range(self.img_size):
                for x in range(self.img_size):
                    # Upward pointing triangle
                    v0 = (cx, cy - size)
                    v1 = (cx - size, cy + size)
                    v2 = (cx + size, cy + size)

                    if self._point_in_triangle((x, y), v0, v1, v2):
                        img[y, x] = 1.0

        return img

    def _point_in_triangle(self, p: Tuple[int, int], v0: Tuple[int, int],
                          v1: Tuple[int, int], v2: Tuple[int, int]) -> bool:
        """Check if point is inside triangle."""
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        d1 = sign(p, v0, v1)
        d2 = sign(p, v1, v2)
        d3 = sign(p, v2, v0)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)

    def generate_motion_sequence(
        self,
        shape_type: str,
        start_pos: Tuple[float, float],
        velocity: Tuple[float, float],
        size: int,
        n_frames: int,
        bounce: bool = False
    ) -> List[np.ndarray]:
        """
        Generate a sequence of frames with moving shape.

        Args:
            shape_type: Type of shape
            start_pos: Starting (x, y) position
            velocity: (vx, vy) velocity per frame
            size: Shape size
            n_frames: Number of frames to generate
            bounce: If True, bounce off walls

        Returns:
            List of frames
        """
        frames = []
        pos = list(start_pos)
        vel = list(velocity)

        for _ in range(n_frames):
            # Create frame with shape at current position
            frame = self.create_shape(shape_type, tuple(pos), size)
            frames.append(frame)

            # Update position
            pos[0] += vel[0]
            pos[1] += vel[1]

            # Handle bouncing
            if bounce:
                if pos[0] - size < 0 or pos[0] + size >= self.img_size:
                    vel[0] = -vel[0]
                    pos[0] = np.clip(pos[0], size, self.img_size - size - 1)

                if pos[1] - size < 0 or pos[1] + size >= self.img_size:
                    vel[1] = -vel[1]
                    pos[1] = np.clip(pos[1], size, self.img_size - size - 1)

        return frames

    def generate_random_sequences(
        self,
        n_sequences: int,
        frames_per_sequence: int = 10,
        shape_types: List[str] = ['circle', 'square', 'triangle'],
        bounce: bool = False
    ) -> List[List[np.ndarray]]:
        """
        Generate random motion sequences for training.

        Args:
            n_sequences: Number of sequences to generate
            frames_per_sequence: Frames per sequence
            shape_types: Which shapes to use
            bounce: Enable bouncing physics

        Returns:
            List of sequences, each is a list of frames
        """
        sequences = []
        margin = 8

        for _ in range(n_sequences):
            # Random shape
            shape = np.random.choice(shape_types)

            # Random starting position (with margin)
            start_pos = (
                np.random.uniform(margin, self.img_size - margin),
                np.random.uniform(margin, self.img_size - margin)
            )

            # Random velocity (not too fast)
            velocity = (
                np.random.uniform(-2.0, 2.0),
                np.random.uniform(-2.0, 2.0)
            )

            # Random size
            size = np.random.randint(3, 6)

            # Generate sequence
            seq = self.generate_motion_sequence(
                shape, start_pos, velocity, size,
                frames_per_sequence, bounce
            )

            sequences.append(seq)

        return sequences


# ============================================================================
# TEMPORAL DATASET
# ============================================================================

class TemporalDataset(Dataset):
    """
    Dataset for temporal prediction.

    Input:  [frame_t-2, frame_t-1, frame_t] (3 frames stacked)
    Output: frame_t+1 (next frame)

    This tests: Can the network learn motion structure?
    """
    def __init__(self, sequences: List[List[np.ndarray]], window_size: int = 3):
        """
        Args:
            sequences: List of frame sequences
            window_size: How many past frames to use as input (default: 3)
        """
        self.window_size = window_size
        self.samples = []

        # Convert sequences to (input_window, target) pairs
        for seq in sequences:
            for i in range(window_size, len(seq)):
                # Input: previous window_size frames
                input_frames = seq[i-window_size:i]
                # Target: next frame
                target_frame = seq[i]

                # Flatten and stack
                input_flat = np.concatenate([f.flatten() for f in input_frames])
                target_flat = target_frame.flatten()

                self.samples.append((input_flat, target_flat))

        # Convert to tensors
        self.inputs = torch.tensor([s[0] for s in self.samples], dtype=torch.float32)
        self.targets = torch.tensor([s[1] for s in self.samples], dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# ============================================================================
# INCREMENTAL LEARNER (Future: Continual Learning)
# ============================================================================

class IncrementalTemporalLearner:
    """
    Temporal learner with incremental update capability.

    Current: Batch training with small batches
    Future: True online learning, context accumulation
    """
    def __init__(
        self,
        model: DualTetrahedralNetwork,
        optimizer: torch.optim.Optimizer,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device

        # Context state (for future continual learning)
        self.context_state = None
        self.examples_seen = 0

    def train_batch(self, batch_x: torch.Tensor, batch_y: torch.Tensor,
                   loss_fn: nn.Module = None) -> float:
        """
        Train on a single batch (can be called incrementally).

        Args:
            batch_x: Input frames
            batch_y: Target frames
            loss_fn: Loss function

        Returns:
            Batch loss
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        self.model.train()

        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)

        # Forward
        output = self.model(batch_x)
        loss = loss_fn(output, batch_y)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.examples_seen += len(batch_x)

        return loss.item()

    def train_epoch(self, dataloader: DataLoader, loss_fn: nn.Module = None) -> float:
        """Train for one epoch."""
        total_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in dataloader:
            loss = self.train_batch(batch_x, batch_y, loss_fn)
            total_loss += loss
            n_batches += 1

        return total_loss / n_batches

    def evaluate(self, dataloader: DataLoader, loss_fn: nn.Module = None) -> float:
        """Evaluate on dataset."""
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                output = self.model(batch_x)
                loss = loss_fn(output, batch_y)

                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_prediction(model: DualTetrahedralNetwork,
                        sequence: List[np.ndarray],
                        img_size: int = 32,
                        n_predict: int = 5):
    """
    Visualize model predictions vs ground truth for a sequence.

    Args:
        model: Trained model
        sequence: Full sequence of frames
        img_size: Image size
        n_predict: How many frames to predict beyond training
    """
    model.eval()

    window_size = 3

    # Use first 3 frames as input
    input_frames = sequence[:window_size]

    print("\n" + "=" * 70)
    print("🎬 TEMPORAL PREDICTION VISUALIZATION")
    print("=" * 70)

    # Predict next frames
    predicted_frames = []
    current_window = [f.copy() for f in input_frames]

    with torch.no_grad():
        for i in range(n_predict):
            # Flatten and concatenate window
            input_flat = np.concatenate([f.flatten() for f in current_window])
            input_tensor = torch.tensor(input_flat, dtype=torch.float32).unsqueeze(0)

            # Predict
            pred_flat = model(input_tensor)
            pred_frame = pred_flat.view(img_size, img_size).numpy()

            predicted_frames.append(pred_frame)

            # Slide window forward (use prediction as next input)
            current_window = current_window[1:] + [pred_frame]

    # Compare with ground truth
    gt_frames = sequence[window_size:window_size + n_predict]

    # Calculate errors
    print(f"\nPredicted {n_predict} frames:")
    for i in range(min(n_predict, len(gt_frames))):
        error = np.abs(predicted_frames[i] - gt_frames[i]).mean()
        print(f"  Frame {i+1}: Error = {error:.6f}")

    # Plot
    fig, axes = plt.subplots(2, n_predict, figsize=(3*n_predict, 6))

    for i in range(n_predict):
        # Predicted
        axes[0, i].imshow(predicted_frames[i], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Predicted t+{i+1}')
        axes[0, i].axis('off')

        # Ground truth (if available)
        if i < len(gt_frames):
            axes[1, i].imshow(gt_frames[i], cmap='gray', vmin=0, vmax=1)
            error = np.abs(predicted_frames[i] - gt_frames[i]).mean()
            axes[1, i].set_title(f'Ground Truth\nError: {error:.4f}')
        else:
            axes[1, i].text(0.5, 0.5, 'No GT', ha='center', va='center')
            axes[1, i].set_title('Beyond training')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

    print("=" * 70)


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    """
    Train dual-tetrahedral network on temporal prediction.

    BASELINE: Moving shapes with constant velocity
    FUTURE: Continuous learning from video streams
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    print("=" * 70)
    print("🎬 DUAL-TETRAHEDRAL TEMPORAL LEARNING")
    print("=" * 70)
    print("\n📚 CONCEPT:")
    print("  Like arithmetic learned ADDITION structure,")
    print("  like rotation learned TRANSFORMATION structure,")
    print("  can this learn MOTION structure?")
    print()
    print("  Input:  3 frames [t-2, t-1, t]")
    print("  Output: Next frame [t+1]")
    print("  Goal:   Learn temporal topology, not memorize sequences")
    print()

    # ========================================================================
    # 1. GENERATE DATA
    # ========================================================================
    print("🎨 Generating motion sequences...")
    img_size = 32
    generator = MovingShapeGenerator(img_size=img_size)

    # Training: Circles and squares moving
    train_sequences = generator.generate_random_sequences(
        n_sequences=200,
        frames_per_sequence=15,
        shape_types=['circle', 'square'],
        bounce=False
    )

    # Test: Include triangles (unseen shape) and same shapes
    test_sequences = generator.generate_random_sequences(
        n_sequences=50,
        frames_per_sequence=15,
        shape_types=['circle', 'square', 'triangle'],
        bounce=False
    )

    print(f"✓ Training sequences: {len(train_sequences)}")
    print(f"✓ Test sequences: {len(test_sequences)}")
    print(f"  Frame size: {img_size}×{img_size}")
    print(f"  Window size: 3 frames")

    # Create datasets
    train_dataset = TemporalDataset(train_sequences, window_size=3)
    test_dataset = TemporalDataset(test_sequences, window_size=3)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")

    # ========================================================================
    # 2. CREATE MODEL
    # ========================================================================
    print("\n🏗️  Building dual-tetrahedral model...")

    input_dim = img_size * img_size * 3  # 3 frames stacked
    output_dim = img_size * img_size      # 1 frame

    model = DualTetrahedralNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        latent_dim=128,  # Larger for temporal complexity
        coupling_strength=0.5,
        output_mode="weighted"
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Input: 3×{img_size}×{img_size} frames ({input_dim} dims)")
    print(f"  Output: 1×{img_size}×{img_size} frame ({output_dim} dims)")
    print(f"  Linear net: Smooth motion trajectories")
    print(f"  Nonlinear net: Discrete events, boundaries")

    # ========================================================================
    # 3. TRAIN (Incrementally capable)
    # ========================================================================
    print("\n⚡ Training...")
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    learner = IncrementalTemporalLearner(model, optimizer, device)

    epochs = 100
    for epoch in range(epochs):
        train_loss = learner.train_epoch(train_loader, loss_fn=nn.MSELoss())
        test_loss = learner.evaluate(test_loader, loss_fn=nn.MSELoss())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Test Loss: {test_loss:.6f}")

    # ========================================================================
    # 4. VISUALIZE
    # ========================================================================
    print("\n🎨 Generating predictions...")
    test_seq = test_sequences[0]
    visualize_prediction(model, test_seq, img_size=img_size, n_predict=5)

    # ========================================================================
    # 5. RESULTS
    # ========================================================================
    final_test_loss = learner.evaluate(test_loader)

    print("\n" + "=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    print(f"\nFinal test loss: {final_test_loss:.6f}")

    if final_test_loss < 0.01:
        print("✅ EXCELLENT: Learned motion structure!")
    elif final_test_loss < 0.05:
        print("✓ GOOD: Can predict motion reasonably")
    else:
        print("⚠️  LEARNING: Network improving (may need more training)")

    print(f"\nExamples processed: {learner.examples_seen}")

    print("\n💡 GENERALIZATION TESTS:")
    print("  1. Does it predict triangles (unseen shape)?")
    print("  2. Does it handle new velocities?")
    print("  3. Can it predict multiple steps ahead?")
    print()
    print("🚀 FUTURE: Continuous learning from video streams")
    print("  • Incremental updates (no retraining)")
    print("  • Context accumulation across videos")
    print("  • Learn physics, causality, object permanence")
    print("=" * 70)

    return model, learner


if __name__ == "__main__":
    model, learner = main()
    print("\n✓ Training complete!")


# ============================================================================
# SUMMARY
# ============================================================================

"""
ZY_temporal_adapter.py - Temporal Topology Learning

GOAL:
  Learn the STRUCTURE of motion and change.
  Like arithmetic learned addition, rotation learned transformation,
  this learns temporal topology.

BASELINE:
  - Moving shapes (circles, squares) with constant velocity
  - Input: 3 consecutive frames
  - Output: Predict next frame
  - Test: Unseen shapes (triangles), new velocities

ARCHITECTURE:
  - Dual tetrahedral networks (linear + nonlinear)
  - Linear: Smooth motion flow, velocity fields
  - Nonlinear: Boundaries, discrete events
  - Designed for incremental learning

FUTURE VISION:
  - Continuous learning from video streams
  - Context accumulation (not retraining)
  - Learn physics, causality from YouTube
  - The manifold of reality itself

THE QUESTION:
  Can geometric topology learn TIME?
  Not just space (rotation) but temporal structure?

  If yes → Path to learning everything from video.
"""
