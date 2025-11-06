"""
ZYZ - CONTINUOUS LEARNING INTERFACE (Coupling subdivision of temporal)
======================================================================

This is ZY (temporal) + Z (interface/coupling).

The interface that couples temporal learning to continuous streams.
Not batch-based. Not epochs. Just... observation → learning → observation.

This is the Z (coupling) that makes temporal learning continuous.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from collections import deque


# ============================================================================
# CONTINUOUS LEARNER - The Temporal Interface/Coupling
# ============================================================================

class ContinuousLearner:
    """
    Interface for continuous learning from temporal streams.

    This is ZYZ: How you COUPLE a model to continuous temporal data.

    Not epochs. Not batches. Just continuous observation.
    Like watching the world unfold.

    Handles:
    - Frame buffering (temporal sliding window)
    - Learning from sequences (temporal coupling)
    - Checkpointing (state persistence)
    - Learning rate scheduling (adaptive coupling strength)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        window_size: int = 3,
        device: str = 'cpu'
    ):
        """
        Args:
            model: The network to train (usually DualTetrahedralNetwork)
            optimizer: Optimizer for learning
            window_size: How many frames to use as input (temporal window)
            device: 'cpu' or 'cuda'
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.window_size = window_size
        self.device = device

        # Frame buffer (temporal sliding window)
        self.frame_buffer = deque(maxlen=window_size + 1)

        # Learning statistics
        self.total_updates = 0
        self.current_loss = 0.0
        self.loss_history = []

        # Learning rate scheduling
        self.initial_lr = optimizer.param_groups[0]['lr']

    def observe_frame(self, frame: np.ndarray) -> Optional[float]:
        """
        Observe a single frame and learn if possible.

        This is the core coupling: frame → learning.

        Args:
            frame: New frame to observe (H, W) array

        Returns:
            Loss if learning occurred, None otherwise
        """
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
        """
        Single learning step (the actual coupling).

        Args:
            input_tensor: Input window (batch, input_dim)
            target_tensor: Target frame (batch, output_dim)

        Returns:
            Loss value
        """
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

        # Adaptive learning rate (adaptive coupling strength)
        if self.total_updates % 1000 == 0:
            self._adjust_learning_rate()

        return self.current_loss

    def _adjust_learning_rate(self):
        """Adjust learning rate based on progress (adaptive coupling)."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.95

    def save_checkpoint(self, path: str):
        """
        Save current state for later continuation.

        Args:
            path: Where to save checkpoint
        """
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'total_updates': self.total_updates,
            'loss_history': self.loss_history
        }
        torch.save(checkpoint, path)
        print(f"✓ Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """
        Load previous state to continue learning.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.total_updates = checkpoint['total_updates']
        self.loss_history = checkpoint['loss_history']
        print(f"✓ Checkpoint loaded: {path}")
        print(f"   Resuming from update {self.total_updates}")

    def get_status(self) -> dict:
        """
        Get current learning status.

        Returns:
            Dictionary with learning statistics
        """
        return {
            'total_updates': self.total_updates,
            'current_loss': self.current_loss,
            'avg_loss_last_100': np.mean(self.loss_history[-100:]) if len(self.loss_history) >= 100 else self.current_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }


# ============================================================================
# SUMMARY
# ============================================================================

"""
ZYZ_continuous_learning.py - Temporal Learning Interface

FRACTAL STRUCTURE:
ZY = temporal adapter
Z = interface/coupling dimension
ZYZ = temporal + interface = continuous learning coupling

KEY INSIGHT:
This is the INTERFACE that couples temporal learning to continuous streams.

Not batch-based. Not epochs. Continuous observation.

WHAT IT DOES:
- Buffers temporal window (3 frames)
- Observes new frame → learns immediately
- Saves/loads state for continuous sessions
- Adapts learning rate over time

USAGE IN COLAB:
After running W, X, Y, Z (core architecture), run this cell.
Then ZYW and ZYX can use it!

```python
# This cell defines the continuous learning interface
from ZYZ_continuous_learning import ContinuousLearner

# Now ZYW and ZYX can import and use it
```

This is Z (coupling) for temporal streams.
The interface that makes temporal learning continuous.
"""
