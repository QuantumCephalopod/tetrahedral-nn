"""
MOTION-BASED INVERSE MODEL - FIXED VERSION
==========================================

Fixed bugs from original:
1. motion_t was always zero (frame_t, frame_t)
2. No proper state representation
3. Missing visualization

New approach:
- State representation: Encode frames directly
- Motion: Extract from frame pairs
- Forward: (state_t, action) → state_t+1
- Inverse: (state_t, state_t+1) → action

Simpler, cleaner, debuggable!

Author: Reality Check Edition
Date: November 10, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
import random

try:
    import gymnasium as gym
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', '-q', 'gymnasium[atari]', 'gymnasium[accept-rom-license]'])
    import gymnasium as gym

φ = (1 + 5**0.5) / 2


# ============================================================================
# STATE ENCODER (Frame → Abstract State)
# ============================================================================

class StateEncoder(nn.Module):
    """
    Encode raw frame to abstract state representation.

    This is simpler and more interpretable than "motion_t".
    """
    def __init__(self, img_size=128, state_dim=128):
        super().__init__()
        self.img_size = img_size
        self.state_dim = state_dim

        # CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 7, 2, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.fc = nn.Linear(256 * 4 * 4, state_dim)

    def forward(self, frame):
        """
        Encode frame to state.

        Args:
            frame: (batch, 3, 128, 128)
        Returns:
            state: (batch, state_dim)
        """
        features = self.encoder(frame)
        features_flat = features.reshape(features.size(0), -1)
        state = self.fc(features_flat)
        return state


# ============================================================================
# ACTION ENCODER
# ============================================================================

class ActionEncoder(nn.Module):
    """Discrete action → continuous embedding"""
    def __init__(self, n_actions, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(n_actions, embed_dim)

    def forward(self, action_idx):
        return self.embedding(action_idx)


# ============================================================================
# FORWARD MODEL: (state, action) → next_state
# ============================================================================

class ForwardModel(nn.Module):
    """
    Forward model: Predict next state from current state and action.

    (state_t, action) → state_t+1
    """
    def __init__(self, state_dim=128, latent_dim=128, n_actions=6):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions

        self.action_encoder = ActionEncoder(n_actions, embed_dim=64)

        # Use dual tetrahedra
        input_dim = state_dim + 64
        output_dim = state_dim

        self.dual_tetra = DualTetrahedralNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            latent_dim=latent_dim,
            coupling_strength=0.5,
            output_mode="weighted"
        )

    def forward(self, state, action):
        """
        Predict next state.

        Args:
            state: (batch, state_dim)
            action: (batch,) action indices
        Returns:
            next_state: (batch, state_dim)
        """
        action_embed = self.action_encoder(action)
        x = torch.cat([state, action_embed], dim=-1)
        next_state = self.dual_tetra(x)
        return next_state


# ============================================================================
# INVERSE MODEL: (state_t, state_t+1) → action
# ============================================================================

class InverseModel(nn.Module):
    """
    Inverse model: Infer action from state transition.

    (state_t, state_t+1) → action
    """
    def __init__(self, state_dim=128, latent_dim=128, n_actions=6):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions

        # Use dual tetrahedra
        input_dim = state_dim * 2
        output_dim = n_actions

        self.dual_tetra = DualTetrahedralNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            latent_dim=latent_dim,
            coupling_strength=0.5,
            output_mode="weighted"
        )

    def forward(self, state_t, state_t1):
        """
        Infer action from state transition.

        Args:
            state_t: (batch, state_dim)
            state_t1: (batch, state_dim)
        Returns:
            action_logits: (batch, n_actions)
        """
        x = torch.cat([state_t, state_t1], dim=-1)
        action_logits = self.dual_tetra(x)
        return action_logits


# ============================================================================
# COUPLED MODEL
# ============================================================================

class CoupledModel(nn.Module):
    """
    Complete coupled forward+inverse model.

    Simpler and more debuggable than original.
    """
    def __init__(self, state_dim=128, latent_dim=128, n_actions=6):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions

        self.state_encoder = StateEncoder(state_dim=state_dim)
        self.forward_model = ForwardModel(state_dim, latent_dim, n_actions)
        self.inverse_model = InverseModel(state_dim, latent_dim, n_actions)

    def encode_state(self, frame):
        """Encode frame to state"""
        return self.state_encoder(frame)

    def compute_losses(self, frame_t, frame_t1, action,
                      forward_weight=1.0,
                      inverse_weight=1.0,
                      consistency_weight=0.5):
        """
        Compute all losses.

        Returns dictionary with losses and metrics.
        """
        # Encode states
        state_t = self.encode_state(frame_t)
        state_t1 = self.encode_state(frame_t1)

        # Forward loss: Can we predict next state?
        pred_state_t1 = self.forward_model(state_t, action)
        loss_forward = F.mse_loss(pred_state_t1, state_t1)

        # Inverse loss: Can we infer action?
        action_logits = self.inverse_model(state_t, state_t1)
        loss_inverse = F.cross_entropy(action_logits, action)

        # Consistency loss: Do models agree?
        action_inferred = torch.argmax(action_logits.detach(), dim=-1)
        pred_state_t1_consistent = self.forward_model(state_t, action_inferred)
        loss_consistency = F.mse_loss(pred_state_t1_consistent, state_t1)

        # Total loss
        total_loss = (forward_weight * loss_forward +
                     inverse_weight * loss_inverse +
                     consistency_weight * loss_consistency)

        # Metrics
        with torch.no_grad():
            action_pred = torch.argmax(action_logits, dim=-1)
            accuracy = (action_pred == action).float().mean()

        return {
            'total': total_loss,
            'forward': loss_forward.item(),
            'inverse': loss_inverse.item(),
            'consistency': loss_consistency.item(),
            'accuracy': accuracy.item()
        }


# ============================================================================
# EXPERIENCE BUFFER
# ============================================================================

class ExperienceBuffer:
    """Store (frame_t, action, frame_t+1) transitions"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, frame_t, action, frame_t1):
        self.buffer.append((frame_t, action, frame_t1))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        frames_t = torch.stack([b[0] for b in batch])
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long)
        frames_t1 = torch.stack([b[2] for b in batch])

        return frames_t, actions, frames_t1

    def __len__(self):
        return len(self.buffer)


# ============================================================================
# TRAINER WITH VISUALIZATION
# ============================================================================

class FixedTrainer:
    """
    Trainer with built-in visualization and diagnostics.

    No more flying blind!
    """
    def __init__(self,
                 env_name='ALE/Pong-v5',
                 state_dim=128,
                 latent_dim=128,
                 base_lr=0.0001,
                 buffer_capacity=10000,
                 batch_size=16,
                 device=None):

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.batch_size = batch_size

        print(f"🖥️  Device: {self.device}")

        # Environment
        print(f"🎮 Environment: {env_name}")
        self.env = gym.make(env_name, render_mode='rgb_array')
        self.n_actions = self.env.action_space.n
        print(f"   Actions: {self.n_actions}")

        # Action names for Pong
        self.action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']

        # Model
        print(f"🧠 Creating coupled model...")
        self.model = CoupledModel(
            state_dim=state_dim,
            latent_dim=latent_dim,
            n_actions=self.n_actions
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"   Parameters: {total_params:,}")

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=base_lr)

        # Buffer
        self.buffer = ExperienceBuffer(capacity=buffer_capacity)

        # Metrics
        self.step_count = 0
        self.episode_count = 0
        self.history = {
            'forward': [], 'inverse': [], 'consistency': [],
            'total': [], 'accuracy': []
        }

    def preprocess_frame(self, frame):
        """Preprocess frame"""
        from PIL import Image
        img = Image.fromarray(frame).resize((128, 128), Image.LANCZOS)
        tensor = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return tensor

    def collect_experience(self, n_steps=100, exploration_rate=1.0):
        """Collect experience"""
        frame_t, _ = self.env.reset()
        frame_t = self.preprocess_frame(frame_t)
        episodes_done = 0

        for step in range(n_steps):
            # Random action for now
            action = random.randint(0, self.n_actions - 1)

            # Execute
            frame_t1_raw, reward, terminated, truncated, info = self.env.step(action)
            frame_t1 = self.preprocess_frame(frame_t1_raw)

            # Store
            self.buffer.add(frame_t, action, frame_t1)

            # Next
            frame_t = frame_t1

            if terminated or truncated:
                frame_t, _ = self.env.reset()
                frame_t = self.preprocess_frame(frame_t)
                episodes_done += 1

        return episodes_done

    def train_step(self):
        """Single training step with metrics"""
        if len(self.buffer) < self.batch_size:
            return None

        # Sample batch
        frames_t, actions, frames_t1 = self.buffer.sample(self.batch_size)
        frames_t = frames_t.to(self.device)
        actions = actions.to(self.device)
        frames_t1 = frames_t1.to(self.device)

        # Compute losses
        losses = self.model.compute_losses(frames_t, frames_t1, actions)

        # Backprop
        self.optimizer.zero_grad()
        losses['total'].backward()
        self.optimizer.step()

        self.step_count += 1

        # Track metrics
        for key in ['forward', 'inverse', 'consistency', 'total', 'accuracy']:
            if key in losses:
                val = losses[key] if key == 'total' else losses[key]
                self.history[key].append(val.item() if torch.is_tensor(val) else val)

        return losses

    def train_loop(self, n_episodes=10, steps_per_episode=50, verbose=True):
        """Training loop with progress tracking"""
        print("\n" + "="*70)
        print("🌀 TRAINING LOOP")
        print("="*70 + "\n")

        for episode in range(n_episodes):
            # Collect
            episodes_done = self.collect_experience(n_steps=100)
            self.episode_count += episodes_done

            # Train
            episode_metrics = {'forward': [], 'inverse': [], 'accuracy': []}

            for _ in range(steps_per_episode):
                metrics = self.train_step()
                if metrics:
                    episode_metrics['forward'].append(metrics['forward'])
                    episode_metrics['inverse'].append(metrics['inverse'])
                    episode_metrics['accuracy'].append(metrics['accuracy'])

            # Log
            if verbose and episode_metrics['forward']:
                avg_forward = np.mean(episode_metrics['forward'])
                avg_inverse = np.mean(episode_metrics['inverse'])
                avg_accuracy = np.mean(episode_metrics['accuracy'])

                print(f"📊 Episode {episode+1}/{n_episodes}")
                print(f"   Steps: {self.step_count}")
                print(f"   Forward loss: {avg_forward:.6f}")
                print(f"   Inverse loss: {avg_inverse:.6f}")
                print(f"   Accuracy: {avg_accuracy*100:.2f}% (random={100/self.n_actions:.2f}%)")
                print(f"   Buffer: {len(self.buffer)}")
                print()

    def plot_training(self):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Forward loss
        axes[0, 0].plot(self.history['forward'])
        axes[0, 0].set_title('Forward Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].grid(alpha=0.3)

        # Inverse loss
        axes[0, 1].plot(self.history['inverse'])
        axes[0, 1].axhline(np.log(self.n_actions), color='r', linestyle='--', label='Random baseline')
        axes[0, 1].set_title('Inverse Loss (Cross-Entropy)')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('CE Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # Accuracy
        axes[1, 0].plot(np.array(self.history['accuracy']) * 100)
        axes[1, 0].axhline(100/self.n_actions, color='r', linestyle='--', label='Random baseline')
        axes[1, 0].set_title('Inverse Model Accuracy')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].legend()
        axes[1, 0].set_ylim([0, 100])
        axes[1, 0].grid(alpha=0.3)

        # Consistency
        axes[1, 1].plot(self.history['consistency'])
        axes[1, 1].set_title('Consistency Loss')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('MSE')
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
        print("✅ Saved: training_progress.png")
        plt.show()

    def show_confusion_matrix(self, n_samples=100):
        """Show confusion matrix for inverse model"""
        import seaborn as sns

        action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']

        self.model.eval()
        frames_t, actions, frames_t1 = self.buffer.sample(min(n_samples, len(self.buffer)))
        frames_t = frames_t.to(self.device)
        frames_t1 = frames_t1.to(self.device)

        with torch.no_grad():
            state_t = self.model.encode_state(frames_t)
            state_t1 = self.model.encode_state(frames_t1)
            logits = self.model.inverse_model(state_t, state_t1)
            pred = torch.argmax(logits, dim=-1)

        # Build confusion matrix
        confusion = np.zeros((self.n_actions, self.n_actions))
        for t, p in zip(actions.cpu().numpy(), pred.cpu().numpy()):
            confusion[t, p] += 1
        confusion /= (confusion.sum(axis=1, keepdims=True) + 1e-8)

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Confusion matrix
        sns.heatmap(confusion, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=action_names, yticklabels=action_names,
                   ax=ax1, vmin=0, vmax=0.5)
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')

        # Per-action accuracy
        diagonal = np.diag(confusion) * 100
        bars = ax2.bar(range(self.n_actions), diagonal)
        ax2.axhline(100/self.n_actions, color='r', linestyle='--', label='Random')
        ax2.set_xticks(range(self.n_actions))
        ax2.set_xticklabels(action_names, rotation=45, ha='right')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Per-Action Accuracy')
        ax2.legend()
        ax2.set_ylim([0, 100])

        # Color bars
        for bar, acc in zip(bars, diagonal/100):
            bar.set_color('green' if acc > 0.25 else 'red' if acc < 0.20 else 'orange')

        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        print("✅ Saved: confusion_matrix.png")
        plt.show()

        self.model.train()

        print(f"\n📊 Diagonal strength: {np.mean(diagonal):.1f}%")
        if np.mean(diagonal) > 25:
            print("✅ LEARNING - Model understands actions!")
        elif np.mean(diagonal) > 20:
            print("🟡 WEAK - Starting to learn")
        else:
            print("⚠️ NOT LEARNING - Near random")


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔧 FIXED MOTION INVERSE MODEL")
    print("="*70)
    print("\nFixed bugs:")
    print("  ✅ motion_t is no longer zero")
    print("  ✅ Proper state encoding")
    print("  ✅ Built-in diagnostics")
    print("  ✅ Visualization tools")
    print("\nUsage:")
    print("  trainer = FixedTrainer()")
    print("  trainer.train_loop(n_episodes=10)")
    print("  trainer.plot_training()")
    print("="*70 + "\n")
