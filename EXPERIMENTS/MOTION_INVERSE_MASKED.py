"""
MOTION-BASED INVERSE MODEL - ACTION MASKED VERSION
==================================================

For continuously learning brains that play multiple games!

Key features:
- Full 18-action Atari space (consistent across all games)
- Per-game action masking (only valid actions contribute to loss)
- Scales to any Atari game without architecture changes

Author: Brain Builder Edition
Date: November 11, 2025
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

# Import the REAL tetrahedral architecture
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Z_COUPLING.Z_interface_coupling import DualTetrahedralNetwork

φ = (1 + 5**0.5) / 2


# ============================================================================
# ATARI ACTION SPACE DEFINITIONS
# ============================================================================

# Full Atari action space (18 actions) - Standard ALE ordering
ATARI_ACTION_NAMES = [
    'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN',
    'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT',
    'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
    'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
]

# Game-specific valid actions (meaningful actions only)
GAME_ACTION_MASKS = {
    'ALE/Pong-v5': [0, 2, 5],  # NOOP, UP, DOWN (paddle moves vertically only)
    'ALE/Breakout-v5': [0, 1, 3, 4],  # NOOP, FIRE, RIGHT, LEFT (horizontal paddle + fire to launch)
    'ALE/SpaceInvaders-v5': [0, 1, 3, 4, 10, 11, 12],  # Movement + fire combos
    'ALE/MontezumaRevenge-v5': list(range(18)),  # Full platformer - all 18 actions
}


def get_action_mask(env_name, n_actions=18):
    """
    Get binary mask for valid actions in a game.

    Returns:
        mask: Binary tensor [n_actions] where 1 = valid, 0 = invalid
        valid_indices: List of valid action indices
    """
    valid_indices = GAME_ACTION_MASKS.get(env_name, list(range(n_actions)))

    mask = torch.zeros(n_actions, dtype=torch.float32)
    mask[valid_indices] = 1.0

    return mask, valid_indices


# ============================================================================
# VISUAL ATTENTION MASKING (Curriculum Learning)
# ============================================================================

def apply_attention_mask(frame, mask_amount, player='right'):
    """
    Apply attention mask to focus on controllable region.

    Developmental curriculum: start with own paddle, gradually expand view.

    Pong is played LEFT-RIGHT (paddles on sides, not top/bottom):
      - Player paddle: right side (usually)
      - Opponent paddle: left side
      - Ball: moves horizontally between paddles

    Args:
        frame: Tensor (C, H, W) or (B, C, H, W)
        mask_amount: 0.0 = no mask, 1.0 = full mask (only player's side visible)
        player: 'right' or 'left' (which side is the agent's paddle)

    Returns:
        Masked frame (same shape as input)
    """
    if mask_amount == 0.0:
        return frame  # No masking

    # Handle both batched and single frames
    is_batched = frame.dim() == 4
    if not is_batched:
        frame = frame.unsqueeze(0)

    batch, channels, height, width = frame.shape

    # Create mask
    mask = torch.ones_like(frame)

    # Mask opponent's region (LEFT-RIGHT for Pong!)
    if player == 'right':
        # Player on right, opponent on left
        # mask_amount=1.0 → mask entire left half (columns 0 to width/2)
        # mask_amount=0.618 → mask left 30.9% of screen
        opponent_region_width = width // 2
        mask_width = int(opponent_region_width * mask_amount)
        mask[:, :, :, :mask_width] = 0.0
    else:
        # Player on left, opponent on right
        opponent_region_width = width // 2
        mask_width = int(opponent_region_width * mask_amount)
        mask[:, :, :, -mask_width:] = 0.0

    masked_frame = frame * mask

    if not is_batched:
        masked_frame = masked_frame.squeeze(0)

    return masked_frame


class MaskScheduler:
    """
    Curriculum scheduler for gradually revealing the screen.

    Start: mask_amount = 1.0 (only see own paddle)
    End: mask_amount = 0.0 (see everything)
    """
    def __init__(self, start_mask=1.0, end_mask=0.0, total_steps=5000):
        self.start_mask = start_mask
        self.end_mask = end_mask
        self.total_steps = total_steps

    def get_mask_amount(self, step):
        """Linear decay from start_mask to end_mask"""
        if step >= self.total_steps:
            return self.end_mask

        progress = step / self.total_steps
        return self.start_mask + (self.end_mask - self.start_mask) * progress


# ============================================================================
# STATE ENCODER (Frame → Abstract State)
# ============================================================================

class StateEncoder(nn.Module):
    """Encode raw frame to abstract state representation."""
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
        """Encode frame to state."""
        features = self.encoder(frame)
        features_flat = features.reshape(features.size(0), -1)
        state = self.fc(features_flat)
        return state


# ============================================================================
# ACTION ENCODER
# ============================================================================

class ActionEncoder(nn.Module):
    """Discrete action → continuous embedding"""
    def __init__(self, n_actions=18, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(n_actions, embed_dim)

    def forward(self, action_idx):
        return self.embedding(action_idx)


# ============================================================================
# REAL DUAL TETRAHEDRAL NETWORK (imported from Z_COUPLING)
# ============================================================================

# DualTetrahedralNetwork is now imported from Z_COUPLING.Z_interface_coupling
# This is the ACTUAL tetrahedral architecture with:
# - Linear tetrahedron (4 vertices, 6 edges, 4 faces, NO ReLU)
# - Nonlinear tetrahedron (4 vertices, 6 edges, 4 faces, WITH ReLU)
# - Inter-face coupling (4 bidirectional attention pairs)
# - Multi-timescale memory (golden ratio decay)


# ============================================================================
# FORWARD MODEL: (state, action) → next_state
# ============================================================================

class ForwardModel(nn.Module):
    """Forward model: Predict next state from current state and action."""
    def __init__(self, state_dim=128, latent_dim=128, n_actions=18):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions

        self.action_encoder = ActionEncoder(n_actions, embed_dim=64)

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
        """Predict next state."""
        action_embed = self.action_encoder(action)
        x = torch.cat([state, action_embed], dim=-1)
        next_state = self.dual_tetra(x)
        return next_state


# ============================================================================
# INVERSE MODEL: (state_t, state_t+1) → action (MASKED)
# ============================================================================

class MaskedInverseModel(nn.Module):
    """
    Inverse model with action masking.

    Only predicts over VALID actions for the current game.
    """
    def __init__(self, state_dim=128, latent_dim=128, n_actions=18):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions

        input_dim = state_dim * 2
        output_dim = n_actions

        self.dual_tetra = DualTetrahedralNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            latent_dim=latent_dim,
            coupling_strength=0.5,
            output_mode="weighted"
        )

    def forward(self, state_t, state_t1, action_mask=None):
        """
        Infer action from state transition.

        Args:
            state_t: (batch, state_dim)
            state_t1: (batch, state_dim)
            action_mask: (n_actions,) or (batch, n_actions) - 1 for valid, 0 for invalid

        Returns:
            action_logits: (batch, n_actions) - masked logits
        """
        x = torch.cat([state_t, state_t1], dim=-1)
        logits = self.dual_tetra(x)

        # Apply mask: set invalid actions to -inf (so they never get chosen)
        if action_mask is not None:
            # Expand mask to match batch size
            batch_size = logits.size(0)
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0).expand(batch_size, -1)  # (batch, n_actions)

            # Use masked_fill for proper broadcasting
            masked_logits = logits.masked_fill(action_mask == 0, -1e9)

            return masked_logits

        return logits


# ============================================================================
# COUPLED MODEL WITH MASKING
# ============================================================================

class MaskedCoupledModel(nn.Module):
    """
    Complete coupled forward+inverse model with action masking.

    Designed for continuous learning across multiple games!
    """
    def __init__(self, state_dim=128, latent_dim=128, n_actions=18):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions

        self.state_encoder = StateEncoder(state_dim=state_dim)
        self.forward_model = ForwardModel(state_dim, latent_dim, n_actions)
        self.inverse_model = MaskedInverseModel(state_dim, latent_dim, n_actions)

    def encode_state(self, frame):
        """Encode frame to state"""
        return self.state_encoder(frame)

    def compute_losses(self, frame_t, frame_t1, action, action_mask,
                      forward_weight=1.0,
                      inverse_weight=1.0,
                      consistency_weight=0.5,
                      visual_mask_amount=0.0,
                      player='right'):
        """
        Compute all losses with action masking and visual masking.

        Args:
            action_mask: (n_actions,) binary mask for valid actions
            visual_mask_amount: 0.0-1.0, amount to mask opponent's side
            player: 'right' or 'left' paddle
        """
        # Apply visual masking to inputs (curriculum learning)
        if visual_mask_amount > 0.0:
            frame_t = apply_attention_mask(frame_t, visual_mask_amount, player)
            frame_t1 = apply_attention_mask(frame_t1, visual_mask_amount, player)

        # Encode states
        state_t = self.encode_state(frame_t)
        state_t1 = self.encode_state(frame_t1)

        # Forward loss: Can we predict next state?
        pred_state_t1 = self.forward_model(state_t, action)
        loss_forward = F.mse_loss(pred_state_t1, state_t1)

        # Inverse loss: Can we infer action? (MASKED)
        action_logits = self.inverse_model(state_t, state_t1, action_mask)

        # Masked cross-entropy: only compute loss over valid actions
        loss_inverse = F.cross_entropy(action_logits, action)

        # Consistency loss: Do models agree?
        action_inferred = torch.argmax(action_logits.detach(), dim=-1)
        pred_state_t1_consistent = self.forward_model(state_t, action_inferred)
        loss_consistency = F.mse_loss(pred_state_t1_consistent, state_t1)

        # Total loss
        total_loss = (forward_weight * loss_forward +
                     inverse_weight * loss_inverse +
                     consistency_weight * loss_consistency)

        # Metrics (masked accuracy - only count valid actions)
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
# MASKED TRAINER WITH LIVE VISUALIZATION
# ============================================================================

class MaskedTrainer:
    """
    Trainer with action masking for multi-game learning.

    Key features:
    - Full 18-action space
    - Per-game valid action masks
    - Live visualization during training
    """
    def __init__(self,
                 env_name='ALE/Pong-v5',
                 state_dim=128,
                 latent_dim=128,
                 base_lr=0.0001,
                 buffer_capacity=10000,
                 batch_size=16,
                 device=None,
                 n_actions=18):  # Always 18 for full Atari space

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.batch_size = batch_size
        self.n_actions = n_actions

        print(f"🖥️  Device: {self.device}")

        # Environment
        print(f"🎮 Environment: {env_name}")
        self.env_name = env_name
        self.env = gym.make(env_name, render_mode='rgb_array', full_action_space=True)
        env_n_actions = self.env.action_space.n
        print(f"   Environment actions: {env_n_actions}")
        print(f"   Model actions (fixed): {n_actions}")

        # Get action mask for this game
        self.action_mask, self.valid_actions = get_action_mask(env_name, n_actions)
        self.action_mask = self.action_mask.to(self.device)

        print(f"   Valid actions: {self.valid_actions}")
        print(f"   Valid action names: {[ATARI_ACTION_NAMES[i] for i in self.valid_actions]}")

        # Model (always 18 actions)
        print(f"🧠 Creating masked coupled model...")
        self.model = MaskedCoupledModel(
            state_dim=state_dim,
            latent_dim=latent_dim,
            n_actions=n_actions
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"   Parameters: {total_params:,}")

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=base_lr)

        # Buffer
        self.buffer = ExperienceBuffer(capacity=buffer_capacity)

        # Visual attention curriculum (gradually reveal screen)
        self.mask_scheduler = MaskScheduler(
            start_mask=1.0,  # Start: only see own paddle
            end_mask=0.0,    # End: see everything
            total_steps=2000  # Reveal over 2000 steps
        )
        self.player_side = 'right'  # Which paddle we control

        # Metrics
        self.step_count = 0
        self.episode_count = 0
        self.history = {
            'forward': [], 'inverse': [], 'consistency': [],
            'total': [], 'accuracy': [], 'mask_amount': []
        }

    def preprocess_frame(self, frame):
        """Preprocess frame"""
        from PIL import Image
        img = Image.fromarray(frame).resize((128, 128), Image.LANCZOS)
        tensor = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return tensor

    def collect_experience(self, n_steps=100):
        """Collect experience (only sample from valid actions)"""
        frame_t, _ = self.env.reset()
        frame_t = self.preprocess_frame(frame_t)
        episodes_done = 0

        for step in range(n_steps):
            # Sample only from valid actions
            action = random.choice(self.valid_actions)

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
        """Single training step with masked metrics"""
        if len(self.buffer) < self.batch_size:
            return None

        # Get current visual mask amount (curriculum)
        visual_mask_amount = self.mask_scheduler.get_mask_amount(self.step_count)

        # Sample batch
        frames_t, actions, frames_t1 = self.buffer.sample(self.batch_size)
        frames_t = frames_t.to(self.device)
        actions = actions.to(self.device)
        frames_t1 = frames_t1.to(self.device)

        # Compute losses with both action and visual masking
        losses = self.model.compute_losses(
            frames_t, frames_t1, actions,
            action_mask=self.action_mask,
            visual_mask_amount=visual_mask_amount,
            player=self.player_side
        )

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

        # Track mask amount
        self.history['mask_amount'].append(visual_mask_amount)

        return losses

    def train_loop(self, n_episodes=10, steps_per_episode=50, verbose=True):
        """Training loop with progress tracking"""
        print("\n" + "="*70)
        print("🌀 MASKED TRAINING LOOP")
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
                current_mask = self.mask_scheduler.get_mask_amount(self.step_count)

                print(f"📊 Episode {episode+1}/{n_episodes}")
                print(f"   Steps: {self.step_count}")
                print(f"   Forward loss: {avg_forward:.6f}")
                print(f"   Inverse loss: {avg_inverse:.6f}")
                print(f"   Accuracy: {avg_accuracy*100:.2f}% (random={100/len(self.valid_actions):.2f}%)")
                print(f"   Visual mask: {current_mask*100:.0f}% (100%=only own paddle, 0%=see all)")
                print(f"   Buffer: {len(self.buffer)}")
                print()

    def live_visualize(self, n_samples=4):
        """
        Live visualization showing:
        - Full frames (ground truth)
        - Masked frames (what model sees - curriculum learning)
        - Predictions
        - Valid vs invalid actions
        """
        from IPython.display import clear_output

        if len(self.buffer) < n_samples:
            print("Not enough data yet...")
            return

        # Get current mask amount
        visual_mask_amount = self.mask_scheduler.get_mask_amount(self.step_count)

        self.model.eval()

        # Reset memory fields in DualTetrahedralNetwork (batch size changes)
        # The multi-timescale memory stores batch-specific state
        # We need to reset it when batch size changes from training (16) to viz (4)
        self.model.forward_model.dual_tetra.fast_field = None
        self.model.forward_model.dual_tetra.medium_field = None
        self.model.forward_model.dual_tetra.slow_field = None
        self.model.inverse_model.dual_tetra.fast_field = None
        self.model.inverse_model.dual_tetra.medium_field = None
        self.model.inverse_model.dual_tetra.slow_field = None

        frames_t, actions, frames_t1 = self.buffer.sample(n_samples)
        frames_t = frames_t.to(self.device)
        frames_t1 = frames_t1.to(self.device)
        actions = actions.to(self.device)

        # Apply visual masking (what model actually sees)
        masked_frames_t = apply_attention_mask(frames_t, visual_mask_amount, self.player_side)
        masked_frames_t1 = apply_attention_mask(frames_t1, visual_mask_amount, self.player_side)

        with torch.no_grad():
            # Encode MASKED frames (like during training)
            state_t = self.model.encode_state(masked_frames_t)
            state_t1 = self.model.encode_state(masked_frames_t1)
            logits = self.model.inverse_model(state_t, state_t1, self.action_mask)
            pred_actions = torch.argmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)

        clear_output(wait=True)

        fig = plt.figure(figsize=(24, n_samples * 3.5))

        for i in range(n_samples):
            # Full Frame t (ground truth)
            ax1 = plt.subplot(n_samples, 5, i*5 + 1)
            ax1.imshow(frames_t[i].cpu().permute(1, 2, 0).numpy())
            ax1.set_title(f'Full Frame t', fontsize=10, fontweight='bold')
            ax1.axis('off')

            # Masked Frame t (what model sees)
            ax2 = plt.subplot(n_samples, 5, i*5 + 2)
            ax2.imshow(masked_frames_t[i].cpu().permute(1, 2, 0).numpy())
            ax2.set_title(f'Model Sees (Mask={visual_mask_amount:.2f})', fontsize=10, fontweight='bold', color='purple')
            ax2.axis('off')

            # Full Frame t+1
            ax3 = plt.subplot(n_samples, 5, i*5 + 3)
            ax3.imshow(frames_t1[i].cpu().permute(1, 2, 0).numpy())
            ax3.set_title(f'Full Frame t+1', fontsize=10, fontweight='bold')
            ax3.axis('off')

            # State diff visualization
            ax4 = plt.subplot(n_samples, 5, i*5 + 4)
            state_diff = (state_t1[i] - state_t[i]).cpu().numpy().reshape(16, 8)
            im = ax4.imshow(state_diff, cmap='RdBu_r', aspect='auto')
            ax4.set_title('State Change', fontsize=10, fontweight='bold')
            ax4.axis('off')
            plt.colorbar(im, ax=ax4, fraction=0.046)

            # Action prediction
            ax5 = plt.subplot(n_samples, 5, i*5 + 5)
            true_action = actions[i].item()
            pred_action = pred_actions[i].item()

            # Only show valid actions
            valid_probs = probs[i][self.valid_actions].cpu().numpy()
            valid_names = [ATARI_ACTION_NAMES[idx] for idx in self.valid_actions]

            bars = ax5.bar(range(len(self.valid_actions)), valid_probs)

            # Color bars
            true_idx_in_valid = self.valid_actions.index(true_action)
            pred_idx_in_valid = self.valid_actions.index(pred_action) if pred_action in self.valid_actions else -1

            bars[true_idx_in_valid].set_color('green')
            if pred_idx_in_valid >= 0:
                bars[pred_idx_in_valid].set_color('red' if pred_action != true_action else 'green')

            ax5.set_xticks(range(len(self.valid_actions)))
            ax5.set_xticklabels(valid_names, rotation=45, ha='right', fontsize=9)
            ax5.set_ylim([0, 1])
            ax5.axhline(1/len(self.valid_actions), color='gray', linestyle='--', alpha=0.5, label='random')

            # Title with color
            correct = pred_action == true_action
            ax5.set_title(f'TRUE: {ATARI_ACTION_NAMES[true_action]}\nPRED: {ATARI_ACTION_NAMES[pred_action]}',
                         fontsize=10, fontweight='bold',
                         color='green' if correct else 'red')

        # Add overall stats at top
        recent_acc = self.history["accuracy"][-1] if self.history["accuracy"] else 0
        mask_pct = visual_mask_amount * 100
        fig.suptitle(f'STEP {self.step_count} | Buffer: {len(self.buffer)} | Accuracy: {recent_acc*100:.1f}% | Mask: {mask_pct:.0f}% | Valid: {len(self.valid_actions)}/{self.n_actions}',
                     fontsize=14, fontweight='bold', y=0.995)

        plt.tight_layout()
        plt.show()

        # Reset memory again before returning to training (batch size will change back to 16)
        self.model.forward_model.dual_tetra.fast_field = None
        self.model.forward_model.dual_tetra.medium_field = None
        self.model.forward_model.dual_tetra.slow_field = None
        self.model.inverse_model.dual_tetra.fast_field = None
        self.model.inverse_model.dual_tetra.medium_field = None
        self.model.inverse_model.dual_tetra.slow_field = None

        self.model.train()

    def train_with_live_viz(self, n_episodes=10, steps_per_episode=50, viz_every=2):
        """Training loop with LIVE visualization every N episodes"""
        print("\n" + "="*70)
        print("🌀 TRAINING WITH LIVE VISUALIZATION")
        print("="*70 + "\n")

        for episode in range(n_episodes):
            # Collect
            self.collect_experience(n_steps=100)

            # Train
            for _ in range(steps_per_episode):
                self.train_step()

            # Show live viz
            if (episode + 1) % viz_every == 0:
                print(f"\n📊 Episode {episode+1}/{n_episodes}")
                self.live_visualize(n_samples=4)

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
        axes[0, 1].axhline(np.log(len(self.valid_actions)), color='r', linestyle='--', label='Random baseline')
        axes[0, 1].set_title('Inverse Loss (Cross-Entropy)')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('CE Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # Accuracy
        axes[1, 0].plot(np.array(self.history['accuracy']) * 100)
        axes[1, 0].axhline(100/len(self.valid_actions), color='r', linestyle='--', label='Random baseline')
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
        plt.savefig('training_masked.png', dpi=150, bbox_inches='tight')
        print("✅ Saved: training_masked.png")
        plt.show()


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧠 MASKED MOTION INVERSE MODEL")
    print("="*70)
    print("\nFeatures:")
    print("  ✅ Full 18-action Atari space")
    print("  ✅ Per-game action masking")
    print("  ✅ Live visualization during training")
    print("  ✅ Scales to any Atari game")
    print("\nUsage:")
    print("  trainer = MaskedTrainer(env_name='ALE/Pong-v5')")
    print("  trainer.train_with_live_viz(n_episodes=10, viz_every=2)")
    print("="*70 + "\n")
