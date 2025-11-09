"""
ACTIVE INFERENCE - ATARI
========================

The Strange Loop Closes: Input → Understanding → Action → Effect → New Input

Philosophy:
  - No reward signal! Only prediction error.
  - Learn the physics of the game world by minimizing surprise.
  - Random exploration initially, world model emerges.
  - Action policy derived from predictions, not scores.

Natural Ratios:
  - φ (golden ratio) timescale factor: 1.618034
  - Fibonacci update frequencies
  - Blended MSE→SSIM for perception

The river flows where it must.

Author: Philipp Remy Bartholomäus
Date: November 9, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
from collections import deque
import random

# Install Gymnasium if needed
try:
    import gymnasium as gym
except ImportError:
    print("Installing Gymnasium with Atari support...")
    import subprocess
    subprocess.check_call(['pip', 'install', '-q', 'gymnasium[atari]', 'gymnasium[accept-rom-license]'])
    import gymnasium as gym

print("✓ Gymnasium loaded")


# ============================================================================
# GOLDEN RATIO CONSTANTS
# ============================================================================

φ = (1 + math.sqrt(5)) / 2  # ≈ 1.618034
print(f"φ (Golden Ratio): {φ:.6f}")

FIBONACCI = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]


# ============================================================================
# ATTENTION CURRICULUM SCHEDULER
# ============================================================================

class AttentionCurriculumScheduler:
    """
    Progressive attention: Learn control → interaction → full world.

    Developmental learning like infants:
      1. Control own body (mask opponent completely)
      2. Understand immediate interaction (ball visible)
      3. Full world model (opponent + strategy)

    Mask shrinks at golden ratio intervals.
    """

    def __init__(self,
                 control_steps=500,      # Phase 1: Pure control
                 interaction_steps=1000,  # Phase 2: Ball interaction
                 understanding_steps=1500, # Phase 3: Full understanding
                 total_steps=3000):       # Phase 4: Complete model
        self.control_steps = control_steps
        self.interaction_steps = interaction_steps
        self.understanding_steps = understanding_steps
        self.total_steps = total_steps

    def get_mask_amount(self, step):
        """
        Returns how much to mask (0.0 = no mask, 1.0 = full mask).

        Shrinks at golden ratio intervals:
          Step 0-500:   1.0 (100% masked - only your paddle)
          Step 500-1000: 0.618 (1/φ - ball visible)
          Step 1000-2000: 0.382 (1/φ² - opponent partially visible)
          Step 2000-3000: 0.236 (1/φ³ - almost full view)
          Step 3000+:    0.0 (no mask - complete view)
        """
        if step < self.control_steps:
            return 1.0
        elif step < self.interaction_steps:
            return 1.0 / φ  # 0.618
        elif step < self.understanding_steps:
            return 1.0 / (φ ** 2)  # 0.382
        elif step < self.total_steps:
            return 1.0 / (φ ** 3)  # 0.236
        else:
            return 0.0

    def use_difference_mode(self, step):
        """
        Use difference prediction during bootstrap phases.

        Difference = next_frame - current_frame makes action effects obvious!
        """
        return step < self.understanding_steps  # First two phases

    def get_phase_name(self, step):
        """Get human-readable phase name."""
        if step < self.control_steps:
            return "Control (learning my paddle)"
        elif step < self.interaction_steps:
            return "Interaction (ball physics)"
        elif step < self.understanding_steps:
            return "Understanding (opponent visible)"
        elif step < self.total_steps:
            return "Integration (full view)"
        else:
            return "Complete (world model)"


# ============================================================================
# FRAME PREPROCESSING
# ============================================================================

def preprocess_frame(frame, size=128):
    """
    Preprocess Atari frame for neural network.

    Atari frames are typically (210, 160, 3) RGB.
    We resize to (size, size, 3) for model.

    Args:
        frame: Numpy array (H, W, 3)
        size: Target size (default 128)

    Returns:
        Tensor (3, size, size) normalized to [0, 1]
    """
    # Resize using PyTorch
    from PIL import Image
    img = Image.fromarray(frame)
    img = img.resize((size, size), Image.LANCZOS)

    # To tensor (3, H, W) normalized
    tensor = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return tensor


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


# ============================================================================
# ACTION ENCODING
# ============================================================================

class ActionEncoder(nn.Module):
    """
    Encode discrete action as continuous vector.

    Action becomes part of input to forward model:
    (frame, action) → next_frame_prediction
    """
    def __init__(self, n_actions, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(n_actions, embed_dim)

    def forward(self, action_idx):
        """
        Args:
            action_idx: Tensor of shape (batch,) with action indices

        Returns:
            Embedded actions (batch, embed_dim)
        """
        return self.embedding(action_idx)


# ============================================================================
# FORWARD MODEL (WORLD MODEL)
# ============================================================================

class ForwardModel(nn.Module):
    """
    Predicts next frame given current frame + action.

    Uses dual-tetrahedral architecture for prediction.
    This is the WORLD MODEL - learns physics of game.
    """
    def __init__(self, img_size=128, latent_dim=128, n_actions=18):
        super().__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.n_actions = n_actions

        # Action encoder
        self.action_encoder = ActionEncoder(n_actions, embed_dim=64)

        # Dual tetrahedra (assumes Z_COUPLING imported)
        input_dim = img_size * img_size * 3 + 64  # Frame + action embedding
        output_dim = img_size * img_size * 3       # Next frame

        self.dual_tetra = DualTetrahedralNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            latent_dim=latent_dim,
            coupling_strength=0.5,
            output_mode="weighted"
        )

    def forward(self, frame, action):
        """
        Predict next frame.

        Args:
            frame: Current frame (batch, 3, H, W)
            action: Action indices (batch,)

        Returns:
            Predicted next frame (batch, 3, H, W)
        """
        batch_size = frame.size(0)

        # Flatten frame
        frame_flat = frame.reshape(batch_size, -1)

        # Encode action
        action_embed = self.action_encoder(action)

        # Concatenate frame + action
        input_vec = torch.cat([frame_flat, action_embed], dim=-1)

        # Predict next frame
        next_frame_flat = self.dual_tetra(input_vec)

        # Reshape to image
        next_frame = next_frame_flat.reshape(batch_size, 3, self.img_size, self.img_size)

        return next_frame


# ============================================================================
# RANDOM POLICY (Exploration)
# ============================================================================

class RandomPolicy:
    """
    Random action selection for exploration.

    Pure exploration - no optimization.
    World model learns from observed transitions.
    """
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def select_action(self, state):
        """
        Args:
            state: Current frame (ignored for random policy)

        Returns:
            Random action index
        """
        return random.randint(0, self.n_actions - 1)


# ============================================================================
# EXPERIENCE BUFFER
# ============================================================================

class ExperienceBuffer:
    """
    Store (frame, action, next_frame) transitions.

    Used for training forward model.
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, frame, action, next_frame):
        """Add transition to buffer."""
        self.buffer.append((frame, action, next_frame))

    def sample(self, batch_size):
        """Sample random batch."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        frames = torch.stack([b[0] for b in batch])
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long)
        next_frames = torch.stack([b[2] for b in batch])

        return frames, actions, next_frames

    def __len__(self):
        return len(self.buffer)


# ============================================================================
# SSIM LOSS
# ============================================================================

def ssim_loss(pred, target, window_size=11):
    """SSIM loss for frame prediction."""
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    def gaussian_window(size, sigma=1.5):
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.unsqueeze(0) * g.unsqueeze(1)

    window = gaussian_window(window_size).to(pred.device)
    window = window.unsqueeze(0).unsqueeze(0)

    def apply_filter(img, window):
        C = img.size(1)
        _window = window.expand(C, 1, -1, -1)
        return F.conv2d(img, _window, padding=window_size // 2, groups=C)

    mu1 = apply_filter(pred, window)
    mu2 = apply_filter(target, window)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = apply_filter(pred ** 2, window) - mu1_sq
    sigma2_sq = apply_filter(target ** 2, window) - mu2_sq
    sigma12 = apply_filter(pred * target, window) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return 1.0 - ssim_map.mean()


# ============================================================================
# BLENDED LOSS SCHEDULER
# ============================================================================

class BlendedLossScheduler:
    """Blend MSE→SSIM during training."""
    def __init__(self, bootstrap_steps=1000, total_steps=10000):
        self.bootstrap_steps = bootstrap_steps
        self.total_steps = total_steps

    def get_weights(self, step):
        """Returns (mse_weight, ssim_weight)."""
        if step < self.bootstrap_steps:
            return 1.0, 0.0

        progress = (step - self.bootstrap_steps) / (self.total_steps - self.bootstrap_steps)
        progress = min(progress, 1.0)

        mse_weight = 1.0 - (0.8 * progress)   # 1.0 → 0.2
        ssim_weight = 0.8 * progress           # 0.0 → 0.8

        return mse_weight, ssim_weight


# ============================================================================
# ACTIVE INFERENCE TRAINER
# ============================================================================

class ActiveInferenceTrainer:
    """
    Train forward model through active inference.

    The Strange Loop:
      1. Observe state
      2. Predict what will happen if action taken
      3. Take action
      4. Observe actual next state
      5. Update model to minimize prediction error
      6. Repeat
    """
    def __init__(
        self,
        env_name='ALE/Pong-v5',
        img_size=128,
        latent_dim=128,
        base_lr=0.0001,
        buffer_capacity=10000,
        batch_size=16,
        device=None
    ):
        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        print(f"🖥️  Device: {self.device}")
        self.img_size = img_size
        self.batch_size = batch_size

        # Create environment
        print(f"\n🎮 Creating environment: {env_name}")
        self.env = gym.make(env_name, render_mode='rgb_array')
        self.n_actions = self.env.action_space.n
        print(f"   Actions: {self.n_actions}")
        print(f"   Observation: {self.env.observation_space.shape}")

        # Create forward model
        print(f"\n🧠 Creating forward model...")
        self.model = ForwardModel(
            img_size=img_size,
            latent_dim=latent_dim,
            n_actions=self.n_actions
        ).to(device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"   Parameters: {total_params:,}")

        # Golden ratio optimizer (nested timescales)
        print(f"\n⚡ Creating optimizer with φ-based timescales...")
        print(f"   φ = {φ:.6f}")

        # Group parameters by hierarchy
        vertex_params = []
        edge_params = []
        face_params = []
        coupling_params = []
        action_params = list(self.model.action_encoder.parameters())

        dual = self.model.dual_tetra

        # Vertex params
        vertex_params.extend(dual.linear_net.embed.parameters())
        vertex_params.extend(dual.nonlinear_net.embed.parameters())
        vertex_params.extend(dual.output_projection.parameters())

        # Edge params
        edge_params.extend(dual.linear_net.edge_modules.parameters())
        edge_params.extend(dual.nonlinear_net.edge_modules.parameters())

        # Face params
        face_params.extend(dual.linear_net.face_modules.parameters())
        face_params.extend(dual.nonlinear_net.face_modules.parameters())

        # Coupling params
        coupling_params.extend(dual.linear_to_nonlinear.parameters())
        coupling_params.extend(dual.nonlinear_to_linear.parameters())

        self.optimizer = optim.Adam([
            {'params': action_params, 'lr': base_lr},                    # Fast
            {'params': vertex_params, 'lr': base_lr},                    # Fast
            {'params': edge_params, 'lr': base_lr / φ},                  # Medium
            {'params': face_params, 'lr': base_lr / (φ ** 2)},          # Slow
            {'params': coupling_params, 'lr': base_lr / (φ ** 3)}       # Slowest
        ])

        print(f"   Action encoder:  LR = {base_lr:.6f}")
        print(f"   Vertices:        LR = {base_lr:.6f}")
        print(f"   Edges:           LR = {base_lr/φ:.6f}")
        print(f"   Faces:           LR = {base_lr/(φ**2):.6f}")
        print(f"   Coupling:        LR = {base_lr/(φ**3):.6f}")

        # Experience buffer
        self.buffer = ExperienceBuffer(capacity=buffer_capacity)

        # Random policy (exploration)
        self.policy = RandomPolicy(self.n_actions)

        # Loss scheduler
        self.loss_scheduler = BlendedLossScheduler()

        # Attention curriculum (developmental learning!)
        self.attention_curriculum = AttentionCurriculumScheduler()

        # Metrics
        self.step_count = 0
        self.episode_count = 0
        self.history = {
            'mse': [], 'ssim': [], 'total': [],
            'mse_weight': [], 'ssim_weight': [],
            'mask_amount': [], 'phase': []
        }

    def collect_experience(self, n_steps=100):
        """
        Collect transitions by taking random actions.

        Returns:
            Number of episodes completed
        """
        print(f"\n📦 Collecting {n_steps} experience transitions...")

        frame, _ = self.env.reset()
        frame = preprocess_frame(frame, self.img_size)
        episodes_done = 0

        for step in range(n_steps):
            # Random action
            action = self.policy.select_action(frame)

            # Take action
            next_frame_raw, reward, terminated, truncated, info = self.env.step(action)
            next_frame = preprocess_frame(next_frame_raw, self.img_size)

            # Store transition
            self.buffer.add(frame, action, next_frame)

            # Next state
            frame = next_frame

            if terminated or truncated:
                frame, _ = self.env.reset()
                frame = preprocess_frame(frame, self.img_size)
                episodes_done += 1

        print(f"   ✓ Collected {len(self.buffer)} total transitions")
        print(f"   ✓ Completed {episodes_done} episodes")
        return episodes_done

    def train_step(self):
        """
        Single training step on batch from buffer.

        Minimizes prediction error (active inference).
        Uses attention curriculum for developmental learning.
        """
        if len(self.buffer) < self.batch_size:
            return None

        # Sample batch
        frames, actions, next_frames = self.buffer.sample(self.batch_size)
        frames = frames.to(self.device)
        actions = actions.to(self.device)
        next_frames = next_frames.to(self.device)

        # === ATTENTION CURRICULUM ===
        # Get current curriculum state
        mask_amount = self.attention_curriculum.get_mask_amount(self.step_count)
        use_difference = self.attention_curriculum.use_difference_mode(self.step_count)
        phase_name = self.attention_curriculum.get_phase_name(self.step_count)

        # Apply attention mask to inputs
        masked_frames = apply_attention_mask(frames, mask_amount, player='right')

        # Predict next frame
        pred_next_frames = self.model(masked_frames, actions)

        # === DIFFERENCE MODE ===
        # During early phases, predict CHANGE not STATE
        if use_difference:
            # IMPORTANT: Compute difference using MASKED frames!
            # Model only sees masked input, so target should also be masked difference
            masked_next_frames = apply_attention_mask(next_frames, mask_amount, player='right')
            target = masked_next_frames - masked_frames  # Masked difference
            prediction = pred_next_frames - masked_frames  # Prediction relative to masked input
        else:
            # Normal state prediction (also masked!)
            target = apply_attention_mask(next_frames, mask_amount, player='right')
            prediction = pred_next_frames

        # === BLENDED LOSS ===
        mse_weight, ssim_weight = self.loss_scheduler.get_weights(self.step_count)

        mse = F.mse_loss(prediction, target)

        if ssim_weight > 0:
            ssim = ssim_loss(prediction, target)
            loss = mse_weight * mse + ssim_weight * ssim
        else:
            ssim = torch.tensor(0.0)
            loss = mse

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1

        # Track metrics
        metrics = {
            'mse': mse.item(),
            'ssim': ssim.item() if ssim_weight > 0 else 0.0,
            'total': loss.item(),
            'mse_weight': mse_weight,
            'ssim_weight': ssim_weight,
            'mask_amount': mask_amount,
            'phase': phase_name,
            'difference_mode': use_difference
        }

        return metrics

    def train_loop(self, n_episodes=10, train_steps_per_episode=50):
        """
        Main training loop.

        Alternates between:
          - Collecting experience (acting in world)
          - Training forward model (minimizing prediction error)
        """
        print("\n" + "="*70)
        print("🌀 ACTIVE INFERENCE TRAINING")
        print("="*70)
        print(f"Episodes: {n_episodes}")
        print(f"Training steps per episode: {train_steps_per_episode}")
        print("="*70 + "\n")

        for episode in range(n_episodes):
            # Collect experience
            episodes_done = self.collect_experience(n_steps=100)
            self.episode_count += episodes_done

            # Train on experience
            print(f"\n🎓 Training for {train_steps_per_episode} steps...")
            episode_metrics = {'mse': [], 'ssim': [], 'total': []}

            for _ in range(train_steps_per_episode):
                metrics = self.train_step()
                if metrics:
                    episode_metrics['mse'].append(metrics['mse'])
                    episode_metrics['ssim'].append(metrics['ssim'])
                    episode_metrics['total'].append(metrics['total'])

                    self.history['mse'].append(metrics['mse'])
                    self.history['ssim'].append(metrics['ssim'])
                    self.history['total'].append(metrics['total'])
                    self.history['mse_weight'].append(metrics['mse_weight'])
                    self.history['ssim_weight'].append(metrics['ssim_weight'])
                    self.history['mask_amount'].append(metrics['mask_amount'])
                    self.history['phase'].append(metrics['phase'])

            # Log
            avg_mse = np.mean(episode_metrics['mse']) if episode_metrics['mse'] else 0
            avg_ssim = np.mean(episode_metrics['ssim']) if episode_metrics['ssim'] else 0
            avg_total = np.mean(episode_metrics['total']) if episode_metrics['total'] else 0

            # Current curriculum state
            current_phase = metrics['phase'] if metrics else 'Initializing'
            current_mask = metrics['mask_amount'] if metrics else 0.0
            is_difference = metrics.get('difference_mode', False) if metrics else False

            print(f"\n📊 Episode {episode+1}/{n_episodes}")
            print(f"   Total episodes seen: {self.episode_count}")
            print(f"   Training steps: {self.step_count}")
            print(f"   🌱 Curriculum: {current_phase}")
            print(f"   👁️  Attention mask: {current_mask*100:.1f}% (0%=full view)")
            print(f"   🔄 Mode: {'Difference' if is_difference else 'State'} prediction")
            print(f"   Loss (MSE): {avg_mse:.6f}")
            print(f"   Loss (SSIM): {avg_ssim:.6f}")
            print(f"   Loss (Total): {avg_total:.6f}")
            print(f"   Buffer size: {len(self.buffer)}")

        print("\n✅ Training complete!\n")

    def visualize_predictions(self, n_samples=4):
        """Visualize model predictions vs actual."""
        if len(self.buffer) < n_samples:
            print("Not enough samples in buffer")
            return

        self.model.eval()

        frames, actions, next_frames = self.buffer.sample(n_samples)
        frames = frames.to(self.device)
        actions = actions.to(self.device)

        with torch.no_grad():
            pred_next_frames = self.model(frames, actions)

        # Plot
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_samples):
            # Current frame
            axes[i, 0].imshow(frames[i].cpu().permute(1, 2, 0).numpy())
            axes[i, 0].set_title(f'Current Frame\nAction: {actions[i].item()}')
            axes[i, 0].axis('off')

            # Predicted next
            axes[i, 1].imshow(pred_next_frames[i].cpu().permute(1, 2, 0).numpy().clip(0, 1))
            axes[i, 1].set_title('Predicted Next')
            axes[i, 1].axis('off')

            # Actual next
            axes[i, 2].imshow(next_frames[i].cpu().permute(1, 2, 0).numpy())
            axes[i, 2].set_title('Actual Next')
            axes[i, 2].axis('off')

        plt.tight_layout()
        plt.savefig('active_inference_predictions.png', dpi=150, bbox_inches='tight')
        print("✓ Saved to 'active_inference_predictions.png'")
        plt.show()

        self.model.train()

    def save_checkpoint(self, path='/content/drive/MyDrive/tetrahedral_checkpoints/atari_checkpoint.pt'):
        """
        Save full checkpoint: model, optimizer, buffer, metrics.

        Use this instead of save_model() to preserve training state.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'buffer': self.buffer.buffer,  # Save the deque
            'history': self.history,
            'img_size': self.img_size,
            'batch_size': self.batch_size,
            'n_actions': self.n_actions,
            'curriculum_phase': self.attention_curriculum.get_phase_name(self.step_count),
            'mask_amount': self.attention_curriculum.get_mask_amount(self.step_count)
        }

        torch.save(checkpoint, path)
        print(f"💾 Checkpoint saved to {path}")
        print(f"   Step: {self.step_count}, Episodes: {self.episode_count}")
        print(f"   🌱 Curriculum: {checkpoint['curriculum_phase']}")
        print(f"   👁️  Attention: {checkpoint['mask_amount']*100:.1f}% masked")

    def load_checkpoint(self, path='/content/drive/MyDrive/tetrahedral_checkpoints/atari_checkpoint.pt'):
        """
        Load full checkpoint and resume training.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.buffer.buffer = checkpoint['buffer']
        self.history = checkpoint['history']

        phase = checkpoint.get('curriculum_phase', 'Unknown')
        mask = checkpoint.get('mask_amount', 0.0)

        print(f"✅ Checkpoint loaded from {path}")
        print(f"   Resuming from step {self.step_count}, episode {self.episode_count}")
        print(f"   🌱 Curriculum: {phase}")
        print(f"   👁️  Attention: {mask*100:.1f}% masked")
        print(f"   Buffer size: {len(self.buffer)}")

    def save_model(self, path='/content/drive/MyDrive/saved_models/forward_model.pth'):
        """Save model weights only (for deployment, not resuming training)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"✓ Model saved to {path}")


# ============================================================================
# QUICK START
# ============================================================================

if __name__ == "__main__":
    """
    Quick start for active inference training.

    Usage in Colab:
      1. Run W, X, Y, Z cells first (load architecture)
      2. Run this cell
      3. trainer = ActiveInferenceTrainer()
      4. trainer.train_loop(n_episodes=10)
      5. trainer.visualize_predictions()
    """

    print("\n" + "="*70)
    print("🌀 ACTIVE INFERENCE - ATARI")
    print("="*70)
    print("\nThe Strange Loop:")
    print("  Input → Understanding → Action → Effect → New Input")
    print("\nPhilosophy:")
    print("  • No reward signal - only prediction error")
    print("  • Learn physics of game world")
    print("  • φ-based nested timescales")
    print("  • Blended MSE→SSIM perception")
    print("\nReady to close the loop! 🎮")
    print("="*70 + "\n")

    print("Example usage:")
    print("  trainer = ActiveInferenceTrainer(env_name='ALE/Pong-v5')")
    print("  trainer.train_loop(n_episodes=10)")
    print("  trainer.visualize_predictions()")
    print("  trainer.save_model()")
