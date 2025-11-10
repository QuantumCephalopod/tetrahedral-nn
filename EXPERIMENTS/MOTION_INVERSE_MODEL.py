"""
MOTION-BASED ACTIVE INFERENCE WITH INVERSE MODEL
=================================================

The Strange Loop ACTUALLY Closes: Agency through Causality

Philosophy:
  - Motion is fundamental (not static frames)
  - Forward model: (motion, action) → next_motion (predict consequences)
  - Inverse model: (motion, next_motion) → action (infer causality)
  - Policy: Synthesize both for action selection with understanding

The True Loop:
  Perceive motion → Infer cause (inverse)
                 → Predict consequence (forward)
                 → Select action (policy with agency)
                 → Execute
                 → Observe motion
                 → Update models

This is how mirror neurons work: understand by simulating!

Author: Philipp Remy Bartholomäus & Claude
Date: November 10, 2025
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

try:
    import gymnasium as gym
except ImportError:
    print("Installing Gymnasium...")
    import subprocess
    subprocess.check_call(['pip', 'install', '-q', 'gymnasium[atari]', 'gymnasium[accept-rom-license]'])
    import gymnasium as gym

print("✓ Gymnasium loaded")

# Golden ratio
φ = (1 + math.sqrt(5)) / 2


# ============================================================================
# MOTION EXTRACTION (Optical Flow Approximation)
# ============================================================================

class MotionExtractor(nn.Module):
    """
    Extract motion representation from frame pairs.

    Approximates optical flow using learned features.
    Motion is the FUNDAMENTAL perceptual unit, not static images.

    Philosophy:
      - Brains sense motion, not images
      - Images derive meaning FROM motion
      - Learning frames = learning 3D from shadows (wrong!)
      - Learning motion = learning causality (right!)
    """
    def __init__(self, img_size=128, motion_dim=128):
        super().__init__()
        self.img_size = img_size
        self.motion_dim = motion_dim

        # Convolutional motion detector
        # Inspired by V1 motion-sensitive cells
        self.motion_conv = nn.Sequential(
            # Spatial feature extraction
            nn.Conv2d(6, 32, kernel_size=7, stride=2, padding=3),  # 6 = 2 frames × 3 channels
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Motion-sensitive features
            nn.AdaptiveAvgPool2d((4, 4)),  # Spatial pooling
        )

        # Motion embedding (spatial → abstract motion)
        self.motion_embed = nn.Linear(256 * 4 * 4, motion_dim)

    def forward(self, frame_t, frame_t1):
        """
        Extract motion from frame pair.

        Args:
            frame_t: Current frame (batch, 3, H, W)
            frame_t1: Next frame (batch, 3, H, W)

        Returns:
            Motion representation (batch, motion_dim)
        """
        # Concatenate frames (temporal context)
        frame_pair = torch.cat([frame_t, frame_t1], dim=1)  # (batch, 6, H, W)

        # Extract motion features
        motion_features = self.motion_conv(frame_pair)  # (batch, 256, 4, 4)
        motion_flat = motion_features.reshape(motion_features.size(0), -1)

        # Abstract motion representation
        motion = self.motion_embed(motion_flat)  # (batch, motion_dim)

        return motion

    def compute_difference(self, frame_t, frame_t1):
        """
        Simple motion baseline: frame difference.

        Returns:
            Difference tensor (batch, 3, H, W)
        """
        return frame_t1 - frame_t


# ============================================================================
# ACTION ENCODER
# ============================================================================

class ActionEncoder(nn.Module):
    """Encode discrete action as continuous vector."""
    def __init__(self, n_actions, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(n_actions, embed_dim)

    def forward(self, action_idx):
        """Args: action_idx (batch,) → embedded (batch, embed_dim)"""
        return self.embedding(action_idx)


# ============================================================================
# FORWARD MODEL (Motion → Motion)
# ============================================================================

class ForwardMotionModel(nn.Module):
    """
    Forward model in MOTION SPACE.

    Learns: (motion_t, action) → motion_t+1

    Predicts sensory consequences of actions.
    "If I do X, what motion will I perceive?"
    """
    def __init__(self, motion_dim=128, latent_dim=128, n_actions=18):
        super().__init__()
        self.motion_dim = motion_dim
        self.latent_dim = latent_dim
        self.n_actions = n_actions

        # Action encoder
        self.action_encoder = ActionEncoder(n_actions, embed_dim=64)

        # Dual tetrahedra for motion prediction
        input_dim = motion_dim + 64  # motion + action
        output_dim = motion_dim       # predicted next motion

        self.dual_tetra = DualTetrahedralNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            latent_dim=latent_dim,
            coupling_strength=0.5,
            output_mode="weighted"
        )

    def forward(self, motion, action):
        """
        Predict next motion from current motion and action.

        Args:
            motion: Current motion (batch, motion_dim)
            action: Action indices (batch,)

        Returns:
            Predicted next motion (batch, motion_dim)
        """
        # Encode action
        action_embed = self.action_encoder(action)

        # Concatenate motion + action
        input_vec = torch.cat([motion, action_embed], dim=-1)

        # Predict next motion
        next_motion = self.dual_tetra(input_vec)

        return next_motion


# ============================================================================
# INVERSE MODEL (Motion → Action)
# ============================================================================

class InverseMotionModel(nn.Module):
    """
    Inverse model in MOTION SPACE.

    Learns: (motion_t, motion_t+1) → action

    Infers causality: "What action caused this motion transition?"

    This is MIRROR NEURONS - understanding actions by simulating them!
    You see motion → infer the action that caused it.
    """
    def __init__(self, motion_dim=128, latent_dim=128, n_actions=18):
        super().__init__()
        self.motion_dim = motion_dim
        self.latent_dim = latent_dim
        self.n_actions = n_actions

        # Dual tetrahedra for action inference
        input_dim = motion_dim * 2  # current + next motion
        output_dim = n_actions       # action logits

        self.dual_tetra = DualTetrahedralNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            latent_dim=latent_dim,
            coupling_strength=0.5,
            output_mode="weighted"
        )

    def forward(self, motion_t, motion_t1):
        """
        Infer action from motion transition.

        Args:
            motion_t: Current motion (batch, motion_dim)
            motion_t1: Next motion (batch, motion_dim)

        Returns:
            Action logits (batch, n_actions)
        """
        # Concatenate motion pair
        motion_transition = torch.cat([motion_t, motion_t1], dim=-1)

        # Infer action
        action_logits = self.dual_tetra(motion_transition)

        return action_logits

    def predict_action(self, motion_t, motion_t1):
        """
        Predict most likely action.

        Returns:
            Action index (batch,)
        """
        logits = self.forward(motion_t, motion_t1)
        return torch.argmax(logits, dim=-1)


# ============================================================================
# COUPLED MODEL (Forward + Inverse)
# ============================================================================

class CoupledMotionModel(nn.Module):
    """
    Coupled forward and inverse models.

    These must be learned TOGETHER with consistency constraint:
      - Inverse: (s, s') → a
      - Forward: (s, a) → s'
      - Consistency: Forward(s, Inverse(s, s')) ≈ s'

    This ensures:
      1. Inverse learns correct action causality
      2. Forward learns correct action consequences
      3. Models agree on physics of world
    """
    def __init__(self, motion_dim=128, latent_dim=128, n_actions=18):
        super().__init__()
        self.motion_dim = motion_dim
        self.latent_dim = latent_dim
        self.n_actions = n_actions

        # Motion extractor
        self.motion_extractor = MotionExtractor(
            img_size=128,
            motion_dim=motion_dim
        )

        # Forward model
        self.forward_model = ForwardMotionModel(
            motion_dim=motion_dim,
            latent_dim=latent_dim,
            n_actions=n_actions
        )

        # Inverse model
        self.inverse_model = InverseMotionModel(
            motion_dim=motion_dim,
            latent_dim=latent_dim,
            n_actions=n_actions
        )

    def extract_motion(self, frame_t, frame_t1):
        """Extract motion from frames."""
        return self.motion_extractor(frame_t, frame_t1)

    def forward_loss(self, motion_t, action, motion_t1):
        """
        Forward model loss: Can we predict motion consequences?

        Args:
            motion_t: Current motion
            action: Action taken
            motion_t1: Actual next motion

        Returns:
            Forward prediction error
        """
        pred_motion_t1 = self.forward_model(motion_t, action)
        return F.mse_loss(pred_motion_t1, motion_t1)

    def inverse_loss(self, motion_t, motion_t1, action):
        """
        Inverse model loss: Can we infer action from motion?

        Args:
            motion_t: Current motion
            motion_t1: Next motion
            action: Actual action (ground truth)

        Returns:
            Action classification error
        """
        action_logits = self.inverse_model(motion_t, motion_t1)
        return F.cross_entropy(action_logits, action)

    def consistency_loss(self, motion_t, motion_t1):
        """
        Consistency loss: Do forward and inverse agree?

        Loop:
          1. Infer action from motion: a_inferred = Inverse(s, s')
          2. Predict next motion: s'_pred = Forward(s, a_inferred)
          3. Error: |s'_pred - s'_actual|

        This ensures models agree on world physics!
        """
        # Infer action from motion transition
        action_logits = self.inverse_model(motion_t, motion_t1)
        action_inferred = torch.argmax(action_logits, dim=-1)

        # Predict motion using inferred action
        motion_t1_pred = self.forward_model(motion_t, action_inferred)

        # Consistency error
        return F.mse_loss(motion_t1_pred, motion_t1)

    def total_loss(self, frame_t, frame_t1, action,
                   forward_weight=1.0,
                   inverse_weight=1.0,
                   consistency_weight=0.5):
        """
        Combined loss for coupled training.

        Returns:
            Dictionary with all loss components
        """
        # Extract motion
        motion_t = self.extract_motion(frame_t, frame_t)  # Static reference
        motion_t1 = self.extract_motion(frame_t, frame_t1)  # Actual motion

        # Component losses
        loss_forward = self.forward_loss(motion_t, action, motion_t1)
        loss_inverse = self.inverse_loss(motion_t, motion_t1, action)
        loss_consistency = self.consistency_loss(motion_t, motion_t1)

        # Total loss
        total = (forward_weight * loss_forward +
                 inverse_weight * loss_inverse +
                 consistency_weight * loss_consistency)

        return {
            'total': total,
            'forward': loss_forward.item(),
            'inverse': loss_inverse.item(),
            'consistency': loss_consistency.item()
        }


# ============================================================================
# POLICY WITH AGENCY
# ============================================================================

class AgenticPolicy:
    """
    Policy that synthesizes forward and inverse models.

    This gives TRUE AGENCY:
      1. Forward: "What will happen if I do X?"
      2. Inverse: "What action leads to desired motion Y?"
      3. Synthesis: Select action that achieves goal with certainty

    Unlike previous policy which just minimized uncertainty,
    this one has INTENTIONALITY - it wants to achieve something!
    """
    def __init__(self,
                 coupled_model: CoupledMotionModel,
                 device: str = 'cpu',
                 temperature: float = 1.0):
        self.model = coupled_model
        self.device = device
        self.temperature = temperature
        self.n_actions = coupled_model.n_actions

    def select_action(self, frame_t, frame_t1_desired=None, exploration_rate=0.1):
        """
        Select action with agency.

        Two modes:
          1. Exploration: Random motion goals, discover what actions do
          2. Exploitation: Given desired motion, select best action

        Args:
            frame_t: Current frame
            frame_t1_desired: Desired next frame (None = explore)
            exploration_rate: Probability of random action

        Returns:
            Selected action index
        """
        # Random exploration
        if random.random() < exploration_rate:
            return random.randint(0, self.n_actions - 1)

        self.model.eval()

        with torch.no_grad():
            if frame_t.dim() == 3:
                frame_t = frame_t.unsqueeze(0)
            frame_t = frame_t.to(self.device)

            # Extract current motion (use frame_t twice for static baseline)
            motion_t = self.model.extract_motion(frame_t, frame_t)

            if frame_t1_desired is None:
                # EXPLORATION MODE
                # Try each action, see which creates interesting motion
                action_scores = []

                for action_idx in range(self.n_actions):
                    action = torch.tensor([action_idx], device=self.device)

                    # Predict resulting motion
                    pred_motion = self.model.forward_model(motion_t, action)

                    # Score by motion magnitude (interesting = high motion)
                    motion_magnitude = torch.norm(pred_motion - motion_t)
                    action_scores.append(motion_magnitude.item())

                # Select action with highest expected motion
                scores_tensor = torch.tensor(action_scores)
                probs = F.softmax(scores_tensor / self.temperature, dim=0)
                action = torch.multinomial(probs, 1).item()

            else:
                # EXPLOITATION MODE
                # Have goal motion, use inverse model to achieve it
                if frame_t1_desired.dim() == 3:
                    frame_t1_desired = frame_t1_desired.unsqueeze(0)
                frame_t1_desired = frame_t1_desired.to(self.device)

                # Extract desired motion
                motion_t1_desired = self.model.extract_motion(frame_t, frame_t1_desired)

                # Infer best action using inverse model
                action_logits = self.model.inverse_model(motion_t, motion_t1_desired)
                action = torch.argmax(action_logits, dim=-1).item()

        self.model.train()
        return action


# ============================================================================
# EXPERIENCE BUFFER
# ============================================================================

class MotionExperienceBuffer:
    """Store (frame_t, action, frame_t+1) transitions."""
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
# TRAINER
# ============================================================================

class MotionInverseTrainer:
    """
    Trainer for coupled motion models.

    Learns forward and inverse models simultaneously.
    """
    def __init__(self,
                 env_name='ALE/Pong-v5',
                 motion_dim=128,
                 latent_dim=128,
                 base_lr=0.0001,
                 buffer_capacity=10000,
                 batch_size=16,
                 device=None):

        # Device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        print(f"🖥️  Device: {self.device}")

        self.batch_size = batch_size

        # Environment
        print(f"\n🎮 Creating environment: {env_name}")
        self.env = gym.make(env_name, render_mode='rgb_array')
        self.n_actions = self.env.action_space.n
        print(f"   Actions: {self.n_actions}")

        # Coupled model
        print(f"\n🧠 Creating coupled forward+inverse model...")
        self.model = CoupledMotionModel(
            motion_dim=motion_dim,
            latent_dim=latent_dim,
            n_actions=self.n_actions
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"   Parameters: {total_params:,}")

        # Optimizer (φ-based hierarchy)
        print(f"\n⚡ Creating optimizer...")
        self.optimizer = optim.Adam(self.model.parameters(), lr=base_lr)

        # Buffer
        self.buffer = MotionExperienceBuffer(capacity=buffer_capacity)

        # Policy (with agency!)
        print(f"\n🎯 Creating Agentic Policy...")
        self.policy = AgenticPolicy(
            coupled_model=self.model,
            device=self.device,
            temperature=1.0
        )

        # Metrics
        self.step_count = 0
        self.episode_count = 0
        self.history = {
            'forward': [], 'inverse': [], 'consistency': [], 'total': []
        }

    def collect_experience(self, n_steps=100, exploration_rate=1.0):
        """Collect transitions using policy."""
        print(f"\n📦 Collecting {n_steps} transitions (exploration={exploration_rate:.2f})...")

        # Preprocess helper
        def preprocess(frame):
            from PIL import Image
            img = Image.fromarray(frame).resize((128, 128), Image.LANCZOS)
            return torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)

        frame_t, _ = self.env.reset()
        frame_t = preprocess(frame_t)
        episodes_done = 0

        for step in range(n_steps):
            # Select action
            action = self.policy.select_action(
                frame_t,
                exploration_rate=exploration_rate
            )

            # Execute
            frame_t1_raw, reward, terminated, truncated, info = self.env.step(action)
            frame_t1 = preprocess(frame_t1_raw)

            # Store
            self.buffer.add(frame_t, action, frame_t1)

            # Next
            frame_t = frame_t1

            if terminated or truncated:
                frame_t, _ = self.env.reset()
                frame_t = preprocess(frame_t)
                episodes_done += 1

        print(f"   ✓ Collected {len(self.buffer)} total transitions")
        print(f"   ✓ Completed {episodes_done} episodes")
        return episodes_done

    def train_step(self, forward_weight=1.0, inverse_weight=1.0, consistency_weight=0.5):
        """Single training step."""
        if len(self.buffer) < self.batch_size:
            return None

        # Sample batch
        frames_t, actions, frames_t1 = self.buffer.sample(self.batch_size)
        frames_t = frames_t.to(self.device)
        actions = actions.to(self.device)
        frames_t1 = frames_t1.to(self.device)

        # Compute loss
        losses = self.model.total_loss(
            frames_t, frames_t1, actions,
            forward_weight=forward_weight,
            inverse_weight=inverse_weight,
            consistency_weight=consistency_weight
        )

        # Backprop
        self.optimizer.zero_grad()
        losses['total'].backward()
        self.optimizer.step()

        self.step_count += 1

        return losses

    def train_loop(self, n_episodes=10, train_steps_per_episode=50, exploration_schedule=None):
        """Main training loop."""
        print("\n" + "="*70)
        print("🌀 MOTION-BASED ACTIVE INFERENCE WITH INVERSE MODEL")
        print("="*70)
        print(f"Episodes: {n_episodes}")
        print(f"Training steps per episode: {train_steps_per_episode}")
        print("="*70 + "\n")

        if exploration_schedule is None:
            exploration_schedule = lambda ep: max(0.1, 1.0 - ep / n_episodes)

        for episode in range(n_episodes):
            exploration_rate = exploration_schedule(episode)

            # Collect experience
            episodes_done = self.collect_experience(
                n_steps=100,
                exploration_rate=exploration_rate
            )
            self.episode_count += episodes_done

            # Train
            print(f"\n🎓 Training for {train_steps_per_episode} steps...")
            episode_metrics = {'forward': [], 'inverse': [], 'consistency': [], 'total': []}

            for _ in range(train_steps_per_episode):
                metrics = self.train_step()
                if metrics:
                    for key in episode_metrics:
                        if key == 'total':
                            episode_metrics[key].append(metrics[key].item())
                        else:
                            episode_metrics[key].append(metrics[key])

                    for key in ['forward', 'inverse', 'consistency', 'total']:
                        self.history[key].append(metrics[key] if key == 'total' else metrics[key])

            # Log
            print(f"\n📊 Episode {episode+1}/{n_episodes}")
            print(f"   Exploration rate: {exploration_rate:.2f}")
            print(f"   Total episodes: {self.episode_count}")
            print(f"   Training steps: {self.step_count}")
            if episode_metrics['forward']:
                print(f"   Forward loss: {np.mean(episode_metrics['forward']):.6f}")
                print(f"   Inverse loss: {np.mean(episode_metrics['inverse']):.6f}")
                print(f"   Consistency loss: {np.mean(episode_metrics['consistency']):.6f}")
                print(f"   Total loss: {np.mean(episode_metrics['total']):.6f}")
            print(f"   Buffer size: {len(self.buffer)}")

        print("\n✅ Training complete!\n")

    def save_checkpoint(self, path='/content/drive/MyDrive/tetrahedral_checkpoints/motion_inverse_checkpoint.pt'):
        """Save checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }

        torch.save(checkpoint, path)
        print(f"💾 Checkpoint saved to {path}")


# ============================================================================
# SUMMARY
# ============================================================================

"""
MOTION_INVERSE_MODEL.py - Agency through Causality

KEY INNOVATIONS:

1. MOTION SPACE (not frame space)
   - Extract motion from frame pairs
   - Learn in motion space (like biological vision)
   - Images derive meaning FROM motion

2. INVERSE MODEL (mirror neurons)
   - (motion_t, motion_t+1) → action
   - Infer causality: "What caused this motion?"
   - Understand actions by simulating them

3. COUPLED MODELS
   - Forward + Inverse learned together
   - Consistency loss ensures agreement
   - Models must agree on world physics

4. AGENTIC POLICY
   - Uses both forward and inverse
   - Has intentionality (not just uncertainty minimization)
   - Can achieve desired motions

This is TRUE active inference with agency!
"""

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🌀 MOTION-BASED ACTIVE INFERENCE")
    print("="*70)
    print("\nThe Strange Loop ACTUALLY Closes:")
    print("  Motion → Infer Cause → Predict Consequence → Act → Observe")
    print("\nKey Features:")
    print("  • Motion space (not frame space)")
    print("  • Inverse model (mirror neurons)")
    print("  • Coupled training (consistency)")
    print("  • Agentic policy (intentionality)")
    print("\nReady to close the loop with agency! 🎮")
    print("="*70 + "\n")

    print("Example usage:")
    print("  trainer = MotionInverseTrainer(env_name='ALE/Pong-v5')")
    print("  trainer.train_loop(n_episodes=10)")
    print("  trainer.save_checkpoint()")
