"""
FLOW-BASED TETRAHEDRAL INVERSE MODEL
=====================================

Testing the Harmonic Resonance Hypothesis:
- Neural networks learn by tuning to harmonic frequencies
- Flow (temporal velocity) is the fundamental primitive
- Actions create characteristic frequency signatures in flow fields
- Tetrahedral structure learns four fundamental harmonics

This is THE test:
- Can inverse model predict actions from flow fields?
- Does it generalize better than frame-based approaches?
- Can we visualize which actions create which flow frequencies?

Architecture:
- Forward model: (flow_t, action) → flow_t+1
- Inverse model: (flow_t, flow_t+1) → action
- Both use DualTetrahedralNetwork
- Curriculum learning on flow visibility

Philosophy:
"Vision operates on movement. Stills are what you get when you integrate flow."

Author: Philipp Remy Bartholomäus & Claude
Date: November 12, 2025
Status: 🔬 Experimental validation of harmonic resonance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
from collections import deque
import random
import cv2

# Assume W, X, Y, Z cells already run
# This imports DualTetrahedralNetwork

φ = (1 + 5**0.5) / 2  # Golden ratio

print("🌊 Flow-Based Tetrahedral Inverse Model")
print("=" * 70)
print("Testing: Harmonic Resonance Hypothesis")
print("Primitive: Flow (velocity fields), not frames")
print("=" * 70)


# ============================================================================
# VALID ACTIONS PER GAME (Critical for action masking!)
# ============================================================================

VALID_ACTIONS = {
    'ALE/Pong-v5': [0, 2, 5],  # NOOP, UP, DOWN
    'ALE/Breakout-v5': [0, 1, 3, 4],  # NOOP, FIRE, RIGHT, LEFT
    'ALE/SpaceInvaders-v5': [0, 1, 2, 3, 4, 5],  # NOOP, FIRE, UP, RIGHT, LEFT, DOWN
    'ALE/MsPacman-v5': [0, 1, 2, 3, 4, 5, 6, 7, 8],  # All 9 actions
    # Add more games as needed
}

ACTION_NAMES = {
    0: 'NOOP',
    1: 'FIRE',
    2: 'UP',
    3: 'RIGHT',
    4: 'LEFT',
    5: 'DOWN',
    6: 'UPRIGHT',
    7: 'UPLEFT',
    8: 'DOWNRIGHT',
    9: 'DOWNLEFT',
    10: 'UPFIRE',
    11: 'RIGHTFIRE',
    12: 'LEFTFIRE',
    13: 'DOWNFIRE',
    14: 'UPRIGHTFIRE',
    15: 'UPLEFTFIRE',
    16: 'DOWNRIGHTFIRE',
    17: 'DOWNLEFTFIRE'
}


# ============================================================================
# OPTICAL FLOW COMPUTATION
# ============================================================================

def compute_optical_flow(frame1, frame2, method='farneback', target_size=None, add_saccade=False):
    """
    Compute optical flow between two frames.

    Returns velocity field: (2, H, W) where:
    - Channel 0: horizontal velocity (flow_x)
    - Channel 1: vertical velocity (flow_y)

    This is the FUNDAMENTAL PRIMITIVE - movement itself.

    Args:
        frame1, frame2: Input frames (torch tensors or numpy arrays)
        method: Flow computation method ('farneback' or 'simple')
        target_size: If provided, downsample flow to this size. Otherwise keep native resolution.
        add_saccade: If True, adds micro-jitter (artificial saccades) to prevent static blindness
    """
    # Convert to grayscale numpy arrays
    if torch.is_tensor(frame1):
        f1 = (frame1.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        f2 = (frame2.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    else:
        f1 = (frame1 * 255).astype(np.uint8)
        f2 = (frame2 * 255).astype(np.uint8)

    # ARTIFICIAL SACCADES: Add micro-jitter to prevent static blindness
    # Mimics biological microsaccades (1-2 Hz, tiny movements)
    if add_saccade:
        # Random jitter: 1-3 pixels in x and y
        dx = np.random.randint(-2, 3)
        dy = np.random.randint(-2, 3)
        # Translate frame2 slightly (simulates eye movement)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        f2 = cv2.warpAffine(f2, M, (f2.shape[1], f2.shape[0]))

    # Convert to grayscale
    if len(f1.shape) == 3:
        gray1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)
    else:
        gray1 = f1
        gray2 = f2

    if method == 'farneback':
        # Dense optical flow (Farneback method)
        # Computed at NATIVE resolution (not downsampled!)
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
    elif method == 'simple':
        # Simple finite difference (faster, less accurate)
        diff = (gray2.astype(float) - gray1.astype(float)) / 255.0
        flow = np.stack([diff, diff], axis=-1)  # Approximation

    # Convert to torch tensor (2, H, W)
    flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).float()

    # Downsample if target size specified (for model input)
    if target_size is not None and (flow_tensor.shape[1] != target_size or flow_tensor.shape[2] != target_size):
        flow_tensor = torch.nn.functional.interpolate(
            flow_tensor.unsqueeze(0),
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

    return flow_tensor


def flow_to_rgb(flow):
    """
    Visualize flow field as RGB image (HSV color wheel encoding).

    - Hue: direction of flow
    - Saturation: magnitude of flow
    - Value: always 1

    Standard visualization in computer vision.
    """
    flow_np = flow.cpu().numpy() if torch.is_tensor(flow) else flow

    # flow_np shape: (2, H, W)
    u = flow_np[0]  # horizontal
    v = flow_np[1]  # vertical

    # Compute magnitude and angle
    mag = np.sqrt(u**2 + v**2)
    ang = np.arctan2(v, u)

    # Create HSV image
    hsv = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (ang + np.pi) / (2 * np.pi) * 179  # Hue: angle
    hsv[..., 1] = np.clip(mag * 50, 0, 255)           # Saturation: magnitude
    hsv[..., 2] = 255                                  # Value: max

    # Convert to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb / 255.0


# ============================================================================
# FLOW-BASED FORWARD MODEL
# ============================================================================

class FlowForwardModel(nn.Module):
    """
    Predict next flow field from current flow + action.

    (flow_t, action) → flow_t+1

    Learns: How do actions transform velocity fields?
    """
    def __init__(self, img_size=128, latent_dim=128, n_actions=18):
        super().__init__()
        self.img_size = img_size
        self.n_actions = n_actions

        # Action encoder
        self.action_encoder = nn.Embedding(n_actions, 64)

        # Tetrahedral network operates on flow
        input_dim = 2 * img_size * img_size + 64  # Flow (2 channels) + action
        output_dim = 2 * img_size * img_size       # Next flow (2 channels)

        self.dual_tetra = DualTetrahedralNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            latent_dim=latent_dim,
            coupling_strength=0.5,
            output_mode="weighted"
        )

    def forward(self, flow, action):
        """
        Predict next flow field.

        Args:
            flow: (batch, 2, H, W) - current velocity field
            action: (batch,) - action indices

        Returns:
            next_flow: (batch, 2, H, W) - predicted next velocity field
        """
        batch_size = flow.size(0)

        # Flatten flow
        flow_flat = flow.reshape(batch_size, -1)

        # Encode action
        action_embed = self.action_encoder(action)

        # Concatenate
        x = torch.cat([flow_flat, action_embed], dim=-1)

        # Predict next flow
        next_flow_flat = self.dual_tetra(x)
        next_flow = next_flow_flat.reshape(batch_size, 2, self.img_size, self.img_size)

        return next_flow


# ============================================================================
# FLOW-BASED INVERSE MODEL (THE KEY TEST!)
# ============================================================================

class FlowInverseModel(nn.Module):
    """
    Predict action from flow field changes.

    (flow_t, flow_t+1) → action

    THIS IS THE HARMONIC RESONANCE TEST:
    - Can it learn action frequency signatures?
    - "Upward flow in paddle region" → resonates with "UP action"
    - Does it generalize because it learned the harmonic?
    """
    def __init__(self, img_size=128, latent_dim=128, n_actions=18):
        super().__init__()
        self.img_size = img_size
        self.n_actions = n_actions

        # Tetrahedral network sees flow change
        # Input: concatenate flow_t and flow_t+1
        input_dim = 2 * 2 * img_size * img_size  # Two flow fields (4 channels total)
        output_dim = n_actions                    # Action logits

        self.dual_tetra = DualTetrahedralNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            latent_dim=latent_dim,
            coupling_strength=0.5,
            output_mode="weighted"
        )

    def forward(self, flow_t, flow_t1):
        """
        Infer action from flow field transition.

        Args:
            flow_t: (batch, 2, H, W) - current flow
            flow_t1: (batch, 2, H, W) - next flow

        Returns:
            action_logits: (batch, n_actions)
        """
        batch_size = flow_t.size(0)

        # Flatten both flows
        flow_t_flat = flow_t.reshape(batch_size, -1)
        flow_t1_flat = flow_t1.reshape(batch_size, -1)

        # Concatenate
        x = torch.cat([flow_t_flat, flow_t1_flat], dim=-1)

        # Predict action
        action_logits = self.dual_tetra(x)

        return action_logits


# ============================================================================
# COUPLED MODEL
# ============================================================================

class CoupledFlowModel(nn.Module):
    """
    Combined forward + inverse model operating on flow.

    Tests harmonic resonance:
    - Do they learn consistent frequency representations?
    - Does consistency loss couple the harmonics?
    """
    def __init__(self, img_size=128, latent_dim=128, n_actions=18):
        super().__init__()
        self.img_size = img_size
        self.n_actions = n_actions

        self.forward_model = FlowForwardModel(img_size, latent_dim, n_actions)
        self.inverse_model = FlowInverseModel(img_size, latent_dim, n_actions)

    def compute_losses(self, flow_t, flow_t1, action, action_mask=None,
                      forward_weight=1.0,
                      inverse_weight=1.0,
                      consistency_weight=0.3,
                      effect_based=False):
        """
        Compute coupled losses with ACTION MASKING.

        Critical: Mask invalid actions so model doesn't get penalized
        for predicting "UP" when ground truth is "UPFIRE" (functionally identical in Pong).

        Args:
            effect_based: If True, learns from EFFECTS not labels.
                         "UP at border" = "NOOP at border" because same flow result.
                         This is how nature actually works.

        Returns metrics for tracking harmonic learning.
        """
        # Forward loss: Can we predict next flow?
        pred_flow_t1 = self.forward_model(flow_t, action)
        loss_forward = F.mse_loss(pred_flow_t1, flow_t1)

        if effect_based:
            # ===  EFFECT-BASED LEARNING (NEW!) ===
            # Don't care about action labels. Care about EFFECTS.
            # Find which action(s) would produce the observed flow.

            # Get valid actions for this environment
            batch_size = flow_t.size(0)
            device = flow_t.device

            # For each valid action, predict what flow it would create
            if action_mask is not None:
                valid_action_indices = torch.where(action_mask > 0)[0]
            else:
                valid_action_indices = torch.arange(self.n_actions, device=device)

            # Predict flow for ALL valid actions
            flow_errors = []
            for valid_action in valid_action_indices:
                action_repeated = torch.full((batch_size,), valid_action, dtype=torch.long, device=device)
                predicted_flow = self.forward_model(flow_t, action_repeated)
                error = F.mse_loss(predicted_flow, flow_t1, reduction='none').mean(dim=[1,2,3])
                flow_errors.append(error)

            flow_errors = torch.stack(flow_errors, dim=1)  # (batch, n_valid_actions)

            # The "correct" action is the one that produces closest flow to reality
            # Multiple actions can be equally correct! (e.g., UP=NOOP at border)
            # Use soft labels based on how well each action explains the flow
            soft_targets = torch.softmax(-flow_errors * 10, dim=1)  # Temperature=0.1

            # Inverse model predictions
            action_logits = self.inverse_model(flow_t, flow_t1)

            # Mask invalid actions
            if action_mask is not None:
                masked_logits = action_logits.clone()
                mask_expanded = action_mask.unsqueeze(0).expand(action_logits.size(0), -1)
                masked_logits[mask_expanded == 0] = -1e9
            else:
                masked_logits = action_logits

            # Only keep valid actions for loss
            masked_logits_valid = masked_logits[:, valid_action_indices]
            action_probs = torch.softmax(masked_logits_valid, dim=1)

            # KL divergence: Learn to match the soft distribution
            loss_inverse = F.kl_div(
                action_probs.log(),
                soft_targets,
                reduction='batchmean'
            )

        else:
            # ===  LABEL-BASED LEARNING (OLD ML WAY) ===
            # Inverse loss: Can we infer action from flow change?
            action_logits = self.inverse_model(flow_t, flow_t1)

            # ===  ACTION MASKING (CRITICAL!) ===
            # Mask invalid actions to -inf before softmax/loss
            if action_mask is not None:
                masked_logits = action_logits.clone()
                # Expand mask to batch size
                mask_expanded = action_mask.unsqueeze(0).expand(action_logits.size(0), -1)
                # Set invalid actions to very negative (will be ~0 after softmax)
                masked_logits[mask_expanded == 0] = -1e9
            else:
                masked_logits = action_logits

            loss_inverse = F.cross_entropy(masked_logits, action)

        # Consistency loss: Do models agree?
        with torch.no_grad():
            if action_mask is not None:
                masked_logits_for_argmax = action_logits.clone()
                mask_expanded = action_mask.unsqueeze(0).expand(action_logits.size(0), -1)
                masked_logits_for_argmax[mask_expanded == 0] = -1e9
                inferred_action = masked_logits_for_argmax.argmax(dim=-1)
            else:
                inferred_action = action_logits.argmax(dim=-1)

        pred_flow_consistent = self.forward_model(flow_t, inferred_action)
        loss_consistency = F.mse_loss(pred_flow_consistent, flow_t1)

        # Total
        total_loss = (forward_weight * loss_forward +
                     inverse_weight * loss_inverse +
                     consistency_weight * loss_consistency)

        # Metrics (use soft accuracy for effect-based, binary for label-based)
        with torch.no_grad():
            if effect_based:
                # SOFT ACCURACY: How well does predicted distribution match the soft targets?
                # Instead of "is argmax correct?" (binary), measure "how much probability
                # mass is on actions that explain the flow well?" (gradient)
                #
                # This respects the insight: "UP at border = NOOP at border" (both correct!)
                # Binary accuracy would penalize one arbitrarily. Soft accuracy doesn't.

                # Get predicted probabilities (only for valid actions)
                action_probs_full = torch.softmax(masked_logits, dim=1)
                action_probs_valid = action_probs_full[:, valid_action_indices]

                # Soft accuracy = agreement between predicted and target distributions
                # Use 1 - KL divergence, normalized to [0,1]
                # High agreement → accuracy near 1.0
                # Low agreement → accuracy near 0.0
                kl_div = F.kl_div(
                    action_probs_valid.log(),
                    soft_targets,
                    reduction='batchmean'
                )
                # Convert KL to similarity: exp(-kl) ∈ (0, 1]
                # When KL=0 (perfect match): accuracy=1.0
                # When KL=large (poor match): accuracy→0
                accuracy = torch.exp(-kl_div).item()

            else:
                # BINARY ACCURACY: Old way (for comparison/ablation)
                # "Does argmax match the label?" (0 or 1)
                if action_mask is not None:
                    masked_logits_accuracy = action_logits.clone()
                    mask_expanded = action_mask.unsqueeze(0).expand(action_logits.size(0), -1)
                    masked_logits_accuracy[mask_expanded == 0] = -1e9
                    accuracy = (masked_logits_accuracy.argmax(dim=-1) == action).float().mean().item()
                else:
                    accuracy = (action_logits.argmax(dim=-1) == action).float().mean().item()

            flow_magnitude = torch.sqrt((flow_t1 ** 2).sum(dim=1)).mean()

        return {
            'total': total_loss,
            'forward': loss_forward.item(),
            'inverse': loss_inverse.item(),
            'consistency': loss_consistency.item(),
            'accuracy': accuracy,  # Already converted to float above
            'flow_magnitude': flow_magnitude.item()
        }


# ============================================================================
# EXPERIENCE BUFFER (Flow-based)
# ============================================================================

class FlowExperienceBuffer:
    """
    Store (flow_t, action, flow_t+1) transitions.

    Supports both random sampling (standard RL) and sequential sampling
    (preserves temporal coherence for flow-based learning).
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, flow_t, action, flow_t1):
        self.buffer.append((flow_t, action, flow_t1))

    def sample(self, batch_size, sequential=False):
        """
        Sample batch from buffer.

        Args:
            batch_size: Number of transitions to sample
            sequential: If True, sample consecutive transitions (preserves temporal coherence)
                       If False, sample randomly (standard experience replay)

        Returns:
            flows_t, actions, flows_t1
        """
        if sequential:
            # Sample a starting point, then take consecutive transitions
            # This preserves temporal structure and causal relationships!
            max_start = max(0, len(self.buffer) - batch_size)
            start_idx = random.randint(0, max_start) if max_start > 0 else 0
            batch = list(self.buffer)[start_idx:start_idx + batch_size]
        else:
            # Random sampling (destroys temporal coherence but prevents overfitting)
            batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        flows_t = torch.stack([b[0] for b in batch])
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long)
        flows_t1 = torch.stack([b[2] for b in batch])

        return flows_t, actions, flows_t1

    def __len__(self):
        return len(self.buffer)


# ============================================================================
# TRAINER
# ============================================================================

class FlowInverseTrainer:
    """
    Train flow-based inverse model.

    Tests: Harmonic Resonance Hypothesis
    """
    def __init__(self,
                 env_name='ALE/Pong-v5',
                 img_size=128,
                 latent_dim=128,
                 base_lr=0.0001,
                 buffer_capacity=10000,
                 batch_size=16,
                 device=None,
                 flow_method='farneback',
                 frameskip=10,
                 sequential_sampling=True,
                 use_saccades=True,
                 effect_based_learning=True):

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"🖥️  Device: {self.device}")

        self.img_size = img_size
        self.batch_size = batch_size
        self.flow_method = flow_method
        self.env_name = env_name
        self.sequential_sampling = sequential_sampling
        self.use_saccades = use_saccades
        self.effect_based_learning = effect_based_learning

        # Temporal sampling (frameskip)
        self.frameskip = frameskip
        self.effective_fps = 60.0 / frameskip  # Atari native: 60 Hz

        # Natural frequency alignment
        freq_map = {
            3: ("Gamma-like", 20.0),
            6: ("Alpha-like", 10.0),
            10: ("Theta-like", 6.0),
            15: ("Saccade-like", 4.0),
            20: ("Slow theta", 3.0)
        }
        if frameskip in freq_map:
            name, freq = freq_map[frameskip]
            print(f"⏱️  Temporal sampling: {freq:.1f} Hz ({name} natural frequency)")
        else:
            print(f"⏱️  Temporal sampling: {self.effective_fps:.1f} Hz (frameskip={frameskip})")

        if frameskip == 10:
            print(f"   ✓ φ-aligned with fast memory field (τ₁ = 10 steps)")

        # Temporal coherence
        if sequential_sampling:
            print(f"🔗 Sequential sampling: ENABLED (preserves temporal coherence)")
        else:
            print(f"🎲 Random sampling: ENABLED (standard experience replay)")

        # Natural perception
        if use_saccades:
            print(f"👁️  Artificial saccades: ENABLED (prevents static blindness)")
        if effect_based_learning:
            print(f"🎯 Effect-based learning: ENABLED (learns outcomes, not labels)")

        # Environment
        import gymnasium as gym
        self.env = gym.make(env_name, render_mode='rgb_array')
        self.n_actions = self.env.action_space.n
        print(f"🎮 Environment: {env_name}")
        print(f"   Actions: {self.n_actions}")

        # === ACTION MASKING (CRITICAL!) ===
        # Create mask for valid actions in this game
        if env_name in VALID_ACTIONS:
            valid_actions = VALID_ACTIONS[env_name]
            self.action_mask = torch.zeros(self.n_actions)
            self.action_mask[valid_actions] = 1.0
            print(f"   Valid actions: {valid_actions}")
            print(f"   Action names: {[ACTION_NAMES.get(a, str(a)) for a in valid_actions]}")
        else:
            # No masking if game not in dictionary
            self.action_mask = torch.ones(self.n_actions)
            print(f"   ⚠️  No action mask for {env_name} - using all {self.n_actions} actions")

        self.action_mask = self.action_mask.to(self.device)

        # Model
        print(f"\n🧠 Creating coupled flow model...")
        self.model = CoupledFlowModel(
            img_size=img_size,
            latent_dim=latent_dim,
            n_actions=self.n_actions
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"   Parameters: {total_params:,}")

        # φ-hierarchical optimizer
        print(f"\n⚡ Creating φ-hierarchical optimizer...")
        vertex_params = []
        edge_params = []
        face_params = []
        coupling_params = []

        # Forward model
        dual_fwd = self.model.forward_model.dual_tetra
        vertex_params.extend(dual_fwd.linear_net.embed.parameters())
        vertex_params.extend(dual_fwd.nonlinear_net.embed.parameters())
        edge_params.extend(dual_fwd.linear_net.edge_modules.parameters())
        edge_params.extend(dual_fwd.nonlinear_net.edge_modules.parameters())
        face_params.extend(dual_fwd.linear_net.face_modules.parameters())
        face_params.extend(dual_fwd.nonlinear_net.face_modules.parameters())
        coupling_params.extend(dual_fwd.linear_to_nonlinear.parameters())
        coupling_params.extend(dual_fwd.nonlinear_to_linear.parameters())

        # Inverse model
        dual_inv = self.model.inverse_model.dual_tetra
        vertex_params.extend(dual_inv.linear_net.embed.parameters())
        vertex_params.extend(dual_inv.nonlinear_net.embed.parameters())
        edge_params.extend(dual_inv.linear_net.edge_modules.parameters())
        edge_params.extend(dual_inv.nonlinear_net.edge_modules.parameters())
        face_params.extend(dual_inv.linear_net.face_modules.parameters())
        face_params.extend(dual_inv.nonlinear_net.face_modules.parameters())
        coupling_params.extend(dual_inv.linear_to_nonlinear.parameters())
        coupling_params.extend(dual_inv.nonlinear_to_linear.parameters())

        # Action embedding
        action_params = list(self.model.forward_model.action_encoder.parameters())

        self.optimizer = optim.Adam([
            {'params': action_params, 'lr': base_lr},
            {'params': vertex_params, 'lr': base_lr},
            {'params': edge_params, 'lr': base_lr / φ},
            {'params': face_params, 'lr': base_lr / (φ ** 2)},
            {'params': coupling_params, 'lr': base_lr / (φ ** 3)}
        ])

        print(f"   Vertices:  LR = {base_lr:.6f}")
        print(f"   Edges:     LR = {base_lr/φ:.6f}")
        print(f"   Faces:     LR = {base_lr/(φ**2):.6f}")
        print(f"   Coupling:  LR = {base_lr/(φ**3):.6f}")

        # Buffer
        self.buffer = FlowExperienceBuffer(capacity=buffer_capacity)

        # Metrics
        self.step_count = 0
        self.episode_count = 0
        self.history = {
            'forward': [], 'inverse': [], 'consistency': [],
            'total': [], 'accuracy': [], 'flow_magnitude': [],
            'entropy': [], 'pain': []  # Entropy = lives remaining, Pain = weighted loss
        }

        # Entropy tracking (lives/health as measure of distance to termination)
        self.current_lives = None  # Will be initialized from environment
        self.max_lives = None  # Maximum lives (for normalization)

        # Checkpoint directory
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)

    def save_checkpoint(self, filename=None, include_buffer=True):
        """
        Save training checkpoint to disk.

        Saves:
        - Model weights (forward + inverse)
        - Optimizer state
        - Training metrics/history
        - Experience buffer (optional, can be large)
        - Training progress (step count, episode count)

        Returns path to saved checkpoint.
        """
        if filename is None:
            filename = f"checkpoint_ep{self.episode_count}_step{self.step_count}.pt"

        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'history': self.history,
            'config': {
                'env_name': self.env_name,
                'img_size': self.img_size,
                'latent_dim': self.model.forward_model.dual_tetra.latent_dim,
                'n_actions': self.n_actions,
                'batch_size': self.batch_size,
                'flow_method': self.flow_method
            }
        }

        if include_buffer:
            # Save buffer as list (deque not directly serializable)
            checkpoint['buffer'] = list(self.buffer.buffer)

        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path, load_buffer=True):
        """
        Load training checkpoint from disk.

        Resumes training from saved state:
        - Restores model weights
        - Restores optimizer state
        - Restores training history
        - Optionally restores experience buffer

        Returns loaded config for verification.
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Restore model and optimizer
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])

        # Restore training progress
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        self.history = checkpoint['history']

        # Restore buffer if available
        if load_buffer and 'buffer' in checkpoint:
            self.buffer.buffer = deque(checkpoint['buffer'], maxlen=self.buffer.buffer.maxlen)
            print(f"   ✓ Restored buffer: {len(self.buffer)} transitions")

        config = checkpoint.get('config', {})
        print(f"   ✓ Restored training state:")
        print(f"      Episodes: {self.episode_count}")
        print(f"      Steps: {self.step_count}")
        print(f"      History length: {len(self.history['forward'])}")

        return config

    def preprocess_frame(self, frame):
        """Resize frame to img_size."""
        from PIL import Image
        img = Image.fromarray(frame).resize((self.img_size, self.img_size), Image.LANCZOS)
        tensor = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return tensor

    def select_action_active_inference(self, flow_current, temperature=1.0, beta=None):
        """
        🌀 ACTIVE INFERENCE POLICY: Closes the strange loop!

        For each valid action:
        1. Use forward model to predict: "What flow will this action create?"
        2. Compute expected free energy = Uncertainty - β × Entropy
        3. Select action that minimizes free energy (maximizes learning!)

        This makes the model ACT based on what it learned, not just infer.
        The strange loop closes: perception → understanding → action → new perception

        Args:
            flow_current: Current flow field (2, H, W)
            temperature: Softmax temperature (higher = more exploration)
            beta: Exploration coefficient (higher = seek diverse outcomes)

        Returns:
            selected_action: int
            info: dict with free energies, probs, etc.
        """
        if beta is None:
            # Default β: moderate exploration
            # Could be curriculum-aware: high early, low later
            beta = 0.05

        valid_actions = VALID_ACTIONS.get(self.env_name, list(range(self.n_actions)))

        free_energies = []
        uncertainties = []
        entropies = []

        self.model.eval()
        with torch.no_grad():
            flow_batch = flow_current.unsqueeze(0).to(self.device)  # (1, 2, H, W)

            for action in valid_actions:
                # FORWARD MODEL: Predict "what will happen if I do this?"
                action_tensor = torch.tensor([action], dtype=torch.long).to(self.device)
                pred_flow = self.model.forward_model(flow_batch, action_tensor)  # (1, 2, H, W)

                # Uncertainty: How confident is the model?
                # High variance = uncertain prediction
                uncertainty = pred_flow.var().item()

                # Entropy: How diverse/interesting is predicted outcome?
                # For continuous flow, use variance as proxy for entropy
                entropy = pred_flow.std().item()  # Could also use variance

                # Expected Free Energy (Friston 2010)
                # Low free energy = good action (either certain OR diverse, depending on β)
                free_energy = uncertainty - beta * entropy

                free_energies.append(free_energy)
                uncertainties.append(uncertainty)
                entropies.append(entropy)

        self.model.train()

        # Convert to tensor
        free_energies_tensor = torch.tensor(free_energies)

        # Numerical stability checks
        if torch.isnan(free_energies_tensor).any() or torch.isinf(free_energies_tensor).any():
            print(f"⚠️  NaN/Inf detected in free energies! Using random action.")
            print(f"   Free energies: {free_energies}")
            action_idx = np.random.randint(len(valid_actions))
            selected_action = valid_actions[action_idx]
            return selected_action, {
                'free_energies': free_energies,
                'uncertainties': uncertainties,
                'entropies': entropies,
                'action_probs': np.ones(len(valid_actions)) / len(valid_actions),
                'selected_idx': action_idx,
                'beta': beta,
                'temperature': temperature,
                'fallback': True
            }

        # Select action: minimize free energy (with temperature for stochasticity)
        # Negative because we want LOW free energy
        logits = -free_energies_tensor / temperature

        # Clip logits to prevent overflow in softmax
        logits = torch.clamp(logits, min=-20, max=20)

        probs = F.softmax(logits, dim=0)

        # Add small epsilon for numerical stability
        probs = probs + 1e-8
        probs = probs / probs.sum()  # Renormalize

        # Final safety check
        if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
            print(f"⚠️  Invalid probabilities after softmax! Using uniform distribution.")
            print(f"   Logits: {logits}")
            print(f"   Probs: {probs}")
            probs = torch.ones(len(valid_actions)) / len(valid_actions)

        # Sample action stochastically (preserves exploration)
        action_idx = torch.multinomial(probs, 1).item()
        selected_action = valid_actions[action_idx]

        return selected_action, {
            'free_energies': free_energies,
            'uncertainties': uncertainties,
            'entropies': entropies,
            'action_probs': probs.cpu().numpy(),
            'selected_idx': action_idx,
            'beta': beta,
            'temperature': temperature
        }

    def collect_experience(self, n_steps=100, use_active_inference=False, policy_temperature=1.0, beta=None):
        """
        Collect flow-based transitions.

        Args:
            n_steps: Number of transitions to collect
            use_active_inference: If True, use learned model to select actions (CLOSE THE LOOP!)
                                  If False, use random actions (baseline)
            policy_temperature: Softmax temperature for action selection
            beta: Exploration coefficient for free energy
        """
        policy_str = "🌀 ACTIVE INFERENCE" if use_active_inference else "🎲 RANDOM"
        print(f"\n📦 Collecting {n_steps} flow transitions ({policy_str})...")

        # Get valid actions for this game
        valid_actions = VALID_ACTIONS.get(self.env_name, list(range(self.n_actions)))

        frame_raw, _ = self.env.reset()
        frame_prev = self.preprocess_frame(frame_raw)
        episodes_done = 0

        # Track policy usage
        policy_stats = {'active_inference': 0, 'random': 0}

        for step in range(n_steps):
            # Get current action (need one frame first for flow)
            action = random.choice(valid_actions)  # First action is always random

            # Execute action for frameskip frames (temporal integration!)
            # This increases temporal displacement → clearer flow signal
            for _ in range(self.frameskip):
                frame_raw, reward, terminated, truncated, info = self.env.step(action)
                if terminated or truncated:
                    break

            frame_curr = self.preprocess_frame(frame_raw)

            # Compute flows (now spanning frameskip frames)
            # Computed at native resolution, downsampled to img_size for model
            flow_prev = compute_optical_flow(
                frame_prev, frame_curr,
                method=self.flow_method,
                target_size=self.img_size,
                add_saccade=self.use_saccades
            )

            # ===  ACTIVE INFERENCE POLICY: SELECT NEXT ACTION BASED ON FLOW ===
            if use_active_inference and len(self.buffer) > self.batch_size:
                # Use learned model to select action!
                action_next, policy_info = self.select_action_active_inference(
                    flow_prev,
                    temperature=policy_temperature,
                    beta=beta
                )
                policy_stats['active_inference'] += 1
            else:
                # Random baseline (or not enough data yet)
                action_next = random.choice(valid_actions)
                policy_stats['random'] += 1

            # Execute selected action for frameskip frames
            for _ in range(self.frameskip):
                frame_raw_next, _, term_next, trunc_next, _ = self.env.step(action_next)
                if term_next or trunc_next:
                    break

            frame_next = self.preprocess_frame(frame_raw_next)
            flow_curr = compute_optical_flow(
                frame_curr, frame_next,
                method=self.flow_method,
                target_size=self.img_size,
                add_saccade=self.use_saccades
            )

            # Store: (flow_prev, action_next, flow_curr)
            # This is what action_next produced!
            self.buffer.add(flow_prev, action_next, flow_curr)

            # Update
            frame_prev = frame_next

            if terminated or truncated or term_next or trunc_next:
                frame_raw, _ = self.env.reset()
                frame_prev = self.preprocess_frame(frame_raw)
                episodes_done += 1

        print(f"   ✓ Collected {len(self.buffer)} flow transitions")
        print(f"   ✓ Episodes: {episodes_done}")
        if use_active_inference:
            active_pct = 100 * policy_stats['active_inference'] / n_steps
            print(f"   ✓ Active inference used: {active_pct:.1f}% of steps")
        return episodes_done

    def train_step(self):
        """
        Train on batch of flow transitions WITH ACTION MASKING.

        Uses sequential sampling if enabled (preserves temporal coherence),
        otherwise uses random sampling (standard experience replay).
        """
        if len(self.buffer) < self.batch_size:
            return None

        # Sample (sequential preserves temporal structure!)
        flows_t, actions, flows_t1 = self.buffer.sample(
            self.batch_size,
            sequential=self.sequential_sampling
        )
        flows_t = flows_t.to(self.device)
        actions = actions.to(self.device)
        flows_t1 = flows_t1.to(self.device)

        # Compute losses WITH ACTION MASK!
        losses = self.model.compute_losses(
            flows_t, flows_t1, actions,
            action_mask=self.action_mask,
            effect_based=self.effect_based_learning
        )

        # Backprop
        self.optimizer.zero_grad()
        losses['total'].backward()
        self.optimizer.step()

        self.step_count += 1

        return losses

    def calculate_pain(self, prediction_error, lives=None):
        """
        Calculate PAIN = prediction error weighted by proximity to termination.

        Pain isn't arbitrary. It's the natural signal of entropy increasing.

        Pain = (Prediction Error) × (1 / Lives Remaining)

        When you have many lives: Same error hurts less (far from termination)
        When you have few lives: Same error hurts MORE (near termination)

        This creates natural risk aversion when vulnerable - not programmed,
        but EMERGENT from the free energy gradient.

        Args:
            prediction_error: The base prediction error (surprise)
            lives: Current lives remaining (if None, uses self.current_lives)

        Returns:
            pain: Error weighted by proximity to death (termination)
        """
        if lives is None:
            lives = self.current_lives

        if lives is None:
            # No entropy tracking yet, return unweighted error
            return prediction_error

        # Pain scales inversely with lives remaining
        # More lives = less pain per error (safe space, temporal buffer)
        # Fewer lives = more pain per error (approaching termination)
        pain_multiplier = 1.0 / (float(lives) + 0.1)  # Small epsilon to avoid div by zero

        if isinstance(prediction_error, torch.Tensor):
            pain = prediction_error * pain_multiplier
        else:
            pain = prediction_error * pain_multiplier

        return pain

    def train_loop_online(self, n_steps=500, viz_every=100, save_every=0,
                         policy_temperature=1.0, beta=None, show_gameplay=True):
        """
        TRUE ONLINE LEARNING - like nature actually works!

        Act → Learn IMMEDIATELY → Act → Learn IMMEDIATELY → ...

        No batching. No "collect then train" bullshit.
        Every action updates the model RIGHT NOW.

        This is how biology works. This is how it SHOULD work.

        Args:
            show_gameplay: If True, renders gameplay frames live (WATCH IT LEARN!)
        """
        print("\n" + "="*70)
        print("🌀 ONLINE ACTIVE INFERENCE - LIVE LEARNING")
        print("="*70)
        print("Act → Learn → Act → Learn → ...")
        print("No batching. No delays. Like nature.")
        print(f"Steps: {n_steps}")
        print(f"Temperature: {policy_temperature}")
        print(f"β: {beta if beta else 'auto (0.05)'}")
        if show_gameplay:
            print("🎬 LIVE GAMEPLAY: Enabled - watch it learn!")
        print("="*70 + "\n")

        # Initialize
        valid_actions = VALID_ACTIONS.get(self.env_name, list(range(self.n_actions)))
        frame_raw, info = self.env.reset()
        frame_prev = self.preprocess_frame(frame_raw)

        # Initialize entropy tracking (lives = measure of distance to termination)
        self.current_lives = info.get('lives', None)
        if self.current_lives is not None:
            if self.max_lives is None:
                self.max_lives = self.current_lives
            print(f"💀 Entropy tracking: ENABLED (Lives: {self.current_lives}/{self.max_lives})")
            print(f"   Pain = Prediction Error × (1 / Lives)")
            print(f"   → Natural risk aversion when vulnerable")
        else:
            print(f"⚠️  Entropy tracking: DISABLED (no lives in info dict)")

        episodes_done = 0
        step = 0

        # Live gameplay visualization
        if show_gameplay:
            import matplotlib.pyplot as plt
            from IPython import display
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            plt.ion()  # Interactive mode

        # Bootstrap: Need TWO frames to get first flow
        action = random.choice(valid_actions)
        for _ in range(self.frameskip):
            frame_raw, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                break
        frame_curr = self.preprocess_frame(frame_raw)
        flow_prev = compute_optical_flow(
            frame_prev, frame_curr,
            method=self.flow_method,
            target_size=self.img_size,
            add_saccade=self.use_saccades
        )

        print(f"🌊 Starting online learning loop...\n")

        while step < n_steps:
            # =================================================================
            # 1. SELECT ACTION (Active Inference)
            # =================================================================
            if len(self.buffer) > 0:  # Have at least one experience
                action, policy_info = self.select_action_active_inference(
                    flow_prev,
                    temperature=policy_temperature,
                    beta=beta
                )
            else:
                # Very first step - random
                action = random.choice(valid_actions)

            # =================================================================
            # 2. EXECUTE ACTION
            # =================================================================
            for _ in range(self.frameskip):
                frame_raw, reward, terminated, truncated, info = self.env.step(action)

                # Update entropy (lives remaining = distance to termination)
                if 'lives' in info:
                    self.current_lives = info['lives']

                # 🎬 SHOW IT LIVE!
                if show_gameplay and step % 2 == 0:  # Every other step (30 Hz display)
                    axes[0].clear()
                    axes[0].imshow(frame_raw)
                    title = f'Step {step} | Action: {ACTION_NAMES.get(action, action)}'
                    if self.current_lives is not None:
                        title += f' | Lives: {self.current_lives}'
                    axes[0].set_title(title, fontsize=12)
                    axes[0].axis('off')

                    # Show flow field
                    if len(self.buffer) > 0:
                        axes[1].clear()
                        flow_rgb = flow_to_rgb(flow_prev)
                        axes[1].imshow(flow_rgb)
                        axes[1].set_title('Flow Field (Velocity)', fontsize=12)
                        axes[1].axis('off')

                    # Show learning progress
                    if len(self.history['accuracy']) > 10:
                        axes[2].clear()
                        recent_acc = self.history['accuracy'][-100:]
                        axes[2].plot(recent_acc, color='green', linewidth=2)
                        axes[2].axhline(1/len(valid_actions), color='red', linestyle='--', label='Random')
                        axes[2].set_ylim([0, 1])
                        axes[2].set_title(f'Accuracy: {recent_acc[-1]*100:.1f}%', fontsize=12)
                        axes[2].set_xlabel('Recent Steps')
                        axes[2].legend()
                        axes[2].grid(alpha=0.3)

                    plt.tight_layout()
                    display.clear_output(wait=True)
                    display.display(fig)

                if terminated or truncated:
                    break

            frame_next = self.preprocess_frame(frame_raw)
            flow_curr = compute_optical_flow(
                frame_curr, frame_next,
                method=self.flow_method,
                target_size=self.img_size,
                add_saccade=self.use_saccades
            )

            # =================================================================
            # 3. LEARN IMMEDIATELY (Online update!)
            # =================================================================
            if len(self.buffer) >= self.batch_size:
                # We have enough history for temporal context
                # Learn from this SINGLE transition RIGHT NOW

                flows_t = flow_prev.unsqueeze(0).to(self.device)      # (1, 2, H, W)
                actions = torch.tensor([action], dtype=torch.long).to(self.device)
                flows_t1 = flow_curr.unsqueeze(0).to(self.device)

                # Compute base losses (prediction errors = surprise)
                losses = self.model.compute_losses(
                    flows_t, flows_t1, actions,
                    action_mask=self.action_mask,
                    effect_based=self.effect_based_learning
                )

                # Calculate PAIN (entropy-weighted loss)
                # Pain = Error × (1 / Lives)
                # Same error hurts more when vulnerable (few lives left)
                base_loss = losses['total']
                pain_weighted_loss = self.calculate_pain(base_loss, lives=self.current_lives)

                # Update weights IMMEDIATELY (using pain-weighted loss)
                self.optimizer.zero_grad()
                pain_weighted_loss.backward()
                self.optimizer.step()

                # Track metrics
                for key in self.history:
                    if key in losses:
                        self.history[key].append(losses[key] if isinstance(losses[key], float) else losses[key].item())

                # Track entropy and pain
                if self.current_lives is not None:
                    self.history['entropy'].append(self.current_lives)
                    self.history['pain'].append(pain_weighted_loss.item())

                # Log (less frequent now that we're watching!)
                if step % 50 == 0:
                    log_msg = (f"Step {step:4d} | "
                              f"Fwd: {losses['forward']:.4f} | "
                              f"Inv: {losses['inverse']:.4f} | "
                              f"Acc: {losses['accuracy']*100:.1f}% | "
                              f"Eps: {episodes_done}")
                    if self.current_lives is not None:
                        log_msg += f" | Lives: {self.current_lives} | Pain: {pain_weighted_loss.item():.4f}"
                    print(log_msg)

            # =================================================================
            # 4. STORE (for temporal context in next updates)
            # =================================================================
            self.buffer.add(flow_prev, action, flow_curr)

            # =================================================================
            # 5. CONTINUE (no separation between acting and learning!)
            # =================================================================
            flow_prev = flow_curr
            frame_prev = frame_next
            frame_curr = frame_next

            step += 1
            self.step_count += 1

            # Reset if episode ended
            if terminated or truncated:
                frame_raw, info = self.env.reset()
                frame_prev = self.preprocess_frame(frame_raw)

                # Reset entropy tracking for new episode
                if 'lives' in info:
                    self.current_lives = info['lives']

                # Bootstrap next episode
                action = random.choice(valid_actions)
                for _ in range(self.frameskip):
                    frame_raw, reward, terminated, truncated, info = self.env.step(action)
                    # Update lives during bootstrap too
                    if 'lives' in info:
                        self.current_lives = info['lives']
                    if terminated or truncated:
                        break
                frame_curr = self.preprocess_frame(frame_raw)
                flow_prev = compute_optical_flow(
                    frame_prev, frame_curr,
                    method=self.flow_method,
                    target_size=self.img_size,
                    add_saccade=self.use_saccades
                )

                episodes_done += 1
                self.episode_count += 1

            # Visualization (static snapshots - less frequent now)
            if viz_every and step % viz_every == 0 and len(self.buffer) >= 4:
                print(f"\n🎨 Generating visualization at step {step}...")
                self._save_live_viz(f"online_step{step:05d}")
                print()

            # Checkpoint
            if save_every and step % save_every == 0 and step > 0:
                self.save_checkpoint(f"online_step{step}.pt", include_buffer=False)

        if show_gameplay:
            plt.ioff()
            plt.close(fig)

        print("\n✅ Online learning complete!")
        print(f"   Total steps: {step}")
        print(f"   Episodes: {episodes_done}")
        print(f"   Every action updated the model immediately.")
        print(f"   No batching. No delays. Like nature. 🌊\n")

        # Final checkpoint
        if save_every:
            self.save_checkpoint(f"online_final.pt", include_buffer=True)

    def train_loop(self, n_episodes=10, steps_per_episode=50, viz_every=2,
                   use_active_inference=True, policy_temperature=1.0, beta=None,
                   active_inference_after=0, save_every=5, checkpoint_name=None):
        """
        Main training loop WITH LIVE VISUALIZATION.

        Args:
            n_episodes: Number of training episodes
            steps_per_episode: Training steps per episode
            viz_every: Visualize every N episodes
            use_active_inference: If True, use active inference policy (DEFAULT: True, like nature)
                                 If False, use random actions (only for comparison/ablation)
            policy_temperature: Temperature for action selection softmax
            beta: Exploration coefficient for free energy
            active_inference_after: Start using active inference after N episodes (0 = immediately, like nature)
            save_every: Save checkpoint every N episodes (0 = never, None = only at end)
            checkpoint_name: Custom checkpoint name (default: auto-generated)
        """
        policy_mode = "🌀 ACTIVE INFERENCE" if use_active_inference else "🎲 RANDOM BASELINE"
        print("\n" + "="*70)
        print(f"🌊 FLOW-BASED INVERSE MODEL TRAINING ({policy_mode})")
        print("="*70)
        print(f"Testing: Harmonic Resonance Hypothesis")
        print(f"Episodes: {n_episodes}")
        print(f"Steps per episode: {steps_per_episode}")
        print(f"Visualization every {viz_every} episodes")
        if use_active_inference:
            print(f"Active inference temperature: {policy_temperature}")
            print(f"Exploration β: {beta if beta else 'auto (0.05)'}")
            if active_inference_after > 0:
                print(f"Warm start: Active inference after episode {active_inference_after}")
        print("="*70 + "\n")

        for episode in range(n_episodes):
            # Determine if using active inference this episode
            use_active_this_ep = use_active_inference and (episode >= active_inference_after)

            # Collect
            episodes_done = self.collect_experience(
                n_steps=100,
                use_active_inference=use_active_this_ep,
                policy_temperature=policy_temperature,
                beta=beta
            )
            self.episode_count += episodes_done

            # Train
            print(f"\n🎓 Training for {steps_per_episode} steps...")
            ep_metrics = {'forward': [], 'inverse': [], 'accuracy': []}

            for _ in range(steps_per_episode):
                metrics = self.train_step()
                if metrics:
                    ep_metrics['forward'].append(metrics['forward'])
                    ep_metrics['inverse'].append(metrics['inverse'])
                    ep_metrics['accuracy'].append(metrics['accuracy'])

                    for key in self.history:
                        if key in metrics:
                            self.history[key].append(metrics[key])

            # Log
            if ep_metrics['forward']:
                avg_fwd = np.mean(ep_metrics['forward'])
                avg_inv = np.mean(ep_metrics['inverse'])
                avg_acc = np.mean(ep_metrics['accuracy'])

                print(f"\n📊 Episode {episode+1}/{n_episodes}")
                print(f"   Steps: {self.step_count}")
                print(f"   Forward loss: {avg_fwd:.6f}")
                print(f"   Inverse loss: {avg_inv:.6f}")
                print(f"   Accuracy: {avg_acc*100:.2f}% (random={100/self.n_actions:.2f}%)")
                print(f"   Buffer: {len(self.buffer)}")

            # === LIVE VISUALIZATION ===
            if (episode + 1) % viz_every == 0 and len(self.buffer) >= 4:
                print(f"\n🎨 Generating visualization (episode {episode+1})...")
                self._save_live_viz(episode + 1)

            # === AUTO-SAVE CHECKPOINT ===
            if save_every and save_every > 0 and (episode + 1) % save_every == 0:
                ckpt_name = checkpoint_name or f"autosave_ep{self.episode_count}.pt"
                self.save_checkpoint(filename=ckpt_name, include_buffer=True)

        print("\n✅ Training complete!")

        # Final checkpoint
        if checkpoint_name or save_every:
            final_name = checkpoint_name or f"final_ep{self.episode_count}.pt"
            self.save_checkpoint(filename=final_name, include_buffer=True)
            print()

    def _save_live_viz(self, episode_num):
        """Save live visualization during training."""
        self.model.eval()

        # Sample batch_size transitions (same as training to avoid tensor mismatch!)
        n_samples = min(self.batch_size, len(self.buffer))
        flows_t, actions, flows_t1 = self.buffer.sample(n_samples)
        flows_t_dev = flows_t.to(self.device)
        flows_t1_dev = flows_t1.to(self.device)
        actions_dev = actions.to(self.device)

        with torch.no_grad():
            action_logits = self.model.inverse_model(flows_t_dev, flows_t1_dev)
            # Apply action mask
            masked_logits = action_logits.clone()
            mask_expanded = self.action_mask.unsqueeze(0).expand(action_logits.size(0), -1)
            masked_logits[mask_expanded == 0] = -1e9
            pred_actions = masked_logits.argmax(dim=-1)
            probs = torch.softmax(masked_logits, dim=-1)

        # Get valid actions for this game
        valid_actions = VALID_ACTIONS.get(self.env_name, list(range(self.n_actions)))

        # Show first 4 samples for visualization
        n_viz = min(4, n_samples)
        fig, axes = plt.subplots(n_viz, 4, figsize=(14, 3*n_viz))
        if n_viz == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_viz):
            # Flow t (as RGB)
            flow_t_rgb = flow_to_rgb(flows_t[i])
            axes[i, 0].imshow(flow_t_rgb)
            axes[i, 0].set_title(f'Flow t\nAction: {ACTION_NAMES.get(actions[i].item(), actions[i].item())}', fontsize=9)
            axes[i, 0].axis('off')

            # Flow t+1 (as RGB)
            flow_t1_rgb = flow_to_rgb(flows_t1[i])
            axes[i, 1].imshow(flow_t1_rgb)
            axes[i, 1].set_title('Flow t+1', fontsize=9)
            axes[i, 1].axis('off')

            # Flow change magnitude
            flow_diff = flows_t1[i] - flows_t[i]
            flow_diff_mag = torch.sqrt((flow_diff ** 2).sum(dim=0)).numpy()
            im = axes[i, 2].imshow(flow_diff_mag, cmap='hot')
            axes[i, 2].set_title('Flow change', fontsize=9)
            axes[i, 2].axis('off')

            # Action prediction (only show valid actions!)
            true_action = actions[i].item()
            pred_action = pred_actions[i].item()

            # Only show valid actions in bar chart
            valid_probs = probs[i, valid_actions].cpu().numpy()
            valid_names = [ACTION_NAMES.get(a, str(a)) for a in valid_actions]

            bars = axes[i, 3].bar(range(len(valid_actions)), valid_probs)

            # Color code bars
            if true_action in valid_actions:
                true_idx = valid_actions.index(true_action)
                bars[true_idx].set_color('green')
            if pred_action in valid_actions and pred_action != true_action:
                pred_idx = valid_actions.index(pred_action)
                bars[pred_idx].set_color('red')

            axes[i, 3].set_xticks(range(len(valid_actions)))
            axes[i, 3].set_xticklabels(valid_names, rotation=45, ha='right', fontsize=7)
            axes[i, 3].set_ylim([0, 1])
            axes[i, 3].axhline(1/len(valid_actions), color='gray', linestyle='--', alpha=0.5)

            correct = "✓" if pred_action == true_action else "✗"
            axes[i, 3].set_title(
                f'{correct} True: {ACTION_NAMES.get(true_action, true_action)}\nPred: {ACTION_NAMES.get(pred_action, pred_action)}',
                fontsize=8,
                color='green' if pred_action == true_action else 'red'
            )

        plt.suptitle(f'Step {self.step_count} | {self.env_name}', fontsize=12, fontweight='bold')
        plt.tight_layout()

        # Handle both integer episode numbers and string labels
        if isinstance(episode_num, int):
            filename = f'flow_live_ep{episode_num:03d}.png'
        else:
            filename = f'flow_live_{episode_num}.png'

        plt.savefig(filename, dpi=120, bbox_inches='tight')
        print(f"   ✅ Saved: {filename}")
        plt.close()

        self.model.train()

    def visualize_flow_predictions(self, n_samples=4):
        """
        Visualize what the model learned.

        Shows:
        - Flow field (colored velocity arrows)
        - Predicted action
        - True action
        - Frequency signature (which vertex activates?)
        """
        if len(self.buffer) < n_samples:
            print("Not enough samples")
            return

        self.model.eval()

        flows_t, actions, flows_t1 = self.buffer.sample(n_samples)
        flows_t = flows_t.to(self.device)
        flows_t1 = flows_t1.to(self.device)
        actions = actions.to(self.device)

        with torch.no_grad():
            action_logits = self.model.inverse_model(flows_t, flows_t1)
            pred_actions = action_logits.argmax(dim=-1)
            probs = torch.softmax(action_logits, dim=-1)

        # Atari action names (Pong subset)
        action_names = ['NOOP', 'FIRE', 'UP', 'DOWN', 'UPFIRE', 'DOWNFIRE']

        fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_samples):
            # Flow t (as RGB)
            flow_t_rgb = flow_to_rgb(flows_t[i].cpu())
            axes[i, 0].imshow(flow_t_rgb)
            axes[i, 0].set_title('Flow t (velocity field)', fontsize=10)
            axes[i, 0].axis('off')

            # Flow t+1 (as RGB)
            flow_t1_rgb = flow_to_rgb(flows_t1[i].cpu())
            axes[i, 1].imshow(flow_t1_rgb)
            axes[i, 1].set_title('Flow t+1', fontsize=10)
            axes[i, 1].axis('off')

            # Flow change magnitude
            flow_diff = flows_t1[i] - flows_t[i]
            flow_diff_mag = torch.sqrt((flow_diff ** 2).sum(dim=0)).cpu().numpy()
            im = axes[i, 2].imshow(flow_diff_mag, cmap='hot')
            axes[i, 2].set_title('Flow change magnitude', fontsize=10)
            axes[i, 2].axis('off')
            plt.colorbar(im, ax=axes[i, 2], fraction=0.046)

            # Action prediction
            true_action = actions[i].item()
            pred_action = pred_actions[i].item()

            action_probs = probs[i].cpu().numpy()[:6]  # First 6 actions
            bars = axes[i, 3].bar(range(6), action_probs)

            # Color code
            bars[true_action].set_color('green')
            if pred_action != true_action and pred_action < 6:
                bars[pred_action].set_color('red')

            axes[i, 3].set_xticks(range(6))
            axes[i, 3].set_xticklabels(action_names, rotation=45, ha='right', fontsize=8)
            axes[i, 3].set_ylim([0, 1])
            axes[i, 3].axhline(1/self.n_actions, color='gray', linestyle='--', alpha=0.5, label='Random')

            correct = "✓" if pred_action == true_action else "✗"
            axes[i, 3].set_title(
                f'{correct} True: {action_names[true_action] if true_action < 6 else str(true_action)}\n'
                f'Pred: {action_names[pred_action] if pred_action < 6 else str(pred_action)}',
                fontsize=10,
                color='green' if pred_action == true_action else 'red'
            )

        plt.tight_layout()
        plt.savefig('flow_inverse_predictions.png', dpi=150, bbox_inches='tight')
        print("✅ Saved: flow_inverse_predictions.png")
        plt.show()

        self.model.train()

    def visualize_active_inference_policy(self, n_samples=4, temperature=1.0, beta=None):
        """
        Visualize active inference policy in action.

        Shows:
        - Current flow field
        - Predicted flows for each action
        - Free energies (uncertainty vs entropy trade-off)
        - Action probabilities
        - Selected action

        This reveals HOW the model decides what to do!
        """
        if len(self.buffer) < n_samples:
            print("Not enough samples")
            return

        flows_t, _, _ = self.buffer.sample(n_samples)
        valid_actions = VALID_ACTIONS.get(self.env_name, list(range(self.n_actions)))

        fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_samples):
            flow = flows_t[i]

            # Use active inference to select action
            selected_action, policy_info = self.select_action_active_inference(
                flow, temperature=temperature, beta=beta
            )

            # Current flow
            flow_rgb = flow_to_rgb(flow)
            axes[i, 0].imshow(flow_rgb)
            axes[i, 0].set_title(f'Current Flow Field\n(Sample {i+1})', fontsize=10)
            axes[i, 0].axis('off')

            # Free energies for each action
            free_energies = policy_info['free_energies']
            action_names = [ACTION_NAMES.get(a, str(a)) for a in valid_actions]

            bars = axes[i, 1].bar(range(len(valid_actions)), free_energies)
            selected_idx = policy_info['selected_idx']
            bars[selected_idx].set_color('green')

            axes[i, 1].set_xticks(range(len(valid_actions)))
            axes[i, 1].set_xticklabels(action_names, rotation=45, ha='right', fontsize=8)
            axes[i, 1].set_ylabel('Expected Free Energy')
            axes[i, 1].axhline(0, color='black', linestyle='-', linewidth=0.5)
            axes[i, 1].set_title('Free Energy per Action\n(Lower = Better)', fontsize=10)
            axes[i, 1].grid(alpha=0.3)

            # Uncertainty vs Entropy
            uncertainties = policy_info['uncertainties']
            entropies = policy_info['entropies']

            axes[i, 2].scatter(uncertainties, entropies, s=100, alpha=0.6)
            axes[i, 2].scatter([uncertainties[selected_idx]], [entropies[selected_idx]],
                             s=200, color='green', marker='*', edgecolors='black', linewidths=2,
                             label=f'Selected: {action_names[selected_idx]}')

            for j, action_name in enumerate(action_names):
                axes[i, 2].annotate(action_name, (uncertainties[j], entropies[j]),
                                   fontsize=7, ha='center', va='bottom')

            axes[i, 2].set_xlabel('Uncertainty (Variance)')
            axes[i, 2].set_ylabel('Entropy (Diversity)')
            axes[i, 2].set_title(f'Epistemic Trade-off\n(β={policy_info["beta"]:.3f})', fontsize=10)
            axes[i, 2].legend(fontsize=8)
            axes[i, 2].grid(alpha=0.3)

            # Action probabilities
            action_probs = policy_info['action_probs']
            bars = axes[i, 3].bar(range(len(valid_actions)), action_probs)
            bars[selected_idx].set_color('green')

            axes[i, 3].set_xticks(range(len(valid_actions)))
            axes[i, 3].set_xticklabels(action_names, rotation=45, ha='right', fontsize=8)
            axes[i, 3].set_ylabel('Probability')
            axes[i, 3].set_ylim([0, 1])
            axes[i, 3].axhline(1/len(valid_actions), color='gray', linestyle='--', alpha=0.5)
            axes[i, 3].set_title(f'Action Distribution\n(T={policy_info["temperature"]:.1f})', fontsize=10)
            axes[i, 3].grid(alpha=0.3)

        plt.suptitle(f'🌀 Active Inference Policy Visualization | {self.env_name}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('flow_active_inference_policy.png', dpi=150, bbox_inches='tight')
        print("✅ Saved: flow_active_inference_policy.png")
        plt.show()

    def plot_training(self):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Forward loss
        axes[0, 0].plot(self.history['forward'])
        axes[0, 0].set_title('Forward Loss (Flow Prediction)')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].grid(alpha=0.3)

        # Inverse loss
        axes[0, 1].plot(self.history['inverse'])
        axes[0, 1].axhline(np.log(self.n_actions), color='r', linestyle='--', label='Random')
        axes[0, 1].set_title('Inverse Loss (Action Prediction)')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Cross Entropy')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # Accuracy
        axes[1, 0].plot(np.array(self.history['accuracy']) * 100)
        axes[1, 0].axhline(100/self.n_actions, color='r', linestyle='--', label='Random')
        axes[1, 0].set_title('Inverse Model Accuracy')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].set_ylim([0, 100])
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # Consistency
        axes[1, 1].plot(self.history['consistency'])
        axes[1, 1].set_title('Consistency Loss (Forward-Inverse Agreement)')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('MSE')
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('flow_training_curves.png', dpi=150, bbox_inches='tight')
        print("✅ Saved: flow_training_curves.png")
        plt.show()


# ============================================================================
# QUICK START
# ============================================================================

print("\n" + "="*70)
print("🌊 FLOW-BASED TETRAHEDRAL INVERSE MODEL - READY")
print("="*70)
print("\n📚 USAGE EXAMPLES:\n")
print("1️⃣  WATCH IT LEARN LIVE! (like you wanted)")
print("   trainer = FlowInverseTrainer(env_name='ALE/Pong-v5', frameskip=10)")
print("   trainer.train_loop_online(n_steps=500, show_gameplay=True)")
print("   # 🎬 LIVE GAMEPLAY: See it play as it learns!")
print("   # Game + Flow field + Learning curve updating in real-time")
print()
print("2️⃣  Adjust exploration (β parameter)")
print("   # High β: Seek diverse outcomes (like infant exploration)")
print("   trainer.train_loop_online(n_steps=500, beta=0.1)")
print("   # Low β: Seek certain outcomes (like skilled performance)")
print("   trainer.train_loop_online(n_steps=500, beta=0.01)")
print()
print("3️⃣  Batch training (if you must - for GPU efficiency)")
print("   trainer.train_loop(n_episodes=10)  # Collects, then trains")
print("   # Still uses active inference, but not truly online")
print()
print("4️⃣  Visualize what model learned")
print("   trainer.visualize_flow_predictions()")
print("   trainer.visualize_active_inference_policy()")
print()
print("5️⃣  Test different natural frequencies")
print("   # Theta (6 Hz) - φ-aligned with memory field - DEFAULT")
print("   trainer_theta = FlowInverseTrainer(frameskip=10)")
print("   # Alpha (10 Hz)")
print("   trainer_alpha = FlowInverseTrainer(frameskip=6)")
print("   # Saccade (4 Hz) - human visual sampling")
print("   trainer_saccade = FlowInverseTrainer(frameskip=15)")
print()
print("6️⃣  Resume from checkpoint")
print("   trainer = FlowInverseTrainer(env_name='ALE/Pong-v5')")
print("   trainer.load_checkpoint('checkpoints/autosave_ep50.pt')")
print("   trainer.train_loop(n_episodes=10)  # Continues with active inference")
print()
print("🔬 FOR ABLATION/COMPARISON ONLY:")
print("   # Random baseline (NOT natural - only for ML paper comparisons)")
print("   trainer_random = FlowInverseTrainer(frameskip=10)")
print("   trainer_random.train_loop(n_episodes=10, use_active_inference=False)")
print("   # Or random sampling (breaks temporal coherence)")
print("   trainer_rand = FlowInverseTrainer(sequential_sampling=False)")
print()
print("="*70)
print("\n🧪 TESTING: Harmonic Resonance Hypothesis")
print("  ✓ Do actions create characteristic flow frequencies?")
print("  ✓ Does tetrahedral structure learn harmonics?")
print("  ✓ Does active inference improve learning efficiency?")
print("  ✓ Can model ACT intelligently based on learned flow patterns?")
print("\n🌀 THE STRANGE LOOP CLOSES:")
print("   Perception → Model → Action → Effect → New Perception → ...")
print("\n🔬 Let's find out!")
print("="*70 + "\n")
