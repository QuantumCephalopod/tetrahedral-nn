"""
COUNCIL OF ADVERSARIES ARCHITECTURE

Philosophy:
- Both networks generate FIELDS of possibilities (not single outputs)
- Each face becomes a specialized judge with its own perspective
- Face coupling = adversarial debate channel
- Consensus emerges from democratic voting
- No privileged "ground truth" - reality negotiated through council
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path

# Import existing architecture
import sys
sys.path.append('.')
from W_geometry import EdgeAttention, FaceAttention, InterFaceAttention

# ============================================================================
# CORE COMPONENTS
# ============================================================================

class FaceJudge(nn.Module):
    """
    Each face judges candidates from its unique perspective.
    Face 0 might care about topology, Face 1 about texture, etc.
    Learns its own criteria through training.
    """
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # Judge network: candidate + face context -> score
        self.judge_net = nn.Sequential(
            nn.Linear(output_dim + latent_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 0-1 score: how good is this candidate?
        )

    def forward(self, candidates, face_state):
        """
        candidates: [batch, num_candidates, output_dim]
        face_state: [batch, latent_dim]
        returns: [batch, num_candidates] - score for each
        """
        bs, num_cand, out_dim = candidates.shape

        # Expand face to match each candidate
        face_expanded = face_state.unsqueeze(1).expand(bs, num_cand, -1)
        # [batch, num_candidates, latent_dim]

        # Concatenate for judging
        judge_input = torch.cat([candidates, face_expanded], dim=-1)
        # [batch, num_candidates, output_dim + latent_dim]

        # Score each candidate
        scores = self.judge_net(judge_input).squeeze(-1)
        return scores  # [batch, num_candidates]


class AdversarialFaceCoupling(nn.Module):
    """
    Enhanced face coupling that transmits CRITIQUE alongside features.
    The debate happens here.
    """
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        # Feature coupling (existing)
        self.coupling = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )

        # Critique integration (new)
        self.critique_gate = nn.Sequential(
            nn.Linear(latent_dim + 1, latent_dim),  # +1 for critique score
            nn.Sigmoid()  # how much to listen to critique
        )

    def forward(self, face_a, face_b, critique_score=None):
        """
        face_a: [batch, latent_dim] - sending face
        face_b: [batch, latent_dim] - receiving face
        critique_score: [batch] - optional critique signal
        """
        # Standard coupling
        coupled = self.coupling(torch.cat([face_a, face_b], dim=-1))

        # If critique provided, modulate the signal
        if critique_score is not None:
            # Reshape critique
            if critique_score.dim() == 1:
                critique_score = critique_score.unsqueeze(-1)

            # Gate based on critique
            gate_input = torch.cat([coupled, critique_score], dim=-1)
            gate = self.critique_gate(gate_input)
            coupled = coupled * gate  # critique modulates coupling strength

        return coupled


class LinearTetrahedron(nn.Module):
    """Linear tetrahedron (smooth, topological)"""
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Input -> 4 vertices
        self.input_projection = nn.Linear(input_dim, latent_dim * 4)

        # Attention layers
        self.edge_attention = EdgeAttention(latent_dim)
        self.face_attention = FaceAttention(latent_dim)
        self.interface_attention = InterFaceAttention(latent_dim)

    def forward(self, x, return_faces=False):
        vertices = self.input_projection(x).reshape(-1, 4, self.latent_dim)
        vertices = self.edge_attention(vertices)
        vertices, faces = self.face_attention(vertices, return_faces=True)
        vertices = self.interface_attention(vertices, faces)

        if return_faces:
            return vertices, faces
        return vertices

    def update_from_faces(self, vertices, face_updates):
        """Update vertices based on face coupling signals"""
        bs = vertices.size(0)
        # Simple averaging for now
        update = torch.stack(face_updates, dim=1).mean(dim=1)
        update = update.reshape(bs, 4, self.latent_dim)
        return vertices + update * 0.1


class NonlinearTetrahedron(nn.Module):
    """Nonlinear tetrahedron (boundaries, perception)"""
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Input -> 4 vertices (with ReLU)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 8),
            nn.ReLU(),
            nn.Linear(latent_dim * 8, latent_dim * 4)
        )

        # Attention layers
        self.edge_attention = EdgeAttention(latent_dim)
        self.face_attention = FaceAttention(latent_dim)
        self.interface_attention = InterFaceAttention(latent_dim)

    def forward(self, x, return_faces=False):
        vertices = self.input_projection(x).reshape(-1, 4, self.latent_dim)
        vertices = self.edge_attention(vertices)
        vertices, faces = self.face_attention(vertices, return_faces=True)
        vertices = self.interface_attention(vertices, faces)

        if return_faces:
            return vertices, faces
        return vertices

    def update_from_faces(self, vertices, face_updates):
        """Update vertices based on face coupling signals"""
        bs = vertices.size(0)
        update = torch.stack(face_updates, dim=1).mean(dim=1)
        update = update.reshape(bs, 4, self.latent_dim)
        return vertices + update * 0.1


# ============================================================================
# COUNCIL OF ADVERSARIES NETWORK
# ============================================================================

class CouncilOfAdversariesNetwork(nn.Module):
    """
    Dual tetrahedral network with council-based output generation.

    Each network generates multiple candidate outputs (field).
    Each face judges all candidates from its perspective.
    Face coupling transmits critiques between networks.
    Final output emerges from consensus voting.
    """
    def __init__(self, input_dim, output_dim, latent_dim,
                 num_candidates=4, coupling_strength=0.5):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.num_candidates = num_candidates
        self.coupling_strength = coupling_strength

        # Dual tetrahedra
        self.linear_net = LinearTetrahedron(input_dim, latent_dim)
        self.nonlinear_net = NonlinearTetrahedron(input_dim, latent_dim)

        # Face coupling (adversarial)
        self.linear_to_nonlinear = nn.ModuleList([
            AdversarialFaceCoupling(latent_dim) for _ in range(4)
        ])
        self.nonlinear_to_linear = nn.ModuleList([
            AdversarialFaceCoupling(latent_dim) for _ in range(4)
        ])

        # FIELD GENERATION: Multiple output heads per network
        self.linear_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim * 4, latent_dim * 2),
                nn.ReLU(),
                nn.Linear(latent_dim * 2, output_dim)
            )
            for _ in range(num_candidates)
        ])

        self.nonlinear_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim * 4, latent_dim * 2),
                nn.ReLU(),
                nn.Linear(latent_dim * 2, output_dim)
            )
            for _ in range(num_candidates)
        ])

        # COUNCIL JUDGES: Each face becomes a critic
        self.linear_face_judges = nn.ModuleList([
            FaceJudge(latent_dim, output_dim) for _ in range(4)
        ])
        self.nonlinear_face_judges = nn.ModuleList([
            FaceJudge(latent_dim, output_dim) for _ in range(4)
        ])

        # Input reconstruction (grounding)
        self.input_reconstructor = nn.Linear(latent_dim * 8, input_dim)

    def forward(self, x, return_field=False):
        bs = x.size(0)

        # ========================================
        # PHASE 1: Process input through both nets
        # ========================================
        lin_v, lin_f = self.linear_net(x, return_faces=True)
        non_v, non_f = self.nonlinear_net(x, return_faces=True)
        # lin_v, non_v: [batch, 4, latent_dim]
        # lin_f, non_f: [batch, 4, latent_dim]

        # ========================================
        # PHASE 2: Generate FIELD of candidates
        # ========================================
        lin_vertices_flat = lin_v.reshape(bs, -1)
        non_vertices_flat = non_v.reshape(bs, -1)

        # Each generator produces one candidate
        lin_candidates = torch.stack([
            gen(lin_vertices_flat) for gen in self.linear_generators
        ], dim=1)  # [batch, num_candidates, output_dim]

        non_candidates = torch.stack([
            gen(non_vertices_flat) for gen in self.nonlinear_generators
        ], dim=1)  # [batch, num_candidates, output_dim]

        # Combine into full field
        all_candidates = torch.cat([lin_candidates, non_candidates], dim=1)
        # [batch, num_candidates*2, output_dim]

        # ========================================
        # PHASE 3: Council voting - each face judges all candidates
        # ========================================
        lin_judgments = []  # Will be [4, batch, num_candidates*2]
        non_judgments = []

        for face_idx in range(4):
            # Linear face judges all candidates
            lin_face_state = lin_f[:, face_idx]  # [batch, latent_dim]
            lin_scores = self.linear_face_judges[face_idx](
                all_candidates, lin_face_state
            )  # [batch, num_candidates*2]
            lin_judgments.append(lin_scores)

            # Nonlinear face judges all candidates
            non_face_state = non_f[:, face_idx]
            non_scores = self.nonlinear_face_judges[face_idx](
                all_candidates, non_face_state
            )
            non_judgments.append(non_scores)

        lin_judgments = torch.stack(lin_judgments, dim=1)  # [batch, 4, num_candidates*2]
        non_judgments = torch.stack(non_judgments, dim=1)

        # ========================================
        # PHASE 4: Adversarial debate through face coupling
        # ========================================
        # Each network's faces critique the OTHER network's candidates

        debate_signals_to_linear = []
        debate_signals_to_nonlinear = []

        for i in range(4):
            # Nonlinear critiques LINEAR's candidates (first num_candidates)
            non_critique_of_lin = non_judgments[:, i, :self.num_candidates].mean(dim=1)
            # Send critique through coupling
            signal_to_lin = self.nonlinear_to_linear[i](
                non_f[:, i],
                lin_f[:, i],
                critique_score=non_critique_of_lin
            )
            debate_signals_to_linear.append(signal_to_lin)

            # Linear critiques NONLINEAR's candidates (last num_candidates)
            lin_critique_of_non = lin_judgments[:, i, self.num_candidates:].mean(dim=1)
            signal_to_non = self.linear_to_nonlinear[i](
                lin_f[:, i],
                non_f[:, i],
                critique_score=lin_critique_of_non
            )
            debate_signals_to_nonlinear.append(signal_to_non)

        # Update vertices based on debate (optional, can enable/disable)
        if self.coupling_strength > 0:
            lin_v = self.linear_net.update_from_faces(
                lin_v,
                [s * self.coupling_strength for s in debate_signals_to_linear]
            )
            non_v = self.nonlinear_net.update_from_faces(
                non_v,
                [s * self.coupling_strength for s in debate_signals_to_nonlinear]
            )

        # ========================================
        # PHASE 5: Consensus emerges from all votes
        # ========================================
        # Average judgments across both networks and all faces
        consensus_scores = (lin_judgments + non_judgments) / 2  # [batch, 4, num_cand*2]
        consensus_scores = consensus_scores.mean(dim=1)  # [batch, num_cand*2]

        # Softmax to get democratic weights
        weights = F.softmax(consensus_scores * 5.0, dim=1)  # temperature=5 for sharpness

        # Weighted combination = consensus output
        final_output = (all_candidates * weights.unsqueeze(-1)).sum(dim=1)

        # Input reconstruction for grounding
        combined_vertices = torch.cat([lin_v.reshape(bs, -1), non_v.reshape(bs, -1)], dim=-1)
        reconstructed_input = self.input_reconstructor(combined_vertices)

        if return_field:
            return final_output, {
                'linear_field': lin_candidates,
                'nonlinear_field': non_candidates,
                'all_candidates': all_candidates,
                'linear_judgments': lin_judgments,
                'nonlinear_judgments': non_judgments,
                'consensus_weights': weights,
                'reconstructed_input': reconstructed_input,
                'debate_signals': (debate_signals_to_linear, debate_signals_to_nonlinear)
            }

        return final_output, reconstructed_input


# ============================================================================
# EXTERNAL DISCRIMINATOR (reality check)
# ============================================================================

class ExternalDiscriminator(nn.Module):
    """
    External discriminator judges if outputs are realistic.
    Separate from council - provides objective "real vs fake" signal.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Returns probability that x is real"""
        return self.net(x)


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_council_step(model, discriminator, batch_x, batch_y,
                       optimizer, disc_optimizer,
                       input_recon_weight=0.1,
                       diversity_weight=0.5,
                       consensus_weight=0.3,
                       adversarial_weight=0.2):
    """
    Single training step for Council of Adversaries.

    Returns: (gen_loss, disc_loss, metrics_dict)
    """
    bs = batch_x.size(0)

    # ========================================
    # GENERATOR STEP
    # ========================================

    output, field_data = model(batch_x, return_field=True)

    lin_field = field_data['linear_field']         # [batch, num_cand, dim]
    non_field = field_data['nonlinear_field']      # [batch, num_cand, dim]
    all_candidates = field_data['all_candidates']  # [batch, num_cand*2, dim]
    lin_judgments = field_data['linear_judgments']
    non_judgments = field_data['nonlinear_judgments']
    weights = field_data['consensus_weights']
    recon_input = field_data['reconstructed_input']

    # LOSS 1: Realism (fool external discriminator)
    real_scores = discriminator(batch_y)

    # Judge all candidates
    all_cand_flat = all_candidates.reshape(-1, model.output_dim)
    fake_scores_all = discriminator(all_cand_flat).reshape(bs, -1)

    # Want at least BEST candidate to fool discriminator
    best_fake_scores, _ = fake_scores_all.max(dim=1)
    gen_realism_loss = -torch.log(best_fake_scores + 1e-8).mean()

    # LOSS 2: Diversity (candidates should explore manifold)
    # Within each network, candidates should be different
    lin_diversity_loss = 0
    non_diversity_loss = 0
    for b in range(bs):
        if lin_field.shape[1] > 1:  # need at least 2 candidates
            lin_dists = torch.pdist(lin_field[b])  # pairwise distances
            lin_diversity_loss -= torch.log(lin_dists.mean() + 1e-8)
        if non_field.shape[1] > 1:
            non_dists = torch.pdist(non_field[b])
            non_diversity_loss -= torch.log(non_dists.mean() + 1e-8)
    diversity_loss = (lin_diversity_loss + non_diversity_loss) / (2 * bs)

    # LOSS 3: Consensus (faces should agree on what's good)
    # Low variance = strong consensus
    consensus_variance = lin_judgments.var(dim=1).mean() + non_judgments.var(dim=1).mean()
    consensus_loss = consensus_variance

    # LOSS 4: Input reconstruction (grounding)
    recon_loss = F.mse_loss(recon_input, batch_x)

    # LOSS 5: Adversarial debate (symmetric)
    # Linear should be harsh on nonlinear's outputs, vice versa
    num_cand = model.num_candidates
    lin_critique_non = lin_judgments[:, :, num_cand:].mean()  # judge nonlinear outputs
    non_critique_lin = non_judgments[:, :, :num_cand].mean()  # judge linear outputs
    # Want these LOW (harsh critics), but need them to provide signal
    # So we actually want them moderate (not too harsh, not too soft)
    target_critique = 0.4  # sweet spot
    adversarial_loss = (
        F.mse_loss(lin_critique_non, torch.tensor(target_critique, device=lin_critique_non.device)) +
        F.mse_loss(non_critique_lin, torch.tensor(target_critique, device=non_critique_lin.device))
    )

    # TOTAL GENERATOR LOSS
    total_gen_loss = (
        1.0 * gen_realism_loss +
        diversity_weight * diversity_loss +
        consensus_weight * consensus_loss +
        input_recon_weight * recon_loss +
        adversarial_weight * adversarial_loss
    )

    optimizer.zero_grad()
    total_gen_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # ========================================
    # DISCRIMINATOR STEP
    # ========================================

    # Re-generate with no grad
    with torch.no_grad():
        output_detached, field_detached = model(batch_x, return_field=True)
        all_cand_detached = field_detached['all_candidates'].reshape(-1, model.output_dim)

    real_scores = discriminator(batch_y)
    fake_scores = discriminator(all_cand_detached)

    disc_loss = (
        -torch.log(real_scores + 1e-8).mean() -
        torch.log(1 - fake_scores + 1e-8).mean()
    )

    disc_optimizer.zero_grad()
    disc_loss.backward()
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
    disc_optimizer.step()

    # ========================================
    # METRICS
    # ========================================

    metrics = {
        'gen_total': total_gen_loss.item(),
        'gen_realism': gen_realism_loss.item(),
        'diversity': diversity_loss.item(),
        'consensus': consensus_loss.item(),
        'recon': recon_loss.item(),
        'adversarial': adversarial_loss.item(),
        'disc_loss': disc_loss.item(),
        'real_score': real_scores.mean().item(),
        'fake_score': best_fake_scores.mean().item(),
        'weight_entropy': -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean().item()
    }

    return total_gen_loss.item(), disc_loss.item(), metrics


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_council_field(model, test_input, test_target, device, epoch):
    """
    Visualize the field of candidates and consensus.
    """
    model.eval()
    with torch.no_grad():
        test_input = test_input.to(device)
        output, field_data = model(test_input[:1], return_field=True)

        lin_field = field_data['linear_field'][0].cpu()  # [num_cand, dim]
        non_field = field_data['nonlinear_field'][0].cpu()
        weights = field_data['consensus_weights'][0].cpu()

        # Reshape to images
        img_size = int(np.sqrt(model.output_dim // 3))

        num_cand = model.num_candidates
        total_cand = num_cand * 2

        fig, axes = plt.subplots(3, total_cand + 2, figsize=(20, 8))

        # Row 0: Linear candidates
        for i in range(num_cand):
            img = lin_field[i].reshape(img_size, img_size, 3).numpy()
            axes[0, i].imshow(np.clip(img, 0, 1))
            axes[0, i].set_title(f'Lin-{i}\nw={weights[i]:.3f}')
            axes[0, i].axis('off')

        # Row 1: Nonlinear candidates
        for i in range(num_cand):
            img = non_field[i].reshape(img_size, img_size, 3).numpy()
            axes[1, i].imshow(np.clip(img, 0, 1))
            axes[1, i].set_title(f'Non-{i}\nw={weights[num_cand+i]:.3f}')
            axes[1, i].axis('off')

        # Row 2: Input, Consensus, Target
        input_img = test_input[0].cpu().reshape(img_size, img_size, 3).numpy()
        output_img = output[0].cpu().reshape(img_size, img_size, 3).numpy()
        target_img = test_target[0].cpu().reshape(img_size, img_size, 3).numpy()

        axes[2, 0].imshow(np.clip(input_img, 0, 1))
        axes[2, 0].set_title('Input')
        axes[2, 0].axis('off')

        axes[2, 1].imshow(np.clip(output_img, 0, 1))
        axes[2, 1].set_title('Consensus')
        axes[2, 1].axis('off')

        axes[2, 2].imshow(np.clip(target_img, 0, 1))
        axes[2, 2].set_title('Target')
        axes[2, 2].axis('off')

        # Hide unused
        for i in range(num_cand, total_cand + 2):
            axes[0, i].axis('off')
            axes[1, i].axis('off')
        for i in range(3, total_cand + 2):
            axes[2, i].axis('off')

        plt.suptitle(f'Council of Adversaries - Epoch {epoch}')
        plt.tight_layout()
        plt.savefig(f'council_field_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.close()

    model.train()


if __name__ == "__main__":
    print("="*70)
    print("🔷 COUNCIL OF ADVERSARIES - Ready to import and use!")
    print("="*70)
    print("\nThis module provides:")
    print("  - CouncilOfAdversariesNetwork: Main architecture")
    print("  - ExternalDiscriminator: Reality judge")
    print("  - train_council_step: Training function")
    print("  - visualize_council_field: Visualization")
    print("\nSee COUNCIL_TRAINING.py for complete training script")
    print("="*70)
