"""
Z - INTER-FACE COUPLING (Integration Vertex)
============================================

Dual-tetrahedral architecture with face-to-face communication.
Coordinates Linear and Nonlinear tetrahedra without vertex contamination.

This is the key innovation: faces become the interface between networks.
The two hemispheres communicate through collective patterns (triangles),
not individual neurons (vertices).

Architecture:
  - LINEAR tetrahedron (4 vertices, no ReLU)
  - NONLINEAR tetrahedron (4 vertices, with ReLU)
  - INTER-FACE coupling (4 face-to-face attention pairs)
  - Combined output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from W_geometry import InterFaceAttention, FACE_INDICES
from X_linear_tetrahedron import LinearTetrahedron
from Y_nonlinear_tetrahedron import NonlinearTetrahedron


class DualTetrahedralNetwork(nn.Module):
    """
    Dual tetrahedral architecture with inter-face coupling.

    This network coordinates two separate tetrahedra:
      1. Linear (left hemisphere): smooth manifolds, deterministic
      2. Nonlinear (right hemisphere): boundaries, statistical

    They communicate through face-to-face attention, allowing pattern-level
    information exchange without vertex contamination.

    Args:
        input_dim: Dimension of input data
        output_dim: Dimension of output data
        latent_dim: Internal representation dimension (default: 64)
        coupling_strength: Weight for inter-face coupling (0.0 to 1.0)
        output_mode: How to combine outputs ("weighted", "linear_only", "nonlinear_only")
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        latent_dim: int = 64,
        coupling_strength: float = 0.5,
        output_mode: str = "weighted"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.coupling_strength = coupling_strength
        self.output_mode = output_mode

        # === TEMPORAL FIELD (What persists across frames) ===
        self.temporal_field = None  # Initialized on first forward pass
        self.field_decay = 0.1  # How much new info vs old context

        # === LINEAR TETRAHEDRON (Left Hemisphere) ===
        self.linear_net = LinearTetrahedron(
            input_dim=input_dim,
            latent_dim=latent_dim,
            output_dim=None  # No direct output, use coupling
        )

        # === NONLINEAR TETRAHEDRON (Right Hemisphere) ===
        self.nonlinear_net = NonlinearTetrahedron(
            input_dim=input_dim,
            latent_dim=latent_dim,
            output_dim=None  # No direct output, use coupling
        )

        # === INTER-FACE COUPLING ===
        # 4 pairs of face-to-face attention (one per face)
        # Linear → Nonlinear
        self.linear_to_nonlinear = nn.ModuleList([
            InterFaceAttention(latent_dim) for _ in range(4)
        ])
        # Nonlinear → Linear
        self.nonlinear_to_linear = nn.ModuleList([
            InterFaceAttention(latent_dim) for _ in range(4)
        ])

        # === OUTPUT COMBINATION ===
        self.output_projection = nn.Linear(latent_dim * 8, output_dim)  # 4 vertices × 2 networks

        # Learnable weights for combining network outputs
        if output_mode == "weighted":
            self.output_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
        else:
            self.register_buffer('output_weights', torch.tensor([1.0, 0.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dual tetrahedral network with inter-face coupling.

        Information flow:
          1. Both networks process input independently
          2. Compute face representations in each network
          3. Inter-face coupling: faces attend to corresponding faces
          4. Update vertices based on coupled face information
          5. Combine outputs from both networks

        Args:
            x: Input tensor, shape (batch, input_dim)

        Returns:
            Output tensor, shape (batch, output_dim)
        """
        batch_size = x.size(0)

        # ====================================================================
        # STEP 1: INDEPENDENT PROCESSING (Edges + Internal Faces)
        # ====================================================================

        # Linear network: forward through edges and faces
        linear_vertices, linear_faces = self.linear_net(x, return_faces=True)
        # linear_vertices: (batch, 4, latent_dim)
        # linear_faces: (batch, 4, latent_dim)

        # Nonlinear network: forward through edges and faces
        nonlinear_vertices, nonlinear_faces = self.nonlinear_net(x, return_faces=True)
        # nonlinear_vertices: (batch, 4, latent_dim)
        # nonlinear_faces: (batch, 4, latent_dim)

        # ====================================================================
        # STEP 1.5: UPDATE TEMPORAL FIELD (What persists?)
        # ====================================================================
        # The field is an exponential moving average of vertex states
        # This is what the network "remembers" across frames
        current_state = linear_vertices.detach()  # Use linear (smooth) vertices

        if self.temporal_field is None:
            # First frame: initialize field
            self.temporal_field = current_state
        else:
            # Subsequent frames: blend new + old
            # α controls how much new info vs context
            self.temporal_field = (self.field_decay * current_state +
                                  (1 - self.field_decay) * self.temporal_field)

        # ====================================================================
        # STEP 2: INTER-FACE COUPLING
        # ====================================================================

        # Linear faces attend to nonlinear faces
        coupled_linear_faces = []
        for i in range(4):
            coupled = self.linear_to_nonlinear[i](
                linear_faces[:, i],      # Linear face i
                nonlinear_faces[:, i]    # Corresponding nonlinear face i
            )
            coupled_linear_faces.append(coupled)

        # Nonlinear faces attend to linear faces
        coupled_nonlinear_faces = []
        for i in range(4):
            coupled = self.nonlinear_to_linear[i](
                nonlinear_faces[:, i],   # Nonlinear face i
                linear_faces[:, i]       # Corresponding linear face i
            )
            coupled_nonlinear_faces.append(coupled)

        # ====================================================================
        # STEP 3: UPDATE VERTICES WITH COUPLED INFORMATION
        # ====================================================================

        # Update linear vertices (NO ReLU - stays continuous)
        linear_vertices = self.linear_net.update_from_faces(
            linear_vertices,
            [f * self.coupling_strength for f in coupled_linear_faces]
        )

        # Update nonlinear vertices (WITH ReLU)
        nonlinear_vertices = self.nonlinear_net.update_from_faces(
            nonlinear_vertices,
            [f * self.coupling_strength for f in coupled_nonlinear_faces]
        )

        # ====================================================================
        # STEP 4: COMBINE OUTPUTS
        # ====================================================================

        if self.output_mode == "linear_only":
            # Use only linear network (for arithmetic verification)
            vertices_combined = linear_vertices.reshape(batch_size, -1)
            # Pad to match expected input size
            padding = torch.zeros(batch_size, self.latent_dim * 4, device=x.device)
            vertices_combined = torch.cat([vertices_combined, padding], dim=-1)

        elif self.output_mode == "nonlinear_only":
            # Use only nonlinear network (for testing)
            vertices_combined = nonlinear_vertices.reshape(batch_size, -1)
            padding = torch.zeros(batch_size, self.latent_dim * 4, device=x.device)
            vertices_combined = torch.cat([padding, vertices_combined], dim=-1)

        else:  # "weighted"
            # Combine both networks with learnable weights
            linear_flat = linear_vertices.reshape(batch_size, -1)
            nonlinear_flat = nonlinear_vertices.reshape(batch_size, -1)
            vertices_combined = torch.cat([linear_flat, nonlinear_flat], dim=-1)

        # Project to output
        output = self.output_projection(vertices_combined)

        # Apply output weights if in weighted mode
        if self.output_mode == "weighted":
            weights = F.softmax(self.output_weights, dim=0)
            # This is a simplified weighting - more sophisticated versions possible
            output = output * weights[0]  # Scale by learned weight

        return output

    def get_network_contributions(self, x: torch.Tensor) -> dict:
        """
        Get separate outputs from each network for analysis.

        Args:
            x: Input tensor, shape (batch, input_dim)

        Returns:
            Dictionary with 'linear', 'nonlinear', 'combined' outputs
        """
        batch_size = x.size(0)

        # Process through both networks
        linear_vertices, _ = self.linear_net(x, return_faces=True)
        nonlinear_vertices, _ = self.nonlinear_net(x, return_faces=True)

        # Create separate outputs
        linear_only = torch.cat([
            linear_vertices.reshape(batch_size, -1),
            torch.zeros(batch_size, self.latent_dim * 4, device=x.device)
        ], dim=-1)

        nonlinear_only = torch.cat([
            torch.zeros(batch_size, self.latent_dim * 4, device=x.device),
            nonlinear_vertices.reshape(batch_size, -1)
        ], dim=-1)

        return {
            'linear': self.output_projection(linear_only),
            'nonlinear': self.output_projection(nonlinear_only),
            'combined': self.forward(x)
        }


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class DualTetrahedralTrainer:
    """
    Training coordinator for dual tetrahedral networks.

    Handles:
      - Training loop
      - Loss computation
      - Metrics tracking
      - Network contribution analysis
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

    def train_epoch(self, train_loader, loss_fn=None) -> float:
        """Train for one epoch."""
        self.model.train()
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        total_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
            output = self.model(batch_x)
            loss = loss_fn(output, batch_y)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def evaluate(self, test_loader, loss_fn=None) -> dict:
        """Evaluate on test set."""
        self.model.eval()
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                output = self.model(batch_x)
                loss = loss_fn(output, batch_y)

                total_loss += loss.item()
                n_batches += 1

        return {'loss': total_loss / n_batches}

    def train(self, train_loader, test_loader, epochs: int, loss_fn=None) -> dict:
        """Full training loop."""
        history = {'train_loss': [], 'test_loss': []}

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, loss_fn)
            test_metrics = self.evaluate(test_loader, loss_fn)

            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_metrics['loss'])

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Test Loss: {test_metrics['loss']:.6f}")

        return history


# ============================================================================
# SUMMARY
# ============================================================================

"""
Z_interface_coupling.py - The Integration Vertex

ARCHITECTURE:
  - DualTetrahedralNetwork: Coordinates linear + nonlinear tetrahedra
  - Inter-face coupling: 4 face-to-face attention pairs (bidirectional)
  - Flexible output combination: weighted, linear_only, or nonlinear_only

KEY INNOVATION:
  Faces are the interface. Networks communicate at the pattern level
  (triangles) without vertex contamination. Linear stays continuous,
  nonlinear can be discontinuous.

INFORMATION FLOW:
  1. Independent processing (edges + faces within each network)
  2. Inter-face coupling (faces attend to corresponding faces)
  3. Vertex update (influenced by coupled faces, respects activation type)
  4. Output combination (flexible weighting)

USAGE:
  model = DualTetrahedralNetwork(input_dim=2, output_dim=1)
  trainer = DualTetrahedralTrainer(model, optimizer)
  history = trainer.train(train_loader, test_loader, epochs=200)

This is the meta-manifold: linear for topology, nonlinear for boundaries.
"""
