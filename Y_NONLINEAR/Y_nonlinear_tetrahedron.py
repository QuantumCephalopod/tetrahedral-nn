"""
Y - NONLINEAR TETRAHEDRON (Right Hemisphere / Emotional / Nondeterministic)
===========================================================================

Nonlinear tetrahedral network WITH ReLU activation.
Learns discontinuous boundaries and handles statistical/fuzzy patterns.

Perfect for:
  - Image processing (object boundaries)
  - Perception tasks (category boundaries)
  - Noisy/statistical data
  - Tasks requiring discrete decision boundaries

This network can create sharp boundaries and handle uncertainty that
doesn't fit smooth manifold structures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from W_geometry import (
    EDGE_INDICES,
    FACE_INDICES,
    EdgeAttention,
    FaceAttention,
    compute_all_edges,
    compute_all_faces,
    aggregate_to_vertices
)


class NonlinearTetrahedron(nn.Module):
    """
    Nonlinear tetrahedral network (Right Hemisphere).

    Architecture:
      - 4 vertices (WITH ReLU activation)
      - 6 edges (pairwise attention with ReLU)
      - 4 faces (3-point attention with ReLU)

    Key property: ReLU activations after aggregation steps.
    This enables:
      - Discrete decision boundaries
      - Sparse activations
      - Handling of discontinuities
      - Statistical pattern recognition

    Args:
        input_dim: Dimension of input data
        latent_dim: Internal vertex representation dimension
        output_dim: Dimension of output (if used standalone)
        activation: Activation function (default: ReLU)
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        output_dim: int = None,
        activation: nn.Module = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.activation = activation if activation is not None else nn.ReLU()

        # Input embedding: map input to 4 vertex initial states
        self.embed = nn.Linear(input_dim, latent_dim * 4)

        # Edge processing: 6 pairwise attention modules
        self.edge_modules = nn.ModuleList([
            EdgeAttention(latent_dim) for _ in range(6)
        ])

        # Face processing: 4 triangular attention modules
        self.face_modules = nn.ModuleList([
            FaceAttention(latent_dim) for _ in range(4)
        ])

        # Output projection (if used standalone)
        if output_dim is not None:
            self.output = nn.Linear(latent_dim * 4, output_dim)
        else:
            self.output = None

    def forward(self, x: torch.Tensor, return_faces: bool = False):
        """
        Forward pass through nonlinear tetrahedron.

        Args:
            x: Input tensor, shape (batch, input_dim)
            return_faces: If True, return (vertices, faces) for coupling

        Returns:
            If return_faces=False: output tensor, shape (batch, output_dim)
            If return_faces=True: (vertices, faces) for inter-face coupling
        """
        batch_size = x.size(0)

        # === STEP 1: EMBED TO VERTICES ===
        vertices = self.embed(x).view(batch_size, 4, self.latent_dim)
        vertices = self.activation(vertices)  # ReLU on initial vertices
        # Shape: (batch, 4, latent_dim)

        # === STEP 2: EDGE PROCESSING ===
        edge_outputs = compute_all_edges(vertices, self.edge_modules)
        # List of 6 tensors, each (batch, latent_dim)

        # Aggregate edge updates to vertices WITH ReLU
        vertices = aggregate_to_vertices(edge_outputs, vertices)
        vertices = self.activation(vertices)  # ReLU after edge aggregation
        # Shape: (batch, 4, latent_dim)

        # === STEP 3: FACE PROCESSING ===
        face_outputs = compute_all_faces(vertices, self.face_modules)
        # List of 4 tensors, each (batch, latent_dim)

        # Aggregate face updates to vertices WITH ReLU
        vertices = aggregate_to_vertices(face_outputs, vertices)
        vertices = self.activation(vertices)  # ReLU after face aggregation
        # Shape: (batch, 4, latent_dim)

        # === STEP 4: OUTPUT ===
        if return_faces:
            # Return vertices and faces for inter-face coupling
            face_tensor = torch.stack(face_outputs, dim=1)  # (batch, 4, latent_dim)
            return vertices, face_tensor
        else:
            # Standalone mode: project to output
            if self.output is None:
                raise ValueError("output_dim must be specified for standalone mode")
            vertices_flat = vertices.reshape(batch_size, -1)
            output = self.output(vertices_flat)
            # Note: No ReLU on final output - let loss function handle it
            return output

    def get_face_states(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        Compute face states from current vertex states.
        Used during inter-face coupling.

        Args:
            vertices: Current vertex states, shape (batch, 4, latent_dim)

        Returns:
            Face states, shape (batch, 4, latent_dim)
        """
        face_outputs = compute_all_faces(vertices, self.face_modules)
        return torch.stack(face_outputs, dim=1)

    def update_from_faces(self, vertices: torch.Tensor, face_updates: list) -> torch.Tensor:
        """
        Update vertices based on coupled face information.
        Used during inter-face coupling.

        Args:
            vertices: Current vertex states, shape (batch, 4, latent_dim)
            face_updates: List of 4 face update tensors, each (batch, latent_dim)

        Returns:
            Updated vertices WITH ReLU, shape (batch, 4, latent_dim)
        """
        updated = aggregate_to_vertices(face_updates, vertices)
        return self.activation(updated)  # Apply ReLU to coupled updates


# ============================================================================
# STANDALONE USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example: Using NonlinearTetrahedron for classification (standalone mode).
    """
    # Create model
    model = NonlinearTetrahedron(input_dim=2, latent_dim=64, output_dim=1)

    # Example input: [feature1, feature2]
    x = torch.tensor([[3.0, 5.0], [7.0, 2.0], [-4.0, 6.0]])

    # Forward pass
    output = model(x)
    print(f"Input: {x}")
    print(f"Output: {output}")
    print(f"Output shape: {output.shape}")  # (batch, 1)

    # Test return_faces mode (for coupling)
    vertices, faces = model(x, return_faces=True)
    print(f"\nVertices shape: {vertices.shape}")  # (batch, 4, 64)
    print(f"Faces shape: {faces.shape}")  # (batch, 4, 64)

    # Check that ReLU creates sparsity
    print(f"\nVertex sparsity (% zeros): {(vertices == 0).float().mean().item() * 100:.1f}%")
    print(f"Face sparsity (% zeros): {(faces == 0).float().mean().item() * 100:.1f}%")


# ============================================================================
# SUMMARY
# ============================================================================

"""
Y_nonlinear_tetrahedron.py - The Right Hemisphere

DESIGN PHILOSOPHY:
  - ReLU activations after each aggregation step
  - Creates discrete decision boundaries
  - Handles discontinuities and statistical patterns
  - Represents emotional, intuitive, holistic processing

KEY DIFFERENCES FROM LINEAR:
  - self.activation(vertices) after embed
  - self.activation(vertices) after edge aggregation
  - self.activation(vertices) after face aggregation
  - Creates sparse representations

KEY METHODS:
  - forward(x, return_faces): Main forward pass with ReLU
  - get_face_states(vertices): Extract face representations
  - update_from_faces(vertices, face_updates): Apply coupled updates with ReLU

USAGE MODES:
  1. Standalone: Set output_dim, use forward(x) directly
  2. Coupled: Use forward(x, return_faces=True) for dual-network setup

This network learns BOUNDARIES and CATEGORIES in the problem space.
"""
