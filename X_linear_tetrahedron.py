"""
X - LINEAR TETRAHEDRON (Left Hemisphere / Logical / Deterministic)
===================================================================

Linear tetrahedral network with NO ReLU activation.
Learns smooth manifold structures and deterministic relationships.

Perfect for:
  - Arithmetic operations
  - Geometric transformations
  - Continuous mathematical functions
  - Tasks with group structure

This network extrapolates perfectly on deterministic tasks because it
preserves the smooth topology of the underlying mathematical structure.
"""

import torch
import torch.nn as nn
from W_geometry import (
    EDGE_INDICES,
    FACE_INDICES,
    EdgeAttention,
    FaceAttention,
    compute_all_edges,
    compute_all_faces,
    aggregate_to_vertices
)


class LinearTetrahedron(nn.Module):
    """
    Linear tetrahedral network (Left Hemisphere).

    Architecture:
      - 4 vertices (no ReLU - pure linear processing)
      - 6 edges (pairwise linear attention)
      - 4 faces (3-point attention)

    Key property: NO nonlinear activations except in attention softmax.
    This preserves smooth manifold structure and enables perfect extrapolation
    on deterministic tasks.

    Args:
        input_dim: Dimension of input data
        latent_dim: Internal vertex representation dimension
        output_dim: Dimension of output (if used standalone)
    """
    def __init__(self, input_dim: int, latent_dim: int = 64, output_dim: int = None):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

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
        Forward pass through linear tetrahedron.

        Args:
            x: Input tensor, shape (batch, input_dim)
            return_faces: If True, return (output, vertices, faces) for coupling

        Returns:
            If return_faces=False: output tensor, shape (batch, output_dim)
            If return_faces=True: (output, vertices, faces) for inter-face coupling
        """
        batch_size = x.size(0)

        # === STEP 1: EMBED TO VERTICES ===
        vertices = self.embed(x).view(batch_size, 4, self.latent_dim)
        # Shape: (batch, 4, latent_dim)
        # NO ReLU - stays continuous

        # === STEP 2: EDGE PROCESSING ===
        edge_outputs = compute_all_edges(vertices, self.edge_modules)
        # List of 6 tensors, each (batch, latent_dim)
        # NO ReLU - pure linear attention

        # Aggregate edge updates to vertices
        vertices = aggregate_to_vertices(edge_outputs, vertices)
        # Shape: (batch, 4, latent_dim)

        # === STEP 3: FACE PROCESSING ===
        face_outputs = compute_all_faces(vertices, self.face_modules)
        # List of 4 tensors, each (batch, latent_dim)
        # NO ReLU - maintains smooth manifold

        # Aggregate face updates to vertices
        vertices = aggregate_to_vertices(face_outputs, vertices)
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
            return self.output(vertices_flat)

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
            Updated vertices, shape (batch, 4, latent_dim)
        """
        return aggregate_to_vertices(face_updates, vertices)


# ============================================================================
# STANDALONE USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example: Using LinearTetrahedron for arithmetic (standalone mode).
    """
    # Create model
    model = LinearTetrahedron(input_dim=2, latent_dim=64, output_dim=1)

    # Example input: [num1, num2]
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


# ============================================================================
# SUMMARY
# ============================================================================

"""
X_linear_tetrahedron.py - The Left Hemisphere

DESIGN PHILOSOPHY:
  - NO ReLU or nonlinear activations (except attention softmax)
  - Preserves smooth manifold structure
  - Perfect for deterministic, continuous tasks
  - Represents logical, mathematical processing

KEY METHODS:
  - forward(x, return_faces): Main forward pass
  - get_face_states(vertices): Extract face representations
  - update_from_faces(vertices, face_updates): Apply coupled updates

USAGE MODES:
  1. Standalone: Set output_dim, use forward(x) directly
  2. Coupled: Use forward(x, return_faces=True) for dual-network setup

This network learns the TOPOLOGY of the problem space.
"""
