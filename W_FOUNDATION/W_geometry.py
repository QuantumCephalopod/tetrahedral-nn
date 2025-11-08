"""
W - GEOMETRY (Foundation Vertex)
================================

Core tetrahedral topology primitives and attention mechanisms.
This is the geometric scaffold that both Linear and Nonlinear tetrahedra build upon.

Structure:
  - 4 vertices per tetrahedron
  - 6 edges (complete graph K4)
  - 4 triangular faces

This module contains ONLY geometry - no network-specific processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# TETRAHEDRAL TOPOLOGY CONSTANTS
# ============================================================================

# Edge pairs: all 6 pairwise connections in a tetrahedron
EDGE_INDICES = [
    (0, 1), (0, 2), (0, 3),  # Edges from vertex 0
    (1, 2), (1, 3),          # Edges from vertex 1
    (2, 3)                   # Edge from vertex 2
]

# Face triplets: all 4 triangular faces
FACE_INDICES = [
    (0, 1, 2),  # Face 0: bottom triangle (if vertex 3 is apex)
    (0, 1, 3),  # Face 1: front triangle
    (0, 2, 3),  # Face 2: right triangle
    (1, 2, 3)   # Face 3: left triangle
]


# ============================================================================
# EDGE ATTENTION (Pairwise Linear Attention)
# ============================================================================

class EdgeAttention(nn.Module):
    """
    Linear attention between two vertices (an edge).

    This implements pairwise interactions without making assumptions about
    activation functions - those are applied by the specific tetrahedron type.

    Args:
        dim: Latent dimension of vertex representations
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Takes concatenated vertex pair, outputs attended representation
        self.attend = nn.Linear(dim * 2, dim)

    def forward(self, v_a: torch.Tensor, v_b: torch.Tensor) -> torch.Tensor:
        """
        Compute edge attention between two vertices.

        Args:
            v_a: Vertex A, shape (batch, dim)
            v_b: Vertex B, shape (batch, dim)

        Returns:
            Edge representation, shape (batch, dim)
        """
        edge_input = torch.cat([v_a, v_b], dim=-1)
        return self.attend(edge_input)


# ============================================================================
# FACE ATTENTION (3-Point Simplicial Attention)
# ============================================================================

class FaceAttention(nn.Module):
    """
    Triangular attention over 3 vertices (a face of the tetrahedron).

    This is the fundamental 3-point attention operation that computes
    attention-weighted combinations of 3 inputs, enabling discovery of
    higher-order relationships.

    Each of the 4 tetrahedral faces uses this operation.

    Args:
        dim: Latent dimension of vertex representations
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Query-Key-Value projection for all 3 vertices at once
        self.to_qkv = nn.Linear(dim * 3, dim * 3)
        self.out = nn.Linear(dim, dim)

    def forward(self, v_a: torch.Tensor, v_b: torch.Tensor, v_c: torch.Tensor) -> torch.Tensor:
        """
        Compute 3-point attention over a triangular face.

        Args:
            v_a, v_b, v_c: Vertices forming the triangle, each shape (batch, dim)

        Returns:
            Face representation, shape (batch, dim)
        """
        # Concatenate all 3 vertices
        x = torch.cat([v_a, v_b, v_c], dim=-1)

        # Project to Q, K, V
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Attention scores (scaled dot-product)
        scores = torch.sum(q * k, dim=-1, keepdim=True) / (self.dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)

        # Combine with mean pooling for stability
        combined = (v_a + v_b + v_c) / 3
        attended = attn * v

        return self.out(attended + combined)


# ============================================================================
# INTER-FACE ATTENTION (Face-to-Face Communication)
# ============================================================================

class InterFaceAttention(nn.Module):
    """
    Attention between corresponding faces of two tetrahedra.

    This is the key innovation for dual-tetrahedral coupling:
    faces from different networks can communicate without sharing vertices.

    Args:
        dim: Latent dimension of face representations
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Cross-attention: one face queries another
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, face_source: torch.Tensor, face_target: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-attention from source face to target face.

        Args:
            face_source: Face from source network, shape (batch, dim)
            face_target: Face from target network, shape (batch, dim)

        Returns:
            Updated face representation incorporating target information, shape (batch, dim)
        """
        q = self.to_q(face_source)
        k = self.to_k(face_target)
        v = self.to_v(face_target)

        # Attention score
        score = torch.sum(q * k, dim=-1, keepdim=True) / (self.dim ** 0.5)
        attn = torch.sigmoid(score)  # Use sigmoid for cross-network attention

        # Weighted combination
        attended = attn * v

        return self.out(attended + face_source)  # Residual connection


# ============================================================================
# GEOMETRIC UTILITIES
# ============================================================================

def compute_all_edges(vertices: torch.Tensor, edge_modules: nn.ModuleList) -> list:
    """
    Compute all 6 edge representations for a tetrahedron.

    Args:
        vertices: Vertex states, shape (batch, 4, dim)
        edge_modules: ModuleList of 6 EdgeAttention modules

    Returns:
        List of 6 edge representations, each shape (batch, dim)
    """
    edge_outputs = []
    for i, (a, b) in enumerate(EDGE_INDICES):
        edge_output = edge_modules[i](vertices[:, a], vertices[:, b])
        edge_outputs.append(edge_output)
    return edge_outputs


def compute_all_faces(vertices: torch.Tensor, face_modules: nn.ModuleList) -> list:
    """
    Compute all 4 face representations for a tetrahedron.

    Args:
        vertices: Vertex states, shape (batch, 4, dim)
        face_modules: ModuleList of 4 FaceAttention modules

    Returns:
        List of 4 face representations, each shape (batch, dim)
    """
    face_outputs = []
    for i, (a, b, c) in enumerate(FACE_INDICES):
        face_output = face_modules[i](vertices[:, a], vertices[:, b], vertices[:, c])
        face_outputs.append(face_output)
    return face_outputs


def aggregate_to_vertices(updates: list, vertices: torch.Tensor) -> torch.Tensor:
    """
    Aggregate edge or face updates back to vertex representations.

    Args:
        updates: List of update tensors, each shape (batch, dim)
        vertices: Current vertex states, shape (batch, 4, dim)

    Returns:
        Updated vertices, shape (batch, 4, dim)
    """
    # Stack updates and take mean
    update_tensor = torch.stack(updates, dim=1)  # (batch, n_updates, dim)
    update_mean = update_tensor.mean(dim=1, keepdim=True)  # (batch, 1, dim)

    # Broadcast and add to all vertices
    return vertices + update_mean.expand(-1, 4, -1)


# ============================================================================
# SUMMARY
# ============================================================================

"""
W_geometry.py provides the foundational geometric primitives:

CONSTANTS:
  - EDGE_INDICES: The 6 edges of a tetrahedron
  - FACE_INDICES: The 4 triangular faces of a tetrahedron

MODULES:
  - EdgeAttention: Pairwise linear attention (for edges)
  - FaceAttention: 3-point simplicial attention (for faces)
  - InterFaceAttention: Cross-tetrahedron face communication (NEW!)

UTILITIES:
  - compute_all_edges(): Process all 6 edges at once
  - compute_all_faces(): Process all 4 faces at once
  - aggregate_to_vertices(): Aggregate updates back to vertices

These primitives are activation-agnostic. Linear and Nonlinear tetrahedra
use them with different activation functions.
"""
