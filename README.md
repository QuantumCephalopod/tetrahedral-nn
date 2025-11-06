# 🔺🔺 Bi-Tetrahedral Neural Network Engine

**Author:** Philipp Remy Bartholomäus
**Date:** October 30, 2025

---

## ⚠️ GOOGLE COLAB RESEARCH REPOSITORY ⚠️

**This repository is optimized for Google Colab workflows.**

Files are self-contained and designed for copy-paste into Colab notebooks, not local development.
See `COLAB_QUICKSTART.md` for quick-start examples.

For the original single-tetrahedron architecture, see: [main repo](https://github.com/QuantumCephalopod/tetrahedral-nn)

---

## Overview

A novel **dual-tetrahedral neural network architecture** that combines two tetrahedra with complementary processing modes:

- **Linear Tetrahedron** (Left/Logical): Deterministic, smooth manifolds
- **Nonlinear Tetrahedron** (Right/Intuitive): Statistical, discontinuous boundaries

**Key Achievement:** Arithmetic generalization from [-9, 9] to trillions (float32 precision limited) using only 361 training samples.

## Architecture

### Single Tetrahedron Structure
Each tetrahedron contains:
- **4 vertices** (core computation nodes)
- **6 edges** (pairwise linear attention interactions)
- **4 faces** (triangular 3-point attention mechanisms)

### Bi-Tetrahedral Configuration
Two tetrahedra work in parallel with inter-face coupling:
- **Linear Tetrahedron (X):** No ReLU, learns smooth continuous structures
- **Nonlinear Tetrahedron (Y):** With ReLU, handles discontinuities and boundaries
- **Inter-Face Coupling (Z):** Face-to-face communication between networks

The tetrahedral topology provides geometric structure for self-organization without task-specific assumptions.

## Key Results

### Arithmetic Generalization
- **Train range:** [-9, 9] (exhaustive dataset)
- **Test range:** Up to 10,000+ (and scales to trillions)
- **Extrapolation:** 1000x — beyond training range
- **Error:** Float32 precision limited (~1e-7 relative error)
- **Mechanism:** Architecture seems to learn Topology

### Inverse Operations (Decomposition)
- **Composition:** (a, b, c, d) → sum (compression: 4 → 1)
- **Decomposition:** num → (num/4, num/4, num/4, num/4) (expansion: 1 → 4)
- **Finding:** Architecture works bidirectionally
- **Implications:** Symmetric processing capabilities

## Repository Structure (Dimensional Organization)

### 📐 Core Dimensions (W, X, Y, Z)
The architecture is organized into 4 fundamental dimensions:

- **`W_geometry.py`** - **Foundation Dimension**
  Pure tetrahedral topology: vertices, edges, faces, attention primitives

- **`X_linear_tetrahedron.py`** - **Linear Dimension (Left Hemisphere)**
  No ReLU activation. Learns smooth manifolds, perfect for arithmetic and deterministic tasks

- **`Y_nonlinear_tetrahedron.py`** - **Nonlinear Dimension (Right Hemisphere)**
  With ReLU activation. Handles boundaries, perception, statistical patterns

- **`Z_interface_coupling.py`** - **Integration Dimension**
  Dual-tetrahedral network with face-to-face coupling between X and Y

### 🔗 Adapter Planes (Inter-Dimensional Interfaces)
Adapters bridge dimensions to handle specific domains:

- **`ZW_arithmetic_adapter.py`** - Arithmetic interface (Z ∩ W plane)
- **`ZX_rotation_adapter.py`** - Rotation learning interface (Z ∩ X plane)
- **`ZY_temporal_adapter.py`** - Temporal prediction interface (Z ∩ Y plane)

### 🎯 Applications & Systems
Complete implementations and validation:

- **`CONTINUOUS_LEARNING_SYSTEM.py`** - Continuous video learning (no epochs, just observation)
- **`GENERALIZATION_TEST.py`** - Arithmetic generalization validation
- **`BASELINE_TEST.py`** - Single vs dual tetrahedron comparison
- **`INFERENCE_SHOWCASE.py`** - Deployment and inference examples
- **`COLAB_INFERENCE_CELL.py`** - Single-cell Colab runner

### 📚 Legacy Files (from original fork)
Original single-tetrahedron implementation:

- **`tetrahedral_core.py`** - Original monolithic architecture
- **`arithmetic_adapter.py`** - Simple arithmetic adapter
- **`tetrahedral_trainer.py`** - Universal training system (still used!)
- **`tetrahedral_tests.py`** - Original test suite
- **`train_example.py`** - Basic training example

## Usage

### Quick Start (Google Colab)

See **`COLAB_QUICKSTART.md`** for ready-to-use Colab examples!

**TL;DR:** Copy any file into a Colab cell and run. Files are self-contained.

### Dual-Tetrahedral Network Example

```python
import torch
import torch.optim as optim
from Z_interface_coupling import DualTetrahedralNetwork

# Create dual network
model = DualTetrahedralNetwork(
    input_dim=2,
    output_dim=1,
    latent_dim=64,
    coupling_strength=0.5,
    output_mode="weighted"
)

# Use with any adapter (ZW, ZX, ZY)
from ZW_arithmetic_adapter import ArithmeticAdapter
adapter = ArithmeticAdapter(n_inputs=2)

# Training works the same as single tetrahedron
# See GENERALIZATION_TEST.py for complete examples
```

### Legacy Single-Tetrahedron Example

```python
from tetrahedral_core import TetrahedralCore
from arithmetic_adapter import ArithmeticAdapter
from tetrahedral_trainer import TetrahedralTrainer

# Original architecture still works!
model = TetrahedralCore(input_dim=2, output_dim=1, latent_dim=64)
# ... (see train_example.py for full code)
```

## Technical Details

### Attention Mechanisms

**Linear Attention (Edges):**
- Pairwise interactions between vertices
- O(N) complexity via linear kernels
- 6 edges connect all vertex pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)

**3-Point Attention (Faces):**
- Triangular attention on 3 vertices per face
- 4 faces per tetrahedron: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
- Enables discovery of higher-order relationships

### Parameter Count
- ~2.5K parameters for 2-input model
- Scales efficiently with input/output dimensions

## Theoretical Foundation

This architecture builds on recent advances in topological deep learning and higher-order neural networks:

### Core References

**Higher-Order Attention:**
- Hajij, M., Zamzmi, G., et al. (2022). "Architectures of Topological Deep Learning: A Survey on Topological Neural Networks." *arXiv:2304.10031*
- Hajij, M., Zamzmi, G., et al. (2022). "Higher-Order Attention Networks." *arXiv:2206.00606*

**Simplicial & Geometric Attention:**
- Giusti, L., et al. (2023). "Simplicial Attention Networks." *OpenReview*
- "2-Simplicial Attention" (2025). *arXiv:2507.02754*
- Farazi, M., et al. (2025). "A Recipe for Geometry-Aware 3D Mesh Transformers." *WACV 2025*

**Linear Attention Mechanisms:**
- Choromanski, K., et al. (2021). "Rethinking Attention with Performers." *ICLR 2021*
- Gu, A., & Dao, T. (2024). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." *arXiv:2312.00752*

**Tetrahedral Structures in Neural Networks:**
- Hu, T., et al. (2022). "TetGAN: A Convolutional Neural Network for Tetrahedral Meshes." *BMVC 2022*
- Gao, L., et al. (2023). "TetCNN: Convolutional Neural Networks on Tetrahedral Meshes." *ACM TOG 2023*

### Conceptual Inspiration

**Morphogenetic Intelligence & Self-Organization:**
- Levin, M. (2019). "The Computational Boundary of a 'Self': Developmental Bioelectricity Drives Multicellularity and Scale-Free Cognition." *Frontiers in Psychology*
- Levin, M. (2022). "Technological Approach to Mind Everywhere: An Experimentally-Grounded Framework for Understanding Diverse Bodies and Minds." *Frontiers in Systems Neuroscience*

Michael Levin's work on morphogenetic fields, bioelectric networks, and self-organizing computational systems fundamentally inspired the architectural philosophy: vertices self-organize their functional specialization through geometric constraints rather than explicit programming.

## Novel Contributions

This work specifically contributes:
1. **Hybrid attention architecture** combining linear (edges) + 3-point (faces) on tetrahedral topology
2. **Extreme generalization proof** (1000× arithmetic extrapolation, float-precision limited)
3. **Bidirectional operation** (compression 4→1 and expansion 1→4)
4. **Domain-agnostic geometric scaffold** for self-organizing computation

The tetrahedral structure itself is established in mesh processing; the novel contribution is its use as a **computational primitive** with hybrid attention mechanisms for general-purpose learning.

## Roadmap

### ✅ Completed
- [x] Arithmetic generalization proof (1000x extrapolation)
- [x] Dual-tetrahedral architecture (linear + nonlinear coupling)
- [x] Rotation transformation learning
- [x] Temporal prediction learning
- [x] Continuous learning system (video streams)
- [x] Inter-face coupling mechanism

### 🚧 In Progress
- [ ] Fractal subdivision (W→WW/WX/WY/WZ, X→XX/XY/XZ/XW, etc.)
- [ ] Multi-modal learning (vision + language)
- [ ] Real-world deployment testing

### 📚 Documentation
See **`EXPLORATIONS.md`** for philosophical foundations, conceptual insights, and the vision behind the architecture.

---

## Related Work

**Original single-tetrahedron architecture:** [github.com/QuantumCephalopod/tetrahedral-nn](https://github.com/QuantumCephalopod/tetrahedral-nn)

**Philosophy & Theory:** See `EXPLORATIONS.md` in this repo
