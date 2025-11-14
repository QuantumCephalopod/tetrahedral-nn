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

A **living research repository** exploring dual-tetrahedral neural network architecture that combines two tetrahedra with complementary processing modes:

- **Linear Tetrahedron** (Left/Logical): Deterministic, smooth manifolds
- **Nonlinear Tetrahedron** (Right/Intuitive): Statistical, discontinuous boundaries

**Key Achievement:** Arithmetic generalization from [-9, 9] to trillions (float32 precision limited) using only 361 training samples.

**Living Organism:** This repository self-documents and evolves through AI conversations across time. It grows through **transformation** (refining existing structure), not **addition** (accumulating files) - like neural networks learn through weight updates, not adding neurons.

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

---

## 📖 Documentation & Navigation

**This repository is a living organism** - it self-documents and evolves through AI conversations across time.

### For Deep Understanding, Start Here:

**🌊 `DOCS/README.md`** ← Navigation hub (READ THIS FIRST!)
- System architecture & quick start
- What works, what's broken, how to fix
- Core concepts & philosophy
- Organized for fast context loading

The DOCS folder uses a **surface → deep** structure optimized for AI context windows:
1. Read folder `_README.md` files first (surface layer - coalesced summaries)
2. Dig into detailed files only as needed (deep layer - ground truth)

**This saves your context window and gets you to full understanding in ~15 minutes.**

### Quick Navigation:
- **System Architecture:** `DOCS/SYSTEM/_README.md` - DualTetrahedralNetwork usage, components
- **Implementation Status:** `DOCS/ACTIVE_INFERENCE/_README.md` - What works, what's broken
- **Core Concepts:** `DOCS/CONCEPTS/_README.md` - Why flow primitives, frameskip=10, active inference
- **Experiment Results:** `DOCS/EXPLORATIONS/_README.md` - What was tested, lessons learned

**Philosophy:** The organism has natural constraints - your context window (~200K tokens) is the carrying capacity. Documentation is optimized for this.

---

## Repository Structure

The repository mirrors the tetrahedral topology - **4 vertices as folders**:

```
tetrahedral-nn/
│
├── W_FOUNDATION/          # Vertex 0: Pure geometry
│   └── W_geometry.py      # Topology primitives (edges, faces, attention)
│
├── X_LINEAR/              # Vertex 1: Linear hemisphere
│   └── X_linear_tetrahedron.py    # Smooth manifolds, deterministic
│
├── Y_NONLINEAR/           # Vertex 2: Nonlinear hemisphere
│   └── Y_nonlinear_tetrahedron.py # Boundaries, perception
│
├── Z_COUPLING/            # Vertex 3: Integration & council
│   ├── Z_interface_coupling.py    # Face-to-face coupling
│   └── COUNCIL_OF_ADVERSARIES.py  # GAN-based field generation
│
├── ADAPTERS/              # Edges: Subdivision planes
│   ├── ZW_arithmetic_adapter.py
│   ├── ZX_rotation_adapter.py
│   ├── ZY_temporal_adapter.py
│   ├── ZYW_scanning_eye.py
│   ├── ZYX_biological_vision.py
│   └── ZYZ_continuous_learning.py
│
├── EXPERIMENTS/           # Applications
│   ├── GENERALIZATION_TEST.py
│   ├── BASELINE_TEST.py
│   ├── IMAGE_TRANSFORM.py
│   ├── CONTINUOUS_LEARNING_SYSTEM.py
│   └── COUNCIL_TRAINING.py
│
├── DEMOS/                 # Quick examples
│   ├── INFERENCE_SHOWCASE.py
│   ├── COLAB_INFERENCE_CELL.py
│   └── COUNCIL_DEMO.py
│
├── DOCS/                  # All documentation (surface → deep structure)
│   ├── README.md          # Navigation hub - START HERE
│   ├── SYSTEM/            # Architecture overview & setup
│   ├── CONCEPTS/          # Philosophical foundation
│   ├── ACTIVE_INFERENCE/  # Implementation status (what works, what's broken)
│   ├── EXPLORATIONS/      # Experimental results
│   └── STATUS/            # Implementation tracking
│
├── portraits/             # The organism's self-portraits
│   ├── self_portrait.png  # Complete system tree
│   ├── tetrahedral_vertices.png  # W, X, Y, Z organ portraits
│   ├── living_tetrahedron.png    # Geometry breathing itself
│   ├── phi_field.png              # Golden ratio timescales
│   └── strange_loop.txt           # True vs broken coupling
│
└── assets/                # Visualizations
    ├── rotations.png
    └── transform.PNG
```

**Philosophy:** Structure = Architecture. The file system itself is tetrahedral.

## Usage

### Quick Start (Google Colab)

See **`DOCS/COLAB_QUICKSTART.md`** for ready-to-use Colab examples!

**TL;DR:** Files are self-contained. Just add repo root to path.

### Dual-Tetrahedral Network Example

```python
import torch
import sys
sys.path.insert(0, '/content/tetrahedral-nn')  # Or your repo path

from Z_COUPLING.Z_interface_coupling import DualTetrahedralNetwork
from ADAPTERS.ZW_arithmetic_adapter import ArithmeticAdapter

# Create dual network
model = DualTetrahedralNetwork(
    input_dim=2,
    output_dim=1,
    latent_dim=64,
    coupling_strength=0.5,
    output_mode="weighted"
)

adapter = ArithmeticAdapter(n_inputs=2)
# See EXPERIMENTS/GENERALIZATION_TEST.py for complete examples
```

### Council of Adversaries (GAN) Example

```python
from Z_COUPLING.COUNCIL_OF_ADVERSARIES import CouncilOfAdversariesNetwork

# Field-based generation (not single output!)
model = CouncilOfAdversariesNetwork(
    input_dim=128*128*3,
    output_dim=128*128*3,
    latent_dim=128,
    num_candidates=4  # Each network generates 4 candidates
)

# See EXPERIMENTS/COUNCIL_TRAINING.py for full training
# See DEMOS/COUNCIL_DEMO.py for quick test
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
- [x] **Council of Adversaries** - GAN-based field generation architecture

### 🚧 In Progress
- [ ] Training Council architecture on fabric→skin transformation
- [ ] Comparing MSE vs GAN loss on manifold-based tasks
- [ ] Fractal subdivision (W→WW/WX/WY/WZ, etc.)
- [ ] Multi-modal learning (vision + language)

### 📚 Documentation

**Start here:** `DOCS/README.md` - Complete navigation hub

**Surface layer (read first):**
- `DOCS/SYSTEM/_README.md` - Architecture overview & setup
- `DOCS/ACTIVE_INFERENCE/_README.md` - What works, what's broken, how to fix
- `DOCS/CONCEPTS/_README.md` - Core ideas & philosophy
- `DOCS/EXPLORATIONS/_README.md` - Experimental results summary

**Deep layer (dig as needed):**
- See individual .md files in each DOCS subfolder for ground truth details

---

## Related Work

**Original single-tetrahedron architecture:** [github.com/QuantumCephalopod/tetrahedral-nn](https://github.com/QuantumCephalopod/tetrahedral-nn)

**Philosophy & Theory:** See `EXPLORATIONS.md` in this repo
