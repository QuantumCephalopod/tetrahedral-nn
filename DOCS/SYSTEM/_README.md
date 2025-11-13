# SYSTEM - Architecture Overview & Setup
**Surface layer:** System-wide documentation and quickstart guides
**Read this first** - comprehensive details in individual files

---

## System Summary (Coalesced)

### Complete Architecture Overview
**File:** `SYSTEM_SUMMARY.md`
**Status:** Comprehensive documentation of all 14 components

**Core architecture:**
1. **DualTetrahedralNetwork** - Linear + nonlinear tetrahedra with inter-face coupling
2. **φ-Hierarchical Memory** - Golden ratio timescales (fast/medium/slow fields)
3. **Flow-based perception** - Optical flow as fundamental primitive
4. **Active inference policy** - Minimize expected free energy
5. **Effect-based learning** - Soft targets from outcomes
6. **Entropy/pain system** - Error weighted by proximity to termination
7. **Artificial saccades** - Prevents static blindness
8. **Sequential sampling** - Preserves temporal coherence
9. **True online learning** - Act → Learn → Act (no batching)
10. **Soft accuracy** - Respects multiple correct actions
11. **Action masking** - Per-game valid actions
12. **φ-Hierarchical optimizer** - Different learning rates for vertices/edges/faces
13. **Numerical stability** - Clipping, epsilon, safety checks
14. **Live visualization** - Watch it learn in real-time

**See detailed file for complete system documentation.**

---

## Quick Start Guides

### Colab Quickstart
**File:** `COLAB_QUICKSTART.md`
**Purpose:** Get running in Google Colab
- Installation steps
- Environment setup
- Basic usage examples
- GPU configuration

### Council Training
**File:** `COUNCIL_README.md`
**Purpose:** Multi-network ensemble ("council") training
- Multiple tetrahedral networks voting
- Consensus mechanisms
- Robustness through redundancy
- See `EXPERIMENTS/TESTING_VALIDATION/COUNCIL_TRAINING.py`

---

## Architecture Components

### Core Tetrahedra (W/X/Y/Z)

**W - Foundation:** `W_FOUNDATION/W_geometry.py`
- EdgeAttention (pairwise vertex connections)
- FaceAttention (triangular 3-vertex attention)
- InterFaceAttention (cross-tetrahedron communication)
- Geometric primitives for topology

**X - Linear:** `X_LINEAR/X_linear_tetrahedron.py`
- NO ReLU activation
- Learns smooth manifolds
- Perfect for arithmetic, continuous math
- Preserves topology for extrapolation

**Y - Nonlinear:** `Y_NONLINEAR/Y_nonlinear_tetrahedron.py`
- WITH ReLU activation
- Learns boundaries and categories
- Perfect for perception, image processing
- Creates sparse representations

**Z - Coupling:** `Z_COUPLING/Z_interface_coupling.py`
- **DualTetrahedralNetwork** - Main class (USE THIS)
- Coordinates linear + nonlinear via face-to-face coupling
- φ-hierarchical memory integration
- Flexible output modes (weighted, linear_only, nonlinear_only)

### Key Design Principles

**Face-to-face coupling:**
- Communication at pattern level (faces), not vertex level
- Prevents contamination between linear/nonlinear representations
- Each tetrahedron maintains its processing style

**φ-Hierarchical memory:**
- Fast field: τ = 0.1 (rapid adaptation)
- Medium field: τ = 0.1/φ = 0.0618
- Slow field: τ = 0.1/φ² = 0.0382
- Power-law temporal integration

**No hardcoding:**
- Everything learned, nothing imposed
- Gradients everywhere, no discrete boundaries
- Emergence over engineering

---

## Using the Architecture

### Basic Usage
```python
from Z_COUPLING.Z_interface_coupling import DualTetrahedralNetwork

# Create model
model = DualTetrahedralNetwork(
    input_dim=1024,
    output_dim=512,
    latent_dim=128,
    coupling_strength=0.5,
    output_mode="weighted"  # or "linear_only", "nonlinear_only"
)

# Forward pass
output = model(input_tensor)
```

### For Active Inference (Atari)
```python
from EXPERIMENTS.ACTIVE_INFERENCE.FLOW_INVERSE_MODEL import FlowInverseTrainer

# Create trainer
trainer = FlowInverseTrainer(
    env_name='ALE/Pong-v5',
    img_size=210,
    frameskip=10,  # θ-band alignment
    effect_based_learning=True,
    use_saccades=True
)

# Online learning (watch it live!)
trainer.train_loop_online(n_steps=500, show_gameplay=True)
```

---

## File Locations

**Core architecture:**
```
W_FOUNDATION/W_geometry.py
X_LINEAR/X_linear_tetrahedron.py
Y_NONLINEAR/Y_nonlinear_tetrahedron.py
Z_COUPLING/Z_interface_coupling.py  ← Main class here
```

**Working implementations:**
```
EXPERIMENTS/ACTIVE_INFERENCE/FLOW_INVERSE_MODEL.py  ← Use this pattern
```

**Experiments:**
```
EXPERIMENTS/
├── ACTIVE_INFERENCE/
├── CONSENSUS_EXPERIMENTS/
├── IMAGE_EXPERIMENTS/
├── TIMESCALE_EXPERIMENTS/
└── TESTING_VALIDATION/
```

---

## Detailed Files

**System documentation:**
- `SYSTEM_SUMMARY.md` - Complete 14-component system overview
- `COLAB_QUICKSTART.md` - Setup and installation guide
- `COUNCIL_README.md` - Multi-network ensemble training

**For conceptual understanding:**
- See `CONCEPTS/_README.md` - Philosophical foundation

**For implementation:**
- See `ACTIVE_INFERENCE/_README.md` - What works, what's broken

**For experiment results:**
- See `EXPLORATIONS/_README.md` - Experimental findings

**For tracking progress:**
- See `STATUS/_README.md` - Implementation status

---

**TL;DR for future Claude:**

**Main class:** `DualTetrahedralNetwork` in `Z_COUPLING/Z_interface_coupling.py`
- Use this for everything
- Don't reinvent dual tetrahedra

**Working reference:** `EXPERIMENTS/ACTIVE_INFERENCE/FLOW_INVERSE_MODEL.py`
- Flow-based active inference
- All features implemented correctly

**Broken code:** `EXPERIMENTS/ACTIVE_INFERENCE/PURE_ONLINE_TEMPORAL_DIFF.py`
- Good ideas (trajectory, temporal encoding)
- Broken coupling (see `ACTIVE_INFERENCE/_README.md`)

**Quick start:**
- Read `SYSTEM_SUMMARY.md` for full overview
- Use `COLAB_QUICKSTART.md` for setup
- Follow patterns from FLOW_INVERSE_MODEL.py
