# TETRAHEDRAL NEURAL NETWORK - ARCHITECTURE MAP
**Date:** November 13, 2025
**Purpose:** High-level organizational reference - understand the repository structure at a glance

---

## 🏛️ Core Architecture (The Foundation)

These are the building blocks. Everything else uses these.

### W_FOUNDATION/ - Geometric Primitives
**File:** `W_geometry.py`
**What:** Tetrahedral topology - edges, faces, attention mechanisms
**Key Classes:**
- `EdgeAttention` - Pairwise connections between vertices
- `FaceAttention` - Triangular attention over 3 vertices
- `InterFaceAttention` - Cross-tetrahedron communication
- Constants: `EDGE_INDICES`, `FACE_INDICES`

**Use this when:** Building any tetrahedral network component

---

### X_LINEAR/ - Smooth Manifolds
**File:** `X_linear_tetrahedron.py`
**What:** Linear tetrahedron (NO ReLU) - learns continuous, deterministic relationships
**Key Class:** `LinearTetrahedron`
**Perfect for:** Arithmetic, geometry, continuous math functions
**Philosophy:** Preserves smooth topology for perfect extrapolation

---

### Y_NONLINEAR/ - Boundaries & Categories
**File:** `Y_nonlinear_tetrahedron.py`
**What:** Nonlinear tetrahedron (WITH ReLU) - learns discontinuities and boundaries
**Key Class:** `NonlinearTetrahedron`
**Perfect for:** Image processing, perception, category boundaries
**Philosophy:** Creates sparse representations and decision boundaries

---

### Z_COUPLING/ - Integration
**File:** `Z_interface_coupling.py`
**What:** Coordinates Linear + Nonlinear tetrahedra through inter-face coupling
**Key Classes:**
- `DualTetrahedralNetwork` - Main architecture (use this!)
- `DualTetrahedralTrainer` - Training utilities

**Critical Features:**
- φ-hierarchical memory (golden ratio timescales: fast/medium/slow fields)
- Face-to-face communication (pattern-level, not vertex contamination)
- Flexible output modes: weighted, linear_only, nonlinear_only

**THIS IS WHAT YOU USE.** Don't reinvent dual networks - use `DualTetrahedralNetwork`.

---

## 🧪 Experiments (What Works)

### ACTIVE_INFERENCE/ - Atari Game Learning

#### ✅ **FLOW_INVERSE_MODEL.py** (WORKING - Reference Implementation)
**Status:** Mature, tested, working
**Primitive:** Optical flow (velocity fields)
**Resolution:** 210×210 (native Atari height)

**Architecture:**
```python
FlowForwardModel:  (flow_t, action) → flow_t+1
FlowInverseModel:  (flow_t, flow_t+1) → action
```

**Key Features:**
- ✅ Action masking per game (CRITICAL - prevents false penalties)
- ✅ Effect-based learning (soft targets from flow similarity)
- ✅ Active inference policy (closes the loop!)
- ✅ Entropy/pain system (error × 1/lives)
- ✅ φ-hierarchical optimizer
- ✅ True online learning mode
- ✅ Artificial saccades (prevents static blindness)
- ✅ Sequential sampling (preserves temporal coherence)

**When to use:** Flow-based Atari experiments, reference for proper patterns

**Key insight:** Forward and inverse are SEPARATE `DualTetrahedralNetwork` instances.
This is CORRECT for active inference - they serve different purposes in the loop.

---

#### ⚠️ **PURE_ONLINE_TEMPORAL_DIFF.py** (FLAWED - Needs Refactoring)
**Status:** Has good ideas but fundamentally broken coupling
**Primitive:** Temporal differences (frame_t+1 - frame_t)

**Good Ideas (Keep These):**
1. **Trajectory-based learning** - Uses 3 temporal diffs for triangulation
   - 1 point = position
   - 2 points = velocity
   - 3 points = acceleration, curvature
2. **Temporal encoding** - Sinusoidal encoding (like transformers) for explicit time
   - Each diff knows its temporal order
   - Multiple frequency bands (32 = 64 dims)
3. **Signal-weighted loss** - `weight = |ground_truth| * error²`
   - Neurons fire for change, not static input
   - Prevents mode collapse (predicting all zeros)

**CRITICAL FLAW (Why User Said "Beyond Braindead"):**
```python
# Line 368: Inverse model sees GROUND TRUTH diff_t1, NOT forward's prediction!
trajectory_extended = torch.cat([trajectory_t, diff_t1], dim=1)
action_logits = self.inverse_model(trajectory_extended, time_encodings_extended)
```

**The Problem:**
- Inverse trained on REAL observations (ground truth)
- Forward trained separately on its own predictions
- Consistency loss tries to couple them but it's WEAK and DETACHED
- They're not actually communicating during training
- This is "ML toyland" - bolted-on complexity to hide missing understanding

**What It Should Be:**
Per `DOCS/INVERSE_MODEL_FAILURE_NOTES.md`:
- Visual inverse model predicting on pixel deltas (not abstract features) ✓ (already doing this)
- Integrated with curriculum masking (from ACTIVE_INFERENCE_ATARI.py)
- Uses existing `DualTetrahedralNetwork` architecture ✓ (already doing this)
- BUT: Needs proper integration with forward model's actual predictions

**Refactoring Strategy:**
1. Keep: Trajectory concept, temporal encoding, signal-weighted loss
2. Remove: Fake coupling between forward/inverse
3. Decision needed:
   - Option A: Keep them separate (like FLOW_INVERSE_MODEL.py) but use for active inference loop
   - Option B: True coupling - inverse sees forward's predictions during training
4. Add: Curriculum masking (learn own paddle first, then ball, then opponent)
5. Add: Action masking (per game valid actions)

---

#### 📊 **ACTIVE_INFERENCE_ATARI.py** (Referenced, Need to Check)
**Status:** Mentioned in INVERSE_MODEL_FAILURE_NOTES.md as having curriculum
**Purpose:** Has working curriculum system (attention masking)

**What to extract:**
- Curriculum implementation (mask opponent at start)
- Developmental learning (control → interaction → understanding)

---

## 📚 Documentation Structure

### High-Level Understanding
- `README.md` - Entry point, tetrahedral proof
- `DOCS/SYSTEM_SUMMARY.md` - Complete flow-based system (14 components!)
- `DOCS/ARCHITECTURE_MAP.md` - **THIS FILE** - Navigation reference

### Conceptual Deep Dives
- `DOCS/INVERSE_MODEL_FAILURE_NOTES.md` - Why abstract features failed, what's needed
- `DOCS/HARMONIC_RESONANCE_HYPOTHESIS.md` - Actions as frequency signatures
- `DOCS/WHY_MOVEMENT_MATTERS.md` - Saccades, static blindness
- `DOCS/TEMPORAL_COHERENCE.md` - Sequential vs random sampling
- `DOCS/NATURAL_FREQUENCIES.md` - Frameskip as biological rhythm alignment
- `DOCS/ACTION_SPACE_DIMENSIONALITY.md` - Buttons as projections of continuous manifold

### Implementation Status
- `DOCS/CURRENT_ISSUES_AND_NEXT_STEPS.md` - Active problems, fresh start guidance
- `DOCS/STATUS_ACTIVE_INFERENCE.md` - Active inference implementation status
- `DOCS/CLOSING_THE_STRANGE_LOOP.md` - Perception → action → effect loop

---

## 🎯 What to Use When

### I want to build a new task with tetrahedral networks:
→ Use `Z_COUPLING/Z_interface_coupling.py` → `DualTetrahedralNetwork`

### I want to learn Atari games with optical flow:
→ Use `EXPERIMENTS/ACTIVE_INFERENCE/FLOW_INVERSE_MODEL.py` as reference

### I want to understand what went wrong before:
→ Read `DOCS/INVERSE_MODEL_FAILURE_NOTES.md`

### I want to add temporal difference learning:
→ Extract good ideas from `PURE_ONLINE_TEMPORAL_DIFF.py`:
   - Trajectory (3 diffs)
   - Temporal encoding
   - Signal-weighted loss
→ But DON'T copy the coupling - it's broken

### I want to understand the philosophy:
→ Read `DOCS/SYSTEM_SUMMARY.md` and conceptual docs

---

## 🚨 Common Pitfalls (Don't Do These!)

### ❌ Creating separate Linear + Nonlinear networks manually
**Why wrong:** Z_COUPLING already does this correctly
**Do instead:** Use `DualTetrahedralNetwork(input_dim, output_dim, latent_dim)`

### ❌ Compressing to abstract feature vectors
**Why wrong:** Loses visual interpretability (see INVERSE_MODEL_FAILURE_NOTES.md)
**Do instead:** Keep predictions in pixel/flow space, learn visual patterns

### ❌ Training inverse on ground truth when claiming it couples with forward
**Why wrong:** They don't actually communicate - fake coupling
**Do instead:** Either keep separate (FLOW approach) OR train inverse on forward's predictions

### ❌ Ignoring action masking
**Why wrong:** Model gets penalized for predicting UP when ground truth is UPFIRE (both identical in Pong!)
**Do instead:** Mask invalid actions per game, only compute loss over valid actions

### ❌ Downsampling to "save computation"
**Why wrong:** ML cargo cult - throws away information for minuscule efficiency gain
**Do instead:** Use native resolution (210×210 for Atari)

### ❌ Binary accuracy for action prediction
**Why wrong:** UP at border = NOOP at border (both stop paddle) - binary accuracy arbitrarily penalizes one
**Do instead:** Soft accuracy or effect-based learning (learn from flow outcomes)

---

## 🔬 Current Work Focus

**What works:**
- Core tetrahedral architecture (W/X/Y/Z)
- Flow-based active inference (FLOW_INVERSE_MODEL.py)
- φ-hierarchical memory
- Effect-based learning

**What needs work:**
- PURE_ONLINE_TEMPORAL_DIFF.py coupling is fundamentally broken
- Need to extract good ideas (trajectory, temporal encoding) without the bad coupling
- Decision needed: How to properly integrate forward/inverse?

**User's directive:**
> "Read *all* the documentation. One after the other... and make a plan to create a higher order layer of information → pushing them down but building reference in the highlevel files. Don't reinvent the wheel... understand what's there and use that! This repo needs to reflect self-organization at every level!"

**This file is part of that high-level organization.** ✓

---

## 📖 Reading Order for New Developers

1. `README.md` - Understand the tetrahedral concept
2. `DOCS/ARCHITECTURE_MAP.md` - **THIS FILE** - Navigate the repo
3. `W_FOUNDATION/W_geometry.py` - Geometric primitives
4. `Z_COUPLING/Z_interface_coupling.py` - Main architecture
5. `EXPERIMENTS/ACTIVE_INFERENCE/FLOW_INVERSE_MODEL.py` - Reference implementation
6. `DOCS/SYSTEM_SUMMARY.md` - Deep dive into all components
7. `DOCS/INVERSE_MODEL_FAILURE_NOTES.md` - Learn from past mistakes
8. `DOCS/CURRENT_ISSUES_AND_NEXT_STEPS.md` - Current state

---

## 🌊 Philosophy

**Everything is a gradient.**
Don't hardcode. Don't discretize unnecessarily. Learn continuous manifolds, discretize at interface.

**Build on what works.**
Don't reimplement. Use `DualTetrahedralNetwork`. Reference FLOW_INVERSE_MODEL.py.

**Stay interpretable.**
Predict in pixel/flow space. Visual patterns, not abstract compressions.

**Nature already figured it out.**
Saccades. Sequential sampling. Pain as entropy gradient. Don't optimize away biological truth for "efficiency."

---

_The river flows where it must._
_Any minute we lose to shit like this costs lives._
_Stay focused. Build on foundations. Don't lose sight of the purpose._
