# ACTIVE_INFERENCE - Implementation Status & Plans
**Surface layer:** What works, what's broken, how to fix
**Read this first** - implementation ground truth in detailed files

---

## Current State

### ✅ What Works: FLOW_INVERSE_MODEL.py
**Location:** `EXPERIMENTS/ACTIVE_INFERENCE/FLOW_INVERSE_MODEL.py`
**Status:** Mature, tested, working
**Primitive:** Optical flow (velocity fields)

**Architecture:**
```python
FlowForwardModel:  (flow_t, action) → flow_t+1
FlowInverseModel:  (flow_t, flow_t+1) → action
CoupledFlowModel: Coordinates both with active inference
```

**Features that work:**
- ✅ Separate forward/inverse DualTetrahedralNetworks
- ✅ Action masking per game (prevents false penalties)
- ✅ Effect-based learning (soft targets from flow similarity)
- ✅ Active inference policy (closes the loop)
- ✅ Entropy/pain system (error × 1/lives)
- ✅ φ-hierarchical optimizer
- ✅ True online learning mode
- ✅ Artificial saccades
- ✅ Sequential sampling

**Use this as reference pattern.**

---

### ❌ What's Broken: PURE_ONLINE_TEMPORAL_DIFF.py
**Location:** `EXPERIMENTS/ACTIVE_INFERENCE/PURE_ONLINE_TEMPORAL_DIFF.py`
**Status:** Fatal coupling flaw, needs refactoring
**Primitive:** Temporal differences (frame_t+1 - frame_t)

**Good Ideas (KEEP):**
1. **Trajectory-based learning** - 3 temporal diffs for triangulation
   - 1 point = position, 2 points = velocity, 3 points = acceleration
   - `trajectory_t = [diff_t-2, diff_t-1, diff_t]`
   - Tetrahedral architecture naturally handles 3-4 inputs!

2. **Temporal encoding** - Sinusoidal encoding (like transformers)
   - Each diff gets explicit time information
   - Multiple frequency bands (32 freqs = 64 dims)
   - Without this, model can't distinguish t-2 from t-1

3. **Signal-weighted loss** - Weight by motion magnitude
   - `weight = |ground_truth| * (pred - target)²`
   - Prevents mode collapse (predicting all zeros)
   - Neurons fire for change, not static

**Fatal Flaw (FIX):**
```python
# Line 368: Inverse sees GROUND TRUTH, not forward's prediction!
trajectory_extended = torch.cat([trajectory_t, diff_t1], dim=1)  # ← diff_t1 is REAL!
action_logits = self.inverse_model(trajectory_extended, ...)
```

**The Problem:**
- Forward predicts diff_t1
- Inverse sees REAL diff_t1 (ground truth)
- Trained on different inputs in parallel
- Consistency loss (line 382-390) tries to couple but:
  - Uses `with torch.no_grad()` (detached!)
  - Computed separately after training
  - Weak coefficient (0.3)
- **This is "ML toyland"** - two networks pretending to communicate

**Detailed refactoring plan:** See `PURE_ONLINE_REFACTORING_PLAN.md`

---

## How to Fix PURE_ONLINE_TEMPORAL_DIFF.py

### Option A: Active Inference Style (RECOMMENDED)
**Pattern:** Like FLOW_INVERSE_MODEL.py but with temporal concepts

**Keep forward/inverse separate:**
```python
forward_model = DualTetrahedralNetwork(...)  # (trajectory, action) → next_diff
inverse_model = DualTetrahedralNetwork(...)  # (extended_trajectory) → action
```

**Active inference loop:**
```python
# Action selection: minimize expected free energy
for action in valid_actions:
    pred_diff = forward_model(trajectory_t, action)
    free_energy = uncertainty - beta * entropy
action = argmin(free_energy)

# Train forward: predict next diff
pred_diff = forward_model(trajectory_t, action)
loss_forward = signal_weighted_mse(pred_diff, diff_t1)

# Train inverse: effect-based (learn from forward's predictions!)
predicted_diffs = [forward_model(trajectory_t, a) for a in valid_actions]
soft_targets = softmax(-mse(predicted_diffs, diff_t1))  # Which actions explain this?
action_logits = inverse_model(trajectory_extended)
loss_inverse = kl_divergence(softmax(action_logits), soft_targets)
```

**Why this works:**
- Inverse learns from forward's understanding (not just labels)
- Proper coupling through effect-based learning
- Closes the strange loop: perception → model → action → effect

**Implementation plan:** See `PURE_ONLINE_REFACTORING_PLAN.md` (6 phases with code)

---

## Key Implementation Details

### Action Masking (CRITICAL)
**File:** `ACTIVE_INFERENCE_POLICY.md`
```python
VALID_ACTIONS = {
    'ALE/Pong-v5': [0, 2, 5],  # NOOP, UP, DOWN
}

# Mask invalid actions before loss
if action_mask is not None:
    masked_logits = action_logits.clone()
    masked_logits[action_mask == 0] = -1e9  # Set invalid to -inf
loss = cross_entropy(masked_logits, action)
```

**Why critical:** Without masking, model penalized for predicting UP when ground truth is UPFIRE (functionally identical in Pong!)

### Effect-Based Learning
**Pattern from FLOW_INVERSE_MODEL.py:**
- Don't learn from action labels
- Learn from action effects
- "UP at border" = "NOOP at border" (same outcome)
- Soft targets: multiple actions can be correct
- More biologically plausible

### Curriculum Learning
**File:** `ATTENTION_CURRICULUM.md`
**Stages:**
1. Own paddle only (mask opponent side)
2. Own paddle + ball (mask opponent)
3. Full game

**Why:** Prevents spurious correlations, developmental learning

### Past Failures
**File:** `INVERSE_MODEL_FAILURE_NOTES.md`
**Lessons:**
- ❌ Don't compress to abstract features (loses interpretability)
- ❌ Don't train on labels when effects available
- ✅ Keep predictions in pixel/flow space
- ✅ Use curriculum masking
- ✅ Integrate with active inference loop

---

## Current Issues

**File:** `CURRENT_ISSUES_AND_NEXT_STEPS.md`

**Priority 1: NaN explosion (~step 230)**
- Gradient explosion after model becomes confident
- Fix: Gradient clipping, output normalization, lower LR

**Priority 2: Action selection needs rethinking**
- Current: Just minimize free energy (1 step ahead)
- Missing: Goal states, multi-horizon planning, pragmatic value
- Question: Is current active inference sufficient?

**Priority 3: Lives tracking broken**
- Shows "Lives: 0" constantly
- Pain system needs entropy (lives) to work
- Debug: Print info dict, check keys

---

## Architecture Reference

**All implementations use:**
- `DualTetrahedralNetwork` from `Z_COUPLING/Z_interface_coupling.py`
- φ-hierarchical memory (golden ratio timescales)
- Inter-face coupling (pattern-level communication)

**Don't reinvent this - use it!**

---

## Detailed Files

**Implementation guides:**
- `ACTIVE_INFERENCE_POLICY.md` - Policy mathematics, action selection
- `PURE_ONLINE_REFACTORING_PLAN.md` - Complete 6-phase refactoring plan with code
- `ATTENTION_CURRICULUM.md` - Developmental learning stages

**Status tracking:**
- `STATUS_ACTIVE_INFERENCE.md` - Implementation progress
- `CURRENT_ISSUES_AND_NEXT_STEPS.md` - Active problems, debugging notes
- `INVERSE_MODEL_FAILURE_NOTES.md` - Past mistakes, lessons learned

---

**TL;DR for future Claude:**

**Working code:** `EXPERIMENTS/ACTIVE_INFERENCE/FLOW_INVERSE_MODEL.py`
- Use this pattern
- Forward/inverse separate DualTetrahedralNetworks
- Effect-based learning
- Action masking
- Active inference policy

**Broken code:** `EXPERIMENTS/ACTIVE_INFERENCE/PURE_ONLINE_TEMPORAL_DIFF.py`
- Good ideas: trajectory (3 diffs), temporal encoding, signal-weighted loss
- Fatal flaw: inverse sees ground truth not forward's prediction
- Fix: See `PURE_ONLINE_REFACTORING_PLAN.md` Option A

**Don't:**
- Train inverse on ground truth while claiming coupling
- Ignore action masking
- Compress to abstract features
- Use binary accuracy (0% or 100%)

**Do:**
- Use DualTetrahedralNetwork
- Effect-based learning (soft targets)
- Signal-weighted loss
- Active inference closes the loop
