# Status: Active Inference on Atari

**Last Updated**: November 11, 2025
**Location**: `EXPERIMENTS/ACTIVE_INFERENCE/`
**Status**: ✅ Working baseline established

---

## Current State

### What Works ✅

**ACTIVE_INFERENCE_ATARI.py** - Complete forward model with developmental curriculum:
- ✅ **Forward model**: Predicts next frame from (current_frame, action)
- ✅ **Curriculum learning**: Progressive attention unmasking
  - Phase 1 (0-500 steps): Only own paddle visible (mask=1.0)
  - Phase 2 (500-1000): Ball interaction visible (mask=0.618 = 1/φ)
  - Phase 3 (1000-2000): Opponent partially visible (mask=0.382 = 1/φ²)
  - Phase 4 (2000+): Full game visible (mask=0.0)
- ✅ **Difference mode**: Learns from visual deltas (next_frame - current_frame)
- ✅ **Golden ratio scheduling**: Natural timescale progression using φ
- ✅ **Visual prediction**: Operates on actual pixels, not abstract features
- ✅ **DualTetrahedralNetwork**: Uses tetrahedral architecture for world modeling

**ACTIVE_INFERENCE_LIVE_VIZ.py** - Real-time visualization
- ✅ Live training visualization
- ✅ Shows curriculum progression
- ✅ Displays prediction quality

### What Doesn't Work ❌

**Inverse Model** (attempted, failed):
- ❌ `MOTION_INVERSE.py` (deleted Nov 11, 2025)
  - **Fatal flaw**: Compressed frames to 128-dim abstract vectors
  - Lost visual interpretability
  - Can't see what the model learned
  - Broke connection to visual active inference framework
  - See `DOCS/INVERSE_MODEL_FAILURE_NOTES.md` for full postmortem

### What's Missing 🎯

1. **Inverse Model** (high priority)
   - Should predict: (frame_t, frame_t+1) → action
   - Must operate on visual deltas, not abstract features
   - Should integrate into existing curriculum
   - Should use action space masking per game

2. **Action Space Masking**
   - Atari has 18-button space, but Pong only uses 3
   - Need per-game valid action masks
   - Prevents false penalties for equivalent actions

3. **Consistency Loss**
   - Forward model: (state, action) → next_state
   - Inverse model: (state, next_state) → action
   - Do they agree? consistency_loss = MSE(forward(state, inferred_action), next_state)

4. **Policy Derivation**
   - Currently random exploration
   - Need action selection from prediction error minimization
   - Active inference principle: minimize expected surprise

---

## Architecture Overview

### Forward Model (Current)

```
Input:  frame_t (3, 128, 128) + action (18-dim one-hot)
        ↓
Network: DualTetrahedralNetwork
        - Linear tetrahedron: Smooth physics
        - Nonlinear tetrahedron: Discrete events
        - Face-to-face coupling: Integration
        ↓
Output: frame_t+1 (3, 128, 128)
```

**Loss**:
- Early training: MSE on deltas (frame_t+1 - frame_t)
- Later training: MSE on absolute frames
- Blended MSE→SSIM for perception

### Inverse Model (Proposed)

```
Input:  frame_t (3, 128, 128) + delta (3, 128, 128)
        - delta = frame_t+1 - frame_t (what changed?)
        ↓
Network: DualTetrahedralNetwork
        - Same architecture as forward model
        - Learns causal relationships
        ↓
Output: action_logits (18,)
        - Masked to valid actions per game
```

**Loss**: CrossEntropy(predicted_action, true_action)

---

## Key Insights

### Why Visual Prediction Matters

**Abstract features lose interpretability:**
```python
# ❌ WRONG (what MOTION_INVERSE.py did)
state = StateEncoder(frame)  # (3, 128, 128) → (128,)
# Can't visualize what the model learned
# Can't see causal relationships
```

**Visual prediction reveals understanding:**
```python
# ✅ CORRECT (what we should do)
delta = frame_t+1 - frame_t  # (3, 128, 128)
action = InverseModel(frame_t, delta)  # (18,)
# Can visualize: "ball moved up → predicted UP action"
```

### Why Curriculum Matters

**Without curriculum:**
- Model sees opponent + own paddle
- Learns spurious correlation: "opponent moved up → I pressed up"
- Can't distinguish controllable vs. uncontrollable

**With curriculum:**
- Phase 1: Only own paddle → learns control
- Phase 2: Add ball → learns interaction
- Phase 3: Add opponent → learns full game
- Developmental learning like biological vision

### Why Tetrahedral Architecture

- **Linear network**: Learns smooth causal relationships (physics)
- **Nonlinear network**: Handles discrete boundaries (collisions, events)
- **Face-to-face coupling**: Integrates both perspectives
- **Multi-timescale memory**: Action effects persist over time

---

## Next Steps

### Immediate Priority: Add Inverse Model

**Requirements:**
1. Extend ACTIVE_INFERENCE_ATARI.py (don't rewrite from scratch!)
2. Use same curriculum system
3. Operate on visual deltas (not abstract features)
4. Add action space masking per game
5. Add consistency loss between forward and inverse
6. Visualize action predictions

**Architecture:**
```python
class CoupledActiveInference(nn.Module):
    def __init__(self):
        self.forward_model = ForwardModel(...)  # Already exists
        self.inverse_model = InverseModel(...)  # NEW

    def forward_loss(self, frame_t, action, frame_t1):
        pred_frame = self.forward_model(frame_t, action)
        return mse_loss(pred_frame, frame_t1)

    def inverse_loss(self, frame_t, frame_t1, action, action_mask):
        delta = frame_t1 - frame_t
        action_logits = self.inverse_model(frame_t, delta)
        masked_logits = action_logits.masked_fill(~action_mask, -1e9)
        return cross_entropy(masked_logits, action)

    def consistency_loss(self, frame_t, frame_t1, action_mask):
        delta = frame_t1 - frame_t
        inferred_action = self.inverse_model(frame_t, delta).argmax()
        predicted_frame = self.forward_model(frame_t, inferred_action)
        return mse_loss(predicted_frame, frame_t1)
```

### Success Metrics

**Early Training (0-500 steps):**
- Forward accuracy: Can predict own paddle motion
- Inverse accuracy: >80% on valid actions (own paddle control)

**Mid Training (500-1500 steps):**
- Forward accuracy: Can predict ball physics
- Inverse accuracy: >70% (ball interaction harder)

**Late Training (1500+ steps):**
- Forward accuracy: Can predict full game
- Inverse accuracy: >60% (opponent adds noise)
- **Key test**: Can distinguish "my action" vs "opponent action" vs "ball physics"

---

## Related Documentation

**Conceptual:**
- `DOCS/ACTIVE_INFERENCE_POLICY.md` - Policy derivation theory
- `DOCS/ATTENTION_CURRICULUM.md` - Curriculum learning philosophy
- `DOCS/EXPLORATIONS.md` - Philosophical foundations

**Failed Attempts:**
- `DOCS/INVERSE_MODEL_FAILURE_NOTES.md` - What NOT to do (abstract features)

**Architecture:**
- `Z_COUPLING/Z_interface_coupling.py` - DualTetrahedralNetwork
- `X_LINEAR/X_linear_tetrahedron.py` - Linear tetrahedron (smooth manifolds)
- `Y_NONLINEAR/Y_nonlinear_tetrahedron.py` - Nonlinear tetrahedron (decision boundaries)

---

## Philosophy

**Active inference is not reinforcement learning.**

- No reward signal
- Only prediction error
- World model emerges from minimizing surprise
- Action policy derived from predictions, not scores

**The curriculum teaches agency.**

Like an infant:
1. First learn: "I control my hand"
2. Then learn: "My hand can affect objects"
3. Finally learn: "Other agents exist"

The masking curriculum enforces this developmental sequence.

**Visual prediction reveals causality.**

Abstract features hide what the model learned.
Visual deltas show: "This action caused that effect."

---

## Code Health

- **Architecture**: Clean, modular, well-documented
- **Curriculum system**: Elegant (golden ratio scheduling)
- **Visual prediction**: Interpretable by design
- **Next step**: Extend, don't rewrite

---

_"The river flows where it must."_
