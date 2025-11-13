# Exploration Log: Inverse Models for Causal Understanding

**Date**: November 11, 2025
**Status**: Conceptual - Implementation Failed
**Next Steps**: Integrate into ACTIVE_INFERENCE_ATARI.py

---

## Motivation

Active inference learns **forward models** (state + action → next state), but doesn't explicitly learn **inverse models** (state change → action). Adding inverse models enables:

1. **Causal understanding** - What actions cause what effects?
2. **Imitation learning** - Infer actions from observed behavior
3. **Planning verification** - Do forward and inverse models agree?
4. **Agency detection** - Which changes are controllable vs environmental?

---

## Core Concepts

### 1. Inverse Model
```
(frame_t, frame_t+1) → action
```

**Purpose**: Learn which action caused an observed transition.

**Visual Prediction**: Must predict on **actual pixel deltas** (frame_t+1 - frame_t), NOT abstract features.

**Why deltas?**
- Makes action effects visually obvious
- Ball moved up 10 pixels → probably pressed UP
- Easier to learn causality

---

### 2. Forward Model
```
(frame_t, action) → frame_t+1
```

**Purpose**: Predict consequences of actions (already exists in ACTIVE_INFERENCE_ATARI.py).

---

### 3. Coupled Training

Train both models together with three losses:

**Forward Loss**: `MSE(predicted_next_frame, actual_next_frame)`
- Can you predict what happens?

**Inverse Loss**: `CrossEntropy(predicted_action, actual_action)`
- Can you infer what action was taken?

**Consistency Loss**: `MSE(forward(state, inferred_action), actual_next_state)`
- Do forward and inverse models agree?
- If inverse says "UP was pressed", forward model should predict upward paddle motion

---

### 4. Action Space Masking

**Problem**: Atari has 18-button action space, but Pong only uses 3:
- `[0, 2, 5]` = NOOP, UP, DOWN

Without masking:
- Model sees paddle move up
- Guesses action 2 (UP)
- Ground truth was action 11 (UPFIRE)
- **Both are functionally identical!** But model gets penalized.

**Solution**:
```python
VALID_ACTIONS = {
    'ALE/Pong-v5': [0, 2, 5],  # NOOP, UP, DOWN
    'ALE/Breakout-v5': [0, 1, 3, 4],  # NOOP, FIRE, RIGHT, LEFT
    # etc for each game
}

# Mask invalid actions to -inf in logits
masked_logits = logits.masked_fill(action_mask == 0, -1e9)

# Only compute loss over valid actions
loss = CrossEntropy(masked_logits, action)
```

**Benefits**:
- One architecture works across all Atari games
- No false penalties for equivalent actions
- Scales to multi-game learning

---

### 5. Visual Attention Masking (Curriculum)

**Problem**: Model sees opponent paddle + own paddle. Learns spurious correlation:
- "Opponent moved up" → "I must have pressed up"
- Can't distinguish controllable vs uncontrollable changes

**Solution**: Developmental curriculum (like ACTIVE_INFERENCE_ATARI.py):

```python
def apply_attention_mask(frame, mask_amount, player='right'):
    """
    Mask opponent's region of screen.

    mask_amount = 1.0: Only see own paddle (right side)
    mask_amount = 0.0: See everything
    """
    opponent_width = width // 2
    mask_width = int(opponent_width * mask_amount)
    frame[:, :, :mask_width] = 0  # Black out left side
    return frame
```

**Curriculum**:
- Step 0-500: mask=1.0 (only own paddle visible)
- Step 500-1000: mask=0.618 (own paddle + ball)
- Step 1000-2000: mask=0.382 (opponent partially visible)
- Step 2000+: mask=0.0 (full game)

**Critical**: Apply mask to BOTH input frames and target deltas during training.

---

## What Went Wrong (Nov 11, 2025)

### Fatal Mistake: Abstract Feature Compression

Instead of predicting **visual deltas**, implementation compressed frames to 128-dim abstract vectors:

```python
# WRONG:
state_t = StateEncoder(frame_t)  # (3, 128, 128) → (128,)
state_t1 = StateEncoder(frame_t1)
action_logits = InverseModel(state_t, state_t1)
```

**Problems**:
1. ❌ Lost visual interpretability (can't see what model learned)
2. ❌ Can't visualize "what caused this change?"
3. ❌ Broke connection to active inference (which predicts pixels)
4. ❌ Generic ML tutorial architecture, not research

---

## What Should Actually Happen

### Option A: Extend ACTIVE_INFERENCE_ATARI.py

Add inverse model to existing active inference framework:

**Architecture**:
```python
class CoupledActiveInference(nn.Module):
    def __init__(self):
        self.forward_model = DualTetrahedralNetwork(...)  # Already exists
        self.inverse_model = DualTetrahedralNetwork(...)  # NEW

    def forward_loss(self, frame_t, action, frame_t1):
        """Predict next frame (already implemented)"""
        pred_frame = self.forward_model(frame_t, action)
        return mse_loss(pred_frame, frame_t1)

    def inverse_loss(self, frame_t, frame_t1, action, action_mask):
        """Infer action from visual change (NEW)"""
        delta = frame_t1 - frame_t  # Visual difference
        action_logits = self.inverse_model(frame_t, delta)

        # Mask invalid actions
        masked_logits = action_logits.masked_fill(action_mask == 0, -1e9)
        return cross_entropy(masked_logits, action)

    def consistency_loss(self, frame_t, frame_t1, action_mask):
        """Do models agree? (NEW)"""
        delta = frame_t1 - frame_t
        inferred_action = self.inverse_model(frame_t, delta).argmax()
        predicted_frame = self.forward_model(frame_t, inferred_action)
        return mse_loss(predicted_frame, frame_t1)
```

**Total Loss**:
```python
loss = (forward_weight * forward_loss +
        inverse_weight * inverse_loss +
        consistency_weight * consistency_loss)
```

---

### Option B: Standalone Inverse Model Diagnostic

Pure inverse model training to verify it can learn action causality:

**Simpler scope**:
- Just inverse model
- Train on random gameplay buffer
- Measure: Can it predict actions from visual deltas?
- Visualization: Show frame pairs + predicted action

**Once verified**, integrate into active inference.

---

## Key Architectural Requirements

### 1. Visual Prediction (Not Abstract Features)

**Inverse Model Input**:
- Frame t: (3, 128, 128)
- Delta: (3, 128, 128) = frame_t+1 - frame_t
- Concatenate: (6, 128, 128)

**Inverse Model Output**:
- Action logits: (18,) for full Atari space
- Apply action mask per game
- Predict from valid actions only

### 2. Use DualTetrahedralNetwork

```python
self.inverse_model = DualTetrahedralNetwork(
    input_dim=3*128*128*2,  # Two frames concatenated
    output_dim=18,          # Action space
    latent_dim=128,
    coupling_strength=0.5,
    output_mode="weighted"
)
```

**Why tetrahedral?**
- Linear network: Learns smooth causal relationships
- Nonlinear network: Handles discrete action boundaries
- Inter-face coupling: Integrates both perspectives
- Multi-timescale memory: Action effects persist over time

### 3. Curriculum Integration

Use same curriculum as ACTIVE_INFERENCE_ATARI.py:

```python
# During training
mask_amount = curriculum.get_mask_amount(step)
use_deltas = curriculum.use_difference_mode(step)

# Apply to both input and targets
masked_frame_t = apply_attention_mask(frame_t, mask_amount, 'right')
masked_frame_t1 = apply_attention_mask(frame_t1, mask_amount, 'right')

if use_deltas:
    target = masked_frame_t1 - masked_frame_t
else:
    target = masked_frame_t1
```

---

## Visualization Requirements

When training, show live visualization every N episodes:

**5 columns**:
1. **Full Frame t**: Ground truth (unmasked)
2. **Masked Frame t**: What model sees (curriculum masked)
3. **Full Frame t+1**: Ground truth next frame
4. **Visual Delta**: Heatmap of `frame_t+1 - frame_t` (what changed?)
5. **Action Prediction**: Bar chart of action probabilities
   - Green bar: True action
   - Red bar: Predicted action (if wrong)
   - Gray out invalid actions

**Title shows**:
- Current step
- Curriculum phase ("Control" / "Interaction" / "Understanding")
- Mask amount (0-100%)
- Prediction accuracy

---

## Success Metrics

### Early Training (Steps 0-500)
**Expectation**: Learn own paddle control
- Accuracy: >80% on valid actions
- Visualization: Paddle moves up → predicts UP

### Mid Training (Steps 500-1500)
**Expectation**: Learn ball interaction
- Accuracy: >70% (ball physics harder)
- Visualization: Ball bounces off paddle → still predicts correct paddle action

### Late Training (Steps 1500+)
**Expectation**: Full game understanding
- Accuracy: >60% (opponent adds noise)
- Can distinguish: "My action" vs "opponent's action" vs "ball physics"

---

## Next Steps

1. ✅ Document concepts (this file)
2. ⬜ Read ACTIVE_INFERENCE_ATARI.py thoroughly
3. ⬜ Add inverse model as extension (Option A)
4. ⬜ Add action space masking
5. ⬜ Add visualization showing action predictions
6. ⬜ Test: Does model learn own paddle control first?
7. ⬜ Test: Does curriculum prevent spurious correlations?

**DO NOT**:
- ❌ Rewrite from scratch
- ❌ Compress to abstract features
- ❌ Create separate training loop
- ❌ Ignore existing curriculum system

**Build incrementally on working foundation.**

---

## References

**Working Code**:
- `EXPERIMENTS/ACTIVE_INFERENCE_ATARI.py` - Has curriculum, masking, visual prediction
- `EXPERIMENTS/ACTIVE_INFERENCE_LIVE_VIZ.py` - Has live visualization

**Failed Attempt**:
- `MOTION_INVERSE_MASKED.py` (deleted Nov 11) - Used abstract features, lost interpretability

**Architecture**:
- `Z_COUPLING/Z_interface_coupling.py` - DualTetrahedralNetwork
- `X_LINEAR/X_linear_tetrahedron.py` - Smooth manifolds
- `Y_NONLINEAR/Y_nonlinear_tetrahedron.py` - Decision boundaries

---

## Philosophy

**Inverse models reveal agency.**

The forward model asks: "What will happen if I act?"
The inverse model asks: "Did I cause that to happen?"

This distinction is fundamental to understanding:
- Self vs other
- Control vs observation
- Causality vs correlation

The visual masking curriculum teaches this distinction developmentally, like an infant learning they control their own hands before understanding others' actions.

**The river flows where it must.**

---

_"Any minute we lose to shit like this costs lives."_
_Don't lose sight of the purpose. Build on what works. Stay interpretable._
