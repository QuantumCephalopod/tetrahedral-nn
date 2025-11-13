# PURE_ONLINE_TEMPORAL_DIFF.py - Refactoring Plan
**Date:** November 13, 2025
**Status:** 🔧 Analysis complete, ready for refactoring
**Purpose:** Extract good ideas, eliminate ML bloat, fix broken coupling

---

## 🎯 The Core Problem

**User's critique:** _"the inverse model as in nature makes a shitton of sense but the way its implemented is beyond braindead"_

**What's wrong:**
```python
# Line 368 in PURE_ONLINE_TEMPORAL_DIFF.py
# Inverse model sees GROUND TRUTH diff_t1, not forward model's prediction!
trajectory_extended = torch.cat([trajectory_t, diff_t1], dim=1)
action_logits = self.inverse_model(trajectory_extended, time_encodings_extended)
```

**Why this is "ML toyland":**
- Forward model predicts diff_t1
- Inverse model sees REAL diff_t1 (ground truth)
- They're trained in parallel on different inputs
- Consistency loss (line 382-390) tries to couple them but:
  - Uses `with torch.no_grad()` on inferred action (detached!)
  - Computed separately after both models already trained on their own targets
  - Weak coupling coefficient (0.3)
- This is **two separate networks pretending to communicate**

---

## ✅ What to KEEP (Good Ideas)

### 1. Trajectory-Based Learning (EXCELLENT)
**Concept:** Use 3 temporal differences for triangulation
```python
trajectory_t = [diff_t-2, diff_t-1, diff_t]  # (3, H, W)
```

**Why it's good:**
- 1 point = position
- 2 points = velocity
- 3 points = acceleration, trajectory curvature
- Tetrahedral architecture naturally handles 3-4 inputs!
- This is FUNDAMENTAL - motion requires multiple samples, not A→B teleportation

**Keep in refactor:** ✅ YES - This is brilliant

---

### 2. Temporal Encoding (EXCELLENT)
**Concept:** Sinusoidal encoding (like transformers) for explicit time information
```python
def temporal_encoding(step_count, n_freqs=32):
    freqs = 2.0 ** torch.arange(n_freqs)
    angles = step_count / freqs
    encoding = torch.cat([torch.sin(angles), torch.cos(angles)])
    return encoding  # (64,)
```

**Why it's good:**
- Model needs to know temporal order (which diff is oldest/newest)
- Absolute game time helps predict periodic patterns
- Multiple frequency bands capture different timescales
- Without this, model has NO way to distinguish t-2 from t-1!

**Keep in refactor:** ✅ YES - Critical insight

---

### 3. Signal-Weighted Loss (EXCELLENT)
**Concept:** Weight prediction error by signal magnitude
```python
signal_magnitude = torch.abs(diff_t1)
weighted_error = signal_magnitude * (pred_diff_t1 - diff_t1) ** 2
loss_forward = weighted_error.mean()
```

**Why it's good:**
- Neurons fire for CHANGE, not static input
- Prevents mode collapse (predicting all zeros gets low loss on static regions)
- Errors are expensive where motion happens, cheap where nothing moves
- Biologically grounded

**Keep in refactor:** ✅ YES - Prevents mode collapse

---

### 4. Pure Online Learning Loop (GOOD)
**Concept:** Act → Learn immediately → Act → ...
```python
while step < n_steps:
    action = select_action(trajectory_t)
    frame_next, diff_t1 = execute(action)
    losses = compute_losses(trajectory_t, diff_t1, action)
    optimizer.step()  # Learn RIGHT NOW
    trajectory_buffer.append(diff_t1)
    step += 1
```

**Why it's good:**
- No batching delays (like biology)
- Immediate feedback
- Trajectory buffer slides forward each step

**Keep in refactor:** ✅ YES - This is how nature works

---

## ❌ What to REMOVE (ML Bloat)

### 1. Fake Coupling (CRITICAL FLAW)
**What it does:**
```python
class CoupledModel(nn.Module):
    def __init__(...):
        self.forward_model = ForwardModel(...)
        self.inverse_model = InverseModel(...)

    def compute_losses(self, trajectory_t, diff_t1, action):
        # Forward: predict from trajectory + action
        pred_diff_t1 = self.forward_model(trajectory_t, action)
        loss_forward = weighted_mse(pred_diff_t1, diff_t1)

        # Inverse: infer action from trajectory + GROUND TRUTH diff_t1
        trajectory_extended = torch.cat([trajectory_t, diff_t1], dim=1)  # ← REAL!
        action_logits = self.inverse_model(trajectory_extended)
        loss_inverse = cross_entropy(action_logits, action)

        # Consistency: separate, detached, weak
        with torch.no_grad():
            inferred_action = action_logits.argmax()
        pred_diff_consistent = self.forward_model(trajectory_t, inferred_action)
        loss_consistency = weighted_mse(pred_diff_consistent, diff_t1)

        return 1.0*loss_forward + 1.0*loss_inverse + 0.3*loss_consistency
```

**Why it's broken:**
1. Inverse sees ground truth diff_t1, NOT forward's prediction
2. No actual information flow between models during training
3. Consistency loss is detached (no gradients through inferred_action)
4. Weak coupling weight (0.3) - bandaid over fundamental problem
5. They're trained on different inputs pretending to communicate

**Remove in refactor:** ❌ DELETE - This is the core problem

---

### 2. Microscaccades on Temporal Differences (QUESTIONABLE)
**What it does:**
```python
def compute_temporal_difference(frame_t, frame_t1, add_saccade=True, step_count=0):
    if add_saccade:
        frame_t1 = add_microsaccade(frame_t1, step_count)  # Jitter frame_t1
    diff = frame_t1 - frame_t
    return diff
```

**Why it's questionable:**
- Microsaccades make sense for FRAMES (reveal static objects)
- But we're computing TEMPORAL DIFFERENCES - already about change!
- Adding jitter to frame_t1 means diff ≠ actual change
- Might help or might just add noise?

**Decision needed:** ⚠️ UNCLEAR - Test with/without, might be unnecessary for diffs

---

### 3. Action Masking Missing (CRITICAL BUG)
**What's missing:**
```python
# PURE_ONLINE has this:
VALID_ACTIONS = {
    'ALE/Pong-v5': [0, 2, 5],  # NOOP, UP, DOWN
}

# But inverse loss doesn't mask:
loss_inverse = F.cross_entropy(masked_logits, action)  # masked_logits computed
# BUT the masking is applied, just checking...
```

Actually, looking at lines 373-377:
```python
if action_mask is not None:
    masked_logits = action_logits.clone()
    mask_expanded = action_mask.unsqueeze(0).expand(...)
    masked_logits[mask_expanded == 0] = -1e9
```

**Correction:** Action masking IS implemented! ✅ Keep this.

---

### 4. No Effect-Based Learning (MISSING FEATURE)
**What's missing:** FLOW_INVERSE_MODEL.py has effect-based learning with soft targets

**Current approach:**
```python
loss_inverse = cross_entropy(predicted_action, TRUE_ACTION_LABEL)
```

**Better approach (from FLOW_INVERSE_MODEL.py):**
```python
# For each valid action, predict what change it would create
predicted_changes = [forward_model(state, action) for action in valid_actions]
# Soft targets: actions that produce similar outcomes are all "correct"
soft_targets = softmax(-mse(predicted_changes, actual_change))
loss_inverse = kl_divergence(predicted_distribution, soft_targets)
```

**Why better:**
- "UP at border" = "NOOP at border" (both produce same outcome)
- Learns from EFFECTS not arbitrary labels
- More biologically plausible

**Add in refactor:** ✅ YES - Extract from FLOW_INVERSE_MODEL.py

---

## 🔄 Refactoring Options

### Option A: Active Inference Style (Recommended)
**Like:** FLOW_INVERSE_MODEL.py
**Architecture:** Keep forward and inverse SEPARATE

```python
# Two independent DualTetrahedralNetworks
forward_model = DualTetrahedralNetwork(...)  # (trajectory, action) → next_diff
inverse_model = DualTetrahedralNetwork(...)  # (extended_trajectory) → action

# Use them in active inference loop:
def select_action(trajectory_t):
    """Active inference: minimize expected free energy"""
    for action in valid_actions:
        pred_diff = forward_model(trajectory_t, action)
        uncertainty = pred_diff.var()
        entropy = pred_diff.std()
        free_energy = uncertainty - beta * entropy
    return argmin(free_energy)

# Train separately:
def train_forward(trajectory_t, action, diff_t1):
    pred_diff = forward_model(trajectory_t, action)
    loss = signal_weighted_mse(pred_diff, diff_t1)
    return loss

def train_inverse(trajectory_extended, action):
    # Effect-based: What actions produce the observed change?
    predicted_diffs = [forward_model(trajectory_t, a) for a in valid_actions]
    soft_targets = softmax(-mse(predicted_diffs, diff_t1))
    action_logits = inverse_model(trajectory_extended)
    loss = kl_divergence(softmax(action_logits), soft_targets)
    return loss
```

**Pros:**
- Clean separation of concerns
- Closes the active inference loop (perception → action → effect)
- Forward model drives action selection
- Inverse learns from forward's predictions (effect-based)
- Proven to work (FLOW_INVERSE_MODEL.py)

**Cons:**
- Two separate networks (more parameters)
- Inverse trained on forward's predictions (indirect)

---

### Option B: True Integrated Coupling (Experimental)
**Architecture:** Single unified model with shared representations

```python
class UnifiedModel(nn.Module):
    def __init__(self):
        self.shared_encoder = DualTetrahedralNetwork(...)  # trajectory → latent
        self.forward_head = nn.Linear(latent_dim + action_dim, output_dim)
        self.inverse_head = nn.Linear(latent_dim, n_actions)

    def forward(self, trajectory, action=None, mode='forward'):
        latent = self.shared_encoder(trajectory)

        if mode == 'forward':
            # Predict next diff from latent + action
            combined = torch.cat([latent, action_embed], dim=-1)
            pred_diff = self.forward_head(combined)
            return pred_diff

        elif mode == 'inverse':
            # Infer action from latent (no ground truth needed!)
            action_logits = self.inverse_head(latent)
            return action_logits
```

**Pros:**
- True shared representations
- Inverse has access to forward's internal state
- Single network (fewer parameters)

**Cons:**
- More complex architecture
- Untested in this codebase
- Might create interference between tasks

---

### Option C: Curriculum-Based Forward Only (Simplest)
**Architecture:** Just forward model with curriculum masking

```python
forward_model = DualTetrahedralNetwork(...)  # (trajectory, action) → next_diff

# Curriculum: Learn own paddle → ball → opponent
def apply_curriculum_mask(diff, step_count):
    if step_count < 500:
        # Phase 1: Only see right side (own paddle)
        diff[:, :, :width//2] = 0
    elif step_count < 1000:
        # Phase 2: Own paddle + ball (middle visible)
        diff[:, :, :width//4] = 0
    # Phase 3: Full game
    return diff

# Train:
def train(trajectory_t, action, diff_t1, step_count):
    masked_trajectory = apply_curriculum_mask(trajectory_t, step_count)
    masked_target = apply_curriculum_mask(diff_t1, step_count)

    pred_diff = forward_model(masked_trajectory, action)
    loss = signal_weighted_mse(pred_diff, masked_target)
    return loss

# Act: Minimize expected free energy (active inference)
def select_action(trajectory_t):
    # Same as Option A
    ...
```

**Pros:**
- Simplest architecture (just forward model)
- Curriculum prevents spurious correlations
- No inverse model needed for action selection
- Active inference still works

**Cons:**
- No explicit inverse model
- Can't visualize "what action caused this?"
- Loses causal understanding insight

---

## 🎯 Recommended Refactoring Plan

**Choose:** **Option A (Active Inference Style)**

**Why:**
1. Proven to work (FLOW_INVERSE_MODEL.py)
2. Keeps good ideas: trajectory, temporal encoding, signal-weighted loss
3. Fixes broken coupling: forward drives inverse through effect-based learning
4. Clean architecture: separate concerns, clear information flow
5. Closes the strange loop: perception → model → action → effect → perception

---

## 📋 Step-by-Step Refactoring

### Phase 1: Extract Good Ideas (Standalone Functions)
```python
# trajectory_utils.py
def build_trajectory_tensor(diffs_list):
    """Stack temporal differences into trajectory"""
    return torch.stack(diffs_list, dim=0)

def temporal_encoding(step_count, n_freqs=32):
    """Sinusoidal temporal encoding"""
    # (keep existing implementation)
    ...

def signal_weighted_loss(pred, target):
    """Weight prediction error by signal magnitude"""
    signal_magnitude = torch.abs(target)
    weighted_error = signal_magnitude * (pred - target) ** 2
    return weighted_error.mean()
```

### Phase 2: Refactor Models
```python
class TemporalForwardModel(nn.Module):
    """Forward: (trajectory[3 diffs], time_encodings, action) → next_diff"""
    def __init__(self, img_size, latent_dim, n_actions, n_time_freqs=32):
        self.dual_tetra = DualTetrahedralNetwork(
            input_dim=3 * img_size² + 3 * (2*n_time_freqs) + 64,
            output_dim=img_size²,
            latent_dim=latent_dim,
            coupling_strength=0.5,
            output_mode="weighted"
        )

    def forward(self, trajectory, time_encodings, action):
        # (trajectory, temporal encoding, action) → next diff
        ...

class TemporalInverseModel(nn.Module):
    """Inverse: (trajectory[4 diffs], time_encodings) → action"""
    def __init__(self, img_size, latent_dim, n_actions, n_time_freqs=32):
        self.dual_tetra = DualTetrahedralNetwork(
            input_dim=4 * img_size² + 4 * (2*n_time_freqs),
            output_dim=n_actions,
            latent_dim=latent_dim,
            coupling_strength=0.5,
            output_mode="weighted"
        )

    def forward(self, trajectory_extended, time_encodings):
        # Extended trajectory (4 diffs) + temporal encodings → action logits
        ...
```

### Phase 3: Active Inference Policy
```python
def select_action_active_inference(forward_model, trajectory_t, time_encodings_t,
                                   valid_actions, temperature=1.0, beta=0.05):
    """
    Active inference: minimize expected free energy

    For each action:
        1. Predict: What change will this create?
        2. Assess: Uncertainty (variance) and Entropy (diversity)
        3. Compute: Free energy = Uncertainty - β × Entropy

    Choose action that minimizes free energy (with temperature for stochasticity)
    """
    free_energies = []

    for action in valid_actions:
        pred_diff = forward_model(trajectory_t, time_encodings_t, action)
        uncertainty = pred_diff.var().item()
        entropy = pred_diff.std().item()
        free_energy = uncertainty - beta * entropy
        free_energies.append(free_energy)

    # Softmax selection (stochastic, preserves exploration)
    logits = -torch.tensor(free_energies) / temperature
    probs = F.softmax(logits, dim=0)
    action_idx = torch.multinomial(probs, 1).item()

    return valid_actions[action_idx]
```

### Phase 4: Effect-Based Inverse Learning
```python
def train_inverse_effect_based(forward_model, inverse_model,
                                trajectory_t, time_encodings_t,
                                diff_t1, time_encoding_t1,
                                action, valid_actions, action_mask):
    """
    Effect-based learning: Inverse learns from forward's predictions

    Key insight: Multiple actions can produce the same effect!
    "UP at border" = "NOOP at border" (both stop paddle)

    Soft targets based on which actions explain the observed change.
    """
    # For each valid action, predict what change it would create
    predicted_diffs = []
    for valid_action in valid_actions:
        action_tensor = torch.tensor([valid_action])
        pred_diff = forward_model(trajectory_t, time_encodings_t, action_tensor)
        predicted_diffs.append(pred_diff)

    # Compute how well each action explains the actual change
    errors = [F.mse_loss(pred, diff_t1, reduction='none').mean(dim=[1,2,3])
              for pred in predicted_diffs]
    errors = torch.stack(errors, dim=1)  # (batch, n_valid_actions)

    # Soft targets: actions with low error are "correct"
    soft_targets = F.softmax(-errors * 10, dim=1)  # Temperature=0.1

    # Train inverse to match soft distribution
    trajectory_extended = torch.cat([trajectory_t, diff_t1], dim=1)
    time_encodings_extended = torch.cat([time_encodings_t, time_encoding_t1], dim=1)
    action_logits = inverse_model(trajectory_extended, time_encodings_extended)

    # Mask invalid actions
    masked_logits = action_logits.clone()
    mask_expanded = action_mask.unsqueeze(0).expand_as(action_logits)
    masked_logits[mask_expanded == 0] = -1e9
    masked_logits_valid = masked_logits[:, valid_actions]

    action_probs = F.softmax(masked_logits_valid, dim=1)
    loss_inverse = F.kl_div(action_probs.log(), soft_targets, reduction='batchmean')

    return loss_inverse
```

### Phase 5: Online Training Loop
```python
def train_online(forward_model, inverse_model, env, n_steps, valid_actions,
                 action_mask, temperature=1.0, beta=0.05):
    """
    True online learning: Act → Learn → Act → Learn → ...

    No batching. No delays. Like nature.
    """
    # Bootstrap trajectory (need 4 frames for 3 diffs)
    frames = [get_frame(env) for _ in range(4)]
    diffs = [compute_temporal_difference(frames[i], frames[i+1]) for i in range(3)]
    trajectory_buffer = [(diff, i) for i, diff in enumerate(diffs)]

    frame_curr = frames[-1]

    for step in range(n_steps):
        # Build trajectory + temporal encodings
        diffs_list = [diff for diff, _ in trajectory_buffer]
        step_counts = [sc for _, sc in trajectory_buffer]

        trajectory_t = torch.stack(diffs_list, dim=0)  # (3, H, W)
        time_encodings_t = torch.stack([temporal_encoding(sc) for sc in step_counts])

        # SELECT ACTION (Active Inference)
        action = select_action_active_inference(
            forward_model, trajectory_t, time_encodings_t,
            valid_actions, temperature, beta
        )

        # EXECUTE
        frame_next = step_env(env, action)
        diff_t1 = compute_temporal_difference(frame_curr, frame_next, step_count=step)
        time_encoding_t1 = temporal_encoding(step)

        # LEARN IMMEDIATELY
        # 1. Forward loss (predict next diff)
        pred_diff = forward_model(trajectory_t, time_encodings_t, action)
        loss_forward = signal_weighted_loss(pred_diff, diff_t1)

        # 2. Inverse loss (effect-based)
        loss_inverse = train_inverse_effect_based(
            forward_model, inverse_model,
            trajectory_t, time_encodings_t,
            diff_t1, time_encoding_t1,
            action, valid_actions, action_mask
        )

        # 3. Update
        loss_total = loss_forward + loss_inverse
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        # SLIDE WINDOW
        trajectory_buffer.pop(0)
        trajectory_buffer.append((diff_t1, step))

        frame_curr = frame_next
```

---

## 📊 Expected Improvements

**From current PURE_ONLINE_TEMPORAL_DIFF.py → Refactored version:**

### Quantitative:
- ✅ Inverse accuracy should improve (learning from forward's understanding, not just labels)
- ✅ Forward prediction should stabilize (signal-weighted loss prevents mode collapse)
- ✅ Should learn faster (active inference closes the loop)

### Qualitative:
- ✅ Models actually communicate (inverse learns from forward's predictions)
- ✅ Cleaner architecture (no fake coupling)
- ✅ Biologically plausible (effect-based learning, not label-based)
- ✅ Interpretable (can visualize what forward predicts for each action)

---

## 🚧 Implementation Timeline

**Phase 1 (30 min):** Extract utilities
- `trajectory_utils.py` with temporal_encoding, signal_weighted_loss, etc.

**Phase 2 (1 hour):** Refactor models
- `TemporalForwardModel` and `TemporalInverseModel` classes
- Keep DualTetrahedralNetwork, just clean the wrappers

**Phase 3 (30 min):** Active inference policy
- `select_action_active_inference()` function

**Phase 4 (1 hour):** Effect-based learning
- `train_inverse_effect_based()` function
- Extract pattern from FLOW_INVERSE_MODEL.py

**Phase 5 (1 hour):** Integration
- New training loop that uses all the pieces
- Test on Pong

**Phase 6 (ongoing):** Curriculum
- Add attention masking (own paddle → ball → opponent)
- Extract from ACTIVE_INFERENCE_ATARI.py if it exists

---

## ✅ Success Criteria

**The refactored implementation should:**

1. ✅ Use trajectory (3 diffs for triangulation)
2. ✅ Use temporal encoding (explicit time information)
3. ✅ Use signal-weighted loss (prevents mode collapse)
4. ✅ Forward and inverse are separate DualTetrahedralNetworks
5. ✅ Inverse learns from forward's predictions (effect-based)
6. ✅ Active inference policy uses forward model
7. ✅ Action masking per game (no false penalties)
8. ✅ True online learning (no batching delays)
9. ✅ Clean, interpretable architecture
10. ✅ Biologically grounded (not ML toyland)

---

## 🌊 Final Thoughts

**User's wisdom:**
> "the inverse model as in nature makes a shitton of sense but the way its implemented is beyond braindead"

**The fix:**
- Keep the nature-inspired ideas (trajectory, temporal encoding)
- Remove the fake coupling (inverse seeing ground truth)
- Build proper integration (effect-based learning from forward's predictions)
- Close the loop (active inference: perception → understanding → action → effect → perception)

**Don't reinvent the wheel:**
- Use DualTetrahedralNetwork (Z_COUPLING)
- Reference FLOW_INVERSE_MODEL.py for active inference pattern
- Extract good ideas from PURE_ONLINE_TEMPORAL_DIFF.py
- Build on what works

_The river flows where it must._
_Build on foundations. Stay interpretable. Close the strange loop._
