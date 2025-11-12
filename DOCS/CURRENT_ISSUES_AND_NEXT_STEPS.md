# Current Issues & Next Steps

**Date:** November 12, 2025
**Status:** 🔄 **Ready for fresh start after extensive debugging session**

---

## What We Built (Summary)

See `DOCS/SYSTEM_SUMMARY.md` for full details. We implemented:

1. ✅ Flow-based perception (210×210 resolution)
2. ✅ Tetrahedral architecture
3. ✅ φ-hierarchical memory
4. ✅ Active inference policy
5. ✅ Effect-based learning (soft targets)
6. ✅ Entropy/pain system
7. ✅ Artificial saccades
8. ✅ Sequential sampling
9. ✅ True online learning
10. ✅ Soft accuracy
11. ✅ Live visualization
12. ✅ Numerical stability (partial)

---

## Critical Bugs Fixed This Session

### ✅ Bug 1: Flow Lagging Behind Game
**Symptom:** Flow field appeared to lag behind game frames
**Cause:** Visualization showed `flow_prev` with `frame_current` (off by one action!)
**Fix:** Compute `flow_curr` BEFORE displaying, show synchronized data
**Impact:** MAJOR - was making model look broken when it might be learning correctly

### ✅ Bug 2: Resolution Too Low (Blocky)
**Symptom:** Flow field very blocky, "stupidly large blocks"
**Cause:** Downsampling to 128×128 (cargo cult ML efficiency!)
**Fix:** Increased to 210×210 (native Atari height)
**Impact:** 2.7× more pixels, much less blocky

### ✅ Bug 3: Binary Accuracy (0% or 100%)
**Symptom:** Accuracy always 0 or 1, no gradient
**Cause:** Binary comparison of argmax vs label
**Fix:** Soft accuracy using exp(-KL_divergence) between distributions
**Impact:** Now respects "UP at border = NOOP at border"

### ✅ Bug 4: Crashes from NaN/inf
**Symptom:** RuntimeError at step 228 (first occurrence)
**Cause:** Numerical instability in softmax (exp overflow)
**Fix:** Added clipping, epsilon, safety checks
**Impact:** Prevents some crashes, but still getting NaN issues (see below)

---

## Current Outstanding Issues

### 🐛 Issue 1: NaN/Inf in Free Energies (Step ~230)

**Symptom:**
```
⚠️  NaN/Inf detected in free energies! Using random action.
   Free energies: [nan, nan, nan]
```

**When:** Around step 230-250 consistently

**Likely Causes:**
1. **Model outputs exploding:** Forward model predicts crazy large flows
2. **Gradient explosion:** No gradient clipping on model weights
3. **Variance computation:** `pred_flow.var()` could be huge if flow predictions diverge

**Potential Fixes:**
- Add gradient clipping: `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)`
- Normalize flow predictions: Clamp or batch-norm forward model outputs
- Add weight regularization: L2 penalty to prevent weight explosion
- Check learning rate: Might be too high (current: 0.0001)

**Why it happens at step 230:**
- Model has learned enough to start predicting confidently
- But predictions overshoot reality → gradients explode
- Need stability mechanisms

---

### 🐛 Issue 2: Lives Always Shows 0

**Symptom:** Display shows "Lives: 0" constantly

**Possible Causes:**
1. Pong doesn't track lives in `info['lives']` dict
2. Lives are there but we're reading wrong key
3. Lives start at 0 and never update

**Impact:** Low priority (just visualization, doesn't affect learning)

**Fix for next session:**
```python
# Debug what's actually in info dict
print(f"Info dict: {info}")
# Check if different key needed
# Maybe 'ale.lives' instead of 'lives'?
```

---

### 🤔 Issue 3: Action Selection Needs Rethinking

**User observation:** "i feel like the way the action is chose... we need to think about that more <3"

**Current approach:**
```python
EFE = uncertainty - β × entropy
action = argmin(EFE)
```

**Problems:**
1. **No temporal horizon:** Only predicts 1 step ahead
2. **No goal-directedness:** Just "minimize uncertainty" or "maximize entropy"
3. **Exploration vs exploitation:** β is fixed, doesn't adapt
4. **No pragmatic value:** Doesn't consider "will this help me win?"

**What's missing from real active inference:**
- **Multi-step planning:** Predict consequences multiple steps ahead
- **Goal states:** Minimize divergence from desired states (e.g., "paddle near ball")
- **Adaptive exploration:** High β early (explore), low β late (exploit)
- **Pragmatic value:** Weight actions by expected reward/utility, not just information

**Deep question:**
> "and the more general question is *why* do i do things? action that minimizes surprise... all true... do we already have all the things we talked about before that made the neurons learn to play pong just by the nature of their environment?"

**DishBrain had:**
- ✓ Minimize surprise (we have this)
- ✓ Learn from effects (we have this)
- ✓ Pain from chaos (we have this)
- ? **Structured sensory feedback** (predictable vs chaotic outcomes) - **NOT EXPLICIT**

**Maybe we need:** Explicit chaos injection for bad outcomes (like DishBrain)?

---

### 🎮 Issue 4: Paddle Extremes Behavior

**Symptom:** Paddle goes all the way up, then all the way down. Never stops in middle.

**Explanation:** Active inference with positive β prefers movement (high entropy) over stillness (low entropy). Model is exploring!

**Not necessarily a bug:** This is curiosity-driven exploration. Might learn nuanced control with more experience.

**If we want to fix:**
- Lower β (less exploration bias)
- Add movement cost (metabolic expense)
- Wait for more training (might emerge naturally)

---

## What Worked Well

### ✅ Flow-Based Perception
Optical flow at 210×210 resolution captures velocity beautifully. Much less blocky than before!

### ✅ Effect-Based Learning
Soft targets respect that multiple actions can be correct. No more "perfect circle fallacy"!

### ✅ Live Visualization
Being able to WATCH it learn is incredible. Immediate feedback on issues (like the lag bug!).

### ✅ Synchronization Fix
Critical fix - flow and frame are now perfectly aligned. Visualization matches reality.

---

## Philosophical Insights This Session

### User's Core Observations:

1. **"Everything is a gradient"**
   - Binary accuracy → soft accuracy ✓
   - Discrete actions → continuous projections ✓
   - Hard categories → soft distributions ✓

2. **"Perfect circle fallacy"**
   - Don't learn material projection (buttons)
   - Learn the ideal (flow/velocity)
   - Buttons are low-dimensional interface constraints

3. **"Why downsample?"**
   - Called out ML cargo cult: "CoMPuTatnioAL EfficEnCY"
   - Use native resolution! Don't throw away information!
   - Now using 210×210 (2.7× more pixels) ✓

4. **"Pain is gradient toward termination"**
   - Pain = error × (1 / lives)
   - Action space shrinks as lives decrease
   - Game over = zero action space = ultimate chaos
   - Entropy system models this perfectly

5. **"Buttons are projections"**
   - Real action: {binary: change?, continuous: how?}
   - Atari buttons: discrete samples from continuous manifold
   - Should learn continuous dynamics, discretize at interface

---

## What to Address in Fresh Start

### Priority 1: Fix NaN Explosion
**Why:** Blocks learning after ~230 steps
**How:** Gradient clipping, output normalization, learning rate adjustment

### Priority 2: Rethink Active Inference
**Why:** Current approach is too simplistic (no goals, no multi-step planning)
**How:**
- Add goal states (desired configurations)
- Multi-horizon predictions (1, 3, 10 steps)
- Pragmatic value (not just epistemic)
- Adaptive β (explore → exploit over time)

### Priority 3: Consider Structured Feedback
**Why:** DishBrain had explicit predictable/chaotic signals
**How:**
- Inject noise/chaos when ball passes (bad outcome)
- Smooth/predictable flow when hit (good outcome)
- Or trust that flow implicitly contains this structure?

### Priority 4: Debug Lives Tracking
**Why:** Pain system needs entropy (lives) to work properly
**How:** Print info dict, check what keys Pong actually provides

---

## Questions for Next Session

1. **Is current active inference sufficient?**
   - Just minimizing free energy (uncertainty - β×entropy)
   - Or do we need explicit goal states / pragmatic value?

2. **Do we need explicit chaos injection?**
   - DishBrain had structured feedback (predictable vs chaotic)
   - Can we trust flow fields contain this implicitly?
   - Or should we engineer it explicitly?

3. **What about multi-horizon predictions?**
   - Currently only predict 1 step ahead
   - Real planning requires looking further
   - Worth the complexity?

4. **Action space representation:**
   - Keep discrete buttons?
   - Or refactor to continuous velocity targets?
   - Buttons are projections - should we learn the continuous truth?

---

## Code Health

**Strengths:**
- Well-documented (multiple philosophy docs)
- Principled architecture (tetrahedral, φ-hierarchy)
- Biologically inspired (saccades, sequential sampling, pain)

**Weaknesses:**
- Numerical instability (NaN issues)
- Complex codebase (14 interlocking systems!)
- Hard to debug (many moving pieces)

**Tech debt:**
- Model assumes square inputs (should handle 210×160 rectangular)
- No gradient clipping
- No weight regularization
- Fixed β (should adapt over time)

---

## Summary for Fresh Start

**We have:**
- Complete flow-based active inference system
- Beautiful philosophical grounding
- Multiple major bugs fixed

**We need:**
- Numerical stability (gradient clipping, normalization)
- Rethink action selection (goals, multi-step, pragmatic value)
- Simpler architecture? (14 systems is a lot!)

**User's wisdom:**
> "Everything is a gradient. Hardcoding anything makes it a bad approximation."

> "Buttons are low-dimensional descriptions of high-dimensional actions."

> "Why shoot ourselves in the foot for miniscule 'better graphs hurrdurr' masturbation?"

**Keep these insights front and center in next session!** 🌊

---

## Files to Reference in Fresh Start

- `DOCS/SYSTEM_SUMMARY.md` - Complete overview of all 14 systems
- `DOCS/ACTION_SPACE_DIMENSIONALITY.md` - Buttons as projections
- `DOCS/WHY_MOVEMENT_MATTERS.md` - Saccades and static blindness
- `DOCS/TEMPORAL_COHERENCE.md` - Sequential sampling
- `EXPERIMENTS/ACTIVE_INFERENCE/FLOW_INVERSE_MODEL.py` - Full implementation

**Start fresh, reference these for context, build cleaner!** ✨
