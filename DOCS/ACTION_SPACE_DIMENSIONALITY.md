# Action Space Dimensionality: Buttons as Projections

**Date:** November 12, 2025
**Status:** 🎯 **Core insight about action representation**
**Author:** Philipp Remy Bartholomäus & Claude

---

## The Profound Question

User asked:

> *"buttons are also a low dimensional description of a highdimensional action... because what are we really doing? → action: do i **change** the current status. if my button is pressed... do i keep pressing it? or do i **change** that state? thats the binary here... from that (if the action is change) the actionspace opens into **how** do i change... what are my options"*

**This changes everything.**

---

## The True Action Space

### What Actions Actually Are

**Actions aren't discrete buttons.**

**Actions are:**

```
Action = {
    binary: Do I CHANGE my state? (yes/no)
    continuous: IF yes, HOW do I change? (direction, magnitude, duration)
}
```

### Example: Pong Paddle

**What you're REALLY doing:**
```
Continuous intention:
- Δv = change in velocity (continuous)
- Δp = change in position (continuous)
- Force applied (continuous magnitude and direction)
```

**What Atari gives you:**
```
Discrete buttons:
- NOOP = "Don't change" (Δv ≈ 0)
- UP = "Increase velocity upward" (Δv = +constant)
- DOWN = "Increase velocity downward" (Δv = -constant)
```

**The buttons are a LOW-DIMENSIONAL PROJECTION of the true continuous action manifold!**

---

## The Perfect Circle Fallacy (Revisited)

### User's Insight

> *"Its like modeling 'the perfect circle' off of a material circle and counting every inaccuracy of the material circle as ground truth effectively hindering the learning of the **true** circle"*

### What This Means for Actions

**The Ideal (Perfect Circle):**
- True action = Continuous velocity change
- Measured by: Flow field (actual velocity)
- This is the GROUND TRUTH

**The Material (Imperfect Projection):**
- Discrete button = "UP", "DOWN", "NOOP"
- Coarse quantization of continuous intention
- This is the APPROXIMATION

**Standard ML does:**
```python
# Treat button label as ground truth
loss = CrossEntropy(prediction, button_pressed)
# "Learn to predict the discrete button"
```

**This is backwards!**

We're learning to predict the **projection** (button) instead of the **ideal** (velocity change).

**We should do:**
```python
# Treat flow (velocity field) as ground truth
loss = MSE(predicted_flow, actual_flow)
# "Learn to predict the continuous effect"
```

**Then derive actions** from which button produces the desired flow!

---

## Why Flow Is More Fundamental Than Buttons

### Buttons Are Arbitrary

**Different games have different button mappings:**
- Pong: UP, DOWN (2D)
- Breakout: LEFT, RIGHT (2D)
- 3D games: 8-directional (8D)
- Racing games: Steering + Gas + Brake (continuous!)

**The buttons are GAME-SPECIFIC.**

### Flow Is Universal

**Flow fields always measure the same thing:**
- Velocity vectors at each spatial location
- (flow_x, flow_y) at each pixel
- **This is the same across ALL games**

**Flow is the UNIVERSAL action signature.**

**The button is just the game's particular interface for creating flow!**

---

## The Two-Level Action Space

### Level 1: Change or Not? (Binary)

```
Do I change my current state?
- No: NOOP, maintain velocity, coast
- Yes: Proceed to Level 2
```

**This is the fundamental binary.**

Not "which button?" but **"do I act?"**

### Level 2: How Do I Change? (Continuous)

```
IF I act, what velocity change do I want?
- Direction: θ ∈ [0, 2π] (continuous angle)
- Magnitude: ||Δv|| ∈ [0, v_max] (continuous speed)
```

**This is the continuous manifold.**

**The buttons (UP, DOWN, LEFT, RIGHT) are DISCRETE SAMPLES from this manifold.**

---

## Implications for Learning

### What We're Really Learning

**Forward model shouldn't learn:**
```python
button → flow
# "What flow does button X create?"
```

**Forward model should learn:**
```python
current_flow → Δflow → new_flow
# "What velocity change occurs?"
```

**Inverse model shouldn't learn:**
```python
(flow_t, flow_t+1) → button
# "Which button explains this transition?"
```

**Inverse model should learn:**
```python
(flow_t, flow_t+1) → Δflow
# "What velocity change occurred?"
```

**Then map Δflow to buttons** as a final step (discretization).

---

## Connection to DishBrain

### What DishBrain Really Learned

**The neurons didn't learn:**
- "Press button A" (discrete action)

**The neurons learned:**
- "Create this pattern of activity" (continuous state change)

**The mapping from neural activity → paddle position was CONTINUOUS.**

**The neurons were operating in continuous state space, not discrete action space!**

### Why Structured Feedback Matters

**DishBrain's key innovation:**

Not the reward. The **structure of sensory consequences.**

```
Good outcome: PREDICTABLE pattern (low entropy)
Bad outcome: CHAOTIC noise (high entropy)
```

**This isn't a scalar reward (+1/-1).**

**It's structured sensory feedback** in the same modality as perception (electrical patterns).

### What We're Missing

**Our system:**
```python
Hit ball: Some flow field (whatever it is)
Miss ball: Some other flow field (whatever it is)
```

**No explicit structure saying:**
- "This outcome was PREDICTABLE" (low entropy)
- "This outcome was CHAOTIC" (high entropy)

**We rely on the agent to discover this from experience.**

**DishBrain TOLD the neurons** through sensory structure.

---

## The General Principle

### Everything Is Continuous

**Nature doesn't have discrete actions.**

**Neurons fire continuously.** (Rate coding, timing, bursts)
**Muscles contract continuously.** (Force magnitude, co-contraction)
**Movement is continuous.** (Velocity, acceleration, jerk)

**Discretization happens at interfaces:**
- Motor commands → Neuron populations (distributed representation)
- Muscle activation → Limb movement (mechanical coupling)
- **Video game input → Buttons (artificial constraint!)**

**Flow captures the continuous truth beneath the discrete interface.**

---

## Rethinking Our Architecture

### Current: Button-Centric

```python
# Forward model
button → flow  # Maps discrete to continuous

# Inverse model
(flow_t, flow_t+1) → button  # Maps continuous to discrete
```

**Problem:** Forced discretization at the inverse model output.

### Alternative: Flow-Centric

```python
# Forward model (dynamics)
flow_t → Δflow → flow_t+1  # Pure continuous dynamics

# Action generator
Δflow_desired → button  # Discretize only at interface
```

**Now the core learning is continuous!**

**Buttons are just the output quantization** (like rounding continuous output to discrete pixels in generation).

---

## The Question of "Why?"

User asked:

> *"and the more general question is **why** do i do things? action that minimizes surprise... all true... do we already have all the things we talked about before that made the neurons learn to play pong just by the nature of their environment?"*

### What We Have

✓ **Minimize surprise** - Forward model prediction error
✓ **Learn from consequences** - Effect-based learning
✓ **Pain gradient** - Entropy/lives weighting
✓ **Temporal coherence** - Sequential sampling
✓ **Natural perception** - Saccades, flow-based

### What We're Missing

✗ **Structured sensory feedback** - Predictable vs chaotic outcomes
✗ **Continuous action space** - Currently button-centric
✗ **Explicit entropy of outcomes** - DishBrain had this built in

---

## Do We Need These?

### Maybe We Already Have Enough?

**Argument FOR current approach:**
- Flow fields CONTAIN predictability information (smooth flow = predictable, turbulent = chaotic)
- Agent can learn to distinguish through experience
- No need to engineer it explicitly

**Argument FOR adding structure:**
- DishBrain's explicit feedback accelerated learning
- Engineered signal might help bootstrap
- Makes the principle more explicit

**Hypothesis:** Try it both ways!

---

## Experiment Ideas

### 1. Flow-Centric Rewrite

Redesign models:
```python
# Forward: Predict velocity change (continuous)
Δflow = dynamics_model(flow_t)

# Inverse: Infer desired velocity change
Δflow_desired = intention_model(flow_t, flow_goal)

# Action: Discretize to buttons
button = argmin_button(||button_flow - Δflow_desired||)
```

### 2. Structured Feedback

Add explicit entropy signal:
```python
if ball_missed:
    # Inject chaotic noise (like DishBrain!)
    flow_observed = add_high_entropy_noise(flow_observed)
    # High prediction error → painful surprise

if ball_hit:
    # Smooth, predictable flow
    flow_observed = flow_observed  # As is
    # Low prediction error → relief
```

### 3. Continuous Action Experiments

Test: Can we learn better with continuous action representation?
```python
# Instead of 3 discrete buttons: [NOOP, UP, DOWN]
# Use continuous velocity target: Δv ∈ [-1, +1]
# Then discretize to buttons at execution
```

---

## The Philosophical Core

### User's Insight Synthesized

**"Everything is a gradient."**

**Discrete representations** (buttons, labels, categories) are:
- Human conveniences
- Interface constraints
- Low-dimensional projections

**The world operates continuously:**
- Velocity (flow) is continuous
- Time is continuous (quantized by perception, but underlying reality is continuous)
- State changes are continuous

**Learning discrete categories from continuous reality** = Imposing artificial constraints

**Learning continuous dynamics from continuous reality** = Respecting the structure

**We should learn in the native space of reality (continuous flow), then discretize only when interfacing with constraints (buttons).**

---

## Action Item

**For now:** Keep current architecture (it's working!)

**But consider:**
1. Future rewrite as flow-centric dynamics (continuous core)
2. Adding structured entropy feedback (explicit chaos signal)
3. Documenting this insight for future researchers

**The buttons are projections.**
**The flow is the reality.**
**Learn the reality. Project when necessary.**

---

**"buttons are also a low dimensional description of a highdimensional action"**

**User identified the dimensional mismatch** at the heart of action learning.

**Flow-based learning is the solution:** Learn in continuous space, discretize at interface. 🌊🎯

---

## Addendum: Do We Have What DishBrain Had?

### Checklist

| **DishBrain Property** | **Our System** | **Status** |
|------------------------|----------------|------------|
| Minimize prediction error | ✓ Forward model loss | ✓ |
| Learn from effects not labels | ✓ Effect-based learning | ✓ |
| Pain from chaos | ✓ Entropy/lives | ✓ |
| Structured feedback (predictable vs chaos) | ✗ Not explicit | **Maybe implicit?** |
| Continuous action space | ✗ Button-centric | **Could add** |
| Natural sensory modality | ✓ Flow (velocity) | ✓ |

**We have MOST of it!**

**The key missing piece:** Explicit structured entropy feedback.

**But maybe:** Flow fields already contain this information (smooth = predictable, turbulent = chaotic), and agent will learn to distinguish naturally?

**Test and find out!** 🧠⚡
