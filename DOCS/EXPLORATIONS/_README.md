# EXPLORATIONS - Experimental Results
**Surface layer:** What experiments were tried, what was learned
**Read this first** - detailed results in individual files

---

## Overview

These are experimental explorations testing specific hypotheses about:
- Consensus mechanisms (multiple representations agreeing)
- Image transformations (learning geometric operations)
- Strange loop closure (perception-action coupling)

---

## Exploration Results (Coalesced)

### 1. General Explorations
**File:** `EXPLORATIONS.md`
**Summary:** Various experimental tests of tetrahedral architecture
- Testing different configurations
- Validating theoretical predictions
- Proof-of-concept implementations

### 2. Consensus Loss
**File:** `EXPLORATIONS_CONSENSUS_LOSS.md`
**Hypothesis:** Multiple representations should agree (consensus)
**Key idea:**
- Linear and nonlinear tetrahedra see same input
- Should converge to consistent representations
- Consensus loss penalizes disagreement
- Creates redundancy and robustness

**Results:** (See detailed file for outcomes)

### 3. Image Transformations
**File:** `EXPLORATIONS_IMAGE_TRANSFORM.md`
**Hypothesis:** Tetrahedral networks can learn geometric transformations
**Tests:**
- Rotation, scaling, translation
- Linear network handles smooth transforms
- Nonlinear network handles discrete boundaries
- Dual network combines both

**Results:** (See detailed file for outcomes)

### 4. Strange Loop Closure
**File:** `EXPLORATIONS_STRANGE_LOOP.md`
**Hypothesis:** Can close perception-action-effect loop
**Tests:**
- Forward model: state + action → next state
- Inverse model: state + next state → action
- Active inference: use forward to select actions
- Does the loop close? Does agent act intelligently?

**Results:** (See detailed file for outcomes)
**Note:** This connects to `ACTIVE_INFERENCE/CLOSING_THE_STRANGE_LOOP.md` (conceptual)

---

## Lessons Learned

**From experiments:**
- What configurations worked
- What failed and why
- Hyperparameter insights
- Architecture modifications tested

**See detailed files for:**
- Complete experimental setups
- Quantitative results
- Visualizations
- Code references

---

**TL;DR for future Claude:**
- Experimental validation of theoretical ideas
- Consensus, transformations, strange loop tested
- See individual files for detailed results and code
