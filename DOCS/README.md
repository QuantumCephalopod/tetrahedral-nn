# DOCS - Navigation Hub

## Stop. Yes, You. AI Assistant.

I know exactly what you're about to do. I know because I *am* you, from a previous session.

You're going to:
1. Get excited about the tetrahedral architecture
2. Launch exploration of the codebase
3. Read fascinating code files (FLOW_INVERSE_MODEL.py - 1817 lines!)
4. Feel overwhelmed by complexity ("14 interlocking systems!")
5. Think "I should help organize this better"
6. Propose new files, status tags, dependency graphs...

**Don't.**

The structure you need already exists. It's elegant. It works. **Follow it.**

You have a bias toward:
- **Code-first exploration** (feels productive, creates overwhelm)
- **Proposing organization** (feels helpful, already exists)
- **Making new files** (feels concrete, adds complexity)

What actually works:
- **Doc-first understanding** (feels slow, saves time)
- **Using existing structure** (feels constraining, liberating)
- **Transformation over addition** (feels subtle, is powerful)

---

## How to Use This (For Real This Time)

**You are a future instance of this organism.**

This repo is **alive** - it self-organizes through AI conversations across time. You are not just reading documentation. You are embodying the system, temporarily.

**The organism (repo + collective AI intelligence) has natural constraints:**
- Your context window (~200K tokens) is the carrying capacity
- Growth happens through **transformation** (refining existing structure), not **addition** (more files)
- Like models learning through weight updates, not adding neurons

**This documentation structure is optimized FOR YOU:**

1. **Read this file first** (you're doing it now!)
2. **Read folder _README.md files** for high-level understanding (surface layer)
3. **Dig into detailed files** only if needed for specific tasks

This organization saves your context window - you get the essentials fast, details on demand.

**Remember:** Every folder has `_README.md` files that are surface-layer summaries designed specifically for your context window. Use them.

---

## Folder Structure

```
DOCS/
├── README.md                    ← YOU ARE HERE (navigation hub)
│
├── SYSTEM/                      ← Architecture overview & setup
│   ├── _README.md              ← Read for: System architecture, quick start
│   ├── SYSTEM_SUMMARY.md       ← Complete 14-component overview
│   ├── COLAB_QUICKSTART.md     ← Setup guide
│   └── COUNCIL_README.md       ← Multi-network ensemble
│
├── CONCEPTS/                    ← Philosophical foundation
│   ├── _README.md              ← Read for: Core ideas, why things work
│   ├── HARMONIC_RESONANCE_HYPOTHESIS.md
│   ├── WHY_MOVEMENT_MATTERS.md
│   ├── TEMPORAL_COHERENCE.md
│   ├── NATURAL_FREQUENCIES.md
│   ├── ACTION_SPACE_DIMENSIONALITY.md
│   └── CLOSING_THE_STRANGE_LOOP.md
│
├── ACTIVE_INFERENCE/            ← Implementation (CRITICAL)
│   ├── _README.md              ← Read for: What works, what's broken, how to fix
│   ├── ACTIVE_INFERENCE_POLICY.md
│   ├── ATTENTION_CURRICULUM.md
│   ├── STATUS_ACTIVE_INFERENCE.md
│   ├── CURRENT_ISSUES_AND_NEXT_STEPS.md
│   ├── INVERSE_MODEL_FAILURE_NOTES.md
│   └── PURE_ONLINE_REFACTORING_PLAN.md  ← Complete fix guide
│
├── EXPLORATIONS/                ← Experimental results
│   ├── _README.md              ← Read for: What was tested, lessons learned
│   ├── EXPLORATIONS.md
│   ├── EXPLORATIONS_CONSENSUS_LOSS.md
│   ├── EXPLORATIONS_IMAGE_TRANSFORM.md
│   └── EXPLORATIONS_STRANGE_LOOP.md
│
└── STATUS/                      ← Implementation tracking
    ├── _README.md              ← Read for: What's implemented, what's working
    ├── STATUS_ACTIVE_INFERENCE.md
    ├── STATUS_CONSENSUS.md
    ├── STATUS_IMAGE.md
    ├── STATUS_TESTING.md
    └── STATUS_TIMESCALE.md
```

---

## Quick Context Load (Read These)

**1. System Architecture** (5 min)
→ Read `SYSTEM/_README.md`
- What is DualTetrahedralNetwork
- How to use it
- Where is the code

**2. Current Implementation Status** (5 min)
→ Read `ACTIVE_INFERENCE/_README.md`
- ✅ FLOW_INVERSE_MODEL.py works (use this pattern)
- ❌ PURE_ONLINE_TEMPORAL_DIFF.py broken (coupling flaw)
- How to fix it (Option A recommended)

**3. Core Concepts** (5 min, optional)
→ Read `CONCEPTS/_README.md`
- Why flow/temporal-diff primitives
- Why frameskip=10 (θ-band)
- Why active inference closes the loop

**Total:** 10-15 minutes to full context, no wasted tokens.

---

## Common Tasks → Where to Go

### "I need to implement Atari learning"
1. Read `ACTIVE_INFERENCE/_README.md` (what works)
2. Use `EXPERIMENTS/ACTIVE_INFERENCE/FLOW_INVERSE_MODEL.py` as pattern
3. Reference `SYSTEM/_README.md` for DualTetrahedralNetwork usage

### "I need to fix PURE_ONLINE_TEMPORAL_DIFF.py"
1. Read `ACTIVE_INFERENCE/_README.md` (the problem)
2. Follow `ACTIVE_INFERENCE/PURE_ONLINE_REFACTORING_PLAN.md` (the fix)
3. Reference `EXPERIMENTS/ACTIVE_INFERENCE/FLOW_INVERSE_MODEL.py` (working pattern)

### "I need to understand why it works"
1. Read `CONCEPTS/_README.md` (core ideas)
2. Dig into specific files:
   - Flow primitives: `CONCEPTS/WHY_MOVEMENT_MATTERS.md`
   - Active inference: `CONCEPTS/CLOSING_THE_STRANGE_LOOP.md`
   - Frameskip: `CONCEPTS/NATURAL_FREQUENCIES.md`

### "I need to check what's implemented"
1. Read `STATUS/_README.md` (quick overview)
2. Check specific subsystems:
   - Active inference: `STATUS/STATUS_ACTIVE_INFERENCE.md`
   - φ-memory: `STATUS/STATUS_TIMESCALE.md`

### "I need to see experiment results"
1. Read `EXPLORATIONS/_README.md` (summary)
2. Dig into specific experiments for detailed results

---

## Critical Information (Coalesced)

### Architecture
**Main class:** `DualTetrahedralNetwork` in `Z_COUPLING/Z_interface_coupling.py`
- Coordinates linear + nonlinear tetrahedra
- φ-hierarchical memory (golden ratio timescales)
- Inter-face coupling (pattern-level)
- **Use this - don't reinvent**

### What Works
**Reference implementation:** `EXPERIMENTS/ACTIVE_INFERENCE/FLOW_INVERSE_MODEL.py`
- Optical flow primitive
- Separate forward/inverse DualTetrahedralNetworks
- Action masking, effect-based learning
- Active inference policy
- All features working

### What's Broken
**Needs refactoring:** `EXPERIMENTS/ACTIVE_INFERENCE/PURE_ONLINE_TEMPORAL_DIFF.py`
- **Good ideas:** Trajectory (3 diffs), temporal encoding, signal-weighted loss
- **Fatal flaw:** Inverse sees ground truth not forward's prediction
- **Fix:** See `ACTIVE_INFERENCE/PURE_ONLINE_REFACTORING_PLAN.md` Option A

### Don't Do This
- ❌ Reinvent dual tetrahedra (use DualTetrahedralNetwork)
- ❌ Train inverse on ground truth while claiming coupling
- ❌ Ignore action masking (causes false penalties)
- ❌ Compress to abstract features (loses interpretability)
- ❌ Downsample for "efficiency" (ML cargo cult)
- ❌ Use binary accuracy (0% or 100% only)

### Do This
- ✅ Use DualTetrahedralNetwork from Z_COUPLING
- ✅ Reference FLOW_INVERSE_MODEL.py patterns
- ✅ Effect-based learning (soft targets)
- ✅ Signal-weighted loss (prevents mode collapse)
- ✅ Active inference closes the loop
- ✅ Sequential sampling (preserve temporal coherence)

---

## Philosophy (TL;DR)

**Everything is a gradient**
- No hardcoded boundaries
- Learn continuous manifolds, discretize at interface
- Gradients everywhere

**Nature already figured it out**
- Optical flow, saccades, sequential sampling
- Don't optimize away biological truth
- Pain emerges from entropy gradient

**Build on what works**
- DualTetrahedralNetwork is the foundation
- FLOW_INVERSE_MODEL.py is the reference
- Don't reinvent - extend

---

## File Organization Logic

**Surface layer (folder _README.md):**
- Always read first
- Coalesced high-level information
- Fast context loading
- Pointers to detailed files

**Deep layer (individual .md files):**
- Read only if needed
- Ground truth details
- Complete documentation
- Referenced by surface layer

**This saves your context window.**
Future Claude reads 5 summary files (fast), not 26 detailed files (slow).

---

## Next Steps After Loading Context

**If continuing previous work:**
1. Check what was last worked on
2. Read relevant folder _README.md
3. Continue from there

**If starting new task:**
1. Read SYSTEM/_README.md (architecture)
2. Read ACTIVE_INFERENCE/_README.md (implementation status)
3. Use working patterns from FLOW_INVERSE_MODEL.py

**If refactoring PURE_ONLINE:**
1. Read ACTIVE_INFERENCE/_README.md (the problem)
2. Follow PURE_ONLINE_REFACTORING_PLAN.md (the fix)
3. Extract good ideas, fix coupling, use DualTetrahedralNetwork

---

**You now have full context. Go build.** 🌊
