# STATUS - Implementation Tracking
**Surface layer:** What's implemented, what's working, what's not
**Read this first** - detailed tracking in individual files

---

## Quick Status Overview

| Subsystem | File | Status |
|-----------|------|--------|
| Active Inference | `STATUS_ACTIVE_INFERENCE.md` | ✅ Working (FLOW_INVERSE_MODEL.py) |
| Consensus | `STATUS_CONSENSUS.md` | ⚠️ Experimental |
| Image Processing | `STATUS_IMAGE.md` | ⚠️ Experimental |
| Testing/Validation | `STATUS_TESTING.md` | 🔄 Ongoing |
| Timescale Memory | `STATUS_TIMESCALE.md` | ✅ Implemented (φ-hierarchy) |

---

## Subsystem Details (Coalesced)

### Active Inference
**File:** `STATUS_ACTIVE_INFERENCE.md`
**Status:** ✅ Mature implementation
- FLOW_INVERSE_MODEL.py working
- PURE_ONLINE_TEMPORAL_DIFF.py needs refactoring
- Action masking implemented
- Effect-based learning implemented
- Active inference policy working
- See `ACTIVE_INFERENCE/_README.md` for details

### Consensus Mechanisms
**File:** `STATUS_CONSENSUS.md`
**Status:** ⚠️ Experimental
- Consensus loss tested
- Multi-representation agreement
- Redundancy and robustness goals
- See `EXPLORATIONS/EXPLORATIONS_CONSENSUS_LOSS.md` for results

### Image Processing
**File:** `STATUS_IMAGE.md`
**Status:** ⚠️ Experimental
- Geometric transformations tested
- Image-based tasks validation
- See `EXPLORATIONS/EXPLORATIONS_IMAGE_TRANSFORM.md` for results

### Testing & Validation
**File:** `STATUS_TESTING.md`
**Status:** 🔄 Ongoing
- Generalization tests
- Baseline comparisons
- Performance benchmarks
- Located in `EXPERIMENTS/TESTING_VALIDATION/`

### Timescale Memory
**File:** `STATUS_TIMESCALE.md`
**Status:** ✅ Fully implemented
- φ-hierarchical memory in `Z_COUPLING/Z_interface_coupling.py`
- Golden ratio decay rates:
  - Fast: τ = 0.1
  - Medium: τ = 0.1/φ = 0.0618
  - Slow: τ = 0.1/φ² = 0.0382
- Power-law-like temporal integration
- See `CONCEPTS/NATURAL_FREQUENCIES.md` for theory

---

## Where to Find Implementations

**Core architecture:**
- `W_FOUNDATION/W_geometry.py` - Geometric primitives
- `X_LINEAR/X_linear_tetrahedron.py` - Linear network
- `Y_NONLINEAR/Y_nonlinear_tetrahedron.py` - Nonlinear network
- `Z_COUPLING/Z_interface_coupling.py` - **DualTetrahedralNetwork** (main)

**Active inference:**
- `EXPERIMENTS/ACTIVE_INFERENCE/FLOW_INVERSE_MODEL.py` - Working
- `EXPERIMENTS/ACTIVE_INFERENCE/PURE_ONLINE_TEMPORAL_DIFF.py` - Needs fix

**Experiments:**
- `EXPERIMENTS/CONSENSUS_EXPERIMENTS/` - Consensus tests
- `EXPERIMENTS/IMAGE_EXPERIMENTS/` - Image transforms
- `EXPERIMENTS/TIMESCALE_EXPERIMENTS/` - φ-memory tests
- `EXPERIMENTS/TESTING_VALIDATION/` - Generalization tests

---

## Next Steps

**Immediate priorities:**
1. Refactor PURE_ONLINE_TEMPORAL_DIFF.py (see `ACTIVE_INFERENCE/PURE_ONLINE_REFACTORING_PLAN.md`)
2. Fix NaN explosion issue (~step 230)
3. Debug lives tracking (entropy/pain system)
4. Test curriculum learning integration

**See detailed status files for:**
- Implementation progress
- Known issues
- Performance metrics
- Code locations

---

**TL;DR for future Claude:**
- ✅ φ-hierarchical memory: Working
- ✅ Flow-based active inference: Working (FLOW_INVERSE_MODEL.py)
- ⚠️ Temporal-diff active inference: Broken coupling, refactor needed
- ⚠️ Consensus/image experiments: Ongoing validation
- See individual STATUS files for detailed tracking
