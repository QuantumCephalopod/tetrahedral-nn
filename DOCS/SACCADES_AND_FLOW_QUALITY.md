# Saccades & Flow Algorithm Quality Analysis

**Date:** November 12, 2025
**Status:** 🔬 Analyzing artificial saccades and optical flow signal quality

---

## PART 1: What Are The Artificial Saccades?

### Current Implementation

```python
# From FLOW_PREPROCESSING_FIX.py, lines 95-99:
if add_saccade:
    dx = np.random.randint(-2, 3)  # Random: -2, -1, 0, 1, 2 pixels
    dy = np.random.randint(-2, 3)  # Random: -2, -1, 0, 1, 2 pixels
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    f2 = cv2.warpAffine(f2, M, (f2.shape[1], f2.shape[0]))
```

**What it does:**
- Shifts frame2 by a random offset (up to 2 pixels in any direction)
- This happens EVERY frame
- Simulates "eye jitter"

---

### Why Do We Need Saccades?

**The Problem: Static Blindness**

In pure flow-based vision:
```python
# Static paddle:
frame1: paddle at position X
frame2: paddle at position X (no movement!)
→ flow = 0 everywhere near paddle
→ In flow visualization: paddle disappears (shows as background)
```

**Flow only shows CHANGE.** Stationary objects have zero velocity → invisible!

**Biological Solution: Microsaccades**

Real eyes make constant tiny movements (microsaccades):
- **Frequency:** 1-2 Hz (1-2 times per second)
- **Amplitude:** 0.5-2 degrees of visual angle (~15-60 arcminutes)
- **Purpose:** Prevent perceptual fading (Troxler effect)

When your eye is perfectly still, stationary objects FADE FROM PERCEPTION within seconds!

**Our Solution: Artificial Jitter**

By shifting frame2 slightly, we create ARTIFICIAL motion for static objects:
```python
# Static paddle WITH saccade:
frame1: paddle at position X
frame2: paddle at position X + 1 pixel (artificial shift!)
→ flow shows small motion
→ Paddle is now visible in flow field!
```

---

### Are Our Saccades Tuned Correctly?

**Current parameters:**
```python
dx, dy = random.randint(-2, 3)  # Uniform distribution: -2, -1, 0, 1, 2
# Magnitude: 0 to 2√2 ≈ 2.83 pixels
# Direction: Random (isotropic)
# Frequency: EVERY FRAME (6 Hz at frameskip=10, 20 Hz at frameskip=3)
```

**Biological microsaccades:**
```python
# Real human microsaccades:
Frequency: 1-2 Hz
Magnitude: ~0.5-2 degrees visual angle
Direction: Often biased (not pure random)
```

### 🔴 POTENTIAL ISSUES:

#### 1. **Too Frequent?**

**Current:** Saccade on EVERY frame (6-20 Hz depending on frameskip)
**Biology:** 1-2 Hz (once or twice per second)

**Problem:** We're jittering WAY more than eyes do!

**Effect:**
- Adds constant noise to flow field
- Static objects show "vibrating" motion instead of stillness
- Might make learning harder (noise vs signal)

**Fix:** Only apply saccades occasionally, not every frame:
```python
if add_saccade and np.random.random() < 0.2:  # 20% of frames = ~1-2 Hz at 10 Hz sampling
    dx = np.random.randint(-2, 3)
    dy = np.random.randint(-2, 3)
    ...
```

#### 2. **Magnitude Too Large?**

**Current:** Up to 2 pixels shift
**At 210×160 resolution:** 2 pixels ≈ 1% of screen width

**Is this too much?** Hard to say without testing, but:
- Biological saccades are typically <1% of visual field
- 2 pixels seems reasonable
- BUT: With frameskip=10, objects ALREADY move a lot, adding 2 more pixels might be excessive

#### 3. **Uniform Distribution?**

**Current:** Uniform random (-2, -1, 0, 1, 2) with equal probability

**Biology:** Microsaccades are NOT uniform:
- Often have preferred directions
- Magnitude follows exponential/gamma distribution (most are small, few are large)
- Not purely random - somewhat correlated with visual attention

**Better distribution:**
```python
# Gaussian-like: most saccades are small, few are large
dx = int(np.random.normal(0, 1))  # Mean=0, std=1 pixel
dy = int(np.random.normal(0, 1))
dx = np.clip(dx, -2, 2)
dy = np.clip(dy, -2, 2)
```

---

### RECOMMENDATION: More Biological Saccades

```python
def add_biological_saccade(frame, saccade_rate=0.2):
    """
    Add saccade with biologically-inspired parameters.

    Args:
        frame: Input frame (numpy array)
        saccade_rate: Probability of saccade this frame (0.2 = ~1-2 Hz at 6-10 Hz sampling)

    Returns:
        Jittered frame (or original if no saccade)
    """
    if np.random.random() > saccade_rate:
        return frame  # No saccade this frame

    # Gaussian distribution (most saccades are small)
    dx = int(np.random.normal(0, 0.7))  # std=0.7 pixels
    dy = int(np.random.normal(0, 0.7))
    dx = np.clip(dx, -2, 2)
    dy = np.clip(dy, -2, 2)

    # Apply shift
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
```

**Benefits:**
- Only jitters 20% of frames (more like biology)
- Smaller typical magnitudes (Gaussian distribution)
- Less noise in flow field
- Static objects still visible (when saccade occurs)

---

---

## PART 2: Is Farneback The Best Flow Algorithm?

### Current Implementation: Farneback

```python
flow = cv2.calcOpticalFlowFarneback(
    gray1, gray2,
    None,
    pyr_scale=0.5,    # Pyramid scaling (0.5 = downsample by 2× each level)
    levels=3,         # Number of pyramid levels
    winsize=15,       # Window size for polynomial expansion
    iterations=3,     # Iterations at each pyramid level
    poly_n=5,         # Size of pixel neighborhood (5 or 7)
    poly_sigma=1.2,   # Gaussian std for polynomial expansion
    flags=0
)
```

**What is Farneback?**
- Dense optical flow (computes flow at EVERY pixel)
- Based on polynomial expansion of image intensities
- Reasonably accurate, moderate speed
- **Published in 2003** - relatively old algorithm!

---

### Alternative Methods (Are They Better?)

#### 1. **DIS Optical Flow** (Dense Inverse Search)

**Published:** 2016 (much newer!)

**Advantages:**
- ✅ **FASTER** than Farneback (2-3× speed)
- ✅ **MORE ACCURATE** than Farneback (especially for large displacements!)
- ✅ Better handles edges and boundaries
- ✅ Dense output (like Farneback)

**When to use:**
- Large displacements (frameskip=10!)
- Real-time applications
- When you want both speed AND accuracy

**Code:**
```python
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
flow = dis.calc(gray1, gray2, None)
```

**Presets:**
- `PRESET_ULTRAFAST`: Fastest, less accurate
- `PRESET_FAST`: Good balance
- `PRESET_MEDIUM`: Better accuracy (recommended!)
- `PRESET_FINE`: Highest accuracy, slower

#### 2. **Deep Flow** (RAFT, FlowNet, etc.)

**Modern deep learning approaches:**
- **RAFT** (2020): State-of-the-art accuracy
- **FlowNet2** (2017): Fast, accurate
- **PWC-Net** (2018): Real-time capable

**Advantages:**
- ✅ Much more accurate than classical methods
- ✅ Handles large displacements well
- ✅ Robust to lighting changes

**Disadvantages:**
- ❌ Requires GPU (slow on CPU)
- ❌ Requires pre-trained models
- ❌ More complex setup
- ❌ Larger memory footprint

**When to use:**
- When accuracy is critical
- GPU available
- Can handle model loading overhead

#### 3. **SparseToDense Methods**

**Hybrid approach:**
1. Compute sparse flow (Lucas-Kanade on keypoints) - FAST
2. Interpolate to dense field - FILL IN

**Advantages:**
- ✅ Very fast
- ✅ Can handle large motions (with keypoint tracking)

**Disadvantages:**
- ❌ Interpolation may be inaccurate in regions without keypoints
- ❌ Not as dense/reliable as true dense methods

---

### What Signal Are We Losing With Farneback?

#### Loss 1: **Large Displacement Inaccuracy**

**Farneback assumes small motion** (polynomial expansion breaks down for large displacement)

**At frameskip=10:**
- Paddle moves 20-50 pixels
- Ball moves 30+ pixels
- **These are LARGE for Farneback!**

**Result:** Flow field shows incorrect velocities or "holes" where motion is too large

**Fix:** Use DIS (better at large displacement) or reduce frameskip

#### Loss 2: **Edge Bleeding**

Farneback can blur flow across object boundaries:
```
Real:        Estimated:
Paddle|Background    Paddle~~Background
↑     |              ↑~~~~~~  (bleeding)
Fast  |Still         Fast→Slower→Still
```

**Result:** Flow at object edges is less sharp than reality

**Fix:** DIS or deep learning methods have better edge preservation

#### Loss 3: **Sub-Pixel Motion**

Farneback has limited sub-pixel accuracy (depends on `poly_n` and `poly_sigma`)

**Small, slow motions** (like subtle paddle adjustments) might be quantized or lost.

**Fix:**
- Tune `poly_sigma` higher (smoother sub-pixel estimates)
- Use deep methods (better sub-pixel accuracy)

#### Loss 4: **Temporal Consistency**

**Farneback processes each frame pair independently** - no temporal smoothness!

**Result:** Flow can "jitter" frame-to-frame even if motion is smooth.

**Example:**
```
t=0→1: flow = 5.2 pixels/frame
t=1→2: flow = 4.8 pixels/frame  (should be ~5.0 if paddle held down!)
t=2→3: flow = 5.3 pixels/frame
```

**Fix:**
- Post-process with temporal smoothing
- Use methods with temporal integration (RAFT has temporal refinement)
- OR: Let the neural network learn to smooth (φ-memory fields!)

---

### RECOMMENDATION: Try DIS Optical Flow!

**Why DIS is probably better for your use case:**

1. **Handles large displacement better** (frameskip=10 problem!)
2. **2-3× faster than Farneback** (more online learning steps!)
3. **Better edge preservation** (paddle boundaries clearer!)
4. **Still dense** (every pixel gets flow)
5. **Easy drop-in replacement** (same API as Farneback)

**Comparison:**

| Method | Speed | Accuracy (small) | Accuracy (large) | Edge Quality | Sub-pixel |
|--------|-------|------------------|------------------|--------------|-----------|
| Farneback | Medium | Good | Poor | Medium | Good |
| DIS | **Fast** | **Good** | **Good** | **Good** | Good |
| RAFT (deep) | Slow (GPU) | Excellent | Excellent | Excellent | Excellent |

**For Atari + frameskip=10: DIS is probably the sweet spot!**

---

### Code: Better Flow Computation

```python
def compute_optical_flow_BEST(frame1, frame2, target_size=None,
                              add_saccade=False, method='DIS'):
    """
    Compute optical flow with BEST settings for learning.

    Args:
        method: 'DIS' (fast, accurate) or 'farneback' (fallback)
    """
    # Preprocess
    f1 = (frame1 * 255).astype(np.uint8)
    f2 = (frame2 * 255).astype(np.uint8)

    # Pad to square
    if target_size is not None:
        f1 = pad_to_square(f1, fill_value=0)
        f2 = pad_to_square(f2, fill_value=0)
        if f1.shape[0] != target_size:
            f1 = cv2.resize(f1, (target_size, target_size))
            f2 = cv2.resize(f2, (target_size, target_size))

    # Biological saccades (less frequent!)
    if add_saccade and np.random.random() < 0.2:  # Only 20% of frames
        dx = int(np.random.normal(0, 0.7))  # Gaussian: most are small
        dy = int(np.random.normal(0, 0.7))
        dx = np.clip(dx, -2, 2)
        dy = np.clip(dy, -2, 2)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        f2 = cv2.warpAffine(f2, M, (f2.shape[1], f2.shape[0]))

    # Compute flow
    if method == 'DIS':
        try:
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            flow = dis.calc(f1, f2, None)
        except AttributeError:
            print("⚠️  DIS not available, falling back to Farneback")
            method = 'farneback'

    if method == 'farneback':
        flow = cv2.calcOpticalFlowFarneback(
            f1, f2, None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

    return torch.from_numpy(flow).permute(2, 0, 1).float()
```

---

## Summary: Are We Losing Signal?

### ✅ **What We're Doing Right:**
- Dense flow (every pixel)
- Reasonable parameters for Farneback
- Saccades to prevent static blindness

### 🔴 **Where We're Losing Signal:**

1. **Aspect ratio stretching** (FIXED! Now using padding)
2. **Frameskip=10 too large** for optical flow algorithms (consider frameskip=3-5)
3. **Farneback struggles with large displacement** (try DIS!)
4. **Saccades too frequent** (every frame instead of 20% of frames)
5. **Saccades uniform distribution** (should be Gaussian)
6. **No temporal consistency** (each frame pair independent - but φ-memory might handle this!)

### 🎯 **High-Impact Fixes (Priority Order):**

1. **Switch to DIS optical flow** (better for large displacement, faster) - EASY WIN
2. **Reduce frameskip to 3-5** (smoother flow) - EASY WIN
3. **Make saccades less frequent** (20% instead of 100%) - EASY
4. **Gaussian saccade distribution** (most are small) - EASY
5. **Consider RAFT** (if GPU available and want state-of-the-art) - HARD but BEST

**Bottom line:** We're probably losing 20-30% of potential signal quality due to:
- Large frameskip (biggest issue!)
- Farneback's limitations on large displacement
- Over-frequent saccades adding noise

**Quick test:** Try DIS + frameskip=3 and see if flow looks much better!

---

**"Nature already figured this out. We just need to copy it correctly."** 🌊👁️
