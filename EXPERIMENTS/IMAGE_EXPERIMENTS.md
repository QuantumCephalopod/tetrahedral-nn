# Image Experiments - Visual Transformation Learning

**Theme:** Learning geometric transformations as test bed for tetrahedral architecture

**Core Philosophy:** "If the network understands geometry, it should predict how images transform"

---

## Quick Summary

Single-file experiment cluster testing whether dual-tetrahedral architecture can learn to predict **image transformations** (rotation, scaling, translation, shearing). This is a controlled test environment before scaling to complex tasks like Atari.

**Key Insight:** Geometric transformations are perfect for testing because:
- Ground truth is well-defined (affine transforms)
- Gradual difficulty scaling (small → large angles)
- Interpretable (can visualize predictions)
- Tests both spatial reasoning and temporal prediction

---

## Experiment

### IMAGE_TRANSFORM.py
**Location:** `IMAGE_EXPERIMENTS/IMAGE_TRANSFORM.py`

**What:** Train tetrahedral network to predict transformed images

**Task:**
```python
Input:  [original_image, transformation_parameters]
Output: transformed_image
```

**Transformations Tested:**
1. **Rotation**: θ ∈ [-45°, +45°]
2. **Scaling**: s ∈ [0.5, 1.5]
3. **Translation**: (dx, dy) ∈ [-20, +20] pixels
4. **Shearing**: h ∈ [-0.5, +0.5]
5. **Composition**: Multiple transforms applied sequentially

**Key Features:**
- Synthetic dataset generation (MNIST, CIFAR-10, or random patterns)
- Progressive difficulty curriculum (small transforms → large)
- Visualization tools (show input, prediction, actual, error)
- MSE + SSIM loss (perceptual quality)

**Architecture Test:**
- Does dual-tetrahedral structure help with spatial reasoning?
- Can vertices/edges/faces specialize for different transform types?
- Does coupling between tetrahedra improve prediction?

---

## Why Image Transforms?

### 1. Controlled Complexity

Unlike real-world data, transformations have:
- **Known ground truth**: Exact mathematical formula
- **Parametric control**: Can dial difficulty up/down
- **Deterministic**: Same input → same output (no noise)
- **Interpretable failures**: Easy to see what went wrong

### 2. Geometric Reasoning Test

Spatial transformations require:
- **Coordinate understanding**: Where is each pixel?
- **Continuity**: Nearby pixels stay nearby
- **Symmetry**: Some transforms are reversible
- **Composition**: T₁ ∘ T₂ = T₃ (can chain transforms)

If network learns these properties, it understands space.

### 3. Precursor to Active Inference

**Connection to Atari:**
```
Image Transform: predict_transform(image, θ) → rotated_image
Active Inference: predict_frame(state, action) → next_state
```

**Same structure!** Action is like a transform parameter. Learning image transforms = learning forward model!

---

## Implementation Details

### Dataset Generation

```python
def generate_transform_dataset(n_samples=10000):
    images = load_mnist()  # or random patterns

    for img in images:
        # Sample random transform
        θ = random.uniform(-45, 45)  # rotation
        s = random.uniform(0.5, 1.5)  # scale

        # Apply transform
        transformed = apply_transform(img, θ, s)

        # Store pair
        dataset.append((img, [θ, s], transformed))

    return dataset
```

### Curriculum Learning

**Phase 1:** Small transforms (±10°, scale 0.8-1.2)
**Phase 2:** Medium transforms (±25°, scale 0.6-1.4)
**Phase 3:** Large transforms (±45°, scale 0.5-1.5)

**Rationale:** Learn simple cases first, generalize to complex

### Visualization

Shows 5-panel view:
1. **Original image**
2. **Transform parameters** (θ, s, etc.)
3. **Ground truth** (actual transformed)
4. **Prediction** (network output)
5. **Error map** (|prediction - actual|)

**Useful for:** Debugging, understanding failure modes, showing to skeptics

---

## Key Findings

1. **Network can learn transforms** with sufficient training
2. **Rotation is harder than translation** (more global)
3. **SSIM loss improves edge quality** over pure MSE
4. **Curriculum helps** (random large transforms → slow learning)
5. **Tetrahedral structure**: Unclear if advantage over standard CNN (needs ablation)

---

## Limitations

1. **Synthetic task**: Real images have texture, lighting, occlusion
2. **Affine only**: No perspective, elastic deformations
3. **Single object**: No multi-object tracking
4. **Static**: No temporal dynamics (yet)

---

## Connection to Other Experiments

**Timescale Experiments:**
- IMAGE_TRANSFORM appears in `TRAINING_10_NESTED_TIMESCALES.py`
- Multi-timescale memory improves transform prediction
- Slow fields = stable object identity, fast fields = motion

**Active Inference:**
- Transform prediction ≈ forward model
- Learn (state, action) → next_state
- Same architecture, different domain

**Consensus:**
- Could use consensus across multiple transform hypotheses
- Uncertainty = disagreement between perspectives
- "What rotation do W/X/Y/Z agree on?"

---

## Usage Example

```python
from EXPERIMENTS.IMAGE_EXPERIMENTS.IMAGE_TRANSFORM import (
    TransformDataset,
    TransformPredictor,
    visualize_predictions
)

# Create dataset
dataset = TransformDataset(
    n_samples=10000,
    transform_types=['rotation', 'scale'],
    difficulty='medium'
)

# Train model
model = TransformPredictor(img_size=128, latent_dim=128)
model.train(dataset, epochs=100)

# Visualize
visualize_predictions(model, dataset, n_samples=5)
```

---

## Future Directions

### 1. 3D Transformations

Currently 2D (image plane), extend to:
- **3D rotation**: Full SO(3) group
- **Perspective**: Depth, camera pose
- **Point clouds**: Rotate 3D meshes

### 2. Inverse Problems

Learn **inverse transforms**:
```python
# Forward: image + θ → rotated_image
# Inverse: rotated_image → original_image + θ
```

**Application:** Image registration, pose estimation

### 3. Equivariance

Make network **equivariant** to transforms:
```python
f(T(x)) = T(f(x))  # Transform commutes with network
```

**Benefit:** Better generalization, fewer parameters

**See:** Group Equivariant CNNs (Cohen & Welling 2016)

### 4. Active Vision

Combine with active inference:
- **Agent controls camera** (rotation, zoom)
- **Learns transform = action**
- **Predicts what will see** if rotate camera

**Result:** Active exploration to understand 3D scene

---

## When to Revisit This Cluster

- Need controlled test bed for architecture changes
- Testing new loss functions (beyond MSE/SSIM)
- Implementing spatial reasoning tasks
- Debugging geometric understanding
- Teaching concepts (clear visualizations)
- Building active vision systems

---

## Related Papers

**Spatial Transformers:**
- Jaderberg et al. (2015) - "Spatial Transformer Networks"
- Dai et al. (2017) - "Deformable Convolutional Networks"

**Image Registration:**
- Balakrishnan et al. (2019) - "VoxelMorph: Learning-based image registration"
- Rohé et al. (2017) - "SVF-Net: Learning deformable image registration"

**Equivariance:**
- Cohen & Welling (2016) - "Group Equivariant CNNs"
- Kondor & Trivedi (2018) - "On the Generalization of Equivariance"

---

**Navigation:**
- Return to: `EXPERIMENTS/` (top level)
- Related: `TIMESCALE_EXPERIMENTS.md` (nested timescales for transforms), `ACTIVE_INFERENCE_ATARI.py` (forward model prediction)
