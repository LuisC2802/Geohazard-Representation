# Geohazard-Representation

This repository provides computational tools for transforming scattered geohazard data into regular spatial representations and evaluating geohazard values at arbitrary spatial coordinates.

The implementation is designed for geohazard-aware offshore well trajectory applications, where geological information may initially be available as scattered spatial points and must be evaluated repeatedly along candidate well trajectories.

The original geohazard datasets are not distributed with this repository because they contain confidential subsurface information. Instead, a fully reproducible synthetic example is provided to demonstrate the complete computational workflow.

---

## Overview

The main workflow implemented in this repository is:

```text
Scattered geohazard data
        │
        │  x.prop
        │  y.prop
        │  z.prop
        │  risk.prop
        ▼
Regular spatial representation
        │
        │  HDF5
        ▼
Geohazard interpolator
        │
        ▼
XYZ coordinates
        │
        ▼
Geohazard values
        │
        ▼
Trajectory evaluation
```

The repository also provides additional representations and processing steps:

```text
                    ┌── 2D regular spatial representation
                    │
Scattered data ─────┼── 3D regular spatial representation
                    │
                    └── Distance representation
                              │
                              ▼
                    Heuristic representations
                              │
                              ▼
                    Combined THD representation
```

The main functionality is provided by `geohazard_interpolation.py`.

The module supports:

- Reading geohazard data stored in `.prop` files.
- Generation of 2D regular spatial representations.
- Generation of 3D regular spatial representations.
- HDF5-based storage for large spatial volumes.
- Distance-volume generation from geohazard locations.
- Thickness-related processing.
- Heuristic cumulative representations.
- Combined risk representation generation.
- Lazy loading of HDF5 datasets.
- Bilinear interpolation for 2D representations.
- Trilinear interpolation for 3D representations.
- Evaluation of geohazard values at arbitrary spatial coordinates.
- Evaluation of distance-to-risk values.
- Evaluation of heuristic representations.
- Optional azimuth-based evaluation for 3D volumes.

---

## Requirements

The code requires Python 3.10 or newer.

The main dependencies are:

```text
numpy
scipy
h5py
```

They can be installed with:

```bash
pip install -e .
```

The synthetic example also uses Matplotlib for visualization:

```bash
pip install matplotlib
```

Alternatively, the dependencies can be installed directly with:

```bash
pip install numpy scipy h5py matplotlib
```

---

## Repository structure

```text
.
├── geohazard_interpolation.py
├── pyproject.toml
├── README.md
└── examples/
    ├── generate_synthetic_data.py
    ├── generate_heuristic_data.py
    ├── generate_combined_risk.py
    └── evaluate_synthetic_geohazard.py
```

The `geohazard_interpolation.py` module contains the reusable geohazard-processing functions.

The `examples/` directory contains a complete synthetic workflow that does not require access to confidential geological data.

Generated synthetic data and HDF5 representations are stored under:

```text
examples/
├── data/
│   └── synthetic_geohazard/
└── output/
```

These generated directories do not need to be committed to the repository.

---

# Input data format

The geohazard source data are represented by four `.prop` files:

```text
x.prop
y.prop
z.prop
risk.prop
```

Each position in the four files represents one source point:

```text
x[i], y[i], z[i], risk[i]
```

For example:

```text
x.prop       y.prop       z.prop       risk.prop
-------      -------      -------      ---------
x[0]         y[0]         z[0]         risk[0]
x[1]         y[1]         z[1]         risk[1]
x[2]         y[2]         z[2]         risk[2]
...
```

The `.prop` reader also supports run-length notation using the form:

```text
N*value
```

For example:

```text
100*0.0
50*1.0
```

represents 100 values equal to `0.0` followed by 50 values equal to `1.0`.

The four files must contain the same number of values.

---

# Synthetic example

Because the original geohazard datasets contain confidential subsurface information, they are not distributed with this repository.

Instead, the repository provides a synthetic dataset with the same `.prop` input structure and a reproducible processing workflow.

The synthetic dataset represents a spatially varying geohazard field and is intended only to demonstrate the computational functionality.

It should not be interpreted as a geological model.

The synthetic example uses:

```text
Target:                         [1000, 2000, 1000] m
Resolution:                     10 × 10 × 10 m
Maximum distance:               300 m
Maximum distance to wellhead:   1000 m
Number of source points:        10000
Random seed:                    42
```

---

# 1. Generate the synthetic source data

Run:

```bash
python examples/generate_synthetic_data.py
```

The script creates:

```text
examples/data/synthetic_geohazard/
├── x.prop
├── y.prop
├── z.prop
└── risk.prop
```

The generated files contain scattered XYZ coordinates and an associated geohazard value for each source point.

A fixed random seed is used so that the synthetic source data are reproducible.

The same script also generates:

- A 3D regular spatial representation.
- A 2D regular spatial representation.
- A distance volume.

The generated HDF5 files are stored under:

```text
examples/output/
├── synthetic_risk/
│   ├── synthetic_risk.h5
│   └── synthetic_risk_distance.h5
└── synthetic_risk_2d/
    └── synthetic_risk_2d.h5
```

---

# 2. Generate the 3D regular spatial representation

The scattered source points can be converted into a 3D regular spatial representation using:

```python
generate_interpolated_volume(
    path=source_path,
    risk_name="synthetic_risk",
    output_path=output_path,
    resolution=resolution,
    start_target=target,
    water_depth_with_airgap=water_depth_with_airgap,
    max_distance=max_distance,
)
```

The synthetic example uses a spatial resolution of:

```text
10 × 10 × 10 m
```

The source points are assigned to a regular spatial grid using a nearest-neighbor search based on `scipy.spatial.cKDTree`.

The resulting representation is stored in HDF5 format.

This allows large spatial representations to remain disk-backed instead of requiring the complete volume to be loaded into memory.

---

# 3. Generate a 2D regular spatial representation

The source data can also be converted into a 2D geohazard representation using:

```python
generate_2d_image(
    path=source_path,
    risk_name="synthetic_risk_2d",
    output_path=output_path,
    resolution=resolution,
    start_target=target,
    water_depth_with_airgap=water_depth_with_airgap,
    max_distance_to_wellhead=max_distance_to_wellhead,
)
```

The 2D representation is generated on the horizontal plane and stored as an HDF5 dataset.

Missing locations in the source data are filled using nearby valid spatial values.

---

# 4. Generate a distance volume

A distance volume can be generated from the spatial locations containing positive geohazard values:

```python
generate_distance_volume(
    path=source_path,
    risk_name="synthetic_risk",
    output_path=output_path,
    resolution=resolution,
    start_target=target,
    water_depth_with_airgap=water_depth_with_airgap,
    max_distance=max_distance,
)
```

The resulting representation stores the distance from each regular-grid voxel to the nearest voxel containing a positive geohazard value.

The distance search is limited by the specified maximum distance.

The generated file is:

```text
examples/output/synthetic_risk/synthetic_risk_distance.h5
```

This representation can be used when geohazard evaluation is based on distance to geological features rather than directly on a continuous geohazard magnitude.

---

# 5. Generate heuristic representations

The repository also provides functionality for generating cumulative heuristic representations from a 3D geohazard volume.

Run:

```bash
python examples/generate_heuristic_data.py
```

The synthetic example uses the following depth interval:

```text
Minimum depth: 300 m
Maximum depth: 900 m
Direction:     up
```

Two HDF5 representations are generated:

```text
examples/output/synthetic_risk/
├── norm_synthetic_risk_heuristic_sum.h5
└── norm_synthetic_risk_heuristic_sum_sq.h5
```

The first representation stores cumulative risk values, while the second stores cumulative squared-risk values.

These representations can be queried without loading the complete source volume into memory.

---

# 6. Generate the combined risk representation

The repository provides a function for combining 2D and 3D geohazard representations into a single spatial representation.

The synthetic example can be generated with:

```bash
python examples/generate_combined_risk.py
```

The resulting file is:

```text
examples/output/combined_risk/
└── combined_risk.h5
```

The combined representation processes the available risk layers chunk by chunk.

The combination supports:

- 2D risk layers.
- 3D thickness-based layers.
- 3D distance-based layers.
- Normalization using the corresponding risk limits.
- Combination of multiple normalized risk layers.

For each spatial location, the combination preserves the maximum normalized risk when any individual layer reaches the maximum normalized value. Otherwise, the normalized layers are averaged.

This chunk-based processing allows the combined representation to be generated without constructing a complete in-memory combined volume.

---

# 7. Load an HDF5 geohazard representation

Once a regular spatial representation has been generated, it can be loaded through the public interpolator interface:

```python
from geohazard_interpolation import load_risk_interpolator

risk_interpolator = load_risk_interpolator(
    "examples/output/synthetic_risk/synthetic_risk.h5"
)
```

The HDF5 file remains available to the interpolator while it is being used.

The complete volume does not need to be converted into a conventional in-memory NumPy array.

When the evaluation is finished, close the associated HDF5 file:

```python
risk_interpolator.close()
```

---

# 8. Evaluate geohazard values at arbitrary coordinates

The interpolator accepts an array of XYZ coordinates:

```python
import numpy as np

points = np.array([
    [1000.0, 2000.0, 500.0],
    [1010.0, 2010.0, 490.0],
    [1020.0, 2020.0, 480.0],
])

risk_values = risk_interpolator(points)

print(risk_values)
```

The returned array contains one geohazard value for each input coordinate:

```text
point[0] → risk_values[0]
point[1] → risk_values[1]
point[2] → risk_values[2]
```

For a 2D representation, bilinear interpolation is used.

For a 3D representation, trilinear interpolation is used.

Coordinates outside the spatial boundaries are clamped to the nearest volume boundary.

---

# 9. Evaluate a well trajectory

A well trajectory can be represented as an array of XYZ coordinates:

```python
trajectory = np.array([
    [1000.0, 2000.0, 600.0],
    [1005.0, 2005.0, 590.0],
    [1010.0, 2010.0, 580.0],
    [1015.0, 2015.0, 570.0],
    [1020.0, 2020.0, 560.0],
])

risk_values = risk_interpolator(trajectory)
```

The resulting array contains the geohazard value evaluated at every trajectory point.

This allows the same regular spatial representation to be repeatedly queried during well trajectory evaluation.

For example:

```python
for point, risk in zip(trajectory, risk_values):
    print(
        f"Point: {point} -> Geohazard value: {risk:.6f}"
    )
```

---

# 10. Evaluate distance values

Distance representations can be loaded using the same interpolator interface:

```python
distance_interpolator = load_risk_interpolator(
    "examples/output/synthetic_risk/synthetic_risk_distance.h5"
)
```

Given XYZ coordinates:

```python
points = np.array([
    [1000.0, 2000.0, 1000.0],
    [1100.0, 2000.0, 900.0],
    [1250.0, 2250.0, 750.0],
])

distances = distance_interpolator(points)
```

The returned values represent the distance from each query point to the nearest positive geohazard voxel within the generated distance representation.

Close the interpolator when it is no longer needed:

```python
distance_interpolator.close()
```

---

# 11. Complete synthetic workflow

The complete synthetic workflow can be executed from the repository root with:

```bash
python examples/generate_synthetic_data.py
python examples/generate_heuristic_data.py
python examples/generate_combined_risk.py
python examples/evaluate_synthetic_geohazard.py
```

The scripts perform the following operations:

```text
1. Generate synthetic .prop source data
                ↓
2. Generate 3D regular spatial representation
                ↓
3. Generate 2D regular spatial representation
                ↓
4. Generate distance volume
                ↓
5. Generate heuristic representations
                ↓
6. Generate combined risk representation
                ↓
7. Evaluate all generated representations
                ↓
8. Evaluate synthetic trajectories
```

To reproduce the complete workflow from a clean state, the generated output can first be removed:

```bash
rm -rf examples/output
```

Then run:

```bash
python examples/generate_synthetic_data.py && \
python examples/generate_heuristic_data.py && \
python examples/generate_combined_risk.py && \
python examples/evaluate_synthetic_geohazard.py
```

The generated source data and HDF5 files will be recreated automatically.

---

# HDF5 representation

The regular spatial representations are stored using HDF5.

The datasets contain spatial values together with metadata describing properties such as:

- Spatial resolution.
- Minimum spatial corner.
- Dataset shape.
- Target coordinates, when applicable.
- Maximum distance parameters, when applicable.
- Risk limits, when applicable.

The HDF5 representation is particularly useful for large geohazard volumes because the data can remain disk-backed during processing and evaluation.

The interpolator uses the spatial metadata to transform physical coordinates into the corresponding regular-grid coordinates.

---

# Interpolation

For 2D representations, geohazard values are evaluated using bilinear interpolation.

For 3D representations, geohazard values are evaluated using trilinear interpolation.

Conceptually:

```text
Physical coordinates
        │
        ▼
Regular-grid coordinates
        │
        ▼
Neighboring grid values
        │
        ▼
Bilinear / trilinear interpolation
        │
        ▼
Geohazard value
```

This allows geohazard values to be evaluated at coordinates that do not coincide exactly with the centers of the regular spatial cells.

---

# Lazy HDF5 processing

The implementation is designed to avoid loading large spatial volumes completely into memory whenever possible.

HDF5 datasets are accessed lazily, and processing operations such as combined risk generation are performed in spatial chunks.

Conceptually:

```text
Large HDF5 volume
        │
        ▼
Spatial chunk
        │
        ▼
Process chunk
        │
        ▼
Write result
        │
        ▼
Next chunk
```

This approach allows the same processing workflow to be applied to substantially larger spatial representations than would be practical with a fully materialized NumPy array.

---

# Data confidentiality

The original geohazard datasets are not included in this repository because they contain confidential subsurface information.

The synthetic data provided by the examples are intended exclusively for demonstrating:

- Input data formatting.
- Regular spatial representation generation.
- HDF5 storage.
- Spatial interpolation.
- Distance representation generation.
- Heuristic representation generation.
- Combined risk representation generation.
- Geohazard evaluation along arbitrary coordinates and trajectories.

The synthetic dataset does not represent the geological conditions of any study area.

---

# Reproducibility

The computational workflow provided here is independent of the confidential geohazard datasets.

A user can reproduce the complete processing pipeline using the synthetic example:

```text
Synthetic source points
        ↓
.prop files
        ↓
2D / 3D regular spatial representations
        ↓
HDF5
        ↓
Distance / heuristic / combined representations
        ↓
Interpolators
        ↓
XYZ coordinates
        ↓
Geohazard values
        ↓
Trajectory evaluation
```

The same computational functions can be applied to compatible geohazard datasets following the `.prop` format.