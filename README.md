# Geohazard-Representation

This repository provides the computational tools used to transform scattered geohazard data into regular spatial representations and to evaluate geohazard values at arbitrary spatial coordinates.

The implementation is designed for geohazard-aware offshore well trajectory applications, where geological information may initially be available as scattered spatial points and must be evaluated repeatedly along candidate well trajectories.

The repository accompanies the study:

> **---**

The geohazard datasets used in the study are not distributed with this repository because they contain confidential subsurface information. Instead, a fully reproducible synthetic example is provided. The example follows the same input format and computational workflow used with the original data.

---

## Overview

The workflow implemented in this repository is:

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

The main functionality is provided by `geohazard_interpolation.py`.

The module supports:

- Reading geohazard data stored in `.prop` files.
- Generation of 2D regular spatial representations.
- Generation of 3D regular spatial representations.
- HDF5-based storage for large spatial volumes.
- Distance-volume generation from geohazard locations.
- Thickness-related processing.
- Lazy loading of HDF5 datasets.
- Bilinear interpolation for 2D representations.
- Trilinear interpolation for 3D representations.
- Evaluation of geohazard values at arbitrary XYZ coordinates.
- Optional azimuth-based evaluation for 3D volumes.

---

## Requirements

The code requires Python 3.10 or newer and the following Python packages:

```text
numpy
scipy
h5py
```

Install the dependencies with:

```bash
pip install numpy scipy h5py
```

For the visualization included in the synthetic example:

```bash
pip install matplotlib
```

---

## Repository structure

```text
.
├── geohazard_interpolation.py
├── README.md
└── examples/
    ├── generate_synthetic_data.py
    └── evaluate_synthetic_geohazard.py
```

`geohazard_interpolation.py` contains the reusable geohazard-processing functions.

The `examples/` directory contains a complete synthetic workflow that does not require access to confidential geological data.

---

# Input data format

The original geohazard data used by the workflow are represented by four `.prop` files:

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

Because the geohazard datasets used in the study are confidential, this repository does not distribute the original geological data.

Instead, the example generates a synthetic geohazard field with the same `.prop` structure.

The synthetic dataset represents a spatially varying geohazard field and is only intended to demonstrate the computational workflow.

It should not be interpreted as a geological model.

---

## 1. Generate the synthetic source data

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

The generated files contain scattered XYZ coordinates and an associated geohazard value for each point.

The script uses a fixed random seed so that the example is reproducible.

---

# 2. Generate the regular spatial representation

The scattered source points can be converted into a regular spatial representation using `generate_interpolated_volume()`.

The synthetic example uses:

```text
Resolution: 10 × 10 × 10 m
Horizontal extent: 600 × 600 m
Vertical extent: 600 m
```

The corresponding HDF5 representation is generated directly from the source `.prop` files.

The relevant operation is:

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

The source points are resampled onto a regular spatial grid using a nearest-neighbor search based on `scipy.spatial.cKDTree`.

The resulting representation is stored in HDF5 format.

This allows large spatial representations to remain disk-backed instead of requiring the complete volume to be loaded into memory.

---

# 3. Generate a 2D regular spatial representation

The same source data can also be converted into a 2D geohazard representation using:

```python
generate_2d_image(
    path=source_path,
    risk_name="synthetic_risk_2d",
    output_path=output_path,
    resolution=resolution,
    start_target=target,
    water_depth_with_airgap=water_depth_with_airgap,
    max_distance_to_wellhead=max_distance,
)
```

The 2D representation is generated on the horizontal plane and stored as an HDF5 dataset.

Missing locations in the source data are filled using nearest valid spatial values.

---

# 4. Load the HDF5 geohazard representation

Once the regular spatial representation has been generated, it can be loaded through the public interpolator interface:

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

# 5. Evaluate geohazard values at arbitrary coordinates

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

# 6. Evaluate a well trajectory

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

# 7. Complete example

The complete synthetic workflow can be executed with:

```bash
python examples/generate_synthetic_data.py
python examples/evaluate_synthetic_geohazard.py
```

The first script generates the synthetic `.prop` files.

The second script:

1. Loads the synthetic source data.
2. Generates the 3D regular spatial representation.
3. Generates the 2D regular spatial representation.
4. Loads the generated HDF5 representation.
5. Evaluates geohazard values at arbitrary coordinates.
6. Evaluates the geohazard along a synthetic well trajectory.
7. Generates a simple visualization of the resulting values.

---

# HDF5 representation

The regular spatial representation is stored using HDF5.

The dataset contains the spatial values together with metadata describing:

- Spatial resolution.
- Minimum spatial corner.
- Dataset shape.

The HDF5 representation is particularly useful for large geohazard volumes because the data can remain disk-backed during evaluation.

The interpolator uses the spatial metadata to transform physical XYZ coordinates into the corresponding regular-grid coordinates.

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

# Distance volumes

The module also provides functionality to generate distance volumes from geohazard locations.

A distance volume represents the distance from each regular-grid voxel to the nearest voxel containing a positive geohazard value.

For example:

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

The distance search is limited by the specified maximum distance.

This functionality can be used when the geohazard representation is based on distance to geological features rather than directly on a continuous geohazard magnitude.

---

# Data confidentiality

The geohazard datasets used in the associated study are not included in this repository because they contain confidential subsurface information.

The synthetic data provided by the examples are therefore intended exclusively for demonstrating:

- Input data formatting.
- Regular spatial representation generation.
- HDF5 storage.
- Spatial interpolation.
- Geohazard evaluation along arbitrary coordinates and trajectories.

The synthetic dataset does not represent the geological conditions of the study area.

---

# Reproducibility

The computational workflow provided here is independent of the confidential datasets.

A user can reproduce the complete processing pipeline using the synthetic example:

```text
Synthetic source points
        ↓
.prop files
        ↓
Regular spatial representation
        ↓
HDF5
        ↓
Interpolator
        ↓
XYZ coordinates
        ↓
Geohazard values
        ↓
Trajectory evaluation
```

The same computational functions can be applied to compatible geohazard datasets following the `.prop` format.

---

# Citation

If you use this software in academic work, please cite the associated publication:

```text
Parra Camacho, L. C., et al.
```

The final bibliographic information should be updated once the article is published.
