# Geohazard Representation

This repository provides a Python implementation of regular spatial representation, interpolation, distance evaluation, heuristic preprocessing, and combined representation of geohazard data.

The repository contains the publicly releasable components of a computational workflow for evaluating geological risks in three-dimensional space. It is designed to provide reproducible examples of the geohazard representation methodology without requiring access to confidential field datasets or the complete proprietary well trajectory optimization framework.

## Requirements

- Python 3.10 or newer
- NumPy
- SciPy
- h5py

The required dependencies can be installed with:

```bash
pip install -e .
```

## Repository structure

```text
Geohazard-Representation/
│
├── geohazard_interpolation.py
│
├── examples/
│   ├── generate_synthetic_data.py
│   ├── generate_heuristic_data.py
│   ├── generate_combined_risk.py
│   ├── evaluate_synthetic_geohazard.py
│   │
│   └── data/
│       └── synthetic_geohazard/
│           ├── x.prop
│           ├── y.prop
│           ├── z.prop
│           ├── synthetic_risk.prop
│           └── synthetic_risk_2d.prop
│
├── pyproject.toml
└── README.md
```

Generated HDF5 files are written to:

```text
examples/output/
```

This directory is excluded from version control through `.gitignore`.

## Synthetic example

The repository includes a complete synthetic workflow that can be executed without access to confidential geological data.

The synthetic example generates a three-dimensional geohazard distribution and a two-dimensional geohazard distribution around a target location. The generated data are used to demonstrate the complete processing workflow, including regular spatial representation, distance evaluation, heuristic preprocessing, combined risk representation, and spatial evaluation.

The synthetic `.prop` files are included in the repository so that the example can be reproduced directly after cloning the project. The `generate_synthetic_data.py` script can also regenerate these input files.

## 1. Regular spatial representation

The repository converts scattered geohazard points into regular spatial representations stored in HDF5 format.

For three-dimensional geohazards, the representation is defined over a regular voxel grid:

```text
X × Y × Z
```

For two-dimensional geohazards, the representation is defined over a regular spatial grid:

```text
X × Y
```

The spatial resolution, grid origin, target location, and other relevant metadata are stored as HDF5 dataset attributes.

The three-dimensional representation can be generated with:

```bash
python examples/generate_synthetic_data.py
```

This generates the synthetic three-dimensional risk volume together with the corresponding two-dimensional and distance representations.

## 2. Two-dimensional geohazard representation

Two-dimensional geohazard information can be interpolated onto a regular X-Y grid.

The resulting HDF5 representation can be evaluated at arbitrary X-Y coordinates using the interpolation utilities provided by the repository.

The synthetic workflow demonstrates this process using `synthetic_risk_2d.prop`.

## 3. Distance representation

For geohazards represented as spatial volumes, the repository can generate a distance-to-risk volume.

For each voxel, the distance to the nearest valid risk voxel is calculated up to a specified maximum distance.

The resulting distance volume is stored in HDF5 format and can be evaluated at arbitrary three-dimensional coordinates.

## 4. Heuristic representations

The repository provides preprocessing functions for generating cumulative risk representations used by heuristic search methods.

For a selected depth interval, cumulative sums and cumulative sums of squared risk values can be generated from a three-dimensional risk volume.

The synthetic example generates these representations with:

```bash
python examples/generate_heuristic_data.py
```

The resulting HDF5 files contain the precomputed values required to efficiently evaluate risk-related heuristic information without repeatedly processing the complete three-dimensional volume.

## 5. Combined risk representation

Multiple geohazard representations can be combined into a single spatial risk representation.

The combination supports:

- two-dimensional geohazard layers;
- three-dimensional thickness-based representations;
- three-dimensional distance-based representations;
- normalization using the corresponding risk limits; and
- spatial combination of the normalized layers.

The synthetic example generates the combined representation with:

```bash
python examples/generate_combined_risk.py
```

The resulting representation is stored as an HDF5 dataset.

## 6. Loading an HDF5 risk representation

The repository provides lazy-loading utilities for HDF5 representations.

A risk interpolator can be created with:

```python
from geohazard_interpolation import load_risk_interpolator

interpolator = load_risk_interpolator(
    "path/to/risk.h5"
)
```

The interpolator provides spatial evaluation without requiring the complete HDF5 volume to be loaded into memory.

## 7. Evaluating arbitrary coordinates

A three-dimensional risk representation can be evaluated at individual coordinates:

```python
risk = interpolator([1000.0, 2000.0, 1000.0])
```

Two-dimensional representations can be evaluated using X-Y coordinates:

```python
risk = interpolator([1000.0, 2000.0])
```

The same approach can be used for evaluating points along a well trajectory.

## 8. Complete synthetic workflow

The complete example can be executed with:

```bash
python examples/generate_synthetic_data.py
python examples/generate_heuristic_data.py
python examples/generate_combined_risk.py
python examples/evaluate_synthetic_geohazard.py
```

Or, equivalently:

```bash
python examples/generate_synthetic_data.py && \
python examples/generate_heuristic_data.py && \
python examples/generate_combined_risk.py && \
python examples/evaluate_synthetic_geohazard.py
```

The final evaluation script demonstrates:

- three-dimensional risk evaluation;
- trajectory-based risk evaluation;
- two-dimensional risk evaluation;
- distance evaluation;
- distance evaluation along a trajectory;
- heuristic evaluation;
- heuristic evaluation along a trajectory;
- combined risk evaluation; and
- combined risk evaluation along a trajectory.

All generated HDF5 files are written to `examples/output/` and are ignored by Git.

## HDF5 representation

HDF5 is used as the storage format for the regular spatial representations.

This allows large three-dimensional datasets to be processed without requiring the complete volume to remain in RAM. The implementation uses HDF5 chunking and lazy access when reading and processing the generated representations.

Relevant spatial metadata, such as resolution, grid origin, target location, and representation-specific parameters, are stored as HDF5 dataset attributes.

## Interpolation

The repository uses spatial interpolation and nearest-neighbor operations to map scattered geohazard information onto regular spatial representations.

The resulting regular representation provides a consistent spatial domain for subsequent risk evaluation and preprocessing.

## Data confidentiality

The repository contains synthetic data only.

The geological datasets used in the corresponding field application are confidential and are therefore not distributed with this repository. Similarly, the complete trajectory optimization framework and other proprietary components are outside the scope of this public repository.

The examples are intended to reproduce the geohazard representation and evaluation workflow using synthetic data without exposing confidential field information.

## Reproducibility

The repository is structured so that the synthetic workflow can be reproduced from a clean installation using the included input data and example scripts.

The `.prop` files under:

```text
examples/data/synthetic_geohazard/
```

are synthetic input data and are intentionally included in version control.

The HDF5 files generated during execution are intermediate/output files and are intentionally excluded from version control through `.gitignore`.

## License

See the repository for the applicable license information.
