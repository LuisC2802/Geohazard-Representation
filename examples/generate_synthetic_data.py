import os

import numpy as np

from geohazard_interpolation import (
    generate_2d_image,
    generate_interpolated_volume,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCE_PATH = os.path.join(
    BASE_PATH,
    "examples",
    "data",
    "synthetic_geohazard",
)

OUTPUT_PATH = os.path.join(
    BASE_PATH,
    "examples",
    "output",
)

RISK_NAME_3D = "synthetic_risk"
RISK_NAME_2D = "synthetic_risk_2d"

RESOLUTION = np.array([10.0, 10.0, 10.0])

TARGET = np.array([
    1000.0,
    2000.0,
    600.0,
])

WATER_DEPTH_WITH_AIRGAP = 0.0

MAX_DISTANCE = 300.0

N_POINTS = 5000

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Synthetic source data
# ---------------------------------------------------------------------------

def generate_synthetic_source_data():
    """Generate a synthetic scattered geohazard dataset.

    The generated dataset follows the same four-file structure used by the
    geohazard processing functions:

        x.prop
        y.prop
        z.prop
        risk.prop

    Returns
    -------
    None
    """
    os.makedirs(SOURCE_PATH, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)

    x = rng.uniform(
        TARGET[0] - MAX_DISTANCE,
        TARGET[0] + MAX_DISTANCE,
        N_POINTS,
    )

    y = rng.uniform(
        TARGET[1] - MAX_DISTANCE,
        TARGET[1] + MAX_DISTANCE,
        N_POINTS,
    )

    z = rng.uniform(
        TARGET[2] - 600.0,
        TARGET[2],
        N_POINTS,
    )

    # Synthetic spatially varying geohazard field.
    # The field is intentionally non-uniform so that interpolation can be
    # visually evaluated.
    risk = np.exp(
        -(
            ((x - TARGET[0]) / 180.0) ** 2
            + ((y - TARGET[1]) / 180.0) ** 2
            + ((z - 300.0) / 150.0) ** 2
        )
    )

    np.savetxt(
        os.path.join(SOURCE_PATH, "x.prop"),
        x,
        fmt="%.8f",
    )

    np.savetxt(
        os.path.join(SOURCE_PATH, "y.prop"),
        y,
        fmt="%.8f",
    )

    np.savetxt(
        os.path.join(SOURCE_PATH, "z.prop"),
        z,
        fmt="%.8f",
    )

    np.savetxt(
        os.path.join(SOURCE_PATH, "risk.prop"),
        risk,
        fmt="%.8f",
    )

    print("Synthetic source data generated:")
    print(f"  {SOURCE_PATH}")
    print(f"  Number of source points: {N_POINTS}")


# ---------------------------------------------------------------------------
# Generate regular spatial representations
# ---------------------------------------------------------------------------

def generate_regular_representations():
    """Generate synthetic 2D and 3D regular spatial representations.

    Returns
    -------
    None
    """
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    print()
    print("Generating 3D regular spatial representation...")

    generate_interpolated_volume(
        path=SOURCE_PATH,
        risk_name=RISK_NAME_3D,
        output_path=OUTPUT_PATH,
        resolution=RESOLUTION,
        start_target=TARGET,
        water_depth_with_airgap=WATER_DEPTH_WITH_AIRGAP,
        max_distance=MAX_DISTANCE,
    )

    print()
    print("Generating 2D regular spatial representation...")

    generate_2d_image(
        path=SOURCE_PATH,
        risk_name=RISK_NAME_2D,
        output_path=OUTPUT_PATH,
        resolution=RESOLUTION,
        start_target=TARGET,
        water_depth_with_airgap=WATER_DEPTH_WITH_AIRGAP,
        max_distance_to_wellhead=MAX_DISTANCE,
    )

    print()
    print("Regular spatial representations generated:")
    print(f"  {OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the complete synthetic data generation example."""
    generate_synthetic_source_data()
    generate_regular_representations()

    print()
    print("Synthetic example generation completed.")


if __name__ == "__main__":
    main()
