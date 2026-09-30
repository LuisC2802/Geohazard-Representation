import os

import numpy as np

from geohazard_interpolation import (
    cumulative_sum_and_sum_sq_hdf5,
    load_hdf5_lazy,
)


BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_PATH = os.path.join(
    BASE_PATH,
    "examples",
    "output",
)

RISK_NAME = "synthetic_risk"

INPUT_PATH = os.path.join(
    OUTPUT_PATH,
    RISK_NAME,
    f"{RISK_NAME}.h5",
)

HEURISTIC_SUM_FILE_NAME = (
    f"norm_{RISK_NAME}_heuristic_sum.h5"
)

HEURISTIC_SUM_SQ_FILE_NAME = (
    f"norm_{RISK_NAME}_heuristic_sum_sq.h5"
)

# Synthetic equivalent of the vertical interval used
# for heuristic accumulation.
KOP_MIN_DEPTH = 300.0
DROPOFF_MAX_DEPTH = 900.0

TARGET = np.array(
    [1000.0, 2000.0, 1000.0],
    dtype=np.float32,
)

DIRECTION = "up"


def generate_heuristic_data():
    """Generate cumulative risk and squared-risk volumes for the heuristic."""

    print("Generating synthetic heuristic data...")

    if not os.path.isfile(INPUT_PATH):
        raise FileNotFoundError(
            f"Input risk volume not found: {INPUT_PATH}\n"
            "Run 'python examples/generate_synthetic_data.py' first."
        )

    dset_in, description, f_in = load_hdf5_lazy(INPUT_PATH)

    try:
        resolution = np.asarray(
            description["resolution"],
            dtype=np.float32,
        )

        shape = description["shape"]

        print(f"Input risk volume: {INPUT_PATH}")
        print(f"Volume shape: {shape}")
        print(f"Resolution: {resolution}")
        print(f"Heuristic depth interval: "
              f"{KOP_MIN_DEPTH:.1f} -> {DROPOFF_MAX_DEPTH:.1f} m")
        print(f"Direction: {DIRECTION}")

        output_folder = os.path.join(
            OUTPUT_PATH,
            RISK_NAME,
        )

        os.makedirs(output_folder, exist_ok=True)

        cumulative_sum_and_sum_sq_hdf5(
            dset_in=dset_in,
            output_path=OUTPUT_PATH,
            risk_folder=RISK_NAME,
            sum_file_name=HEURISTIC_SUM_FILE_NAME,
            sum_sq_file_name=HEURISTIC_SUM_SQ_FILE_NAME,
            z_min_real=KOP_MIN_DEPTH,
            z_max_real=DROPOFF_MAX_DEPTH,
            resolution=resolution,
            start_target=TARGET,
            direction=DIRECTION,
        )

    finally:
        f_in.close()

    heuristic_sum_path = os.path.join(
        OUTPUT_PATH,
        RISK_NAME,
        HEURISTIC_SUM_FILE_NAME,
    )

    heuristic_sum_sq_path = os.path.join(
        OUTPUT_PATH,
        RISK_NAME,
        HEURISTIC_SUM_SQ_FILE_NAME,
    )

    print("\nHeuristic data generated:")
    print(f"  Risk sum: {heuristic_sum_path}")
    print(f"  Risk sum squared: {heuristic_sum_sq_path}")


def main():
    generate_heuristic_data()

    print("\nSynthetic heuristic data generation completed.")


if __name__ == "__main__":
    main()