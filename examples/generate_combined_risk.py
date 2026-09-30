import os

import numpy as np

from geohazard_interpolation import (
    combine_images_to_hdf5,
    load_hdf5_lazy,
)


BASE_PATH = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

OUTPUT_PATH = os.path.join(
    BASE_PATH,
    "examples",
    "output",
)

# =========================================================
# INPUT REPRESENTATIONS
# =========================================================

RISK_NAME_2D = "synthetic_risk_2d"
RISK_NAME_3D = "synthetic_risk"

INPUT_2D_PATH = os.path.join(
    OUTPUT_PATH,
    RISK_NAME_2D,
    f"{RISK_NAME_2D}.h5",
)

INPUT_3D_PATH = os.path.join(
    OUTPUT_PATH,
    RISK_NAME_3D,
    f"{RISK_NAME_3D}.h5",
)

# =========================================================
# OUTPUT
# =========================================================

COMBINED_FOLDER = os.path.join(
    OUTPUT_PATH,
    "combined_risk",
)

COMBINED_PATH = os.path.join(
    COMBINED_FOLDER,
    "combined_risk.h5",
)

# =========================================================
# SYNTHETIC CONFIGURATION
# =========================================================

TARGET = np.array(
    [1000.0, 2000.0, 1000.0],
    dtype=np.float32,
)

KOP_MIN_DEPTH = 300.0

MAX_DISTANCE_TO_WELLHEAD = 1000.0

CHUNK_SIZE = 64


def generate_combined_risk():
    """Generate the combined 2D risk representation."""

    print(
        "Generating combined synthetic risk representation..."
    )

    if not os.path.isfile(INPUT_2D_PATH):

        raise FileNotFoundError(
            f"2D risk representation not found: "
            f"{INPUT_2D_PATH}\n"
            "Run 'python examples/generate_synthetic_data.py' first."
        )

    if not os.path.isfile(INPUT_3D_PATH):

        raise FileNotFoundError(
            f"3D risk representation not found: "
            f"{INPUT_3D_PATH}\n"
            "Run 'python examples/generate_synthetic_data.py' first."
        )

    os.makedirs(
        COMBINED_FOLDER,
        exist_ok=True
    )

    # =====================================================
    # READ 3D METADATA
    # =====================================================

    dset_3d, description_3d, file_3d = (
        load_hdf5_lazy(
            INPUT_3D_PATH
        )
    )

    try:

        resolution = np.asarray(
            description_3d["resolution"],
            dtype=np.float32
        )

        min_corner = np.asarray(
            description_3d["min_corner"],
            dtype=np.float32
        )

        shape = description_3d["shape"]

        print(
            f"3D input: {INPUT_3D_PATH}"
        )

        print(
            f"3D shape: {shape}"
        )

        print(
            f"Resolution: {resolution}"
        )

        print(
            f"Minimum corner: {min_corner}"
        )

    finally:

        file_3d.close()

    # =====================================================
    # READ 2D METADATA
    # =====================================================

    dset_2d, description_2d, file_2d = (
        load_hdf5_lazy(
            INPUT_2D_PATH
        )
    )

    try:

        print(
            f"2D input: {INPUT_2D_PATH}"
        )

        print(
            f"2D shape: {description_2d['shape']}"
        )

    finally:

        file_2d.close()

    # =====================================================
    # COMBINATION
    # =====================================================

    combine_images_to_hdf5(
        output_path=COMBINED_PATH,
        images_2d=[
            INPUT_2D_PATH
        ],
        limits_2d=[
            [0.0, 1.0]
        ],
        volumes_3d=[
            INPUT_3D_PATH
        ],
        limits_3d=[
            [0.0, 1.0]
        ],
        check_type_3d=[
            "Thickness"
        ],
        z_max_real=KOP_MIN_DEPTH,
        min_corner_real=min_corner,
        resolution=resolution,
        start_target=TARGET,
        max_distance_to_wellhead=(
            MAX_DISTANCE_TO_WELLHEAD
        ),
        chunk_size=CHUNK_SIZE,
    )

    print(
        "\nCombined risk generated:"
    )

    print(
        f"  {COMBINED_PATH}"
    )

    print(
        "\nCombined synthetic risk generation completed."
    )


def main():
    generate_combined_risk()


if __name__ == "__main__":
    main()