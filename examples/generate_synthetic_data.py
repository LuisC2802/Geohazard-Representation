import os

import numpy as np

from geohazard_interpolation import (
    generate_2d_image,
    generate_distance_volume,
    generate_interpolated_volume,
)


BASE_PATH = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

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


# ---------------------------------------------------------------------
# Synthetic geohazard configuration
# ---------------------------------------------------------------------

RESOLUTION = np.array(
    [10.0, 10.0, 10.0]
)

TARGET = np.array(
    [
        1000.0,
        2000.0,
        1000.0,
    ]
)

WATER_DEPTH_WITH_AIRGAP = 0.0

# Padding distance used during 3D interpolation and distance search.
MAX_DISTANCE = 300.0

# Half-width of the target-centered X-Y domain.
MAX_DISTANCE_TO_WELLHEAD = 1000.0

N_POINTS = 10000

RANDOM_SEED = 42


# ---------------------------------------------------------------------
# Synthetic source data
# ---------------------------------------------------------------------

def generate_synthetic_source_data():
    """
    Generate synthetic geohazard property files.

    The generated points cover the complete target-centered X-Y domain
    and the vertical interval used by the public examples.
    """

    os.makedirs(
        SOURCE_PATH,
        exist_ok=True,
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    x = rng.uniform(
        TARGET[0] - MAX_DISTANCE_TO_WELLHEAD,
        TARGET[0] + MAX_DISTANCE_TO_WELLHEAD,
        N_POINTS,
    )

    y = rng.uniform(
        TARGET[1] - MAX_DISTANCE_TO_WELLHEAD,
        TARGET[1] + MAX_DISTANCE_TO_WELLHEAD,
        N_POINTS,
    )

    z = rng.uniform(
        WATER_DEPTH_WITH_AIRGAP,
        TARGET[2],
        N_POINTS,
    )

    # Synthetic 3D geohazard distribution.
    #
    # The maximum risk is located inside the generated volume rather
    # than at its boundary so that the interpolation and distance
    # examples are visually and numerically meaningful.
    risk = np.exp(
        -(
            ((x - TARGET[0]) / 350.0) ** 2
            + ((y - TARGET[1]) / 350.0) ** 2
            + (
                (z - 600.0) / 250.0
            ) ** 2
        )
    )

    # Synthetic 2D risk values.
    #
    # The same point distribution is reused because the public
    # generate_2d_image function only uses X/Y to construct the map.
    risk_2d = np.exp(
        -(
            ((x - TARGET[0]) / 350.0) ** 2
            + ((y - TARGET[1]) / 350.0) ** 2
        )
    )

    np.savetxt(
        os.path.join(
            SOURCE_PATH,
            "x.prop",
        ),
        x,
        fmt="%.6f",
    )

    np.savetxt(
        os.path.join(
            SOURCE_PATH,
            "y.prop",
        ),
        y,
        fmt="%.6f",
    )

    np.savetxt(
        os.path.join(
            SOURCE_PATH,
            "z.prop",
        ),
        z,
        fmt="%.6f",
    )

    np.savetxt(
        os.path.join(
            SOURCE_PATH,
            RISK_NAME_3D + ".prop",
        ),
        risk,
        fmt="%.8f",
    )

    np.savetxt(
        os.path.join(
            SOURCE_PATH,
            RISK_NAME_2D + ".prop",
        ),
        risk_2d,
        fmt="%.8f",
    )

    print(
        "Synthetic source data generated:"
    )

    print(
        f"  Source directory: {SOURCE_PATH}"
    )

    print(
        f"  Number of points: {N_POINTS}"
    )

    print(
        f"  X range: "
        f"{x.min():.1f} -> {x.max():.1f}"
    )

    print(
        f"  Y range: "
        f"{y.min():.1f} -> {y.max():.1f}"
    )

    print(
        f"  Z range: "
        f"{z.min():.1f} -> {z.max():.1f}"
    )


# ---------------------------------------------------------------------
# Regular spatial representations
# ---------------------------------------------------------------------

def generate_regular_representations():
    """
    Generate the 3D, 2D and distance-based HDF5 representations.
    """

    print(
        "\nGenerating 3D regular spatial representation..."
    )

    output_3d = generate_interpolated_volume(
        path=SOURCE_PATH,
        risk_name=RISK_NAME_3D,
        output_path=OUTPUT_PATH,
        max_distance=MAX_DISTANCE,
        resolution=RESOLUTION,
        start_target=TARGET,
        water_depth_with_airgap=WATER_DEPTH_WITH_AIRGAP,
        max_distance_to_wellhead=MAX_DISTANCE_TO_WELLHEAD,
    )

    print(
        f"3D representation generated: {output_3d}"
    )

    print(
        "\nGenerating 2D regular spatial representation..."
    )

    output_2d = generate_2d_image(
        path=SOURCE_PATH,
        risk_name=RISK_NAME_2D,
        output_path=OUTPUT_PATH,
        resolution=RESOLUTION,
        start_target=TARGET,
        water_depth_with_airgap=WATER_DEPTH_WITH_AIRGAP,
        max_distance_to_wellhead=MAX_DISTANCE_TO_WELLHEAD,
    )

    print(
        f"2D representation generated: {output_2d}"
    )

    print(
        "\nGenerating distance volume..."
    )

    output_distance = generate_distance_volume(
        path=SOURCE_PATH,
        risk_name=RISK_NAME_3D,
        output_path=OUTPUT_PATH,
        max_distance=MAX_DISTANCE,
        resolution=RESOLUTION,
        start_target=TARGET,
        water_depth_with_airgap=WATER_DEPTH_WITH_AIRGAP,
        max_distance_to_wellhead=MAX_DISTANCE_TO_WELLHEAD,
    )

    print(
        f"Distance volume generated: {output_distance}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print(
        "Generating synthetic geohazard example..."
    )

    print(
        f"Target: {TARGET}"
    )

    print(
        f"Resolution: {RESOLUTION}"
    )

    print(
        f"Maximum distance: {MAX_DISTANCE} m"
    )

    print(
        "Maximum distance to wellhead: "
        f"{MAX_DISTANCE_TO_WELLHEAD} m"
    )

    generate_synthetic_source_data()

    generate_regular_representations()

    print(
        "\nSynthetic geohazard example completed."
    )

    print(
        "\nGenerated files:"
    )

    print(
        f"  3D risk: "
        f"{os.path.join(OUTPUT_PATH, RISK_NAME_3D, RISK_NAME_3D + '.h5')}"
    )

    print(
        f"  2D risk: "
        f"{os.path.join(OUTPUT_PATH, RISK_NAME_2D, RISK_NAME_2D + '.h5')}"
    )

    print(
        f"  Distance: "
        f"{os.path.join(OUTPUT_PATH, RISK_NAME_3D, RISK_NAME_3D + '_distance.h5')}"
    )


if __name__ == "__main__":
    main()