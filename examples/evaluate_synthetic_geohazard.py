import os

import numpy as np

from geohazard_interpolation import load_risk_interpolator


BASE_PATH = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

OUTPUT_PATH = os.path.join(
    BASE_PATH,
    "examples",
    "output"
)

RISK_NAME_3D = "synthetic_risk"
RISK_NAME_2D = "synthetic_risk_2d"
DISTANCE_NAME = "synthetic_risk_distance"
COMBINED_NAME = "combined_risk"

HEURISTIC_SUM_NAME = (
    "norm_synthetic_risk_heuristic_sum"
)

HEURISTIC_SUM_SQ_NAME = (
    "norm_synthetic_risk_heuristic_sum_sq"
)

TARGET = np.array(
    [1000.0, 2000.0, 1000.0],
    dtype=np.float32
)


def evaluate_3d_points():
    """Evaluate the 3D risk representation at selected XYZ coordinates."""

    h5_path = os.path.join(
        OUTPUT_PATH,
        RISK_NAME_3D,
        f"{RISK_NAME_3D}.h5",
    )

    interpolator = load_risk_interpolator(
        h5_path,
        preload=False,
    )

    try:

        points = np.array(
            [
                TARGET,
                TARGET + [50.0, 50.0, -50.0],
                TARGET + [-100.0, 100.0, -100.0],
            ],
            dtype=np.float32,
        )

        values = interpolator(points)

        print("\n3D risk evaluation:")

        for point, value in zip(
            points,
            values,
        ):

            print(
                f"  Point [{point[0]:.1f}, "
                f"{point[1]:.1f}, "
                f"{point[2]:.1f}]"
                f" -> risk = {value:.6f}"
            )

    finally:

        interpolator.close()


def evaluate_trajectory():
    """Evaluate the 3D risk representation along a synthetic well trajectory."""

    h5_path = os.path.join(
        OUTPUT_PATH,
        RISK_NAME_3D,
        f"{RISK_NAME_3D}.h5",
    )

    interpolator = load_risk_interpolator(
        h5_path,
        preload=False,
    )

    try:

        n_points = 100

        trajectory = np.column_stack(
            (
                np.linspace(
                    TARGET[0] - 200.0,
                    TARGET[0] + 200.0,
                    n_points,
                ),
                np.linspace(
                    TARGET[1] - 150.0,
                    TARGET[1] + 150.0,
                    n_points,
                ),
                np.linspace(
                    TARGET[2],
                    TARGET[2] - 500.0,
                    n_points,
                ),
            )
        ).astype(np.float32)

        risks = interpolator(
            trajectory
        )

        print("\n3D trajectory evaluation:")

        print(
            f"  Number of trajectory points: "
            f"{len(trajectory)}"
        )

        print(
            f"  Minimum risk: "
            f"{np.min(risks):.6f}"
        )

        print(
            f"  Maximum risk: "
            f"{np.max(risks):.6f}"
        )

        print(
            f"  Mean risk: "
            f"{np.mean(risks):.6f}"
        )

    finally:

        interpolator.close()


def evaluate_2d_points():
    """Evaluate the 2D risk representation at selected XY coordinates."""

    h5_path = os.path.join(
        OUTPUT_PATH,
        RISK_NAME_2D,
        f"{RISK_NAME_2D}.h5",
    )

    interpolator = load_risk_interpolator(
        h5_path,
        preload=False,
    )

    try:

        points = np.array(
            [
                [TARGET[0], TARGET[1]],
                [
                    TARGET[0] + 100.0,
                    TARGET[1] + 100.0,
                ],
                [
                    TARGET[0] - 100.0,
                    TARGET[1] - 100.0,
                ],
            ],
            dtype=np.float32,
        )

        values = interpolator(
            points
        )

        print("\n2D risk evaluation:")

        for point, value in zip(
            points,
            values,
        ):

            print(
                f"  Point [{point[0]:.1f}, "
                f"{point[1]:.1f}]"
                f" -> risk = {value:.6f}"
            )

    finally:

        interpolator.close()


def evaluate_distance_points():
    """Evaluate the distance-to-risk volume at selected XYZ coordinates."""

    h5_path = os.path.join(
        OUTPUT_PATH,
        RISK_NAME_3D,
        f"{DISTANCE_NAME}.h5",
    )

    interpolator = load_risk_interpolator(
        h5_path,
        preload=False,
    )

    try:

        points = np.array(
            [
                TARGET,
                TARGET + [100.0, 0.0, -100.0],
                TARGET + [250.0, 250.0, -250.0],
            ],
            dtype=np.float32,
        )

        distances = interpolator(
            points
        )

        print("\nDistance-to-risk evaluation:")

        for point, distance in zip(
            points,
            distances,
        ):

            print(
                f"  Point [{point[0]:.1f}, "
                f"{point[1]:.1f}, "
                f"{point[2]:.1f}]"
                f" -> distance = {distance:.2f} m"
            )

    finally:

        interpolator.close()


def evaluate_distance_along_trajectory():
    """Evaluate the distance to the risk volume along a synthetic trajectory."""

    h5_path = os.path.join(
        OUTPUT_PATH,
        RISK_NAME_3D,
        f"{DISTANCE_NAME}.h5",
    )

    interpolator = load_risk_interpolator(
        h5_path,
        preload=False,
    )

    try:

        n_points = 100

        trajectory = np.column_stack(
            (
                np.linspace(
                    TARGET[0] - 400.0,
                    TARGET[0] + 400.0,
                    n_points,
                ),
                np.linspace(
                    TARGET[1] - 300.0,
                    TARGET[1] + 300.0,
                    n_points,
                ),
                np.linspace(
                    TARGET[2],
                    TARGET[2] - 700.0,
                    n_points,
                ),
            )
        ).astype(np.float32)

        distances = interpolator(
            trajectory
        )

        print(
            "\nDistance-to-risk trajectory evaluation:"
        )

        print(
            f"  Number of trajectory points: "
            f"{len(trajectory)}"
        )

        print(
            f"  Minimum distance: "
            f"{np.min(distances):.2f} m"
        )

        print(
            f"  Maximum distance: "
            f"{np.max(distances):.2f} m"
        )

        print(
            f"  Mean distance: "
            f"{np.mean(distances):.2f} m"
        )

    finally:

        interpolator.close()


def evaluate_heuristic_points():
    """Evaluate the heuristic sum and squared-sum representations."""

    sum_h5_path = os.path.join(
        OUTPUT_PATH,
        RISK_NAME_3D,
        f"{HEURISTIC_SUM_NAME}.h5",
    )

    sum_sq_h5_path = os.path.join(
        OUTPUT_PATH,
        RISK_NAME_3D,
        f"{HEURISTIC_SUM_SQ_NAME}.h5",
    )

    sum_interpolator = load_risk_interpolator(
        sum_h5_path,
        preload=False,
    )

    sum_sq_interpolator = load_risk_interpolator(
        sum_sq_h5_path,
        preload=False,
    )

    try:

        points = np.array(
            [
                TARGET,
                TARGET + [50.0, 50.0, -50.0],
                TARGET + [-100.0, 100.0, -100.0],
            ],
            dtype=np.float32,
        )

        sum_values = sum_interpolator(
            points
        )

        sum_sq_values = sum_sq_interpolator(
            points
        )

        print(
            "\nHeuristic representation evaluation:"
        )

        for (
            point,
            sum_value,
            sum_sq_value,
        ) in zip(
            points,
            sum_values,
            sum_sq_values,
        ):

            print(
                f"  Point [{point[0]:.1f}, "
                f"{point[1]:.1f}, "
                f"{point[2]:.1f}]"
                f" -> sum = {sum_value:.6f},"
                f" sum_sq = {sum_sq_value:.6f}"
            )

    finally:

        sum_interpolator.close()
        sum_sq_interpolator.close()


def evaluate_heuristic_along_trajectory():
    """Evaluate heuristic representations along a trajectory."""

    sum_h5_path = os.path.join(
        OUTPUT_PATH,
        RISK_NAME_3D,
        f"{HEURISTIC_SUM_NAME}.h5",
    )

    sum_sq_h5_path = os.path.join(
        OUTPUT_PATH,
        RISK_NAME_3D,
        f"{HEURISTIC_SUM_SQ_NAME}.h5",
    )

    sum_interpolator = load_risk_interpolator(
        sum_h5_path,
        preload=False,
    )

    sum_sq_interpolator = load_risk_interpolator(
        sum_sq_h5_path,
        preload=False,
    )

    try:

        n_points = 100

        trajectory = np.column_stack(
            (
                np.linspace(
                    TARGET[0] - 200.0,
                    TARGET[0] + 200.0,
                    n_points,
                ),
                np.linspace(
                    TARGET[1] - 150.0,
                    TARGET[1] + 150.0,
                    n_points,
                ),
                np.linspace(
                    TARGET[2],
                    TARGET[2] - 500.0,
                    n_points,
                ),
            )
        ).astype(np.float32)

        sum_values = sum_interpolator(
            trajectory
        )

        sum_sq_values = sum_sq_interpolator(
            trajectory
        )

        print(
            "\nHeuristic trajectory evaluation:"
        )

        print(
            f"  Number of trajectory points: "
            f"{len(trajectory)}"
        )

        print(
            f"  Sum - minimum: "
            f"{np.min(sum_values):.6f}"
        )

        print(
            f"  Sum - maximum: "
            f"{np.max(sum_values):.6f}"
        )

        print(
            f"  Sum - mean: "
            f"{np.mean(sum_values):.6f}"
        )

        print(
            f"  Sum squared - minimum: "
            f"{np.min(sum_sq_values):.6f}"
        )

        print(
            f"  Sum squared - maximum: "
            f"{np.max(sum_sq_values):.6f}"
        )

        print(
            f"  Sum squared - mean: "
            f"{np.mean(sum_sq_values):.6f}"
        )

    finally:

        sum_interpolator.close()
        sum_sq_interpolator.close()


def evaluate_combined_points():
    """Evaluate the combined 2D risk representation at selected XY coordinates."""

    h5_path = os.path.join(
        OUTPUT_PATH,
        COMBINED_NAME,
        f"{COMBINED_NAME}.h5",
    )

    interpolator = load_risk_interpolator(
        h5_path,
        preload=False,
    )

    try:

        points = np.array(
            [
                [TARGET[0], TARGET[1]],
                [
                    TARGET[0] + 100.0,
                    TARGET[1] + 100.0,
                ],
                [
                    TARGET[0] - 100.0,
                    TARGET[1] - 100.0,
                ],
            ],
            dtype=np.float32,
        )

        values = interpolator(
            points
        )

        print(
            "\nCombined risk evaluation:"
        )

        for point, value in zip(
            points,
            values,
        ):

            print(
                f"  Point [{point[0]:.1f}, "
                f"{point[1]:.1f}]"
                f" -> combined risk = {value:.6f}"
            )

    finally:

        interpolator.close()


def evaluate_combined_along_trajectory():
    """Evaluate the combined risk along a synthetic XY trajectory."""

    h5_path = os.path.join(
        OUTPUT_PATH,
        COMBINED_NAME,
        f"{COMBINED_NAME}.h5",
    )

    interpolator = load_risk_interpolator(
        h5_path,
        preload=False,
    )

    try:

        n_points = 100

        trajectory = np.column_stack(
            (
                np.linspace(
                    TARGET[0] - 400.0,
                    TARGET[0] + 400.0,
                    n_points,
                ),
                np.linspace(
                    TARGET[1] - 300.0,
                    TARGET[1] + 300.0,
                    n_points,
                ),
            )
        ).astype(np.float32)

        risks = interpolator(
            trajectory
        )

        print(
            "\nCombined risk trajectory evaluation:"
        )

        print(
            f"  Number of trajectory points: "
            f"{len(trajectory)}"
        )

        print(
            f"  Minimum combined risk: "
            f"{np.min(risks):.6f}"
        )

        print(
            f"  Maximum combined risk: "
            f"{np.max(risks):.6f}"
        )

        print(
            f"  Mean combined risk: "
            f"{np.mean(risks):.6f}"
        )

    finally:

        interpolator.close()


def main():
    print(
        "Evaluating synthetic geohazard representations..."
    )

    evaluate_3d_points()
    evaluate_trajectory()

    evaluate_2d_points()

    evaluate_distance_points()
    evaluate_distance_along_trajectory()

    evaluate_heuristic_points()
    evaluate_heuristic_along_trajectory()

    evaluate_combined_points()
    evaluate_combined_along_trajectory()

    print(
        "\nSynthetic geohazard evaluation completed."
    )


if __name__ == "__main__":
    main()