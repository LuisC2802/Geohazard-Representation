import os

import matplotlib.pyplot as plt
import numpy as np

from geohazard_interpolation import load_risk_interpolator


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_PATH = os.path.join(
    BASE_PATH,
    "examples",
    "output",
)

RISK_VOLUME_PATH = os.path.join(
    OUTPUT_PATH,
    "synthetic_risk",
    "synthetic_risk.h5",
)


# ---------------------------------------------------------------------------
# Point evaluation
# ---------------------------------------------------------------------------

def evaluate_arbitrary_points(risk_interpolator):
    """Evaluate the geohazard at arbitrary XYZ coordinates.

    Parameters
    ----------
    risk_interpolator : callable
        Geohazard interpolator returned by ``load_risk_interpolator``.

    Returns
    -------
    np.ndarray
        Geohazard values at the requested coordinates.
    """
    points = np.array([
        [1000.0, 2000.0, 600.0],
        [1010.0, 2010.0, 590.0],
        [1020.0, 2020.0, 580.0],
        [1030.0, 2030.0, 570.0],
        [1040.0, 2040.0, 560.0],
    ])

    risk_values = risk_interpolator(points)

    print()
    print("Geohazard evaluation at arbitrary coordinates:")
    print()

    for point, risk in zip(points, risk_values):
        print(
            f"  XYZ = ({point[0]:.1f}, "
            f"{point[1]:.1f}, "
            f"{point[2]:.1f}) "
            f"-> risk = {risk:.6f}"
        )

    return risk_values


# ---------------------------------------------------------------------------
# Trajectory evaluation
# ---------------------------------------------------------------------------

def evaluate_synthetic_trajectory(risk_interpolator):
    """Evaluate the geohazard along a synthetic well trajectory.

    Parameters
    ----------
    risk_interpolator : callable
        Geohazard interpolator returned by ``load_risk_interpolator``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Trajectory coordinates and corresponding geohazard values.
    """
    trajectory = np.array([
        [1000.0, 2000.0, 600.0],
        [1005.0, 2005.0, 590.0],
        [1010.0, 2010.0, 580.0],
        [1015.0, 2015.0, 570.0],
        [1020.0, 2020.0, 560.0],
        [1025.0, 2025.0, 550.0],
        [1030.0, 2030.0, 540.0],
        [1035.0, 2035.0, 530.0],
        [1040.0, 2040.0, 520.0],
        [1045.0, 2045.0, 510.0],
        [1050.0, 2050.0, 500.0],
    ])

    risk_values = risk_interpolator(trajectory)

    print()
    print("Geohazard evaluation along a synthetic trajectory:")
    print()

    for index, (point, risk) in enumerate(
        zip(trajectory, risk_values)
    ):
        print(
            f"  Point {index:02d}: "
            f"XYZ = ({point[0]:.1f}, "
            f"{point[1]:.1f}, "
            f"{point[2]:.1f}) "
            f"-> risk = {risk:.6f}"
        )

    return trajectory, risk_values


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_trajectory_risk(risk_values):
    """Plot the geohazard values along the trajectory.

    Parameters
    ----------
    risk_values : np.ndarray
        Geohazard values evaluated along the trajectory.

    Returns
    -------
    None
    """
    plt.figure(figsize=(8, 4))

    plt.plot(
        np.arange(len(risk_values)),
        risk_values,
        marker="o",
    )

    plt.xlabel("Trajectory point")
    plt.ylabel("Geohazard value")
    plt.title("Synthetic geohazard along a well trajectory")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the synthetic geohazard evaluation example."""
    if not os.path.exists(RISK_VOLUME_PATH):
        raise FileNotFoundError(
            "The synthetic HDF5 volume was not found.\n"
            "Run the following command first:\n\n"
            "    python examples/generate_synthetic_data.py"
        )

    print("Loading synthetic geohazard representation:")
    print(f"  {RISK_VOLUME_PATH}")

    risk_interpolator = load_risk_interpolator(
        RISK_VOLUME_PATH
    )

    try:
        evaluate_arbitrary_points(
            risk_interpolator
        )

        trajectory, risk_values = evaluate_synthetic_trajectory(
            risk_interpolator
        )

        plot_trajectory_risk(
            risk_values
        )

    finally:
        risk_interpolator.close()

    print()
    print("Synthetic geohazard evaluation completed.")


if __name__ == "__main__":
    main()
