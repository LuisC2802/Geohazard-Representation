import gc
import logging
import math
import os
from collections import OrderedDict

import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree


logger = logging.getLogger(__name__)


def read_prop_1d(filepath):
    """
    Read a one-dimensional property file.

    The property format supports both individual floating-point values
    and run-length encoded values using the ``N*VALUE`` notation.

    For example, the following input::

        3*5.0
        2.0
        4.0

    is interpreted as::

        [5.0, 5.0, 5.0, 2.0, 4.0]

    Args:
        filepath (str):
            Path to the property file.

    Returns:
        numpy.ndarray:
            One-dimensional array containing the decoded values.

    Raises:
        FileNotFoundError:
            If the property file does not exist.
        ValueError:
            If a value cannot be converted to a floating-point number.
    """
    values = []

    with open(filepath, "r") as file:
        for line in file:
            tokens = line.strip().split()

            for token in tokens:
                if "*" in token:
                    count, value = token.split("*")
                    values.extend([float(value)] * int(count))
                else:
                    values.append(float(token))

    return np.asarray(values)


def _get_chunks(shape):
    """
    Determine the default HDF5 chunk shape.

    The chunk sizes follow the configuration used by the original
    implementation.

    Args:
        shape (tuple):
            Dataset shape.

    Returns:
        tuple or None:
            Chunk dimensions suitable for the given dataset shape.
    """
    if len(shape) == 3:
        return (
            min(64, shape[0]),
            min(64, shape[1]),
            min(64, shape[2]),
        )

    if len(shape) == 2:
        return (
            min(128, shape[0]),
            min(128, shape[1]),
        )

    return None


def create_hdf5_dataset(
    output_path,
    shape,
    resolution,
    min_corner,
    dtype=np.float32,
    chunks=None,
    dataset_name=None,
):
    """
    Create a compressed HDF5 dataset and store spatial metadata.

    The dataset is created with gzip compression and spatial metadata
    describing the regular grid.

    Args:
        output_path (str):
            Destination HDF5 file.
        shape (tuple):
            Dataset dimensions.
        resolution (array-like):
            Voxel resolution along each spatial axis.
        min_corner (array-like):
            World-coordinate position of the minimum grid corner.
        dtype (numpy.dtype, optional):
            Dataset data type. Defaults to ``numpy.float32``.
        chunks (tuple, optional):
            HDF5 chunk dimensions. If omitted, they are determined
            automatically by ``_get_chunks``.
        dataset_name (str, optional):
            HDF5 dataset name. If omitted, the file name without the
            extension is used.

    Returns:
        tuple:
            ``(h5py.File, h5py.Dataset)`` containing the open file and
            created dataset.
    """
    output_directory = os.path.dirname(output_path)

    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    if dataset_name is None:
        dataset_name = os.path.splitext(
            os.path.basename(output_path)
        )[0]

    if chunks is None:
        chunks = _get_chunks(shape)

    file_handle = h5py.File(output_path, "w")

    dataset = file_handle.create_dataset(
        dataset_name,
        shape=shape,
        dtype=dtype,
        compression="gzip",
        chunks=chunks,
    )

    dataset.attrs["resolution"] = np.asarray(
        resolution,
        dtype=np.float32,
    )

    dataset.attrs["min_corner"] = np.asarray(
        min_corner,
        dtype=np.float32,
    )

    dataset.attrs["shape"] = shape

    return file_handle, dataset


def save_hdf5(
    volume,
    output_path,
    resolution=None,
    min_corner=None,
    dataset_name=None,
    extra_attrs=None,
):
    """
    Save an in-memory volume to a compressed HDF5 dataset.

    This function is intended for relatively small arrays that can
    safely be held in memory. Large spatial volumes should instead be
    generated directly into HDF5 using the streaming generation
    functions.

    Args:
        volume (numpy.ndarray):
            Volume or image to store.
        output_path (str):
            Destination HDF5 file.
        resolution (array-like, optional):
            Spatial resolution. Defaults to ``(2, 2, 2)``.
        min_corner (array-like, optional):
            Minimum spatial coordinate. Defaults to ``(0, 0, 0)``.
        dataset_name (str, optional):
            Name of the HDF5 dataset.
        extra_attrs (dict, optional):
            Additional HDF5 attributes to store.

    Returns:
        None
    """
    volume = np.asarray(volume)

    if resolution is None:
        resolution = (2.0, 2.0, 2.0)

    if min_corner is None:
        min_corner = (0.0, 0.0, 0.0)

    if dataset_name is None:
        dataset_name = os.path.splitext(
            os.path.basename(output_path)
        )[0]

    chunks = _get_chunks(volume.shape)

    file_handle, dataset = create_hdf5_dataset(
        output_path=output_path,
        shape=volume.shape,
        resolution=resolution,
        min_corner=min_corner,
        dtype=np.float32,
        chunks=chunks,
        dataset_name=dataset_name,
    )

    try:
        dataset[:] = volume

        if extra_attrs:
            for key, value in extra_attrs.items():
                dataset.attrs[key] = value

        file_handle.flush()

    finally:
        file_handle.close()


def load_hdf5_lazy(h5_path):
    """
    Open an HDF5 spatial dataset without loading its data into memory.

    The returned dataset remains valid while the returned HDF5 file
    handle remains open.

    Args:
        h5_path (str):
            Path to the HDF5 file.

    Returns:
        tuple:
            ``(dataset, metadata, file_handle)`` where:

            - ``dataset`` is the lazy HDF5 dataset;
            - ``metadata`` contains resolution, minimum corner and shape;
            - ``file_handle`` must remain open while the dataset is used.
    """
    dataset_name = os.path.splitext(
        os.path.basename(h5_path)
    )[0]

    file_handle = h5py.File(h5_path, "r")
    dataset = file_handle[dataset_name]

    metadata = {
        "resolution": np.asarray(
            dataset.attrs["resolution"]
        ),
        "min_corner": np.asarray(
            dataset.attrs["min_corner"]
        ),
        "shape": dataset.shape,
    }

    return dataset, metadata, file_handle


def load_hdf5_with_metadata(h5_path, load_data=True):
    """
    Load an HDF5 dataset together with its spatial metadata.

    When ``load_data`` is True, the complete dataset is loaded into
    memory. When it is False, only metadata is returned and the HDF5
    file is closed after the function returns.

    This function should therefore not be used for lazy access. Use
    ``load_hdf5_lazy`` when a live HDF5 dataset is required.

    Args:
        h5_path (str):
            Path to the HDF5 file.
        load_data (bool, optional):
            Whether to load the complete dataset into memory.
            Defaults to True.

    Returns:
        tuple:
            ``(data, metadata)``.

            If ``load_data`` is True, ``data`` is a NumPy array.
            Otherwise, ``data`` is None.

    Raises:
        KeyError:
            If the expected dataset is not present in the file.
    """
    dataset_name = os.path.splitext(
        os.path.basename(h5_path)
    )[0]

    with h5py.File(h5_path, "r") as file_handle:
        dataset = file_handle[dataset_name]

        metadata = {
            "resolution": np.asarray(
                dataset.attrs["resolution"]
            ),
            "min_corner": np.asarray(
                dataset.attrs["min_corner"]
            ),
            "shape": dataset.shape,
        }

        data = dataset[:] if load_data else None

    return data, metadata


def _normalize_values(values, limits):
    """
    Normalize values using the configured minimum and maximum limits.

    The operation reproduces the normalization used by the original
    WellPath implementation:

        normalized = (value - minimum) / (maximum - minimum)

    followed by clipping to the [0, 1] interval.

    Args:
        values (numpy.ndarray):
            Input values.
        limits (array-like):
            Two-element sequence containing minimum and maximum values.

    Returns:
        numpy.ndarray:
            Normalized values clipped to [0, 1].
    """
    min_value, max_value = limits
    denominator = max_value - min_value

    normalized = (
        values - min_value
    ) / denominator

    return np.clip(normalized, 0, 1)


def matriz_normalization_to_hdf5_thickness(
    dset_in,
    dset_out,
    slab_z=64,
):
    """
    Convert a thickness volume to a binary HDF5 volume.

    Each voxel is assigned 1 when its original value is greater than
    zero and 0 otherwise. Processing is performed slab by slab along
    the Z axis to avoid loading the complete volume into memory.

    Args:
        dset_in (h5py.Dataset):
            Input HDF5 dataset.
        dset_out (h5py.Dataset):
            Output HDF5 dataset.
        slab_z (int, optional):
            Number of Z slices processed per iteration.
            Defaults to 64.

    Returns:
        None
    """
    shape = dset_in.shape

    for z0 in range(0, shape[2], slab_z):
        z1 = min(z0 + slab_z, shape[2])

        slab = dset_in[:, :, z0:z1]
        slab = (slab > 0).astype(np.float32)

        dset_out[:, :, z0:z1] = slab

        del slab


def matriz_normalization_to_hdf5(
    dset_in,
    dset_out,
    limits,
    slab_z=64,
):
    """
    Normalize an HDF5 volume slab by slab.

    The values are normalized using the supplied minimum and maximum
    limits and clipped to the [0, 1] interval.

    Args:
        dset_in (h5py.Dataset):
            Input HDF5 dataset.
        dset_out (h5py.Dataset):
            Output HDF5 dataset.
        limits (array-like):
            Two-element sequence containing minimum and maximum values.
        slab_z (int, optional):
            Number of Z slices processed per iteration.
            Defaults to 64.

    Returns:
        None
    """
    shape = dset_in.shape

    for z0 in range(0, shape[2], slab_z):
        z1 = min(z0 + slab_z, shape[2])

        slab = dset_in[:, :, z0:z1]

        slab = _normalize_values(
            slab,
            limits,
        )

        dset_out[:, :, z0:z1] = slab

        del slab


def cumulative_sum_and_sum_sq_hdf5(
    dset_in,
    output_path,
    risk_folder,
    sum_file_name,
    sum_sq_file_name,
    z_min_real,
    z_max_real,
    resolution,
    start_target,
    direction="up",
    slab_z=64,
):
    """
    Generate cumulative sums and cumulative squared sums from an HDF5 volume.

    The calculation is performed slab by slab along the Z axis. The
    cumulative direction can be either upward or downward.

    This function reproduces the heuristic preprocessing used by the
    original implementation.

    Args:
        dset_in (h5py.Dataset):
            Input risk volume.
        output_path (str):
            Root output directory.
        risk_folder (str):
            Risk-specific output folder.
        sum_file_name (str):
            File name for the cumulative sum volume.
        sum_sq_file_name (str):
            File name for the cumulative squared sum volume.
        z_min_real (float):
            Minimum real Z coordinate of the active interval.
        z_max_real (float):
            Maximum real Z coordinate of the active interval.
        resolution (array-like):
            Spatial resolution.
        start_target (array-like):
            Target coordinates used to define the volume origin.
        direction (str, optional):
            Accumulation direction, either ``"up"`` or ``"down"``.
            Defaults to ``"up"``.
        slab_z (int, optional):
            Number of Z slices processed per iteration.
            Defaults to 64.

    Returns:
        None

    Raises:
        ValueError:
            If ``direction`` is neither ``"up"`` nor ``"down"``.
    """
    shape = dset_in.shape

    size_z = (
        shape[2]
        * resolution[2]
    )

    min_corner_z = (
        start_target[2]
        - size_z
    )

    z_idx_min = int(
        np.floor(
            (z_min_real - min_corner_z)
            / resolution[2]
        )
    )

    z_idx_max = int(
        np.floor(
            (z_max_real - min_corner_z)
            / resolution[2]
        )
    )

    z_idx_min = max(
        z_idx_min,
        0,
    )

    z_idx_max = min(
        z_idx_max,
        shape[2] - 1,
    )

    min_corner = dset_in.attrs["min_corner"]

    sum_path = os.path.join(
        output_path,
        risk_folder,
        sum_file_name,
    )

    sum_sq_path = os.path.join(
        output_path,
        risk_folder,
        sum_sq_file_name,
    )

    file_sum, dset_sum = create_hdf5_dataset(
        sum_path,
        shape,
        resolution,
        min_corner,
    )

    file_sq, dset_sq = create_hdf5_dataset(
        sum_sq_path,
        shape,
        resolution,
        min_corner,
    )

    running_sum = None
    running_sq = None

    if direction == "up":
        z_starts = range(
            z_idx_min,
            z_idx_max + 1,
            slab_z,
        )

    elif direction == "down":
        z_starts = range(
            z_idx_max,
            z_idx_min - 1,
            -slab_z,
        )

    else:
        file_sum.close()
        file_sq.close()

        raise ValueError(
            "direction must be 'up' or 'down'"
        )

    try:
        for z0 in z_starts:

            if direction == "up":
                z1 = min(
                    z0 + slab_z,
                    z_idx_max + 1,
                )

                slab = dset_in[
                    :,
                    :,
                    z0:z1,
                ]

            else:
                z1 = max(
                    z0 - slab_z + 1,
                    z_idx_min,
                )

                slab = dset_in[
                    :,
                    :,
                    z1:z0 + 1,
                ][:, :, ::-1]

            slab = np.asarray(
                slab,
                dtype=np.float32,
            )

            slab_cumsum = np.cumsum(
                slab,
                axis=2,
            )

            slab_sq_cumsum = np.cumsum(
                slab * slab,
                axis=2,
            )

            if running_sum is None:
                running_sum = slab_cumsum
                running_sq = slab_sq_cumsum

            else:
                running_sum = (
                    running_sum[:, :, -1:]
                    + slab_cumsum
                )

                running_sq = (
                    running_sq[:, :, -1:]
                    + slab_sq_cumsum
                )

            if direction == "down":
                write_sum = running_sum[:, :, ::-1]
                write_sq = running_sq[:, :, ::-1]

                write_z0 = z1
                write_z1 = z0 + 1

            else:
                write_sum = running_sum
                write_sq = running_sq

                write_z0 = z0
                write_z1 = z1

            dset_sum[
                :,
                :,
                write_z0:write_z1,
            ] = write_sum

            dset_sq[
                :,
                :,
                write_z0:write_z1,
            ] = write_sq

            del slab
            del slab_cumsum
            del slab_sq_cumsum

    finally:
        file_sum.close()
        file_sq.close()


def normalize_and_average_volumes(
    volumes_3d,
    limits_3d,
):
    """
    Load, normalize and average multiple 3D volumes.

    This function reproduces the original in-memory normalization
    workflow and is intended for datasets that can safely fit in memory.

    Args:
        volumes_3d (list[str]):
            Paths to HDF5 volumes.
        limits_3d (list[array-like]):
            Normalization limits corresponding to each volume.

    Returns:
        tuple:
            ``(normalized_volumes, average_volume)``.
    """
    if len(volumes_3d) != len(limits_3d):
        raise ValueError(
            "volumes_3d and limits_3d must have the same length"
        )

    normalized_volumes = []
    average_volume = None

    for path, limits in zip(
        volumes_3d,
        limits_3d,
    ):
        image, _ = load_hdf5_with_metadata(
            path,
            load_data=True,
        )

        normalized = _normalize_values(
            image,
            limits,
        )

        normalized_volumes.append(
            normalized
        )

        if average_volume is None:
            average_volume = normalized.astype(
                float
            )

        else:
            average_volume += normalized

    average_volume /= len(volumes_3d)

    return normalized_volumes, average_volume


def find_file_recursive(
    base_path,
    filename_to_find,
):
    """
    Recursively search for a file and return its containing directory.

    Args:
        base_path (str):
            Root directory where the search starts.
        filename_to_find (str):
            File name to search for.

    Returns:
        str or None:
            Directory containing the requested file, or None if the file
            cannot be found.
    """
    for root, _, files in os.walk(base_path):
        if filename_to_find in files:
            return root

    return None


def generate_2d_image(
    path,
    risk_name,
    output_path,
    resolution,
    start_target,
    water_depth_with_airgap,
    max_distance_to_wellhead,
):
    """
    Generate a regular 2D geohazard representation from property files.

    The input property points are first mapped to an extended regular
    grid. Missing grid cells are filled using the nearest available
    value determined by a Euclidean distance transform. The extended
    grid is then cropped to the target-centered domain used by the
    original implementation.

    The resulting image is stored as a three-dimensional HDF5 dataset
    with a singleton Z dimension: ``(X, Y, 1)``.

    Args:
        path (str):
            Directory containing ``x.prop``, ``y.prop``, ``z.prop`` and
            the requested risk property file.
        risk_name (str):
            Name of the risk property file without the ``.prop``
            extension.
        output_path (str):
            Root output directory.
        resolution (array-like):
            Spatial resolution. Only X and Y are used.
        start_target (array-like):
            Target coordinates. X and Y define the center of the image.
        water_depth_with_airgap (float):
            Z coordinate associated with the 2D hazard map.
        max_distance_to_wellhead (float):
            Half-width of the target-centered spatial domain.

    Returns:
        str:
            Path to the generated HDF5 file.
    """
    logger.info(
        "Starting 2D image generation for risk '%s'.",
        risk_name,
    )

    resolution = np.asarray(
        resolution,
        dtype=np.float32,
    )[:2]

    x_shape = int(
        2 * math.ceil(
            max_distance_to_wellhead
            / resolution[0]
        )
    )

    y_shape = int(
        2 * math.ceil(
            max_distance_to_wellhead
            / resolution[1]
        )
    )

    shape = (
        x_shape,
        y_shape,
    )

    target = np.asarray(
        [
            start_target[0],
            start_target[1],
            water_depth_with_airgap,
        ],
        dtype=np.float32,
    )

    half_size = (
        0.5
        * np.asarray(shape)
        * resolution
    )

    min_corner = (
        target[:2]
        - half_size
    )

    max_corner = (
        target[:2]
        + half_size
    )

    margin = half_size

    extended_min_corner = (
        min_corner
        - margin
    )

    extended_max_corner = (
        max_corner
        + margin
    )

    extended_shape = (
        int(
            np.ceil(
                (
                    extended_max_corner[0]
                    - extended_min_corner[0]
                )
                / resolution[0]
            )
        ),
        int(
            np.ceil(
                (
                    extended_max_corner[1]
                    - extended_min_corner[1]
                )
                / resolution[1]
            )
        ),
    )

    x_vals = read_prop_1d(
        os.path.join(path, "x.prop")
    )

    y_vals = read_prop_1d(
        os.path.join(path, "y.prop")
    )

    z_vals = read_prop_1d(
        os.path.join(path, "z.prop")
    )

    values = read_prop_1d(
        os.path.join(
            path,
            risk_name + ".prop",
        )
    )

    coords_xyz = np.vstack(
        [
            x_vals,
            y_vals,
            z_vals,
        ]
    ).T

    in_bounds_mask = np.all(
        (
            coords_xyz[:, :2]
            >= extended_min_corner
        )
        &
        (
            coords_xyz[:, :2]
            < extended_max_corner
        ),
        axis=1,
    )

    coords_inside = coords_xyz[
        in_bounds_mask
    ]

    values_inside = values[
        in_bounds_mask
    ]

    image_extended = np.full(
        extended_shape,
        np.nan,
        dtype=np.float32,
    )

    if len(coords_inside) > 0:

        indices = (
            (
                coords_inside[:, :2]
                - extended_min_corner
            )
            / resolution
        ).astype(int)

        valid_mask = (
            (indices[:, 0] >= 0)
            &
            (indices[:, 0] < extended_shape[0])
            &
            (indices[:, 1] >= 0)
            &
            (indices[:, 1] < extended_shape[1])
        )

        indices = indices[
            valid_mask
        ]

        values_inside = values_inside[
            valid_mask
        ]

        image_extended[
            indices[:, 0],
            indices[:, 1],
        ] = values_inside

    del (
        x_vals,
        y_vals,
        z_vals,
        values,
        coords_xyz,
        coords_inside,
        values_inside,
    )

    gc.collect()

    if np.all(
        np.isnan(image_extended)
    ):
        raise ValueError(
            f"Empty 2D image for risk '{risk_name}'."
        )

    nan_mask = np.isnan(
        image_extended
    )

    if np.any(nan_mask):

        _, indices = distance_transform_edt(
            nan_mask,
            return_indices=True,
        )

        image_extended[nan_mask] = (
            image_extended[
                indices[0][nan_mask],
                indices[1][nan_mask],
            ]
        )

    crop_start_x = int(
        round(
            (
                min_corner[0]
                - extended_min_corner[0]
            )
            / resolution[0]
        )
    )

    crop_start_y = int(
        round(
            (
                min_corner[1]
                - extended_min_corner[1]
            )
            / resolution[1]
        )
    )

    crop_end_x = (
        crop_start_x
        + shape[0]
    )

    crop_end_y = (
        crop_start_y
        + shape[1]
    )

    image_2d = image_extended[
        crop_start_x:crop_end_x,
        crop_start_y:crop_end_y,
    ]

    del image_extended
    gc.collect()

    image_2d = image_2d[
        :,
        :,
        np.newaxis,
    ]

    risk_directory = os.path.join(
        output_path,
        risk_name,
    )

    os.makedirs(
        risk_directory,
        exist_ok=True,
    )

    output_name = os.path.join(
        risk_directory,
        risk_name + ".h5",
    )

    tmp_path = output_name + ".tmp"

    file_handle = None
    success = False

    try:
        file_handle, dataset = create_hdf5_dataset(
            output_path=tmp_path,
            shape=image_2d.shape,
            resolution=resolution,
            min_corner=min_corner,
            dtype=np.float32,
            chunks=(
                min(64, image_2d.shape[0]),
                min(64, image_2d.shape[1]),
                1,
            ),
            dataset_name=risk_name,
        )

        dataset[:] = image_2d
        dataset.attrs["target"] = target

        file_handle.flush()
        success = True

    finally:
        if file_handle is not None:
            file_handle.close()

        if success:
            os.replace(
                tmp_path,
                output_name,
            )

        elif os.path.exists(tmp_path):
            os.remove(tmp_path)

    logger.info(
        "Finished 2D image generation for risk '%s'.",
        risk_name,
    )

    return output_name


def generate_interpolated_volume(
    path,
    risk_name,
    output_path,
    max_distance,
    resolution,
    start_target,
    water_depth_with_airgap,
    max_distance_to_wellhead,
):
    """
    Generate a regular 3D geohazard volume using nearest-neighbor interpolation.

    The input property points are stored in a KD-tree and queried for
    every voxel of the target-centered regular grid. Processing is
    performed by Z slabs to avoid creating the complete voxel-coordinate
    array in memory.

    A padded temporary volume is generated first. The final volume is
    then cropped to the target-centered domain.

    Args:
        path (str):
            Directory containing the coordinate property files and risk
            property file.
        risk_name (str):
            Name of the risk property file without the ``.prop``
            extension.
        output_path (str):
            Root output directory.
        max_distance (float):
            Padding distance in meters used to extend the domain before
            nearest-neighbor interpolation.
        resolution (array-like):
            Spatial resolution along X, Y and Z.
        start_target (array-like):
            Target coordinates defining the center and upper Z boundary
            of the volume.
        water_depth_with_airgap (float):
            Upper Z boundary of the generated volume.
        max_distance_to_wellhead (float):
            Half-width of the target-centered X-Y domain.

    Returns:
        str:
            Path to the final HDF5 volume.
    """
    logger.info(
        "Starting 3D interpolated volume generation for risk '%s'.",
        risk_name,
    )

    resolution = np.asarray(
        resolution,
        dtype=np.float32,
    )

    x_shape = (
        2
        * math.ceil(
            max_distance_to_wellhead
            / resolution[0]
        )
    )

    y_shape = (
        2
        * math.ceil(
            max_distance_to_wellhead
            / resolution[1]
        )
    )

    z_shape = math.ceil(
        (
            start_target[2]
            - water_depth_with_airgap
        )
        / resolution[2]
    )

    shape = (
        x_shape,
        y_shape,
        z_shape,
    )

    target = np.asarray(
        start_target,
        dtype=np.float32,
    )

    x_vals = read_prop_1d(
        os.path.join(path, "x.prop")
    )

    y_vals = read_prop_1d(
        os.path.join(path, "y.prop")
    )

    z_vals = read_prop_1d(
        os.path.join(path, "z.prop")
    )

    values = read_prop_1d(
        os.path.join(
            path,
            risk_name + ".prop",
        )
    ).astype(np.float32)

    coords_xyz = np.vstack(
        (
            x_vals,
            y_vals,
            z_vals,
        )
    ).T.astype(np.float32)

    del x_vals, y_vals, z_vals
    gc.collect()

    padding_voxels = np.ceil(
        max_distance / resolution
    ).astype(int)

    half_size_xy = (
        0.5
        * np.asarray(shape[:2])
        * resolution[:2]
    )

    size_z = (
        shape[2]
        * resolution[2]
    )

    min_corner_original = np.asarray(
        [
            target[0] - half_size_xy[0],
            target[1] - half_size_xy[1],
            target[2] - size_z,
        ],
        dtype=np.float32,
    )

    min_corner_padded = (
        min_corner_original
        - padding_voxels * resolution
    )

    shape_padded = tuple(
        shape[i]
        + 2 * padding_voxels[i]
        for i in range(3)
    )

    tree = cKDTree(
        coords_xyz
    )

    del coords_xyz
    gc.collect()

    chunk_size = (
        128,
        128,
        32,
    )

    risk_directory = os.path.join(
        output_path,
        risk_name,
    )

    os.makedirs(
        risk_directory,
        exist_ok=True,
    )

    output_path_orig = os.path.join(
        risk_directory,
        f"{risk_name}_orig.h5",
    )

    file_handle, dataset = create_hdf5_dataset(
        output_path=output_path_orig,
        shape=shape_padded,
        resolution=resolution,
        min_corner=min_corner_padded,
        chunks=chunk_size,
        dataset_name=f"{risk_name}_orig",
    )

    try:
        nx = shape_padded[0]
        ny = shape_padded[1]

        for z0 in range(
            0,
            shape_padded[2],
            chunk_size[2],
        ):
            z1 = min(
                z0 + chunk_size[2],
                shape_padded[2],
            )

            depth = z1 - z0

            xs = np.repeat(
                np.arange(nx),
                ny * depth,
            )

            ys = np.tile(
                np.repeat(
                    np.arange(ny),
                    depth,
                ),
                nx,
            )

            zs = np.tile(
                np.arange(z0, z1),
                nx * ny,
            )

            world_x = (
                min_corner_padded[0]
                + xs * resolution[0]
            )

            world_y = (
                min_corner_padded[1]
                + ys * resolution[1]
            )

            world_z = (
                min_corner_padded[2]
                + zs * resolution[2]
            )

            points = np.column_stack(
                (
                    world_x,
                    world_y,
                    world_z,
                )
            ).astype(np.float32)

            _, indices = tree.query(
                points,
                k=1,
                workers=-1,
            )

            slab = values[
                indices
            ].reshape(
                nx,
                ny,
                depth,
            )

            dataset[
                :,
                :,
                z0:z1,
            ] = slab.astype(
                np.float32
            )

            del (
                xs,
                ys,
                zs,
                world_x,
                world_y,
                world_z,
                points,
                indices,
                slab,
            )

    finally:
        file_handle.close()

    x0, y0, z0 = padding_voxels

    x1 = x0 + shape[0]
    y1 = y0 + shape[1]
    z1 = z0 + shape[2]

    output_path_h5 = os.path.join(
        risk_directory,
        f"{risk_name}.h5",
    )

    dataset_in, _, file_in = load_hdf5_lazy(
        output_path_orig
    )

    file_out, dataset_out = create_hdf5_dataset(
        output_path=output_path_h5,
        shape=shape,
        resolution=resolution,
        min_corner=min_corner_original,
        chunks=chunk_size,
        dataset_name=risk_name,
    )

    try:
        for z in range(
            z0,
            z1,
            chunk_size[2],
        ):
            z_slice = slice(
                z,
                min(
                    z + chunk_size[2],
                    z1,
                ),
            )

            subvolume = dataset_in[
                x0:x1,
                y0:y1,
                z_slice,
            ]

            dataset_out[
                :,
                :,
                z_slice.start - z0:
                z_slice.stop - z0,
            ] = subvolume

        file_out.flush()

    finally:
        file_in.close()
        file_out.close()

    logger.info(
        "Finished 3D interpolated volume generation for risk '%s'.",
        risk_name,
    )

    return output_path_h5


def generate_distance_volume(
    path,
    risk_name,
    output_path,
    max_distance,
    resolution,
    start_target,
    water_depth_with_airgap,
    max_distance_to_wellhead,
):
    """
    Generate a regular 3D volume containing distance to risk points.

    Only input points with finite positive risk values are considered
    risk points. For every voxel in the target-centered volume, the
    distance to the nearest risk point is calculated using a KD-tree.

    Voxels without a risk point within ``max_distance`` are assigned
    ``max_distance + 1``, reproducing the original implementation.

    Args:
        path (str):
            Directory containing the coordinate property files and risk
            property file.
        risk_name (str):
            Name of the risk property file without the ``.prop``
            extension.
        output_path (str):
            Root output directory.
        max_distance (float):
            Maximum distance considered when querying the KD-tree.
        resolution (array-like):
            Spatial resolution along X, Y and Z.
        start_target (array-like):
            Target coordinates.
        water_depth_with_airgap (float):
            Upper Z boundary of the generated volume.
        max_distance_to_wellhead (float):
            Half-width of the target-centered X-Y domain.

    Returns:
        str:
            Path to the generated HDF5 distance volume.

    Raises:
        ValueError:
            If no valid positive risk points are available.
    """
    logger.info(
        "Starting distance volume generation for risk '%s'.",
        risk_name,
    )

    resolution = np.asarray(
        resolution,
        dtype=np.float32,
    )

    x_shape = (
        2
        * math.ceil(
            max_distance_to_wellhead
            / resolution[0]
        )
    )

    y_shape = (
        2
        * math.ceil(
            max_distance_to_wellhead
            / resolution[1]
        )
    )

    z_shape = math.ceil(
        (
            start_target[2]
            - water_depth_with_airgap
        )
        / resolution[2]
    )

    shape = (
        x_shape,
        y_shape,
        z_shape,
    )

    target = np.asarray(
        start_target,
        dtype=np.float32,
    )

    half_size_xy = (
        0.5
        * np.asarray(shape[:2])
        * resolution[:2]
    )

    size_z = (
        shape[2]
        * resolution[2]
    )

    min_corner = np.asarray(
        [
            target[0] - half_size_xy[0],
            target[1] - half_size_xy[1],
            target[2] - size_z,
        ],
        dtype=np.float32,
    )

    x_vals = read_prop_1d(
        os.path.join(path, "x.prop")
    )

    y_vals = read_prop_1d(
        os.path.join(path, "y.prop")
    )

    z_vals = read_prop_1d(
        os.path.join(path, "z.prop")
    )

    values = read_prop_1d(
        os.path.join(
            path,
            risk_name + ".prop",
        )
    )

    coords_xyz = np.vstack(
        [
            x_vals,
            y_vals,
            z_vals,
        ]
    ).T.astype(np.float32)

    del x_vals, y_vals, z_vals
    gc.collect()

    risk_mask = (
        np.isfinite(values)
        & (values > 0)
    )

    risk_points = coords_xyz[
        risk_mask
    ]

    if len(risk_points) == 0:
        raise ValueError(
            f"No valid risk points found for '{risk_name}'."
        )

    del coords_xyz
    del values
    gc.collect()

    tree = cKDTree(
        risk_points
    )

    chunk_size = (
        128,
        128,
        64,
    )

    output_path_h5 = os.path.join(
        output_path,
        risk_name,
        f"{risk_name}_distance.h5",
    )

    file_out, dataset_out = create_hdf5_dataset(
        output_path=output_path_h5,
        shape=shape,
        resolution=resolution,
        min_corner=min_corner,
        chunks=chunk_size,
        dataset_name=f"{risk_name}_distance",
    )

    try:
        nx, ny, nz = shape

        for z0 in range(
            0,
            nz,
            chunk_size[2],
        ):
            z1 = min(
                z0 + chunk_size[2],
                nz,
            )

            depth = z1 - z0

            xs = np.repeat(
                np.arange(nx),
                ny * depth,
            )

            ys = np.tile(
                np.repeat(
                    np.arange(ny),
                    depth,
                ),
                nx,
            )

            zs = np.tile(
                np.arange(z0, z1),
                nx * ny,
            )

            voxel_indices = np.column_stack(
                [
                    xs,
                    ys,
                    zs,
                ]
            )

            voxel_coordinates = (
                min_corner
                + voxel_indices * resolution
            )

            distances, _ = tree.query(
                voxel_coordinates,
                k=1,
                distance_upper_bound=max_distance,
            )

            distances = distances.astype(
                np.float32
            )

            distances[
                ~np.isfinite(distances)
            ] = (
                max_distance + 1
            )

            slab = distances.reshape(
                nx,
                ny,
                depth,
            )

            dataset_out[
                :,
                :,
                z0:z1,
            ] = slab

            del (
                xs,
                ys,
                zs,
                voxel_indices,
                voxel_coordinates,
                distances,
                slab,
            )

        dataset_out.attrs[
            "max_distance"
        ] = max_distance

        file_out.flush()

    finally:
        file_out.close()

    logger.info(
        "Finished distance volume generation for risk '%s'.",
        risk_name,
    )

    return output_path_h5


def combine_images_to_hdf5(
    output_path,
    images_2d,
    limits_2d,
    volumes_3d,
    limits_3d,
    check_type_3d,
    z_max_real,
    min_corner_real,
    resolution,
    chunk_size=64,
):
    """
    Combine normalized 2D and 3D hazard representations into one HDF5 dataset.

    The combination reproduces the original WellPath logic:

    - 2D hazards are normalized directly.
    - Thickness volumes are summed along Z before normalization.
    - Distance volumes are normalized voxel-wise; if any voxel along Z
      reaches normalized value 1, the resulting value is 1, otherwise
      the mean along Z is used.
    - The final value is 1 wherever any hazard reaches 1; otherwise the
      mean of the available normalized hazard layers is used.

    Args:
        output_path (str):
            Destination HDF5 file.
        images_2d (list[str]):
            Paths to 2D HDF5 hazard representations.
        limits_2d (list[array-like]):
            Normalization limits for the 2D representations.
        volumes_3d (list[str]):
            Paths to 3D HDF5 hazard representations.
        limits_3d (list[array-like]):
            Normalization limits for the 3D representations.
        check_type_3d (list[str]):
            Hazard types corresponding to ``volumes_3d``.
        z_max_real (float):
            Maximum real Z coordinate used by the combination.
        min_corner_real (array-like):
            Minimum corner of the reference volume.
        resolution (array-like):
            Spatial resolution.
        chunk_size (int, optional):
            X-Y processing chunk size. Defaults to 64.

    Returns:
        str:
            Path to the combined HDF5 dataset.

    Raises:
        ValueError:
            If no 2D or 3D input volumes are provided.
    """
    if images_2d:
        reference_path = images_2d[0]
        is_2d = True

    elif volumes_3d:
        reference_path = volumes_3d[0]
        is_2d = False

    else:
        raise ValueError(
            "No input data (2D or 3D)."
        )

    reference_dset, reference_desc, reference_file = (
        load_hdf5_lazy(
            reference_path
        )
    )

    resolution = reference_desc[
        "resolution"
    ]

    if is_2d:
        shape = reference_dset.shape

    else:
        shape = (
            reference_dset.shape[0],
            reference_dset.shape[1],
            1,
        )

    output_min_corner = np.asarray(
        min_corner_real[:2]
    )

    file_out, dataset_out = create_hdf5_dataset(
        output_path=output_path,
        shape=shape,
        resolution=resolution,
        min_corner=output_min_corner,
        dtype=np.float32,
        chunks=(
            min(chunk_size, shape[0]),
            min(chunk_size, shape[1]),
            1,
        ),
        dataset_name="combined_risk",
    )

    min_corner_z = min_corner_real[2]

    z_idx_max = int(
        np.floor(
            (
                z_max_real
                - min_corner_z
            )
            / resolution[2]
        )
    )

    if volumes_3d:
        temporary_dset, _, temporary_file = (
            load_hdf5_lazy(
                volumes_3d[0]
            )
        )

        z_idx_max = min(
            z_idx_max,
            temporary_dset.shape[2] - 1,
        )

        temporary_file.close()

    if z_idx_max < 0:
        reference_file.close()
        file_out.close()

        raise ValueError(
            "Invalid z_max_real."
        )

    dsets_2d = []
    files_2d = []

    dsets_3d = []
    files_3d = []

    try:
        for path in images_2d:
            dataset, _, file_handle = (
                load_hdf5_lazy(path)
            )

            dsets_2d.append(dataset)
            files_2d.append(file_handle)

        for path in volumes_3d:
            dataset, _, file_handle = (
                load_hdf5_lazy(path)
            )

            dsets_3d.append(dataset)
            files_3d.append(file_handle)

        for i in range(
            0,
            shape[0],
            chunk_size,
        ):
            for j in range(
                0,
                shape[1],
                chunk_size,
            ):
                i_end = min(
                    i + chunk_size,
                    shape[0],
                )

                j_end = min(
                    j + chunk_size,
                    shape[1],
                )

                normalized_layers = []

                for dataset, limits in zip(
                    dsets_2d,
                    limits_2d,
                ):
                    chunk = dataset[
                        i:i_end,
                        j:j_end,
                        0,
                    ]

                    normalized = _normalize_values(
                        chunk,
                        limits,
                    )

                    normalized_layers.append(
                        normalized
                    )

                for (
                    dataset,
                    limits,
                    hazard_type,
                ) in zip(
                    dsets_3d,
                    limits_3d,
                    check_type_3d,
                ):
                    subvolume = dataset[
                        i:i_end,
                        j:j_end,
                        :z_idx_max + 1,
                    ]

                    if hazard_type == "Thickness":
                        thickness = np.sum(
                            subvolume,
                            axis=2,
                        )

                        normalized = _normalize_values(
                            thickness,
                            limits,
                        )

                    elif hazard_type == "Distance":
                        normalized_volume = (
                            _normalize_values(
                                subvolume,
                                limits,
                            )
                        )

                        mask = np.any(
                            normalized_volume == 1,
                            axis=2,
                        )

                        mean_values = np.mean(
                            normalized_volume,
                            axis=2,
                        )

                        normalized = np.where(
                            mask,
                            1,
                            mean_values,
                        )

                    else:
                        continue

                    normalized_layers.append(
                        normalized
                    )

                if not normalized_layers:
                    raise ValueError(
                        "No hazard layers available for combination."
                    )

                stacked = np.stack(
                    normalized_layers
                )

                mask_ones = np.any(
                    stacked == 1,
                    axis=0,
                )

                mean_values = np.mean(
                    stacked,
                    axis=0,
                )

                final_chunk = np.where(
                    mask_ones,
                    1,
                    mean_values,
                )

                dataset_out[
                    i:i_end,
                    j:j_end,
                    0,
                ] = final_chunk

        file_out.flush()

    finally:
        for file_handle in files_2d:
            file_handle.close()

        for file_handle in files_3d:
            file_handle.close()

        file_out.close()
        reference_file.close()

    return output_path


def _load_local_subvolume_2d(
    dset,
    points,
    resolution,
    min_corner,
    margin=2,
    preload=False,
):
    """
    Load the smallest 2D HDF5 subvolume containing the requested points.

    Args:
        dset (h5py.Dataset):
            HDF5 dataset.
        points (array-like):
            XY or XYZ coordinates to evaluate.
        resolution (array-like):
            Spatial resolution.
        min_corner (array-like):
            Minimum grid coordinate.
        margin (int, optional):
            Number of voxels added around the requested region.
            Defaults to 2.
        preload (bool, optional):
            If True, return the complete dataset. Defaults to False.

    Returns:
        tuple:
            ``(subvolume, (x0, y0))``.
    """
    if preload:
        return dset, (0, 0)

    points = np.asarray(
        points
    )

    fx = (
        points[:, 0]
        - min_corner[0]
    ) / resolution[0]

    fy = (
        points[:, 1]
        - min_corner[1]
    ) / resolution[1]

    ix = np.floor(
        fx
    ).astype(np.int64)

    iy = np.floor(
        fy
    ).astype(np.int64)

    x0 = max(
        ix.min() - margin,
        0,
    )

    y0 = max(
        iy.min() - margin,
        0,
    )

    x1 = min(
        ix.max() + margin + 2,
        dset.shape[0],
    )

    y1 = min(
        iy.max() + margin + 2,
        dset.shape[1],
    )

    subvolume = dset[
        x0:x1,
        y0:y1,
    ]

    return subvolume, (x0, y0)


def _load_local_subvolume_3d(
    dset,
    points,
    resolution,
    min_corner,
    azimuth=None,
    margin=2,
    slice_cache=None,
    max_cached_azimuths=5,
    preload=False,
    start_target=None,
    max_distance_to_wellhead=None,
):
    """
    Load a local 3D HDF5 subvolume for interpolation.

    When no azimuth is supplied, only the bounding region containing
    the requested points is loaded. When an azimuth is supplied, an LRU
    cache stores a target-centered slice region associated with the
    azimuth, reproducing the original optimization used by the
    interpolator.

    Args:
        dset (h5py.Dataset):
            HDF5 dataset.
        points (array-like):
            XYZ coordinates.
        resolution (array-like):
            Spatial resolution.
        min_corner (array-like):
            Minimum grid coordinate.
        azimuth (float, optional):
            Azimuth used by the cached radial mode.
        margin (int, optional):
            Number of voxels added around the requested region.
            Defaults to 2.
        slice_cache (OrderedDict, optional):
            LRU cache for azimuth-specific subvolumes.
        max_cached_azimuths (int, optional):
            Maximum number of cached azimuths. Defaults to 5.
        preload (bool, optional):
            If True, return the complete dataset.
        start_target (array-like, optional):
            Target coordinates required by azimuth mode.
        max_distance_to_wellhead (float, optional):
            Maximum radial distance required by azimuth mode.

    Returns:
        tuple:
            ``(subvolume, (x0, y0, z0))``.

    Raises:
        RuntimeError:
            If the azimuth-specific ray contains no valid voxels.
        ValueError:
            If azimuth mode is requested without the required target
            or maximum-distance parameters.
    """
    points = np.asarray(
        points
    )

    if preload:
        return dset, (0, 0, 0)

    if azimuth is None:

        fx = (
            points[:, 0]
            - min_corner[0]
        ) / resolution[0]

        fy = (
            points[:, 1]
            - min_corner[1]
        ) / resolution[1]

        fz = (
            points[:, 2]
            - min_corner[2]
        ) / resolution[2]

        ix = np.floor(
            fx
        ).astype(np.int64)

        iy = np.floor(
            fy
        ).astype(np.int64)

        iz = np.floor(
            fz
        ).astype(np.int64)

        x0 = max(
            ix.min() - margin,
            0,
        )

        y0 = max(
            iy.min() - margin,
            0,
        )

        z0 = max(
            iz.min() - margin,
            0,
        )

        x1 = min(
            ix.max() + margin + 2,
            dset.shape[0],
        )

        y1 = min(
            iy.max() + margin + 2,
            dset.shape[1],
        )

        z1 = min(
            iz.max() + margin + 2,
            dset.shape[2],
        )

        subvolume = dset[
            x0:x1,
            y0:y1,
            z0:z1,
        ]

        return subvolume, (x0, y0, z0)

    if start_target is None:
        raise ValueError(
            "start_target is required when azimuth is provided."
        )

    if max_distance_to_wellhead is None:
        raise ValueError(
            "max_distance_to_wellhead is required when azimuth is provided."
        )

    if slice_cache is None:
        slice_cache = OrderedDict()

    azimuth_key = int(
        round(azimuth)
    )

    if azimuth_key in slice_cache:
        slice_cache.move_to_end(
            azimuth_key
        )

    else:
        rx, ry, _ = resolution

        nx, ny, nz = dset.shape

        target = np.asarray(
            start_target
        )

        theta = np.deg2rad(
            azimuth
        )

        direction = np.array(
            [
                np.cos(theta),
                np.sin(theta),
            ]
        )

        distances = np.linspace(
            0,
            max_distance_to_wellhead,
            int(
                max_distance_to_wellhead / rx
            ) + 1,
        )

        xs = (
            target[0]
            + distances * direction[0]
        )

        ys = (
            target[1]
            + distances * direction[1]
        )

        ix = (
            (
                xs - min_corner[0]
            )
            / rx
        ).astype(
            np.int64
        )

        iy = (
            (
                ys - min_corner[1]
            )
            / ry
        ).astype(
            np.int64
        )

        valid = (
            (ix >= 0)
            & (ix < nx)
            & (iy >= 0)
            & (iy < ny)
        )

        ix = ix[valid]
        iy = iy[valid]

        if len(ix) == 0:
            raise RuntimeError(
                f"No valid voxels for azimuth {azimuth}."
            )

        x0_cache = max(
            ix.min() - margin,
            0,
        )

        x1_cache = min(
            ix.max() + margin + 2,
            nx,
        )

        y0_cache = max(
            iy.min() - margin,
            0,
        )

        y1_cache = min(
            iy.max() + margin + 2,
            ny,
        )

        z0_cache = 0
        z1_cache = nz

        subvolume_cache = dset[
            x0_cache:x1_cache,
            y0_cache:y1_cache,
            z0_cache:z1_cache,
        ]

        slice_cache[azimuth_key] = {
            "subvol": subvolume_cache,
            "x0": x0_cache,
            "y0": y0_cache,
            "z0": z0_cache,
        }

        while len(slice_cache) > max_cached_azimuths:
            slice_cache.popitem(
                last=False
            )

    cache = slice_cache[
        azimuth_key
    ]

    fx = (
        points[:, 0]
        - min_corner[0]
    ) / resolution[0]

    fy = (
        points[:, 1]
        - min_corner[1]
    ) / resolution[1]

    fz = (
        points[:, 2]
        - min_corner[2]
    ) / resolution[2]

    ix = np.floor(
        fx
    ).astype(np.int64)

    iy = np.floor(
        fy
    ).astype(np.int64)

    iz = np.floor(
        fz
    ).astype(np.int64)

    x0 = max(
        ix.min() - margin,
        cache["x0"],
    )

    y0 = max(
        iy.min() - margin,
        cache["y0"],
    )

    z0 = max(
        iz.min() - margin,
        cache["z0"],
    )

    x1 = min(
        ix.max() + margin + 2,
        cache["x0"]
        + cache["subvol"].shape[0],
    )

    y1 = min(
        iy.max() + margin + 2,
        cache["y0"]
        + cache["subvol"].shape[1],
    )

    z1 = min(
        iz.max() + margin + 2,
        cache["z0"]
        + cache["subvol"].shape[2],
    )

    subvolume = cache["subvol"][
        x0 - cache["x0"]:
        x1 - cache["x0"],
        y0 - cache["y0"]:
        y1 - cache["y0"],
        z0 - cache["z0"]:
        z1 - cache["z0"],
    ]

    return subvolume, (x0, y0, z0)


def create_interpolation_function_lazy(
    dset,
    resolution,
    min_corner,
    preload=False,
    start_target=None,
    max_distance_to_wellhead=None,
):
    """
    Create a lazy bilinear or trilinear interpolation function.

    The interpolation function operates directly on an HDF5 dataset.
    For 2D datasets, bilinear interpolation is used. For 3D datasets,
    trilinear interpolation is used.

    Coordinates outside the spatial domain are clamped to the nearest
    valid grid boundary, reproducing the behavior of the original
    implementation.

    For 3D data, the function can optionally use the original azimuth
    cache mode.

    Args:
        dset (h5py.Dataset):
            HDF5 dataset containing the spatial representation.
        resolution (array-like):
            Spatial resolution.
        min_corner (array-like):
            Minimum world-coordinate corner.
        preload (bool, optional):
            Whether to preload the full 3D dataset for interpolation.
            Defaults to False.
        start_target (array-like, optional):
            Target coordinates required for azimuth mode.
        max_distance_to_wellhead (float, optional):
            Maximum radial distance required for azimuth mode.

    Returns:
        callable:
            Function accepting ``points`` and optionally ``azimuth``.
            The returned array contains interpolated hazard values.
    """
    is_3d = (
        len(dset.shape) == 3
        and dset.shape[2] > 1
    )

    slice_cache = OrderedDict()

    def interp(points, azimuth=None):
        """
        Evaluate the interpolated hazard representation at coordinates.

        Args:
            points (array-like):
                Array of spatial coordinates with shape ``(N, 3)``.
            azimuth (float, optional):
                Azimuth used by the optional 3D cached interpolation mode.

        Returns:
            numpy.ndarray:
                Interpolated hazard values as float32.
        """
        if len(points) == 0:
            return np.array(
                [],
                dtype=np.float32,
            )

        points = np.asarray(
            points,
            dtype=np.float32,
        )

        if not is_3d:

            shape_2d = np.asarray(
                dset.shape[:2]
            )

            resolution_2d = np.asarray(
                resolution[:2]
            )

            min_corner_2d = np.asarray(
                min_corner[:2]
            )

            max_corner_2d = (
                min_corner_2d
                + shape_2d
                * resolution_2d
            )

            points_clamped = points.copy()

            points_clamped[:, 0] = np.clip(
                points[:, 0],
                min_corner_2d[0],
                max_corner_2d[0]
                - resolution_2d[0],
            )

            points_clamped[:, 1] = np.clip(
                points[:, 1],
                min_corner_2d[1],
                max_corner_2d[1]
                - resolution_2d[1],
            )

            points_xy = (
                points_clamped[:, :2]
            )

            subvolume, (
                x0,
                y0,
            ) = _load_local_subvolume_2d(
                dset,
                points_xy,
                resolution,
                min_corner,
                preload=True,
            )

            if subvolume.ndim == 3:
                subvolume = subvolume[
                    :,
                    :,
                    0,
                ]

            fx = (
                (
                    points_clamped[:, 0]
                    - min_corner[0]
                )
                / resolution[0]
                - x0
            )

            fy = (
                (
                    points_clamped[:, 1]
                    - min_corner[1]
                )
                / resolution[1]
                - y0
            )

            fx = np.clip(
                fx,
                0,
                subvolume.shape[0]
                - 1.0001,
            )

            fy = np.clip(
                fy,
                0,
                subvolume.shape[1]
                - 1.0001,
            )

            ix = np.floor(
                fx
            ).astype(np.int64)

            iy = np.floor(
                fy
            ).astype(np.int64)

            dx = fx - ix
            dy = fy - iy

            ix = np.clip(
                ix,
                0,
                subvolume.shape[0] - 2,
            )

            iy = np.clip(
                iy,
                0,
                subvolume.shape[1] - 2,
            )

            c00 = subvolume[
                ix,
                iy,
            ]

            c10 = subvolume[
                ix + 1,
                iy,
            ]

            c01 = subvolume[
                ix,
                iy + 1,
            ]

            c11 = subvolume[
                ix + 1,
                iy + 1,
            ]

            c0 = (
                c00 * (1 - dx)
                + c10 * dx
            )

            c1 = (
                c01 * (1 - dx)
                + c11 * dx
            )

            result = (
                c0 * (1 - dy)
                + c1 * dy
            )

            return result.astype(
                np.float32
            )

        shape_3d = np.asarray(
            dset.shape
        )

        resolution_3d = np.asarray(
            resolution
        )

        min_corner_3d = np.asarray(
            min_corner
        )

        max_corner_3d = (
            min_corner_3d
            + shape_3d
            * resolution_3d
        )

        points_clamped = points.copy()

        points_clamped[:, 0] = np.clip(
            points[:, 0],
            min_corner_3d[0],
            max_corner_3d[0]
            - resolution_3d[0],
        )

        points_clamped[:, 1] = np.clip(
            points[:, 1],
            min_corner_3d[1],
            max_corner_3d[1]
            - resolution_3d[1],
        )

        points_clamped[:, 2] = np.clip(
            points[:, 2],
            min_corner_3d[2],
            max_corner_3d[2]
            - resolution_3d[2],
        )

        subvolume, (
            x0,
            y0,
            z0,
        ) = _load_local_subvolume_3d(
            dset,
            points_clamped,
            resolution,
            min_corner,
            azimuth=azimuth,
            slice_cache=slice_cache,
            max_cached_azimuths=1,
            preload=preload,
            start_target=start_target,
            max_distance_to_wellhead=max_distance_to_wellhead,
        )

        if azimuth is None:

            fx = (
                (
                    points_clamped[:, 0]
                    - min_corner[0]
                )
                / resolution[0]
                - x0
            )

            fy = (
                (
                    points_clamped[:, 1]
                    - min_corner[1]
                )
                / resolution[1]
                - y0
            )

            fz = (
                (
                    points_clamped[:, 2]
                    - min_corner[2]
                )
                / resolution[2]
                - z0
            )

        else:

            theta = np.deg2rad(
                azimuth
            )

            direction = np.array(
                [
                    np.cos(theta),
                    np.sin(theta),
                ],
                dtype=np.float32,
            )

            target_xy = np.asarray(
                start_target[:2],
                dtype=np.float32,
            )

            relative_xy = (
                points_clamped[:, :2]
                - target_xy
            )

            projected_distance = (
                relative_xy
                @ direction
            )

            fx = (
                (
                    max_distance_to_wellhead
                    - projected_distance
                )
                / resolution[0]
                - x0
            )

            fy = np.zeros_like(
                fx
            )

            fz = (
                (
                    points_clamped[:, 2]
                    - min_corner[2]
                )
                / resolution[2]
                - z0
            )

        fx = np.clip(
            fx,
            0,
            subvolume.shape[0]
            - 1.0001,
        )

        fy = np.clip(
            fy,
            0,
            subvolume.shape[1]
            - 1.0001,
        )

        fz = np.clip(
            fz,
            0,
            subvolume.shape[2]
            - 1.0001,
        )

        ix = np.floor(
            fx
        ).astype(np.int64)

        iy = np.floor(
            fy
        ).astype(np.int64)

        iz = np.floor(
            fz
        ).astype(np.int64)

        dx = fx - ix
        dy = fy - iy
        dz = fz - iz

        ix = np.clip(
            ix,
            0,
            subvolume.shape[0] - 2,
        )

        iy = np.clip(
            iy,
            0,
            subvolume.shape[1] - 2,
        )

        iz = np.clip(
            iz,
            0,
            subvolume.shape[2] - 2,
        )

        c000 = subvolume[
            ix,
            iy,
            iz,
        ]

        c100 = subvolume[
            ix + 1,
            iy,
            iz,
        ]

        c010 = subvolume[
            ix,
            iy + 1,
            iz,
        ]

        c110 = subvolume[
            ix + 1,
            iy + 1,
            iz,
        ]

        c001 = subvolume[
            ix,
            iy,
            iz + 1,
        ]

        c101 = subvolume[
            ix + 1,
            iy,
            iz + 1,
        ]

        c011 = subvolume[
            ix,
            iy + 1,
            iz + 1,
        ]

        c111 = subvolume[
            ix + 1,
            iy + 1,
            iz + 1,
        ]

        c00 = (
            c000 * (1 - dx)
            + c100 * dx
        )

        c01 = (
            c001 * (1 - dx)
            + c101 * dx
        )

        c10 = (
            c010 * (1 - dx)
            + c110 * dx
        )

        c11 = (
            c011 * (1 - dx)
            + c111 * dx
        )

        c0 = (
            c00 * (1 - dy)
            + c10 * dy
        )

        c1 = (
            c01 * (1 - dy)
            + c11 * dy
        )

        result = (
            c0 * (1 - dz)
            + c1 * dz
        )

        return result.astype(
            np.float32
        )

    return interp


def load_risk_interpolator(
    h5_path,
    preload=False,
    start_target=None,
    max_distance_to_wellhead=None,
):
    """
    Load an HDF5 geohazard representation and create its interpolator.

    The returned object is a callable function. The underlying HDF5
    file remains open so that interpolation can be performed lazily
    without loading the complete volume into memory.

    The returned function exposes a ``close()`` method that must be
    called when the interpolator is no longer needed.

    Args:
        h5_path (str):
            Path to the HDF5 risk representation.
        preload (bool, optional):
            Whether the 3D representation should be preloaded during
            interpolation. Defaults to False.
        start_target (array-like, optional):
            Target coordinates required if azimuth-based interpolation
            is used.
        max_distance_to_wellhead (float, optional):
            Maximum radial distance required if azimuth-based
            interpolation is used.

    Returns:
        callable:
            Interpolation function accepting an array of coordinates.

            Example::

                interpolator = load_risk_interpolator(
                    "igneas.h5"
                )

                values = interpolator(points)

                interpolator.close()
    """
    dataset, metadata, file_handle = load_hdf5_lazy(
        h5_path
    )

    try:
        interpolator = create_interpolation_function_lazy(
            dataset,
            metadata["resolution"],
            metadata["min_corner"],
            preload=preload,
            start_target=start_target,
            max_distance_to_wellhead=max_distance_to_wellhead,
        )

    except Exception:
        file_handle.close()
        raise

    def close():
        """Close the HDF5 file associated with the interpolator."""
        if file_handle.id.valid:
            file_handle.close()

    interpolator.close = close
    interpolator.metadata = metadata
    interpolator.h5_path = h5_path

    return interpolator
