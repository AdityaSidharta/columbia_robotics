from numba import njit, prange
import numpy as np


def transform_is_valid(t, tolerance=1e-3):
    """Check if array is a valid transform.

    Args:
        t (numpy.array [4, 4]): Transform candidate.
        tolerance (float, optional): maximum absolute difference
            for two numbers to be considered close enough to each
            other. Defaults to 1e-3.

    Returns:
        bool: True if array is a valid transform else False.
    """
    r = t[:-1, :-1]
    d = t[:-1, -1]
    rest = t[-1, :]

    valid = True

    if not t.shape == (4,4):
        print('A')
        valid = False
    if not np.all(np.isreal(r)):
        print('B')
        valid = False
    if not np.all(np.isreal(d)):
        print('C')
        valid = False
    if not (np.all(rest[:-1] == 0) and rest[-1] == 1):
        print('D')
        valid = False
    if not np.all(np.isclose(r @ np.transpose(r), np.transpose(r) @ r)):
        print('E')
        valid = False
    if not np.all(np.isclose(r @ np.transpose(r), np.eye(r.shape[0]).astype(float))):
        print(r @ np.transpose(r))
        valid = False
    if not np.all(np.isclose(np.linalg.det(r), 1.)):
        print(np.linalg.det(r))
        valid = False
    return valid



def transform_concat(t1, t2):
    """[summary]

    Args:
        t1 (numpy.array [4, 4]): SE3 transform.
        t2 (numpy.array [4, 4]): SE3 transform.

    Raises:
        ValueError: t1 is invalid.
        ValueError: t2 is invalid.

    Returns:
        numpy.array [4, 4]: t1 * t2.
    """
    if not transform_is_valid(t1):
        raise ValueError('Invalid input transform t1')
    if not transform_is_valid(t2):
        raise ValueError('Invalid input transform t2')
    return t1 @ t2


def transform_point3s(t, ps):
    """Transform 3D points from one space to another.

    Args:
        t (numpy.array [4, 4]): SE3 transform.
        ps (numpy.array [n, 3]): Array of n 3D points (x, y, z).

    Raises:
        ValueError: If t is not a valid transform.
        ValueError: If ps does not have correct shape.

    Returns:
        numpy.array [n, 3]: Transformed 3D points.
    """
    if not transform_is_valid(t):
        raise ValueError('Invalid input transform t')
    if len(ps.shape) != 2 or ps.shape[1] != 3:
        raise ValueError('Invalid input points ps')
    ps_transpose = np.transpose(np.pad(ps, [(0,0),(0,1)], constant_values=1.))
    tmp_result = t @ ps_transpose
    result = np.transpose(tmp_result[:-1])
    return result


def transform_inverse(t):
    """Find the inverse of the transform.

    Args:
        t (numpy.array [4, 4]): SE3 transform.

    Raises:
        ValueError: If t is not a valid transform.

    Returns:
        numpy.array [4, 4]: Inverse of the input transform.
    """
    if not transform_is_valid(t):
        raise ValueError('Invalid input transform t')
    return np.linalg.inv(t)


@njit(parallel=True)
def camera_to_image(intrinsics, camera_points):
    """Project points in camera space to the image plane.

    Args:
        intrinsics (numpy.array [3, 3]): Pinhole intrinsics.
        camera_points (numpy.array [n, 3]): n 3D points (x, y, z) in camera coordinates.

    Raises:
        ValueError: If intrinsics are not the correct shape.
        ValueError: If camera points are not the correct shape.

    Returns:
        numpy.array [n, 2]: n 2D projections of the input points on the image plane.
    """
    if intrinsics.shape != (3, 3):
        raise ValueError('Invalid input intrinsics')
    if len(camera_points.shape) != 2 or camera_points.shape[1] != 3:
        raise ValueError('Invalid camera point')

    u0 = intrinsics[0, 2]
    v0 = intrinsics[1, 2]
    fu = intrinsics[0, 0]
    fv = intrinsics[1, 1]

    # find u, v int coords
    image_coordinates = np.empty((camera_points.shape[0], 2), dtype=np.int64)
    for i in prange(camera_points.shape[0]):
        image_coordinates[i, 0] = int(np.round((camera_points[i, 0] * fu / camera_points[i, 2]) + u0))
        image_coordinates[i, 1] = int(np.round((camera_points[i, 1] * fv / camera_points[i, 2]) + v0))

    return image_coordinates


def depth_to_point_cloud(intrinsics, depth_image):
    """Back project a depth image to a point cloud.
        Note: point clouds are unordered, so any permutation of points in the list is acceptable.
        Note: Only output those points whose depth > 0.

    Args:
        intrinsics (numpy.array [3, 3]): given as [[fu, 0, u0], [0, fv, v0], [0, 0, 1]]
        depth_image (numpy.array [h, w]): each entry is a z depth value.

    Returns:
        numpy.array [n, 3]: each row represents a different valid 3D point.
    """
    assert intrinsics.shape == (3,3)
    fu = intrinsics[0][0]
    fv = intrinsics[1][1]
    u0 = intrinsics[0][2]
    v0 = intrinsics[1][2]

    result = []

    for v in range(len(depth_image)):
        for u in range(len(depth_image[v])):
            z = depth_image[v][u]
            x = (u - u0) / fu * z
            y = (v - v0) / fv * z
            if z > 0:
                result.append([x, y, z])
    return np.array(result)