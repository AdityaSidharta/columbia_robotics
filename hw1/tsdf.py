from skimage import measure

import transforms
from transforms import *


class TSDFVolume:
    """Volumetric TSDF Fusion of RGB-D Images."""

    def __init__(self, volume_bounds, voxel_size):
        """Initialize tsdf volume instance variables.

        Args:
            volume_bounds (numpy.array [3, 2]): rows index [x, y, z] and cols index [min_bound, max_bound].
                Note: units are in meters.
            voxel_size (float): The side length of each voxel in meters.

        Raises:
            ValueError: If volume bounds are not the correct shape.
            ValueError: If voxel size is not positive.
        """
        volume_bounds = np.asarray(volume_bounds)
        if volume_bounds.shape != (3, 2):
            raise ValueError("volume_bounds should be of shape (3, 2).")

        if voxel_size <= 0.0:
            raise ValueError("voxel size must be positive.")

        # Define voxel volume parameters
        self._volume_bounds = volume_bounds
        self._voxel_size = float(voxel_size)
        self._truncation_margin = 2 * self._voxel_size  # truncation on SDF (max alowable distance away from a surface)

        # Adjust volume bounds and ensure C-order contiguous
        # and calculate voxel bounds taking the voxel size into consideration
        self._voxel_bounds = (
            np.ceil((self._volume_bounds[:, 1] - self._volume_bounds[:, 0]) / self._voxel_size)
            .copy(order="C")
            .astype(int)
        )
        self._volume_bounds[:, 1] = self._volume_bounds[:, 0] + self._voxel_bounds * self._voxel_size

        # volume min bound is the origin of the volume in world coordinates
        self._volume_origin = self._volume_bounds[:, 0].copy(order="C").astype(np.float32)

        print(
            "Voxel volume size: {} x {} x {} - # voxels: {:,}".format(
                self._voxel_bounds[0],
                self._voxel_bounds[1],
                self._voxel_bounds[2],
                self._voxel_bounds[0] * self._voxel_bounds[1] * self._voxel_bounds[2],
            )
        )

        # Initialize pointers to voxel volume in memory
        self._tsdf_volume = np.ones(self._voxel_bounds).astype(np.float32)

        # for computing the cumulative moving average of observations per voxel
        self._weight_volume = np.zeros(self._voxel_bounds).astype(np.float32)
        color_bounds = np.append(self._voxel_bounds, 3)
        self._color_volume = np.zeros(color_bounds).astype(np.float32)  # rgb order

        # Get voxel grid coordinates, get all possible iteration of voxel coordinates
        xv, yv, zv = np.meshgrid(
            range(self._voxel_bounds[0]), range(self._voxel_bounds[1]), range(self._voxel_bounds[2]), indexing="ij"
        )
        self._voxel_coords = (
            np.concatenate([xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1)], axis=0).astype(int).T
        )

    def get_volume(self):
        """Get the tsdf and color volumes.

        Returns:
            numpy.array [l, w, h]: l, w, h are the dimensions of the voxel grid in voxel space.
                Each entry contains the integrated tsdf value.
            numpy.array [l, w, h, 3]: l, w, h are the dimensions of the voxel grid in voxel space.
                3 is the channel number in the order r, g, then b.
        """
        return self._tsdf_volume, self._color_volume

    def get_mesh(self):
        """Run marching cubes over the constructed tsdf volume to get a mesh representation.

        Returns:
            numpy.array [n, 3]: each row represents a 3D point.
            numpy.array [k, 3]: each row is a list of point indices used to render triangles.
            numpy.array [n, 3]: each row represents the normal vector for the corresponding 3D point.
            numpy.array [n, 3]: each row represents the color of the corresponding 3D point.
        """
        tsdf_volume, color_vol = self.get_volume()

        # Marching cubes
        voxel_points, triangles, normals, _ = measure.marching_cubes(tsdf_volume, method="lewiner", level=0)
        points_ind = np.round(voxel_points).astype(int)
        points = self.voxel_to_world(self._volume_origin, voxel_points, self._voxel_size)

        # Get vertex colors.
        rgb_vals = color_vol[points_ind[:, 0], points_ind[:, 1], points_ind[:, 2]]
        colors_r = rgb_vals[:, 0]
        colors_g = rgb_vals[:, 1]
        colors_b = rgb_vals[:, 2]
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
        colors = colors.astype(np.uint8)

        return points, triangles, normals, colors

    """
    *******************************************************************************
    ****************************** ASSIGNMENT BEGINS ******************************
    *******************************************************************************
    """

    @staticmethod
    @njit(parallel=True)
    def voxel_to_world(volume_origin, voxel_coords, voxel_size):
        """Convert from voxel coordinates to world coordinates
            (in effect scaling voxel_coords by voxel_size).

        Args:
            volume_origin (numpy.array [3, ]): The origin of the voxel
                grid in world coordinate space.
            voxel_coords (numpy.array [n, 3]): Each row gives the 3D coordinates of a voxel.
            voxel_size (float): The side length of each voxel in meters.

        Returns:
            numpy.array [n, 3]: World coordinate representation of each of the n 3D points.
        """
        volume_origin = volume_origin.astype(np.float32)
        voxel_coords = voxel_coords.astype(np.float32)
        world_points = np.empty_like(voxel_coords, dtype=np.float32)

        # NOTE: prange is used instead of range(...) to take advantage of parallelism.
        for i in prange(voxel_coords.shape[0]):
            x, y, z = voxel_coords[i]
            world_x = volume_origin[0] + (x * voxel_size)
            world_y = volume_origin[1] + (y * voxel_size)
            world_z = volume_origin[2] + (z * voxel_size)
            world_points[i] = np.array([world_x, world_y, world_z])
        return world_points

    @staticmethod
    @njit(parallel=True)
    def get_new_tsdf_and_weights(tsdf_old, margin_distance, w_old, observation_weight):
        """[summary]

        Args:
            tsdf_old (numpy.array [v, ]): v is equal to the number of voxels to be
                integrated at this timestamp. Old tsdf values that need to be
                updated based on the current observation.
            margin_distance (numpy.array [v, ]): The tsdf values of the current observation.
                It should be of type numpy.array [v, ], where v is the number
                of valid voxels.
            w_old (numpy.array [v, ]): old weight values.
            observation_weight (float): Weight to give each new observation.

        Returns:
            numpy.array [v, ]: new tsdf values for entries in tsdf_old
            numpy.array [v, ]: new weights to be used in the future.
        """
        tsdf_new = np.empty_like(tsdf_old, dtype=np.float32)
        w_new = np.empty_like(w_old, dtype=np.float32)

        for i in prange(len(tsdf_old)):
            w_new[i] = w_old[i] + observation_weight
            tsdf_new[i] = (w_old[i] * tsdf_old[i] + observation_weight * margin_distance[i]) / w_new[i]
        return tsdf_new, w_new

    def get_values(self, image, voxel_u, voxel_v, valid_voxel, dtype=float):
        result = np.zeros_like(valid_voxel, dtype=dtype)
        round_voxel_u = np.round(voxel_u[valid_voxel])
        round_voxel_v = np.round(voxel_v[valid_voxel])
        print_debug("Image.shape : {}".format(image.shape))
        valid_result = image[round_voxel_v, round_voxel_u]
        result[valid_voxel] = valid_result
        print_debug("len valid_result : {}".format(len(valid_result)))
        print_debug("len zero result : {}".format(np.sum(result == 0.0)))
        print_debug("len nonzero result : {}".format(np.sum(result != 0.0)))
        return result

    def get_valid_points(self, depth_image, voxel_u, voxel_v, voxel_z):
        """Compute a boolean array for indexing the voxel volume and other variables.
        Note that every time the method integrate(...) is called, not every voxel in
        the volume will be updated. This method returns a boolean matrix called
        valid_points with dimension (n, ), where n = # of voxels. Index i of
        valid_points will be true if this voxel will be updated, false if the voxel
        needs not to be updated.

        The criteria for checking if a voxel is valid or not is shown below.

        Args:
            depth_image (numpy.array [h, w]): A z depth image.
            voxel_u (numpy.array [v, ]): Voxel coordinate projected into camera coordinate, axis is u
            voxel_v (numpy.array [v, ]): Voxel coordinate projected into camera coordinate, axis is v
            voxel_z (numpy.array [v, ]): Voxel coordinate projected into world coordinate axis z
        Returns:
            valid_points numpy.array [v, ]: A boolean matrix that will be
            used to index into the voxel grid. Note the dimension of this
            variable.
        """

        image_height, image_width = depth_image.shape

        # TODO 1:
        #  Eliminate pixels not in the image bounds or that are behind the image plane
        valid_voxel_u = (voxel_u >= 0) & (voxel_u < image_width - 0.5)
        valid_voxel_v = (voxel_v >= 0) & (voxel_v < image_height - 0.5)
        valid_voxel = valid_voxel_u & valid_voxel_v

        print_debug("max_voxel_u : {}".format(max(voxel_u)))
        print_debug("min_voxel_u : {}".format(min(voxel_u)))
        print_debug("max_voxel_v : {}".format(max(voxel_v)))
        print_debug("min_voxel_v : {}".format(min(voxel_v)))
        print_debug("max_valid_voxel_u : {}".format(max(voxel_u[valid_voxel])))
        print_debug("min_valid_voxel_u : {}".format(min(voxel_u[valid_voxel])))
        print_debug("max_valid_voxel_v : {}".format(max(voxel_v[valid_voxel])))
        print_debug("min_valid_voxel_v : {}".format(min(voxel_v[valid_voxel])))

        # TODO 2.1:
        #  Get depths for valid coordinates u, v from the depth image. Zero elsewhere.
        depths = self.get_values(depth_image, voxel_u, voxel_v, valid_voxel)

        # TODO 2.3:
        #  Filter out zero depth values and cases where depth + truncation margin >= voxel_z
        return depths > 0

    def get_new_colors_with_weights(self, color_old, color_new, w_old, w_new, observation_weight=1.0):
        """Compute the new RGB values for the color volume given the current values
        in the color volume, the RGB image pixels, and the old and new weights.

        Args:
            color_old (numpy.array [n, 3]): Old colors from self._color_volume in RGB.
            color_new (numpy.array [n, 3]): Newly observed colors from the image in RGB
            w_old (numpy.array [n, ]): Old weights from the self._tsdf_volume
            w_new (numpy.array [n, ]): New weights from calling get_new_tsdf_and_weights
            observation_weight (float, optional):  The weight to assign for the current
                observation. Defaults to 1.
        Returns:
            valid_points numpy.array [n, 3]: The newly computed colors in RGB. Note that
            the input color and output color should have the same dimensions.
        """
        valid_points = np.empty_like(color_old, dtype=np.float32)

        for i in prange(len(color_old)):
            valid_points[i] = (w_old[i] * color_old[i] + observation_weight * color_new[i]) / w_new[i]

        return valid_points

    def integrate(self, color_image, depth_image, camera_intrinsics, camera_pose, observation_weight=1.0):
        """Integrate an RGB-D observation into the TSDF volume, by updating the weight volume,
            tsdf volume, and color volume.

        Args:
            color_image (numpy.array [h, w, 3]): An rgb image.
            depth_image (numpy.array [h, w]): A z depth image.
            camera_intrinsics (numpy.array [3, 3]): given as [[fu, 0, u0], [0, fv, v0], [0, 0, 1]]
            camera_pose (numpy.array [4, 4]): SE3 transform representing pose (camera to world)
            observation_weight (float, optional):  The weight to assign for the current
                observation. Defaults to 1.
        """
        color_image = color_image.astype(np.float32)
        observation_weight = np.float32(observation_weight)
        n = len(self._voxel_coords)

        # TODO: 1. Project the voxel grid coordinates to the world
        #  space by calling `voxel_to_world`. Then, transform the points
        #  in world coordinate to camera coordinates, which are in (u, v).
        #  You might want to save the voxel z coordinate for later use.
        voxel_coords = self._voxel_coords
        world_coords = self.voxel_to_world(self._volume_origin, voxel_coords, self._voxel_size)
        camera_coords = transforms.transform_point3s(transforms.transform_inverse(camera_pose), world_coords)
        image_coords = transforms.camera_to_image(camera_intrinsics, camera_coords)
        voxel_u = image_coords[:, 0]
        voxel_v = image_coords[:, 1]
        voxel_z = camera_coords[:, 2]

        assert voxel_coords.shape == (n, 3)
        assert world_coords.shape == (n, 3)
        # assert camera_coords == (n, 3)
        assert image_coords.shape == (n, 2)
        assert voxel_u.shape == (n,), voxel_u.shape
        assert voxel_v.shape == (n,)
        assert voxel_z.shape == (n,)

        # TODO: 2.
        #  Get all of the valid points in the voxel grid by implementing
        #  the helper get_valid_points. Be sure to pass in the correct parameters.
        valid_coords = self.get_valid_points(depth_image, voxel_u, voxel_v, voxel_z)
        v = np.sum(valid_coords)

        image_depth = self.get_values(depth_image, voxel_u, voxel_v, valid_coords)
        image_r = self.get_values(color_image[:, :, 0], voxel_u, voxel_v, valid_coords)
        image_g = self.get_values(color_image[:, :, 1], voxel_u, voxel_v, valid_coords)
        image_b = self.get_values(color_image[:, :, 2], voxel_u, voxel_v, valid_coords)

        assert valid_coords.shape == (n,)
        assert image_depth.shape == (n,)
        assert image_r.shape == (n,)
        assert image_g.shape == (n,)
        assert image_b.shape == (n,)

        # TODO: 3.
        #  With the valid_points array as your indexing array, index into
        #  the self._voxel_coords variable to get the valid voxel x, y, and z.

        valid_voxel_idx_x = voxel_coords[:, 0][valid_coords]
        valid_voxel_idx_y = voxel_coords[:, 1][valid_coords]
        valid_voxel_idx_z = voxel_coords[:, 2][valid_coords]

        assert valid_voxel_idx_x.shape == (v,)
        assert valid_voxel_idx_y.shape == (v,)
        assert valid_voxel_idx_z.shape == (v,)

        # TODO: 4. With the valid_points array as your indexing array,
        #  get the valid pixels. Use those valid pixels to index into
        #  the depth_image, and find the valid margin distance.
        valid_voxel_z = voxel_z[valid_coords]
        valid_image_depth = image_depth[valid_coords]
        margin_distance = valid_image_depth - valid_voxel_z
        margin_distance = margin_distance.astype(np.float32)

        assert valid_voxel_z.shape == (v,)
        assert valid_image_depth.shape == (v,)
        assert margin_distance.shape == (v,)

        # TODO: 5.
        #  Compute the new weight volume and tsdf volume by calling
        #  `get_new_tsdf_and_weights`. Then update the weight volume
        #  and tsdf volume.

        tsdf_old = self._tsdf_volume[valid_voxel_idx_x, valid_voxel_idx_y, valid_voxel_idx_z]
        weight_old = self._weight_volume[valid_voxel_idx_x, valid_voxel_idx_y, valid_voxel_idx_z]
        tsdf_old = tsdf_old.astype(np.float32)
        weight_old = weight_old.astype(np.float32)

        tsdf_new, weight_new = self.get_new_tsdf_and_weights(tsdf_old, margin_distance, weight_old, observation_weight)

        print(
            "shape of tsdf_volume to be updated : {}".format(
                self._tsdf_volume[valid_voxel_idx_x, valid_voxel_idx_y, valid_voxel_idx_z].shape
            )
        )
        print("shape of tsdf_new : {}".format(tsdf_new.shape))

        self._tsdf_volume[valid_voxel_idx_x, valid_voxel_idx_y, valid_voxel_idx_z] = tsdf_new
        self._weight_volume[valid_voxel_idx_x, valid_voxel_idx_y, valid_voxel_idx_z] = weight_new

        assert tsdf_old.shape == (v,)
        assert weight_old.shape == (v,)
        assert tsdf_new.shape == (v,)
        assert weight_new.shape == (v,)

        # TODO: 6.
        #  Compute the new colors for only the valid voxels by using
        #  get_new_colors_with_weights, and update the current color volume
        #  with the new colors. The color_old and color_new parameters can
        #  be obtained by indexing the valid voxels in the color volume and
        #  indexing the valid pixels in the rgb image.

        valid_image_r = image_r[valid_coords]
        valid_image_g = image_g[valid_coords]
        valid_image_b = image_b[valid_coords]
        color_old = self._color_volume[valid_voxel_idx_x, valid_voxel_idx_y, valid_voxel_idx_z, :]
        color_new = np.transpose(np.vstack([valid_image_r, valid_image_g, valid_image_b]))
        result_color_new = self.get_new_colors_with_weights(
            color_old, color_new, weight_old, weight_new, observation_weight
        )
        self._color_volume[valid_voxel_idx_x, valid_voxel_idx_y, valid_voxel_idx_z] = result_color_new

        assert valid_image_r.shape == (v,)
        assert valid_image_b.shape == (v,)
        assert valid_image_b.shape == (v,)
        assert color_old.shape == (v, 3)
        assert color_new.shape == (v, 3)
        # assert result_color_new == (v, 3), result_color_new.shape

    """
    *******************************************************************************
    ******************************* ASSIGNMENT ENDS *******************************
    *******************************************************************************
    """