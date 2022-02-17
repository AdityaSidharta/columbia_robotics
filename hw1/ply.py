import os

import numpy as np


class Ply(object):
    """Class to represent a ply in memory, read plys, and write plys."""

    def __init__(self, ply_path=None, triangles=None, points=None, normals=None, colors=None):
        """Initialize the in memory ply representation.

        Args:
            ply_path (str, optional): Path to .ply file to read (note only
                supports text mode, not binary mode). Defaults to None.
            triangles (numpy.array [k, 3], optional): each row is a list of point indices used to
                render triangles. Defaults to None.
            points (numpy.array [n, 3], optional): each row represents a 3D point. Defaults to None.
            normals (numpy.array [n, 3], optional): each row represents the normal vector for the
                corresponding 3D point. Defaults to None.
            colors (numpy.array [n, 3], optional): each row represents the color of the
                corresponding 3D point. Defaults to None.
        """
        super().__init__()
        self.ply_path = ply_path
        self.triangles = triangles
        self.points = points
        self.normals = normals
        self.colors = colors

        if self.ply_path is not None:
            self.read(self.ply_path)
        self.validate()

    def print(self):
        print("ply_path : {}".format(self.ply_path))
        print("triangles : {}".format(self.triangles))
        print("points : {}".format(self.points))
        print("normals : {}".format(self.normals))
        print("colors: {}".format(self.colors))

    def validate(self):
        if self.ply_path is not None:
            assert os.path.exists(self.ply_path)
        if self.triangles is not None:
            assert self.triangles.shape[1] == 3
        if self.points is not None:
            assert self.points.shape[1] == 3
        if self.normals is not None:
            assert self.points.shape == self.normals.shape
        if self.colors is not None:
            assert self.normals.shape == self.colors.shape

    def write(self, ply_path):
        """Write mesh, point cloud, or oriented point cloud to ply file.

        Args:
            ply_path (str): Output ply path.
        """
        self.validate()
        is_normal = False
        is_color = False
        is_face = False
        n_faces = 0
        result = []
        assert self.points is not None
        n_points = len(self.points)
        if self.normals is not None:
            is_normal = True
        if self.colors is not None:
            is_color = True
        if self.triangles is not None:
            is_face = True
            n_faces = len(self.triangles)

        result.append("ply")
        result.append("format ascii 1.0")
        result.append("element vertex {}".format(n_points))
        result.append("property float x")
        result.append("property float y")
        result.append("property float z")
        if is_normal:
            result.append("property float nx")
            result.append("property float ny")
            result.append("property float nz")
        if is_color:
            result.append("property uchar red")
            result.append("property uchar green")
            result.append("property uchar blue")
        if is_face:
            result.append("element face {}".format(n_faces))
            result.append("property list uchar int vertex_index")
        result.append("end_header")
        for idx in range(n_points):
            tmp_result = []
            tmp_result.extend(self.points[idx].astype(str))
            if is_normal:
                tmp_result.extend(self.normals[idx].astype(str))
            if is_color:
                tmp_result.extend(self.colors[idx].astype(str))
            result.append(" ".join(tmp_result))
        for idx in range(n_faces):
            result.append(" ".join(["3"] + self.triangles[idx].astype(str).tolist()))
        result.append("")

        with open(ply_path, "w") as f:
            f.write("\n".join(result))
        self.ply_path = ply_path

    def read(self, ply_path):
        """Read a ply into memory.

        Args:
            ply_path (str): ply to read in.
        """
        self.ply_path = ply_path
        with open(ply_path, "r") as f:
            file = f.read()
        lines = [x for x in file.split("\n") if x != ""]
        is_normal = False
        is_color = False
        is_face = False
        n_faces = 0
        self.triangles = None
        self.points = None
        self.normals = None
        self.colors = None

        if "property float nx" in lines:
            is_normal = True
        if "property uchar red" in lines:
            is_color = True
        if "property list uchar int vertex_index" in lines:
            is_face = True

        assert lines.pop(0) == "ply"
        assert lines.pop(0) == "format ascii 1.0"

        vertices = lines.pop(0).split(" ")
        assert len(vertices) == 3
        assert vertices[0] == "element"
        assert vertices[1] == "vertex"
        n_points = int(vertices[2])

        assert lines.pop(0) == "property float x"
        assert lines.pop(0) == "property float y"
        assert lines.pop(0) == "property float z"

        if is_normal:
            assert lines.pop(0) == "property float nx"
            assert lines.pop(0) == "property float ny"
            assert lines.pop(0) == "property float nz"
        if is_color:
            assert lines.pop(0) == "property uchar red"
            assert lines.pop(0) == "property uchar green"
            assert lines.pop(0) == "property uchar blue"
        if is_face:
            faces = lines.pop(0).split(" ")
            assert len(faces) == 3
            assert faces[0] == "element"
            assert faces[1] == "face", faces
            n_faces = int(faces[2])
            assert lines.pop(0) == "property list uchar int vertex_index", lines
        assert lines.pop(0) == "end_header"

        self.points = np.zeros((n_points, 3), float)
        if is_normal:
            self.normals = np.zeros((n_points, 3), float)
        if is_color:
            self.colors = np.zeros((n_points, 3), int)
        if is_face:
            self.triangles = np.zeros((n_faces, 3), int)

        for idx in range(n_points):
            points = lines.pop(0).split(" ")
            self.points[idx][0] = float(points.pop(0))
            self.points[idx][1] = float(points.pop(0))
            self.points[idx][2] = float(points.pop(0))
            if is_normal:
                self.normals[idx][0] = float(points.pop(0))
                self.normals[idx][1] = float(points.pop(0))
                self.normals[idx][2] = float(points.pop(0))
            if is_color:
                self.colors[idx][0] = int(points.pop(0))
                self.colors[idx][1] = int(points.pop(0))
                self.colors[idx][2] = int(points.pop(0))
            assert len(points) == 0
        for idx in range(n_faces):
            faces = lines.pop(0).split(" ")
            assert faces.pop(0) == "3"
            self.triangles[idx][0] = int(faces.pop(0))
            self.triangles[idx][1] = int(faces.pop(0))
            self.triangles[idx][2] = int(faces.pop(0))
        assert len(lines) == 0
        self.validate()
