from collections import deque

import numpy as np


class RRTGraph:
    def __init__(self, q_init, distance_method="l2"):
        self.graph = dict()
        self.add_vertex(q_init)
        self.distance_method = distance_method
        self.distances = {}
        self.parents = {}
        self.explored = {}

    def add_vertex(self, array_vertex):
        vertex = tuple(array_vertex)
        if vertex not in self.graph:
            self.graph[vertex] = []

    def add_edge(self, array_vertex_a, array_vertex_b):
        vertex_a = tuple(array_vertex_a)
        vertex_b = tuple(array_vertex_b)
        assert vertex_a in self.graph
        assert vertex_b in self.graph
        if vertex_b not in self.graph[vertex_a]:
            self.graph[vertex_a].append(vertex_b)
        if vertex_a not in self.graph[vertex_b]:
            self.graph[vertex_b].append(vertex_a)

    @staticmethod
    def random_sample():
        return np.random.uniform(low=-np.pi * 2, high=np.pi * 2, size=(6,))

    def distance(self, point_a, point_b):
        point_a = np.array(point_a)
        point_b = np.array(point_b)
        if self.distance_method == "l1":
            return np.sum(np.abs(point_a - point_b))
        elif self.distance_method == "l2":
            return np.sqrt(np.sum(np.power((point_a - point_b), 2)))
        elif self.distance_method == "l3":
            return np.sum(np.minimum((2 * np.pi - np.abs(point_a - point_b)), (np.abs(point_a - point_b))))

    def nearest(self, target_vertex):
        vertices = [np.array(vertex) for vertex in self.graph]
        distances = [self.distance(vertex, target_vertex) for vertex in vertices]
        idx = np.argmin(np.array(distances))
        return vertices[idx], distances[idx]

    def steer(self, nearest_vertex, target_vertex, nearest_distance, delta_q):
        new_vertex = nearest_vertex + ((target_vertex - nearest_vertex) * delta_q / nearest_distance)
        new_vertex = np.maximum(new_vertex, -np.pi * 2)
        new_vertex = np.minimum(new_vertex, np.pi * 2)
        return new_vertex

    def find_path(self, array_source, array_target):
        source = tuple(array_source)
        target = tuple(array_target)
        self.distances = {}
        self.parents = {}
        self.explored = {}
        for vertex in self.graph:
            self.distances[vertex] = 0
            self.parents[vertex] = None
            self.explored[vertex] = False
        q = deque()
        q.append(source)
        self.explored[source] = True
        self.distances[source] = 0
        self.parents[source] = None
        while len(q):
            u = q.popleft()
            for v in self.graph[u]:
                if not self.explored[v]:
                    self.explored[v] = True
                    self.distances[v] = self.distances[u] + 1
                    self.parents[v] = u
                    q.append(v)
                    if v == target:
                        path = []
                        current_node = target
                        while current_node is not None:
                            path.append(np.array(current_node))
                            current_node = self.parents[current_node]
                        path.reverse()
                        return path
        raise ValueError()
