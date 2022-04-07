class Graph:
    def __init__(self, vertices, edges):
        self.graph = {}
        for vertex in vertices:
            self.graph[vertex] = []
        for edge in edges:
            src, dst = edge
            assert src in self.graph
            assert dst in self.graph
            self.graph[src] = 