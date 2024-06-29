import numpy as np
import heapq
from tqdm import tqdm
from collections import defaultdict

class Dijkstra:
    def __init__(self, graph):
        self.graph = graph
        self.n_vertices = graph.shape[0]
        self.adj_list = self.create_adj_list(graph)
        self.distances = np.full((self.n_vertices, self.n_vertices), np.inf)
        self.predecessors = np.full((self.n_vertices, self.n_vertices), -1)

    def create_adj_list(self, graph):
        adj_list = defaultdict(list)
        for i in range(graph.shape[0]):
            for j in range(graph.shape[1]):
                if graph[i][j] > 0:
                    adj_list[i].append((j, graph[i][j]))
        return adj_list

    def calculate_shortest_paths_from_vertex(self, start_vertex):
        self.distances[start_vertex] = np.full(self.n_vertices, np.inf)
        self.distances[start_vertex][start_vertex] = 0
        min_heap = [(0, start_vertex)]  # (distance, vertex)
        predecessors = np.full(self.n_vertices, -1)

        while min_heap:
            current_distance, current_vertex = heapq.heappop(min_heap)

            if current_distance > self.distances[start_vertex][current_vertex]:
                continue

            for neighbor, weight in self.adj_list[current_vertex]:
                distance = current_distance + weight

                if distance < self.distances[start_vertex][neighbor]:
                    self.distances[start_vertex][neighbor] = distance
                    predecessors[neighbor] = current_vertex
                    heapq.heappush(min_heap, (distance, neighbor))

        self.predecessors[start_vertex] = predecessors

    def calculate_all_shortest_paths(self):
        for vertex in tqdm(range(self.n_vertices), desc="Calculating shortest paths"):
            self.calculate_shortest_paths_from_vertex(vertex)

    def get_distances(self):
        return self.distances
    
    def get_distance_from_to(self, start_vertex, end_vertex):
        return self.distances[start_vertex][end_vertex]

    def get_shortest_path(self, start_vertex, end_vertex):
        path = []
        current_vertex = end_vertex
        while current_vertex != -1:
            path.append(current_vertex)
            current_vertex = self.predecessors[start_vertex][current_vertex]
        path = path[::-1]  # Reverse the path to get it from start to end
        if path[0] == start_vertex:
            return path
        else:
            return None  # If the start vertex isn't at the beginning, there's no path

    def set_graph(self, graph):
        self.graph = graph
        self.n_vertices = graph.shape[0]
        self.adj_list = self.create_adj_list(graph)
        self.distances = np.full((self.n_vertices, self.n_vertices), np.inf)
        self.predecessors = np.full((self.n_vertices, self.n_vertices), -1)

if __name__ == "__main__":
    # Example graph represented as an adjacency matrix
    graph = np.array([
        [0, 7, 9, 0, 0, 14],
        [7, 0, 10, 15, 0, 0],
        [9, 10, 0, 11, 0, 2],
        [0, 15, 11, 0, 6, 0],
        [0, 0, 0, 6, 0, 9],
        [14, 0, 2, 0, 9, 0]
    ])

    dijkstra = Dijkstra(graph)
    dijkstra.calculate_all_shortest_paths()
    distances = dijkstra.get_distances()
    print("All distances:", distances, type(distances))


    for start_vertex in range(graph.shape[0]):
        print(f"Distances from vertex {start_vertex}: {distances[start_vertex]}")
        for end_vertex in range(graph.shape[0]):
            path = dijkstra.get_shortest_path(start_vertex, end_vertex)
            if path:
                print(f"  Shortest path to vertex {end_vertex}: {path} with cost {distances[start_vertex][end_vertex]}")
            else:
                print(f"  No path found to vertex {end_vertex}")
    
    # Get the distance from vertex 0 to vertex 4
    start_vertex = 0 # any vertex
    end_vertex = 4  # any vertex
    distance = dijkstra.get_distance_from_to(start_vertex, end_vertex)
    print(f"Distance from vertex {start_vertex} to vertex {end_vertex}: {distance}")

    