import os

if __name__ == "__main__":
    current_path = os.path.dirname(os.path.realpath(__file__))

    import sys
    sys.path.append(os.path.join(current_path, "Voronoi"))
    sys.path.append(os.path.join(current_path, ".."))

# Standard imports
import random
random.seed(42)
import numpy as np
from scipy.spatial import Voronoi as SciVoronoi
from scipy.sparse import lil_matrix, csr_matrix # sparse matrices
from scipy.sparse import save_npz, load_npz
import matplotlib.pyplot as plt


from plagueSpread.utils.GeometryUtils import (
    calculate_triangle_centroids, 
    # get_triangles_of_grid_points, barycentric_interpolate_height_grid,
    get_triangle_of_point, barycentric_interpolate_height, CH_quickhull,
    is_inside_polygon_2d
)
from plagueSpread.utils.DijkstraAlgorithm import Dijkstra
from plagueSpread.KDTree import KdNode

from vvrpywork.constants import Key, Mouse, Color
from vvrpywork.scene import Scene3D, world_space
from vvrpywork.shapes import (
    Point3D, Line3D, Arrow3D, Sphere3D, Cuboid3D, Cuboid3DGeneralized,
    PointSet3D, LineSet3D, Mesh3D
)

from noise import pnoise2
from time import time
from tqdm import tqdm
# import pickle


WIDTH_3D = 1400
HEIGHT_3D = 800

DEBUG = True # False
CONSOLE_TALK = True # False
TRIAL_MODE = False # False

class PlagueSpread3DTerrain(Scene3D):
    def __init__(self, WIDTH, HEIGHT):
        super().__init__(WIDTH, HEIGHT, "Plague Spread 3D", output=True)
        self._scenario_mode_init()

        self.scenario_parameters_init()

        # setting up grid essentials
        self.load_terrain() 
        self.create_grid() if not self.mesh else None # create the grid if we don't have a mesh
        self.triangulate_grid(self.grid.points, self.GRID_SIZE, -1, 1) if not self.mesh else None # triangulate the grid if we don't have a mesh
        centroids_need_update, dist_need_update, adj_need_update, short_paths_need_update,\
              self.el_dist_matrix_need_update, self.el_short_paths_need_update = self.perform_file_checks() # check if we need to update the files
        self.calculate_centroids(centroids_need_update) 
        self.create_centroids_kd_tree() # create the KD-Tree for the centroids
        self.setup_geography_specifics() # set up the specifics of the geography before calculating the matrices
        self.create_adjacency_matrix(adj_need_update)
        self.create_distances_matrix(dist_need_update) # centroid distances matrix is the graph for Dijkstra
        self.create_shortest_paths_matrix(short_paths_need_update)
        console_log("Terrain set up\n-----------------")

        self.construct_scenario() if not self.TRIAL_MODE else self.construct_mini_scenario()
        if self.wells_pcd.points.size > 1:
            self.Voronoi = self.getVoronoi(self.wells_pcd.points)


        self._print_instructions()
        self.my_mouse_pos = Point3D((0, 0, 0))
        # self.addShape(self.my_mouse_pos, "mouse")

        # debug
        # self.addShape(Point3D((-1, -1, 0), size=1, color=Color.RED), "down_left")
        # self.addShape(Point3D((1, 1, 0), size=1, color=Color.GREEN), "up_right")
        # self.addShape(Point3D((1, -1, 0), size=1, color=Color.BLUE), "down_right")
        # self.addShape(Point3D((-1, 1, 0), size=1, color=Color.YELLOW), "up_left")
        # self._get_triangle_of_test_point(pt=np.array([-0.119, -0.755, 0]))
        # self._get_triangle_of_test_point([0.5, 0.5, 0])
        # self._highest_points_test()
        # self._barycentric_interpolation_test(pt=np.array([0.5, 0.5, 0]))

    def _check_grid_mismatch(self, idx):
        '''Check if get_triangles_of_grid_points and self.triangle_indices are consistent.'''
        console_log(f"{idx} triangle indices: {self.triangle_indices[idx]}")
        console_log(f"{idx} triangle {self.grid.points[self.triangle_indices[idx][0]]}, {self.grid.points[self.triangle_indices[idx][1]]}, {self.grid.points[self.triangle_indices[idx][2]]}")
        p1 = self.grid.points[self.triangle_indices[idx][0]]
        p2 = self.grid.points[self.triangle_indices[idx][1]]
        p3 = self.grid.points[self.triangle_indices[idx][2]]

        # self.addShape(Point3D(p1, size=1, color=Color.RED), f"p1_{idx}")
        # self.addShape(Point3D(p2, size=1, color=Color.RED), f"p2_{idx}")
        # self.addShape(Point3D(p3, size=1, color=Color.RED), f"p3_{idx}")

        # get the centroid of the triangle
        centroid = self.centroids[idx]
        console_log(f"Centroid: {centroid}")
        # self.addShape(Point3D(centroid, size=1, color=Color.DARKGREEN), f"centroid_{idx}") 

        function_triangle_indices = get_triangle_of_point(centroid[:,:2], self.triangle_indices, self.mesh.vertices)
        function_triangle = self.grid.points[function_triangle_indices]
        console_log(f"Function triangle: {function_triangle}")
        f1, f2, f3 = function_triangle
        # self.addShape(Point3D(f1, size=1, color=Color.BLUE), f"f1_{idx}")
        # self.addShape(Point3D(f2, size=1, color=Color.BLUE), f"f2_{idx}")
        # self.addShape(Point3D(f3, size=1, color=Color.BLUE), f"f3_{idx}")

        # are they equal?
        if not np.array_equal(function_triangle, self.grid.points[self.triangle_indices[idx]]):
            console_log(f"Triangle indices {idx} mismatch!")
            console_log(f"Function triangle: {function_triangle}")
            console_log(f"Self triangle: {self.triangle_indices[idx]}")

        # debug triangle
        pp = np.array([[0.15789474, -0.78947368, -0.1404871],
              [0.26315789, -0.68421053, -0.19282421],
              [0.15789474, -0.68421053, -0.10920157]])
        self.addShape(Point3D(pp[0], size=1, color=Color.YELLOW), f"pp1_{idx}")
        self.addShape(Point3D(pp[1], size=1, color=Color.YELLOW), f"pp2_{idx}")
        self.addShape(Point3D(pp[2], size=1, color=Color.YELLOW), f"pp3_{idx}")

        self.addShape(Point3D([ 0.19731697, -0.68796272, -0.14163439], size=0.2, color=Color.ORANGE), f"p_{idx}")

    def _show_matrices(self, color=Color.BLACK):
        '''Add the adjacenct triangles to the scene.'''
        
        ### Adjacency matrix
        # list_of_names = []
        # adjacency_matrix = self.adjacency_matrix.toarray()
        # for i in range(len(adjacency_matrix)//16):
        #     for j in range(len(adjacency_matrix)//16):
        #         if adjacency_matrix[i, j] == 1:
        #             p1 = self.centroids[i]
        #             p2 = self.centroids[j]
        #             name = f"adj_{i}_{j}"
        #             list_of_names.append(name)
        #             self.addShape(Line3D(p1, p2, width = 0.1, color=color), name)
        
        ### Elevated Distance matrix
        # el_dist_matrix = self.create_elevation_distance_matrix().toarray()
        # # retrieve the line indices
        # lines = []
        # for i in range(len(el_dist_matrix)):
        #     for j in range(i + 1, len(el_dist_matrix)):
        #         if el_dist_matrix[i, j] != 0:
        #             lines.append((i, j))
        # lineset = LineSet3D(self.centroids, lines, width=1, color=color)
        # self.addShape(lineset, "elevation_distances")
        # list_of_names.append("elevation_distances")

        ### Distance Map
        # pointset_centroids = PointSet3D(self.centroids, size=1, color=color)
        # self.addShape(pointset_centroids, "centroids")
        # list_of_names.append("centroids")
        # if not self.ELEVATION_DISTANCE:
        #     # get the sum by row of all columns 
        #     sum_by_row = np.sum(self.shortest_paths_matrix, axis=1).reshape(-1, 1)
        #     # for each point, RGBA normalized by the sum of the row
        #     # minmax normalize the sum by row
        #     sum_by_row = (sum_by_row - np.min(sum_by_row)) / (np.max(sum_by_row) - np.min(sum_by_row))
        #     sum_by_row = np.repeat(sum_by_row, 3, axis=1)
        #     sum_by_row = np.hstack([sum_by_row, np.ones((len(sum_by_row), 1))])
        #     pointset_centroids.colors = sum_by_row

        #     self.removeShape("centroids")
        #     self.addShape(pointset_centroids, "centroids")
        # else:
        #     # get the sum by row of all columns 
        #     sum_by_row = np.sum(self.elevated_shortest_paths_matrix, axis=1).reshape(-1, 1)
        #     # for each point, RGBA normalized by the sum of the row
        #     # minmax normalize the sum by row
        #     sum_by_row = (sum_by_row - np.min(sum_by_row)) / (np.max(sum_by_row) - np.min(sum_by_row))
        #     sum_by_row = np.repeat(sum_by_row, 3, axis=1)
        #     sum_by_row = np.hstack([sum_by_row, np.ones((len(sum_by_row), 1))])
        #     pointset_centroids.colors = sum_by_row

        #     self.removeShape("centroids")
        #     self.addShape(pointset_centroids, "centroids")

        ### Visualizing calculated elevation
        list_of_names = []
        if self.ELEVATION_DISTANCE and self.uphill_indices_matrix is not None and self.downhill_indices_matrix is not None:
            uphill_distances = self.uphill_indices_matrix.toarray() # boolean matrix
            downhill_distances = self.downhill_indices_matrix.toarray() # boolean matrix
            
            console_log("Showing uphill and downhill lines...")
            uphill_lines = []
            downhill_lines = []
            for i in range(len(uphill_distances)):
                for j in range(len(uphill_distances)):
                    if uphill_distances[i, j]:
                        uphill_lines.append((i, j))
                    if downhill_distances[i, j]:
                        downhill_lines.append((i, j))
                    
            #increase the z value of the points to be more visible 
            uphill_points = self.centroids + np.array([0, 0, 0.01])
            uphill_lineset = LineSet3D(uphill_points, uphill_lines, width=1, color=Color.RED)
            downhill_points = self.centroids + np.array([0, 0, +0.01])
            downhill_lineset = LineSet3D(downhill_points, downhill_lines, width=1, color=Color.BLUE)
            self.addShape(uphill_lineset, "uphill_lines")
            self.addShape(downhill_lineset, "downhill_lines")
            list_of_names.append("uphill_lines")
            list_of_names.append("downhill_lines")
            console_log("Done.")
        elif self.ELEVATION_DISTANCE:
            console_log("Uphill and downhill matrices are not available.")

        return list_of_names

    def _show_path(self, start, end):
        '''Show the shortest path between two vertices.'''
        centroid_distances_matrix = self.centroid_distances_matrix.toarray()
        dijkstra = Dijkstra(centroid_distances_matrix)  
        dijkstra.calculate_shortest_paths_from_vertex(start)
        shortest_paths = dijkstra.get_shortest_path(start, end)
        shortest_costs = dijkstra.get_distances()
        self.terminal_log(f"Shortest path cost: {shortest_costs[start][end]}")

        list_of_names = []
        lines = []
        for i in range(len(shortest_paths) - 1):
            lines.append((i, i + 1))
        path_points = self.centroids[shortest_paths]
        # increase the z value of the path points to be more visible
        path_points[:, 2] += 0.01
        lineset = LineSet3D(path_points, lines, width=5, color=Color.RED)
        self.removeShape("shortest_paths")
        self.addShape(lineset, "shortest_paths")
        list_of_names.append("shortest_paths")
        
        if self.ELEVATION_DISTANCE:
            if self.elevation_distance_matrix is None:
                self.elevation_distance_matrix = self.create_elevation_distance_matrix()
            elevation_distance_matrix = self.elevation_distance_matrix.toarray()
            dijkstra_elev = Dijkstra(elevation_distance_matrix)
            dijkstra_elev.calculate_shortest_paths_from_vertex(start)
            shortest_costs_elev = dijkstra_elev.get_distances()
            self.terminal_log(f"Elevated shortest path cost: {shortest_costs_elev[start][end]}")
            shortest_paths_elev = dijkstra_elev.get_shortest_path(start, end)

            lines = []
            for i in range(len(shortest_paths_elev) - 1):
                lines.append((i, i + 1))
            elevated_points = self.centroids[shortest_paths_elev]
            # increase the z value of the elevated points to be more visible
            elevated_points[:, 2] += 0.01
            lineset = LineSet3D(elevated_points, lines, width=5, color=Color.YELLOWGREEN)
            self.removeShape("shortest_paths_elev")
            self.addShape(lineset, "shortest_paths_elev")
            list_of_names.append("shortest_paths_elev")

        if self.EUCLIDEAN:
            # the euclidean path between the two points
            # is simply the straight line between them
            name = "euclidean_path"
            euclidean_line = Line3D(self.centroids[start], self.centroids[end], width=1, color=Color.BLACK)
            self.removeShape(name)
            self.addShape(euclidean_line, name)
            self.terminal_log(f"Euclidean distance: {np.linalg.norm(self.centroids[start] - self.centroids[end])}")
            list_of_names.append(name)
        
        return list_of_names

    def _get_triangle_of_test_point(self, pt=np.array([-0.119, -0.755, 0])):
        test_pt = np.array(pt)
        # vertical line through test point
        lower_limit = np.array([test_pt[0], test_pt[1], -1])
        upper_limit = np.array([test_pt[0], test_pt[1], 1])
        self.addShape(Line3D(lower_limit, upper_limit, width=0.02, color=Color.BLACK), f"vertical_line_{test_pt}")
        self.addShape(Point3D(test_pt, size=0.02, color=Color.BLUE), f"test_point_{test_pt}")

        # get the k nearest neighbors
        # k = 70
        # nearest_centroids = KdNode.nearestK(test_pt, self.kd_centroid_root, k)
        # for i, centroid in enumerate(nearest_centroids):
        #     self.addShape(Point3D(centroid.point, size=0.2, color=Color.RED), f"nearest_{i}")
        
        triangle_index, _ = get_triangle_of_point(test_pt, self.triangle_indices, self.mesh.vertices)
        triangle_vertices = self.grid.points[self.triangle_indices[triangle_index]]
        self.addShape(Point3D(triangle_vertices[0], size=0.05, color=Color.RED), f"triangle_vertex_1_{test_pt}")
        self.addShape(Point3D(triangle_vertices[1], size=0.05, color=Color.RED), f"triangle_vertex_2_{test_pt}")
        self.addShape(Point3D(triangle_vertices[2], size=0.05, color=Color.RED), f"triangle_vertex_3_{test_pt}")
        # self.addShape(Line3D(test_pt, self.centroids[triangle_index], color=Color.ORANGE), f"line_to_centroid_triangle_{test_pt}")

    def _highest_points_test(self):
        start_time = time()
        # Get the 100 highest points
        highest_indices = np.argsort(self.grid.points[:, 2])[-50:]
        highest_points = self.grid.points[highest_indices]

        # Start from the highest point (assume sorted in ascending order)
        highest_point = highest_points[-1]
        
        # Filter out points within the required distance from the highest point
        filtered_points = np.array([point for point in highest_points 
                        if np.linalg.norm(point - highest_point) <= 0.3])
        
        # for all the filtered points, get the centroids within a certain radius
        final_points = []
        for point in filtered_points:
            nearest_centroids = np.where(np.linalg.norm(self.centroids - point, axis=1) <= 0.2)[0]
            nearest_centroids = self.centroids[nearest_centroids]
            final_points.extend(nearest_centroids)

        # Add the final points to the scene
        pointset = PointSet3D(final_points, size=1, color=Color.RED)
        self.addShape(pointset, "final_points")
        end_time = time()
        console_log(f"Time taken for highest points test: {end_time - start_time} seconds.")

    def _barycentric_interpolation_test(self, pt=np.array([0.5, 0.5, 0])):
        triangle_index, _ = get_triangle_of_point(pt, self.triangle_indices, self.mesh.vertices)
        triangle_vertices = self.grid.points[self.triangle_indices[triangle_index]]
        self.addShape(Point3D(triangle_vertices[0], size=0.1, color=Color.RED), f"triangle_vertex_1_{pt}")
        self.addShape(Point3D(triangle_vertices[1], size=0.1, color=Color.RED), f"triangle_vertex_2_{pt}")
        self.addShape(Point3D(triangle_vertices[2], size=0.1, color=Color.RED), f"triangle_vertex_3_{pt}")
        # # check if the order of the triangle vertices is correct, i.e. counter-clockwise
        # self.addShape(Arrow3D(triangle_vertices[0], triangle_vertices[1], color=Color.RED), f"arrow_1_{pt}")
        # self.addShape(Arrow3D(triangle_vertices[1], triangle_vertices[2], color=Color.BLUE), f"arrow_2_{pt}")
        # self.addShape(Arrow3D(triangle_vertices[2], triangle_vertices[0], color=Color.GREEN), f"arrow_3_{pt}")

        self.addShape(Point3D(pt, size=0.5, color=Color.BLUE), f"test_point_{pt}")
        # self.addShape(Line3D(pt, self.centroids[triangle_index], color=Color.ORANGE), f"line_to_centroid_triangle_{pt}")
        # straight vertical line through the original point
        lower_limit = np.array([pt[0], pt[1], -1])
        upper_limit = np.array([pt[0], pt[1], 1])
        self.addShape(Line3D(lower_limit, upper_limit, width=0.02, color=Color.BLACK), f"vertical_line_{pt}")


        # barycentric interpolation
        # create a 2d array as input for the function
        pt = pt.reshape(1, -1)

        height, debug_point_list, debug_triangle_list = barycentric_interpolate_height(pt[:, :2], self.mesh.triangles, self.mesh.vertices, debug=True)
        console_log(f"Height at point {pt}: {height}, {pt[0][2]}")
        # assign the height to the point
        pt[0][2] = height   
        self.addShape(Point3D(pt[0], size=0.05, color=Color.GREEN), f"interpolated_point_{pt}")
        print(f"Debug triangle list: {debug_triangle_list}")
        print(f"Debug point list: {debug_point_list}")
        # for point in debug_point_list:

            
        for triangle in debug_triangle_list:
            tri_idx = self.mesh.triangles[triangle]
            p1, p2, p3 = self.mesh.vertices[tri_idx]
            self.addShape(Point3D(p1, size=0.1, color=Color.MAGENTA), f"tri_vertex_1_{p1}")
            self.addShape(Point3D(p2, size=0.1, color=Color.MAGENTA), f"tri_vertex_2_{p2}")
            self.addShape(Point3D(p3, size=0.1, color=Color.MAGENTA), f"tri_vertex_3_{p3}")

            tri_idx_minus_1 = self.mesh.triangles[triangle - 1]
            tri_idx_plus_1 = self.mesh.triangles[triangle + 1]
            p1, p2, p3 = self.mesh.vertices[tri_idx_minus_1]
            # self.addShape(Point3D(p1, size=0.1, color=Color.BLACK), f"tri_vertex_1_{p1}")
            # self.addShape(Point3D(p2, size=0.1, color=Color.BLACK), f"tri_vertex_2_{p2}")
            # self.addShape(Point3D(p3, size=0.1, color=Color.BLACK), f"tri_vertex_3_{p3}")

            # p1, p2, p3 = self.mesh.vertices[tri_idx_plus_1]
            # self.addShape(Point3D(p1, size=0.1, color=Color.YELLOW), f"tri_vertex_1_{p1}")
            # self.addShape(Point3D(p2, size=0.1, color=Color.YELLOW), f"tri_vertex_2_{p2}")
            # self.addShape(Point3D(p3, size=0.1, color=Color.YELLOW), f"tri_vertex_3_{p3}")
        for idx in debug_point_list:
            self.addShape(Point3D(pt[idx], size=0.1, color=Color.BLACK), f"debug_point_{idx}")


    def load_terrain(self):
        '''Load the terrain from a saved terrain.'''
        console_log("Loading the terrain from the saved obj...")
        terrain_path = os.path.join(os.path.dirname(__file__), "resources", "terrain.obj")
        start_time = time()
        self.mesh = Mesh3D(terrain_path, color=Color.GREY)
        end_time = time()
        console_log(f"Terrain loaded in {end_time - start_time} seconds.")
        
        self.mesh.remove_duplicated_vertices()
        self.mesh.remove_unreferenced_vertices()
        vertices = self.mesh.vertices
        vertices -= np.mean(vertices, axis=0) # center the mesh
        distanceSq = (vertices ** 2).sum(axis=-1) # calculate the squared distance
        max_dist = np.sqrt(np.max(distanceSq)) # get the maximum distance
        self.mesh.vertices = vertices / max_dist # normalize the vertices
        self.mesh.vertices *= 1.5 # scale the mesh
        # Corners
        # Min: (-1.0504099513130472, -0.14137424978632518, -1.0066074624794825), 
        # Max: (1.0217544773397103, 0.22599769287879068, 1.0655569661732747)
        # rotate the mesh 90 degrees around the x-axis
        rotation_matrix_x = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        self.mesh.vertices = np.dot(self.mesh.vertices, rotation_matrix_x.T)
        self.addShape(self.mesh, "mesh")

        self.wireframe = LineSet3D.create_from_mesh(self.mesh)
        self.addShape(self.wireframe, "wireframe")

        pointset_size = 1 if self.GRID_SIZE < 50 else 0.5
        self.grid = PointSet3D(self.mesh.vertices, size=pointset_size, color=Color.GREY)
        self.grid_lines = self.wireframe # synonym
        self.triangle_indices = self.mesh.triangles
    
    def setup_mountainous_area(self):
        # Get the 50 highest points
        highest_indices = np.argsort(self.grid.points[:, 2])[-50:]
        highest_points = self.grid.points[highest_indices]

        highest_point = highest_points[-1]
        
        # Filter out points that are a certain distance away from the highest point
        filtered_points = np.array([point for point in highest_points 
                        if np.linalg.norm(point - highest_point) <= 0.3])
        
        # for all the filtered points, get the centroids within a certain radius
        final_points = []
        final_points_indexes = []
        for point in filtered_points:
            nearest_centroids = np.where(np.linalg.norm(self.centroids - point, axis=1) <= 0.2)[0]
            final_points_indexes.extend(nearest_centroids)
            nearest_centroids = self.centroids[nearest_centroids]
            final_points.extend(nearest_centroids)

        # Add the final points to the scene
        mountainous_area = PointSet3D(final_points, size=1, color=Color.BLACK)
        self.mountainous_area = mountainous_area
        self.mountainous_area_idxs = final_points_indexes
        self.addShape(mountainous_area, "mountainous_area")

    def setup_tunnels(self):
        ''' Set up the tunneled pathways.'''
        tunnel_left = np.array([-0.51082277, -0.02691042, -0.05437386])
        tunnel_right = np.array([-0.17765336, -0.04306034, -0.01655441])
        # get closest centroids to the tunnel points
        tunnel_left_centroid = KdNode.nearestNeighbor(tunnel_left, self.kd_centroid_root).point
        tunnel_right_centroid = KdNode.nearestNeighbor(tunnel_right, self.kd_centroid_root).point
        tunnel_left_centroid_idx = np.where(np.all(self.centroids == tunnel_left_centroid, axis=1))[0][0]
        tunnel_right_centroid_idx = np.where(np.all(self.centroids == tunnel_right_centroid, axis=1))[0][0]

        line = Line3D(tunnel_left_centroid, tunnel_right_centroid, width=3, color=Color.BLACK)
        self.addShape(line, "tunnel_line_1")
        self.tunnel_lines = [line]
        self.tunnel_lines_idxs = [[tunnel_left_centroid_idx, tunnel_right_centroid_idx]]

    def setup_geography_specifics(self):
        '''Set up the specifics of the geography that concern the graph
        for shortest paths calculation.'''
        self.setup_mountainous_area()

        self.setup_tunnels()

        # add as needed...

    def toggle_mesh(self, action=-1):
        '''Toggle which parts of the mesh to show.
        - 0: show only the wireframe
        - 1: show only the mesh
        - anything else: show both'''
        self.removeShape("mesh")
        self.removeShape("wireframe")
        if action == 0:
            # show only the wireframe
            self.addShape(self.wireframe, "wireframe")
        elif action == 1:
            # show only the mesh
            self.addShape(self.mesh, "mesh")
        else:
            # show both
            self.addShape(self.mesh, "mesh")
            self.addShape(self.wireframe, "wireframe") 

    def create_grid(self):
        '''Creates a 3D grid on the z=0 plane'''
        # create a grid of evenly-spaced points
        grid = np.linspace(-1, 1, self.GRID_SIZE) # split from -1 to 1 into GRID_SIZE parts
        grid_x, grid_y = np.meshgrid(grid, grid) # create a meshgrid, i.e., a 2D grid of points
        grid_x_flat = grid_x.ravel() # flatten the grid
        grid_y_flat = grid_y.ravel() 
        
        # Generate Perlin noise values for the grid points
        z_values = np.array([pnoise2(x, y, octaves=1, persistence=0.5, lacunarity=2.0, repeatx=self.GRID_SIZE, repeaty=self.GRID_SIZE, base=0)
                            for x, y in zip(grid_x_flat, grid_y_flat)])

        # Combine x, y, and z coordinates
        grid = np.column_stack([grid_x_flat, grid_y_flat, z_values])
        pointset_size = 1 if self.GRID_SIZE < 50 else 0.5
        grid = PointSet3D(grid, size=pointset_size, color=self.GRID_COLOR)
        self.grid = grid
        self.addShape(self.grid, "grid")

    def triangulate_grid(self, grid, size, x_min, x_max):
        '''Triangulates the grid to form a mesh.'''
        list_of_indexed_triangles = []
        for i in range(len(grid)):
            next = i + 1
            upper = i - size
            diagonal = i + size + 1
            if next % size != 0:
                if upper >= 0:
                    list_of_indexed_triangles.append(np.array([i, upper, next]))
                if diagonal < len(grid):
                    list_of_indexed_triangles.append(np.array([i, next, diagonal]))
        
        # store the indices of the triangles
        self.triangle_indices = list_of_indexed_triangles

        line_indices = []
        for i, triangle_index in enumerate(list_of_indexed_triangles):
            for j in range(3):
                line_indices.append((triangle_index[j], triangle_index[(j + 1) % 3]))

        lineset_width = 1 if self.GRID_SIZE < 50 else 0.5
        lineset = LineSet3D(grid, line_indices, width=lineset_width,color=self.GRID_COLOR)
        self.grid_lines = lineset
        # add the lineset to the scene
        self.addShape(lineset, "grid_lines")

        self.triangles_lineset =lineset
    
    def perform_file_checks(self):
        update_centroids = True
        update_distances = True
        update_adjacency = True
        update_shortest_paths = True
        update_elevation_distances = True
        update_elevated_shortest_paths = True

        # if we don't have a mesh, we need to check if the grid has changed
        if self.mesh is None:
            # file paths
            if self.GRID_SIZE == 100:
                path = os.path.join(os.path.dirname(__file__), "resources", "grid_100")
                grid_100_file_path = os.path.join(path, "grid.npy")
                if os.path.exists(grid_100_file_path):
                    pass # success
                else:
                    path=os.path.join(os.path.dirname(__file__), "resources", "terrain") # revert
            else:
                path = os.path.join(os.path.dirname(__file__), "resources", "terrain")
            grid_file_path = os.path.join(path, "grid.npy")

            # check if grid.npy exists
            console_log("Checking if the grid file exists...")
            grid_file_check = os.path.exists(grid_file_path)
            if not grid_file_check:
                console_log("Grid file does not exist. Saving grid...\n---------")
                start_time = time()
                np.save(grid_file_path, self.grid.points)
                end_time = time()
                console_log(f"Grid saved in {end_time - start_time} seconds.")
            
            # check if the instance grid is the same as the saved grid
            if grid_file_check:
                console_log("Checking if the grid is the same as the stored grid...")
                start_time = time()
                saved_grid = np.load(grid_file_path)
                end_time = time()
                console_log(f"Stored grid loaded in {end_time - start_time} seconds.")
                is_same_grid = np.array_equal(saved_grid, self.grid.points)
                console_log(f"The instance grid is the same as the stored grid? {is_same_grid}")

                # if the grids are not the same, store the new grid, both need updating
                if not is_same_grid:
                    console_log("Grid has changed, storing new one...\n---------")
                    np.save(grid_file_path, self.grid.points)
                    return update_centroids, update_distances, update_adjacency, update_shortest_paths,\
                        update_elevation_distances, update_elevated_shortest_paths 
            # else, if the grid file didn't exist, we've already assured that the grid is the same as the saved grid
            else:
                is_same_grid = True
        # if we do have a mesh, we consider we have the same "grid"
        else:
            path = os.path.join(os.path.dirname(__file__), "resources", "terrain")
            is_same_grid = True
            
        file_ext = "npy" if not self.mesh else "npz"
        console_log("Checking if the centroids exist...")
        centroids_file_path = os.path.join(path, f"centroids.npy")

        # if the grid is the same as the saved grid and we have the centroids saved, load them
        if os.path.exists(centroids_file_path) and is_same_grid:
            console_log("Centroids exist, grid hasn't changed.")
            update_centroids = False
        # else, if the grid is the same as the saved grid but we don't have the centroids saved, calculate them, save them, and load them
        elif not os.path.exists(centroids_file_path) and is_same_grid:
            console_log("Centroids do not exist, grid hasn't changed.")
            update_centroids = True

        '''DEPRECATED
        console_log("Checking if the centroid-to-triangle-index mapping exists...")
        map_file_path = os.path.join(path, "centroid_to_triangle_index_map.pkl")

        # if we don't need to update the centroids, check if the mapping exists
        if not update_centroids:
            if os.path.exists(map_file_path):
                console_log("Centroid-to-triangle-index mapping exists.")
                update_centroids = False # all good, no need to update
            else:
                console_log("Centroid-to-triangle-index mapping does not exist.")
                update_centroids = True # need to update
        else:
            pass # no need to check if the mapping exists if we need to update the shortest paths anyway
            '''

        console_log("Checking if the adjacency matrix exists...")
        adj_file_path = os.path.join(path, f"adjacency.{file_ext}")

        if os.path.exists(adj_file_path) and is_same_grid:
            console_log("Adjacency matrix exists, grid hasn't changed.")
            update_adjacency = False
        elif not os.path.exists(adj_file_path) and is_same_grid:
            console_log("Adjacency matrix does not exist, grid hasn't changed.")
            update_adjacency = True

        console_log("Checking if the distances matrix exists...")
        
        distances_file_path = os.path.join(path, f"centroid_distances.{file_ext}")
    
        # if the grid is the same as the saved grid and we have the distances matrix saved, load it
        if os.path.exists(distances_file_path) and is_same_grid:
            console_log("Distances matrix exists, grid hasn't changed.")
            update_distances = False
        # else, if the grid is the same as the saved grid but we don't have the distances matrix saved, calculate it, save it, and load it
        elif not os.path.exists(distances_file_path) and is_same_grid:
            console_log("Distances matrix does not exist, grid hasn't changed.")
            update_distances = True
        
        console_log("Checking if the shortest paths exist...")
        shortest_paths_file_path = os.path.join(path, "shortest_paths.npy")
        # if the grid is the same as the saved grid, and we have the shortest paths saved, load them
        if os.path.exists(shortest_paths_file_path) and is_same_grid:
            console_log("Shortest paths exist, grid hasn't changed.")
            update_shortest_paths = False
        # else, if the grid is the same as the saved grid but we don't have the shortest paths saved, calculate them, save them, and load them
        elif not os.path.exists(shortest_paths_file_path) and is_same_grid:
            console_log("Shortest paths do not exist, grid hasn't changed.")
            update_shortest_paths = True

        console_log("Checking if the elevation distance matrix exists...")
        elev_dist_file_path = os.path.join(path, f"elevation_distances.{file_ext}")
        # if the grid is the same as the saved grid, and we have the elevation distances saved, load them
        if os.path.exists(elev_dist_file_path) and is_same_grid:
            console_log("Elevation distances exist, grid hasn't changed.")
            update_elevation_distances = False
        # else, if the grid is the same as the saved grid but we don't have the elevation distances saved, calculate them, save them, and load them
        elif not os.path.exists(elev_dist_file_path) and is_same_grid:
            console_log("Elevation distances do not exist, grid hasn't changed.")
            update_elevation_distances = True

        console_log("Checking if the elevated shortest paths exist...")
        elev_shortest_paths_file_path = os.path.join(path, "elevated_shortest_paths.npy")
        # if the grid is the same as the saved grid, and we have the elevated shortest paths saved, load them
        if os.path.exists(elev_shortest_paths_file_path) and is_same_grid:
            console_log("Elevated shortest paths exist, grid hasn't changed.")
            update_elevated_shortest_paths = False
        
        return update_centroids, update_distances, update_adjacency, update_shortest_paths,\
              update_elevation_distances, update_elevated_shortest_paths
    
    def calculate_centroids(self, update:bool=False):
        '''Calculates the centroids of the triangles.'''

        # if self.GRID_SIZE == 100:
        #     path = os.path.join(os.path.dirname(__file__), "resources", "grid_100")
        #     centroids_100_file_path = os.path.join(path, "centroids.npy")
        #     if os.path.exists(centroids_100_file_path):
        #         pass # success
        #     else:
        #         path=os.path.join(os.path.dirname(__file__), "resources", "terrain") # revert
        # else:
        path = os.path.join(os.path.dirname(__file__), "resources", "terrain")
        centroids_file_path = os.path.join(path, "centroids.npy")
        # map_file_path = os.path.join(path, "centroid_to_triangle_index_map.pkl") # DEPRECATED
        if not update:
            console_log("Centroids exist.")
            console_log("Loading the centroids...")
            start_time = time()
            centroids = np.load(centroids_file_path)
            # Load the map # DEPRECATED
            # with open(map_file_path, 'rb') as map_file:
            #     centroid_to_triangle_index_map = pickle.load(map_file)
            end_time = time()
            console_log(f"Centroids loaded in {end_time - start_time} seconds.")
        else:
            centroids = calculate_triangle_centroids(self.triangle_indices, self.grid.points)
            centroids = np.array(centroids)
            console_log("Saving the centroids...")
            start_time = time()
            np.save(centroids_file_path, centroids)
            # Save the map # DEPRECATED
            # with open(map_file_path, 'wb') as map_file:
            #     pickle.dump(centroid_to_triangle_index_map, map_file)
            end_time = time()
            console_log(f"Centroids saved in {end_time - start_time} seconds.")

        console_log(f"Shape of the centroids array: {centroids.shape}")
        self.centroids = centroids
        # self.centroid_to_triangle_index_map = centroid_to_triangle_index_map # DEPRECATED
        return centroids
    
    def create_centroids_kd_tree(self):
        '''Creates a KD-Tree for quick nearest neighbor search of the centroids.'''
        console_log("Creating the KD-Tree for centroids...")
        start_time = time()
        self.kd_centroid_root = KdNode.build_kd_node(self.centroids)
        end_time = time()
        console_log(f"KD-Tree for centroids created in {end_time - start_time} seconds.")
        return self.kd_centroid_root

    def calculate_adjacency_matrix(self):
        '''Calculates the adjacency matrix for the grid.'''
        console_log("Calculating the adjacency matrix...")
        adjacency_matrix = np.zeros((len(self.triangle_indices), len(self.triangle_indices)))
        for i, triangle in enumerate(tqdm(self.triangle_indices, desc="Calculating adjacency matrix")):
            # for every triangle, we consider its adjacent triangles only if they share an edge
            for j, other_triangle in enumerate(self.triangle_indices):
                if i == j:
                    continue
                if i in self.tunnel_lines_idxs[0] and j in self.tunnel_lines_idxs[0]:
                    adjacency_matrix[i, j] = 1
                    adjacency_matrix[j, i] = 1
                    continue
                # if the triangles share an edge, they are adjacent
                if len(set(triangle) & set(other_triangle)) == 2:
                    adjacency_matrix[i, j] = 1
        console_log(f"Shape of the adjacency matrix: {adjacency_matrix.shape}")
        return adjacency_matrix
    
    def calculate_adjacency_matrix_optimized(self, triangle_indices):
        '''Calculates the adjacency matrix for the grid.'''
        num_triangles = len(triangle_indices)
        adjacency_matrix = lil_matrix((num_triangles, num_triangles), dtype=np.int8)  # for construction

        # dictionary to map edges to triangles
        edge_to_triangles = {}
        
        for i, triangle in enumerate(triangle_indices):
            edges = [
                tuple(sorted([triangle[0], triangle[1]])),
                tuple(sorted([triangle[1], triangle[2]])),
                tuple(sorted([triangle[2], triangle[0]]))
            ]
            for edge in edges:
                if edge not in edge_to_triangles:
                    edge_to_triangles[edge] = []
                edge_to_triangles[edge].append(i)
        
        # for every edge, if two triangles share the edge, they are adjacent
        for edge, triangles in tqdm(edge_to_triangles.items(), desc="Calculating adjacency matrix"):
            for i in range(len(triangles)):
                for j in range(i + 1, len(triangles)):
                    adjacency_matrix[triangles[i], triangles[j]] = 1
                    adjacency_matrix[triangles[j], triangles[i]] = 1

        # add the tunnel lines
        for tunnel in self.tunnel_lines_idxs:
            adjacency_matrix[tunnel[0], tunnel[1]] = 1
            adjacency_matrix[tunnel[1], tunnel[0]] = 1
        
        return adjacency_matrix.tocsr()

    def create_adjacency_matrix(self, update:bool=False):
        '''Creates an adjacency matrix for the grid.'''
        adjacency_matrix = None

        # file paths
        # if self.GRID_SIZE == 100 and self.mesh is None:
        #     path = os.path.join(os.path.dirname(__file__), "resources", "grid_100")
        #     adj_100_file_path = os.path.join(path, "adjacency.npy")
        #     if os.path.exists(adj_100_file_path):
        #         pass # success
        #     else:
        #         path=os.path.join(os.path.dirname(__file__), "resources", "terrain") # revert
        # else:
        path = os.path.join(os.path.dirname(__file__), "resources", "terrain")
        adj_file_path = os.path.join(path, "adjacency.npz") 
        if not update:
            console_log("Adjacency matrix exists.")
            
            console_log("Loading the adjacency matrix...")
            start_time = time()
            adjacency_matrix = load_npz(adj_file_path)
            end_time = time()
            console_log(f"Adjacency matrix loaded in {end_time - start_time} seconds.")
        else:
            console_log("Adjacency matrix: calculating and storing it...")

            # calculate the adjacency matrix
            adjacency_matrix = self.calculate_adjacency_matrix_optimized(self.triangle_indices) # returns a csr_matrix
            console_log("Saving the adjacency matrix...")
            start_time = time()
            save_npz(adj_file_path, adjacency_matrix)
            end_time = time()
            console_log(f"Adjacency matrix saved in {end_time - start_time} seconds.")

        self.adjacency_matrix = adjacency_matrix
        return adjacency_matrix

    def calculate_distances_matrix(self):
        '''Calculates a matrix of distances between the centroids of the triangles.'''

        # for all the centroids, calculate the distances between them
        centroids = self.centroids
        distances_matrix = np.zeros((len(centroids), len(centroids)))
        for i, centroid1 in enumerate(tqdm(centroids, desc="Calculating distances", leave=True)):
            for j, centroid2 in enumerate(centroids):
                # if the triangle centroids are adjacent, calculate the distance between them, otherwise 0 (same triangle or not adjacent)
                if self.adjacency_matrix[i, j] == 1:
                    distances_matrix[i, j] = np.linalg.norm(centroid1 - centroid2)
                else:
                    distances_matrix[i, j] = 0
        # shape of the distances matrix for 80x80 grid should be (80, 80)
        console_log(f"Shape of the distances matrix: {distances_matrix.shape}")
        return distances_matrix
    
    def calculate_distances_matrix_optimized(self, centroids, adjacency_matrix):
        '''Calculates a matrix of distances between the centroids of the triangles.'''

        num_centroids = len(centroids)
        distances_matrix = lil_matrix((num_centroids, num_centroids)) # for construction

        # Calculate distances only for adjacent centroids
        for i in tqdm(range(num_centroids), desc="Calculating distances", leave=True):
            adjacent_indices = adjacency_matrix[i].indices
            # check if there is geography specific information
            # if the centroid is in the mountainous area, we consider it to be impassable
            if i in self.mountainous_area_idxs:
                distances_matrix[i, adjacent_indices] = np.inf
                continue
            distances = np.linalg.norm(centroids[i] - centroids[adjacent_indices], axis=1)
            # if the centroid is in the tunnel, we consider it to be inexpensive to traverse
            if i in self.tunnel_lines_idxs[0]:
                distances_matrix[i, adjacent_indices] = distances/10
                continue
            distances_matrix[i, adjacent_indices] = distances

        distances_matrix = distances_matrix.tocsr()
        console_log(f"Shape of the distances matrix: {distances_matrix.shape}")
        return distances_matrix
        
    def create_distances_matrix(self, update:bool=False):
        '''Creates a matrix of distances between the centroids of the triangles.'''
        centroid_distances_matrix = None
        
        # if self.GRID_SIZE == 100 and self.mesh is None:
        #     path = os.path.join(os.path.dirname(__file__), "resources", "grid_100")
        #     distances_100_file_path = os.path.join(path, "centroid_distances.npy")
        #     if os.path.exists(distances_100_file_path):
        #         pass # success
        #     else:
        #         path=os.path.join(os.path.dirname(__file__), "resources", "terrain") # revert
        # else:
        path = os.path.join(os.path.dirname(__file__), "resources", "terrain")
        distances_file_path = os.path.join(path, "centroid_distances.npz")
        if not update:
            console_log("Loading the distances matrix...")
            start_time = time()
            centroid_distances_matrix = load_npz(distances_file_path)
            end_time = time()
            console_log(f"Distances matrix loaded in {end_time - start_time} seconds.")
        else:
            console_log("Need to calculate the distances matrix...")
            start_time = time()
            centroid_distances_matrix = self.calculate_distances_matrix_optimized(self.centroids, self.adjacency_matrix) # self.calculate_distances_matrix()
            end_time = time()
            console_log(f"Distances matrix calculated in {end_time - start_time} seconds.")
            console_log("Saving the distances matrix...")
            start_time = time()
            save_npz(distances_file_path, centroid_distances_matrix)
            end_time = time()
            console_log(f"Distances matrix saved in {end_time - start_time} seconds.")
        
        self.centroid_distances_matrix = centroid_distances_matrix
        return centroid_distances_matrix
    
    def create_shortest_paths_matrix(self, update:bool=False):
        '''Creates a matrix of shortest paths between the centroids of the triangles.'''
        shortest_paths_matrix = None
        if self.GRID_SIZE == 100 and self.mesh is None:
            path = os.path.join(os.path.dirname(__file__), "resources", "grid_100")
            shortest_paths_100_file_path = os.path.join(path, "shortest_paths.npy")
            if os.path.exists(shortest_paths_100_file_path):
                pass # success
            else:
                path=os.path.join(os.path.dirname(__file__), "resources", "terrain") # revert
        else:
            path = os.path.join(os.path.dirname(__file__), "resources", "terrain")
        shortest_paths_file_path = os.path.join(path, "shortest_paths.npy")
        if not update:
            console_log("Loading the shortest paths matrix...")
            start_time = time()
            shortest_paths_matrix = np.load(shortest_paths_file_path, allow_pickle=True)
            end_time = time()
            console_log(f"Shortest paths matrix loaded in {end_time - start_time} seconds.")
        else:
            console_log("Calculating the shortest paths matrix...")
            start_time = time()
            self.dijkstra = Dijkstra(self.centroid_distances_matrix.toarray()) # Dijkstra object for finding shortest paths between centroids
            self.dijkstra.calculate_all_shortest_paths()
            shortest_paths_matrix = self.dijkstra.get_distances()
            end_time = time()
            console_log(f"Shortest paths matrix calculated in {end_time - start_time} seconds.")
            console_log("Saving the shortest paths matrix...")
            start_time = time()
            np.save(shortest_paths_file_path, shortest_paths_matrix)
            end_time = time()
            console_log(f"Shortest paths matrix saved in {end_time - start_time} seconds.")
        
        console_log("Shape of the shortest paths matrix: ", shortest_paths_matrix.shape)
        self.shortest_paths_matrix = shortest_paths_matrix
        return shortest_paths_matrix
    
    # def calculate_elevation_distance_matrix(self, centroids, adjacency_matrix, uphill_weight=1, downhill_weight=1):
    #     '''Calculates the elevation distance matrix between the centroids of the triangles.'''
    #     num_centroids = len(centroids)
    #     elevation_distance_matrix = lil_matrix((num_centroids, num_centroids)) # for construction
    #     # Initialize matrices to track uphill and downhill indices
    #     uphill_indices_matrix = lil_matrix((num_centroids, num_centroids), dtype=bool)
    #     downhill_indices_matrix = lil_matrix((num_centroids, num_centroids), dtype=bool)

    #     for i in tqdm(range(num_centroids), desc="Calculating elevation distances", leave=True):
    #         adjacent_indices = adjacency_matrix[i].indices
    #         # check if there is geography specific information
    #         # if the centroid is in the mountainous area, we consider it to be impassable
    #         if i in self.mountainous_area_idxs:
    #             elevation_distance_matrix[i, adjacent_indices] = np.inf
    #             continue
    #         # Horizontal distances in the x and y plane
    #         horizontal_distances = np.linalg.norm(centroids[i][:2] - centroids[adjacent_indices][:, :2], axis=1)
    #         # Elevation differences
    #         delta_z = (centroids[i][2] - centroids[adjacent_indices][:, 2]) # direction of uphill or downhill decided by the sign
    #         # positive sign means downhill (z_i > z_adj) and negative sign means uphill (z_i < z_adj)
    #         downhill_values = delta_z > 0
    #         vertical_distances = np.copy(delta_z)
    #         vertical_distances[downhill_values] = -downhill_weight * delta_z[downhill_values] # reduce the gain for downhill movement
    #         vertical_distances[~downhill_values] = uphill_weight * np.abs(delta_z[~downhill_values])
    #         # Calculate the elevation distance
    #         # if the centroid is in the tunnel, we consider it to be massively preferable
    #         if i in self.tunnel_lines_idxs[0]:
    #             elevation_distance_matrix[i, adjacent_indices] = (horizontal_distances + vertical_distances)/10
    #             continue
    #         elevation_distance_matrix[i, adjacent_indices] = horizontal_distances + vertical_distances
            
    #         #---
    #         # Mark uphill and downhill indices
    #         uphill_indices_matrix[i, adjacent_indices[~downhill_values]] = True
    #         downhill_indices_matrix[i, adjacent_indices[downhill_values]] = True

    #     elevation_distance_matrix = elevation_distance_matrix.tocsr()
    #     self.uphill_indices_matrix = uphill_indices_matrix.tocsr() # debug
    #     self.downhill_indices_matrix = downhill_indices_matrix.tocsr() # debug
    #     console_log(f"Shape of the elevation distance matrix: {elevation_distance_matrix.shape}")
    #     return elevation_distance_matrix

    def calculate_elevation_distance_matrix(self, centroids, adjacency_matrix, uphill_weight=1, downhill_weight=1):
        '''Calculates the elevation distance matrix between the centroids of the triangles.'''
        num_centroids = len(centroids)
        elevation_distance_matrix = lil_matrix((num_centroids, num_centroids)) # for construction
        uphill_indices_matrix = lil_matrix((num_centroids, num_centroids), dtype=bool)
        downhill_indices_matrix = lil_matrix((num_centroids, num_centroids), dtype=bool)

        for i in tqdm(range(num_centroids), desc="Calculating elevation distances", leave=True):
            adjacent_indices = adjacency_matrix[i].indices
            
            if i in self.mountainous_area_idxs:
                elevation_distance_matrix[i, adjacent_indices] = np.inf
                continue
            
            horizontal_distances = np.linalg.norm(centroids[i][:2] - centroids[adjacent_indices][:, :2], axis=1)
            delta_y = centroids[i][1] - centroids[adjacent_indices][:, 1]  # difference in y (latitude)
            delta_z = centroids[i][2] - centroids[adjacent_indices][:, 2]  # difference in z (elevation)

            uphill_condition = (delta_y > 0) & (delta_z > 0)  # y increasing and z increasing -> uphill
            downhill_condition = (delta_y > 0) & (delta_z < 0)  # y increasing and z decreasing -> downhill

            vertical_distances = np.zeros_like(delta_z)
            vertical_distances[uphill_condition] = uphill_weight * delta_z[uphill_condition]
            vertical_distances[downhill_condition] = -downhill_weight * delta_z[downhill_condition]

            if i in self.tunnel_lines_idxs[0]:
                elevation_distance_matrix[i, adjacent_indices] = (horizontal_distances + vertical_distances) / 10
            else:
                elevation_distance_matrix[i, adjacent_indices] = horizontal_distances + vertical_distances

            uphill_indices_matrix[i, adjacent_indices[uphill_condition]] = True
            downhill_indices_matrix[i, adjacent_indices[downhill_condition]] = True

        elevation_distance_matrix = elevation_distance_matrix.tocsr()
        #==========================================
        self.uphill_indices_matrix = uphill_indices_matrix.tocsr()
        self.downhill_indices_matrix = downhill_indices_matrix.tocsr()
        # store the uphill and downhill indices for debugging
        path = os.path.join(os.path.dirname(__file__), "resources", "terrain")
        uphill_indices_file_path = os.path.join(path, "uphill_indices.npz")
        downhill_indices_file_path = os.path.join(path, "downhill_indices.npz")
        save_npz(uphill_indices_file_path, self.uphill_indices_matrix)
        save_npz(downhill_indices_file_path, self.downhill_indices_matrix)
        #==========================================
        console_log(f"Shape of the elevation distance matrix: {elevation_distance_matrix.shape}")
        return elevation_distance_matrix
    
    def create_elevation_distance_matrix(self, update:bool=False):
        '''Creates the elevation distance matrix between the centroids of the triangles.'''
        if self.elevation_distance_matrix is not None:
            return self.elevation_distance_matrix
        
        elevation_distance_matrix = None
        # if self.GRID_SIZE == 100:
        #     path = os.path.join(os.path.dirname(__file__), "resources", "grid_100")
        #     elev_dist_100_file_path = os.path.join(path, "elevation_distances.npy")
        #     if os.path.exists(elev_dist_100_file_path):
        #         pass # success
        #     else:
        #         path=os.path.join(os.path.dirname(__file__), "resources", "terrain") # revert
        # else:
        path = os.path.join(os.path.dirname(__file__), "resources", "terrain")
        elev_dist_file_path = os.path.join(path, "elevation_distances.npz")
        if not update:
            console_log("Loading the elevation distance matrix...")
            start_time = time()
            elevation_distance_matrix = load_npz(elev_dist_file_path)
            try:
                self.uphill_indices_matrix = load_npz(os.path.join(path, "uphill_indices.npz"))
                self.downhill_indices_matrix = load_npz(os.path.join(path, "downhill_indices.npz"))
            except:
                console_log("Uphill and downhill indices matrices not found.")
            end_time = time()
            console_log(f"Elevation distance matrix loaded in {end_time - start_time} seconds.")
        else:
            console_log("Calculating the elevation distance matrix...")
            start_time = time()
            elevation_distance_matrix = self.calculate_elevation_distance_matrix(self.centroids, self.adjacency_matrix,\
                                                                                  uphill_weight=8, downhill_weight=0.5) 
            end_time = time()
            console_log(f"Elevation distance matrix calculated in {end_time - start_time} seconds.")
            console_log("Saving the elevation distance matrix...")
            start_time = time()
            save_npz(elev_dist_file_path, elevation_distance_matrix)
            end_time = time()
            console_log(f"Elevation distance matrix saved in {end_time - start_time} seconds.")

        self.elevation_distance_matrix = elevation_distance_matrix
        return elevation_distance_matrix
    
    def create_elevated_shortest_paths_matrix(self, update:bool=False):
        '''Creates a matrix of shortest paths between the centroids of the triangles.'''
        if self.elevated_shortest_paths_matrix is not None:
            return self.elevated_shortest_paths_matrix
        
        shortest_paths_matrix = None
        # if self.GRID_SIZE == 100:
        #     path = os.path.join(os.path.dirname(__file__), "resources", "grid_100")
        #     shortest_paths_100_file_path = os.path.join(path, "elevated_shortest_paths.npy")
        #     if os.path.exists(shortest_paths_100_file_path):
        #         pass # success
        #     else:
        #         path=os.path.join(os.path.dirname(__file__), "resources", "terrain") # revert
        # else:
        path = os.path.join(os.path.dirname(__file__), "resources", "terrain")
        shortest_paths_file_path = os.path.join(path, "elevated_shortest_paths.npy")
        if not update:
            console_log("Loading the elevated shortest paths matrix...")
            start_time = time()
            shortest_paths_matrix = np.load(shortest_paths_file_path, allow_pickle=True)
            end_time = time()
            console_log(f"Elevated shortest paths matrix loaded in {end_time - start_time} seconds.")
        else:
            console_log("Calculating the shortest paths matrix...")
            start_time = time()
            self.dijkstra = Dijkstra(self.elevation_distance_matrix.toarray()) # Dijkstra object for finding shortest paths between centroids with elevation
            self.dijkstra.calculate_all_shortest_paths()
            shortest_paths_matrix = self.dijkstra.get_distances()
            end_time = time()
            console_log(f"Elevated shortest paths matrix calculated in {end_time - start_time} seconds.")
            console_log("Saving the elevated shortest paths matrix...")
            start_time = time()
            np.save(shortest_paths_file_path, shortest_paths_matrix)
            end_time = time()
            console_log(f"Shortest paths matrix saved in {end_time - start_time} seconds.")
        
        console_log("Shape of the elevated shortest paths matrix: ", shortest_paths_matrix.shape)
        self.elevated_shortest_paths_matrix = shortest_paths_matrix
        return shortest_paths_matrix
    
    def _scenario_mode_init(self):
        self.DEBUG = DEBUG
        self.CONSOLE_TALK = CONSOLE_TALK
        self.TRIAL_MODE = TRIAL_MODE

    def _console_log_scenario(self):
        self.terminal_log("---")
        self.terminal_log(f"DEBUG: {self.DEBUG}, CONSOLE_TALK: {self.CONSOLE_TALK}, TRIAL_MODE: {self.TRIAL_MODE}")
        self.terminal_log(f"Population: {self.POPULATION}, Wells: {self.WELLS}, Number of infected wells: {len(self.infected_wells_indices)}, Infected wells indices: {self.infected_wells_indices}")
        self.terminal_log(f"DENSE REGIONS: {self.DENSE_REGIONS}")
        self.terminal_log(f"EUCLIDEAN: {self.EUCLIDEAN}")
        self.terminal_log(f"ELEVATION DIST: {self.ELEVATION_DISTANCE}")
        self.terminal_log(f"RANDOM_SELECTION: {self.RANDOM_SELECTION}")
        self.terminal_log(f"Chances of choosing the closest well: {self.P1}, Chances of choosing the second closest well: {self.P2}, Chances of choosing the third closest well: {self.P3}") if self.RANDOM_SELECTION else None
        self.terminal_log(f"Number of infected people: {len(self.infected_people_indices)}")
        self.terminal_log("---")

        # console_log("---")
        # console_log(f"DEBUG: {self.DEBUG}, CONSOLE_TALK: {self.CONSOLE_TALK}, TRIAL_MODE: {self.TRIAL_MODE}")
        # console_log(f"Population: {self.POPULATION}, Wells: {self.WELLS}, Number of infected wells: {len(self.infected_wells_indices)}, Infected wells indices: {self.infected_wells_indices}")
        # console_log(f"RANDOM_SELECTION: {self.RANDOM_SELECTION}")
        # console_log(f"Chances of choosing the closest well: {self.P1}, Chances of choosing the second closest well: {self.P2}, Chances of choosing the third closest well: {self.P3}") if self.RANDOM_SELECTION else None
        # console_log(f"Number of infected people: {len(self.infected_people_indices)}")
        # console_log("---")

    @world_space
    def on_mouse_press(self, x, y, z, button, modifiers):
        if (button == Mouse.MOUSELEFT or button == Mouse.MOUSERIGHT) and (modifiers & Key.MOD_SHIFT or modifiers & Key.MOD_ALT):
            if np.isinf(z) or np.isinf(x) or np.isinf(y):
                console_log("Mouse pressed outside the bounds, ", x, y, z)
                return
            self.my_mouse_pos.x = x
            self.my_mouse_pos.y = y
            self.my_mouse_pos.z = z
            # self.my_mouse_pos.color = Color.WHITE
            self.my_mouse_pos.color = [1, 1, 1, 0]

            # if the mouse is released within the bound...
            if self.within_bound(x, y):
                console_log(f"Mouse released at ({x}, {y}, {z})")
                # if the right mouse button was released...
                if button == Mouse.MOUSERIGHT and modifiers & Key.MOD_SHIFT:
                    # find the closest well to the mouse position
                    closest_well_index = np.argmin(np.linalg.norm(np.array(self.wells_pcd.points) - np.array([x, y, z]), axis=1))
                    # if its within a certain distance...
                    if np.linalg.norm(np.array(self.wells_pcd.points[closest_well_index]) - np.array([x, y, z])) < 0.05:
                        # check if the closest well is not already infected
                        if closest_well_index not in self.infected_wells_indices:
                            ## infect the closest well
                            # get the current infected percentage
                            infected_percentage = len(self.infected_people_indices) / self.POPULATION
                            self.infect_single_well(closest_well_index)
                            self.find_infected_people() if not self.RANDOM_SELECTION else self.find_infected_people_stochastic()
                            # get the new infected percentage
                            new_infected_percentage = len(self.infected_people_indices) / self.POPULATION
                            # print the percentage increase
                            self.terminal_log(f"Percentage impact: {(new_infected_percentage - infected_percentage)*100}")
                        else:
                            ## disenfect the closest well
                            # get the current infected percentage
                            infected_percentage = len(self.infected_people_indices) / self.POPULATION
                            self.disinfect_single_well(closest_well_index)
                            self.find_infected_people() if not self.RANDOM_SELECTION else self.find_infected_people_stochastic()
                            # get the new infected percentage
                            new_infected_percentage = len(self.infected_people_indices) / self.POPULATION
                            # print the percentage decrease
                            self.terminal_log(f"Percentage impact: {(new_infected_percentage - infected_percentage)*100}")
                # else, if the left mouse button was released...
                elif button == Mouse.MOUSELEFT and modifiers & Key.MOD_SHIFT:
                    infected_percentage = len(self.infected_people_indices) / self.POPULATION
                    # find the closest well to the mouse position
                    closest_well_index = np.argmin(np.linalg.norm(np.array(self.wells_pcd.points) - np.array([x, y, z]), axis=1))
                    # if its within a certain distance...
                    if np.linalg.norm(np.array(self.wells_pcd.points[closest_well_index]) - np.array([x, y, z])) < 0.05:
                        # remove the closest well
                        self.remove_single_well(closest_well_index)
                    else:
                        # add a new well at the mouse position
                        self.add_single_well(x, y)
                    self.find_infected_people() if not self.RANDOM_SELECTION else self.find_infected_people_stochastic()
                    new_infected_percentage = len(self.infected_people_indices) / self.POPULATION
                    self.terminal_log(f"Percentage impact: {(new_infected_percentage - infected_percentage)*100}")
                    self.resetVoronoi()

                # debug
                elif modifiers & Key.MOD_ALT:
                    if button == Mouse.MOUSELEFT and modifiers & Key.MOD_ALT:
                        # find the closest centroid to the mouse position
                        start_point = np.array([x, y, z])
                        closest_centroid = KdNode.nearestNeighbor(start_point, self.kd_centroid_root)
                        closest_centroid = closest_centroid.point
                        closest_centroid_index = np.where(np.all(self.centroids == closest_centroid, axis=1))[0][0]
                        self.removeShape("show_path_start")
                        self.addShape(Point3D(closest_centroid, size=0.3, color=Color.RED), "show_path_start")
                        self.debug_names.append("show_path_start")
                        self.show_path_start = closest_centroid_index
                        console_log(f"Start point: {closest_centroid}")
                    if button == Mouse.MOUSERIGHT and modifiers & Key.MOD_ALT:
                        # find the closest centroid to the mouse position
                        end_point = np.array([x, y, z])
                        closest_centroid = KdNode.nearestNeighbor(end_point, self.kd_centroid_root)
                        closest_centroid = closest_centroid.point
                        closest_centroid_index = np.where(np.all(self.centroids == closest_centroid, axis=1))[0][0]
                        self.removeShape("show_path_end")
                        self.addShape(Point3D(closest_centroid, size=0.3, color=Color.BLUE), "show_path_end")
                        self.debug_names.append("show_path_end")
                        self.show_path_end = closest_centroid_index
                        console_log(f"End point: {closest_centroid}")
                    if (self.show_path_start is not None and self.show_path_end is not None) and not np.allclose(self.show_path_start, self.show_path_end):
                        console_log(f"Finding path...")
                        self.debug_names = self._show_path(self.show_path_start, self.show_path_end)
                        console_log(f"Path set")
                    
            # self.updateShape("mouse")

    def within_bound(self, x, y):
        '''Checks if the point (x, y) is within the bounding box.'''
        # return x >= self.bbx[0][0] and x <= self.bbx[1][0] and y >= self.bbx[0][1] and y <= self.bbx[1][1]
        polygon = np.array([[self.bound.x_min, self.bound.y_min], [self.bound.x_max, self.bound.y_min],\
                             [self.bound.x_max, self.bound.y_max], [self.bound.x_min, self.bound.y_max]])
        return is_inside_polygon_2d(np.array([[x, y]]), polygon)

    def on_key_press(self, symbol, modifiers):

        def version_1():
            self.POPULATION = 1000 if not self.TRIAL_MODE else 5
            self.POPULATION_SIZE = 1 if not self.TRIAL_MODE else 0.7
            self.WELLS = 15 if not self.TRIAL_MODE else 3
            self.reset_scene()

        def version_2():
            self.POPULATION = 10000 if not self.TRIAL_MODE else 10
            self.POPULATION_SIZE = 0.8 if not self.TRIAL_MODE else 0.7
            self.WELLS = 30 if not self.TRIAL_MODE else 5
            self.reset_scene()
        
        def version_3():
            self.POPULATION = 30000 if not self.TRIAL_MODE else 15
            self.POPULATION_SIZE = 0.7 if not self.TRIAL_MODE else 0.7
            self.WELLS = 45 if not self.TRIAL_MODE else 7
            self.reset_scene()

        if not modifiers & Key.MOD_ALT:
            # print the scenario parameters
            if symbol == Key.BACKSPACE:
                self._console_log_scenario()
            # reset the scene
            if symbol == Key.ENTER:
                self.reset_scene()
                self._print_instructions()
            # toggle between trial mode and normal mode
            if symbol == Key.UP:
                self.TRIAL_MODE = not self.TRIAL_MODE
                version_1()
            # increase or decrease the number of wells
            if symbol == Key.RIGHT:
                self.WELLS += 1
                self.reset_scene()
            if symbol == Key.LEFT:
                self.WELLS -= 1
                self.reset_scene()
            # increase or decrease the population
            if symbol == Key.M:
                self.POPULATION += 100 if not self.TRIAL_MODE else 10
                self.reset_scene()
            if symbol == Key.N:
                self.POPULATION -= 100 if not self.TRIAL_MODE else 10
                self.reset_scene()
            # toggle between dense regions
            if symbol == Key.W:
                self.DENSE_REGIONS = not self.DENSE_REGIONS
                self.reset_scene()
            if symbol == Key.E:
                self.EUCLIDEAN = not self.EUCLIDEAN
                self.removeShape("infected_people")
                self.find_infected_people() if not self.RANDOM_SELECTION else self.find_infected_people_stochastic()
            # toggle between random selection and stochastic selection
            if symbol == Key.R:
                self.RANDOM_SELECTION = not self.RANDOM_SELECTION
                self.P1 = 0.8
                self.P2 = 0.15
                self.P3 = 0.05
                # self.reset_scene() # for some reason, resetting the scene chooses new infected well population, unlike 2D
                self.removeShape("infected_people")
                self.find_infected_people() if not self.RANDOM_SELECTION else self.find_infected_people_stochastic()
                self._print_instructions()
            # increase or decrease the probabilities of choosing the closest well
            if symbol == Key.P:
                if modifiers & Key.MOD_SHIFT:
                    self.P1 += 0.05
                    self.P2 -= 0.025
                    self.P3 -= 0.025
                else:
                    self.P1 -= 0.05
                    self.P2 += 0.025
                    self.P3 += 0.025
                # make sure the probabilities are between 0 and 1
                self.P1 = 0 if self.P1 < 0 else self.P1
                self.P1 = 1 if self.P1 > 1 else self.P1

                self.P2 = 0 if self.P2 < 0 else self.P2
                self.P2 = 1 if self.P2 > 1 else self.P2

                self.P3 = 0 if self.P3 < 0 else self.P3
                self.P3 = 1 if self.P3 > 1 else self.P3

                # ensure the probabilities always sum to 1
                total = self.P1 + self.P2 + self.P3
                self.P1 /= total
                self.P2 /= total
                self.P3 /= total

                # self.reset_scene() # for some reason, resetting the scene chooses new infected well population, unlike 2D
                self.removeShape("infected_people")
                self.find_infected_people() if not self.RANDOM_SELECTION else self.find_infected_people_stochastic()
            if symbol == Key.V and not modifiers & Key.MOD_SHIFT:
                self.VORONOI_VISIBLE = not self.VORONOI_VISIBLE
                # self.Voronoi.generate(self.wells_pcd.points, WIDTH, HEIGHT)
                # edges = self.Voronoi.getEdges()
                if self.Voronoi is not None and self.VORONOI_VISIBLE:
                    # self.Voronoi = self.getVoronoi(self.wells_pcd.points)
                    self.drawVoronoi()
                else:
                    self.removeShape("Voronoi")
                    self.removeShape("Voronoi Points")
            if symbol == Key.V and modifiers & Key.MOD_SHIFT:
                self.COMPUTE_WITH_VORONOI = not self.COMPUTE_WITH_VORONOI
                if self.COMPUTE_WITH_VORONOI:
                    self.Voronoi = self.getVoronoi(self.wells_pcd.points)
                self.reset_scene()
            # use the elevation distance function
            if symbol == Key.G:
                self.ELEVATION_DISTANCE = not self.ELEVATION_DISTANCE
                self.create_elevation_distance_matrix(self.el_dist_matrix_need_update)
                self.create_elevated_shortest_paths_matrix(self.el_short_paths_need_update)
                # self.reset_scene()
                self.removeShape("infected_people")
                self.find_infected_people() if not self.RANDOM_SELECTION else self.find_infected_people_stochastic()
        else:
            ## debug
            roster_colors = [Color.RED, Color.BLUE, Color.GREEN,\
                                Color.CYAN, Color.WHITE, Color.BLACK,\
                                Color.GREEN, Color.YELLOW, Color.DARKRED,\
                                Color.DARKGREEN, Color.YELLOWGREEN, Color.GRAY]
            if symbol == Key.UP and modifiers & Key.MOD_ALT:
                ''' Show the matrices for debugging.'''
                self.DEBUG = True
                if self.DEBUG:
                    names = self._show_matrices(roster_colors[self.current_color_pointer])
                    for name in names:
                        self.debug_names.append(name)
                    console_log(f"DEBUG {self.debug_names}")
            if symbol == Key.RIGHT and modifiers & Key.MOD_ALT:
                ''' Change the color of the grid clockwise.'''
                self.current_color_pointer += 1
                self.current_color_pointer = self.current_color_pointer % 12
                
                if not self.wireframe_toggle:
                    self.grid.colors = np.full_like(self.grid.colors, roster_colors[self.current_color_pointer])
                    self.updateShape("grid")
                    self.grid_lines.colors = np.full_like(self.grid_lines.colors, roster_colors[self.current_color_pointer])
                    self.updateShape("grid_lines") if self.mesh is None else self.updateShape("wireframe")
                elif self.wireframe_toggle and self.mesh is not None:
                    self.mesh.color = roster_colors[self.current_color_pointer]
                    self.updateShape("mesh")
            if symbol == Key.LEFT and modifiers & Key.MOD_ALT:
                ''' Change the color of the grid anti-clockwise.'''
                self.current_color_pointer -= 1
                self.current_color_pointer %= 12

                if not self.wireframe_toggle:
                    self.grid.colors = np.full_like(self.grid.colors, roster_colors[self.current_color_pointer])
                    self.updateShape("grid")
                    self.grid_lines.colors = np.full_like(self.grid_lines.colors, roster_colors[self.current_color_pointer])
                    self.updateShape("grid_lines") if self.mesh is None else self.updateShape("wireframe")
                elif self.wireframe_toggle and self.mesh is not None:
                    self.mesh.color = roster_colors[self.current_color_pointer]
                    self.updateShape("mesh")       
            if symbol == Key.C and modifiers & Key.MOD_ALT:
                '''Toggle the mesh visibility.'''
                if self.mesh is not None:
                    self.wireframe_toggle = not self.wireframe_toggle
                    if self.wireframe_toggle:
                        self.toggle_mesh(1)
                    else:
                        self.toggle_mesh(0)
            if symbol == Key.SPACE and modifiers & Key.MOD_ALT:
                '''Undo all debug changes.'''
                self.DEBUG = False
                self.toggle_mesh(-1)
                if self.wireframe_toggle:
                    self.mesh.color = Color.GREY
                    self.updateShape("mesh")
                else:
                    self.grid.colors = np.full_like(self.grid.colors, Color.GRAY)
                    self.updateShape("grid")
                    self.grid_lines.colors = np.full_like(self.grid_lines.colors, Color.GRAY)
                    self.updateShape("grid_lines") if self.mesh is None else self.updateShape("wireframe")
                for name in self.debug_names:
                    self.removeShape(name)
                self.show_path_end = None
                self.show_path_start = None
            
        # set the scenario to version 1 or 2 or 3
        if symbol == Key._1:
            version_1()
        if symbol == Key._2:
            version_2()
        if symbol == Key._3:
            version_3()

    def scenario_parameters_init(self):
        self.GRID_SIZE = 20 # will create a grid of N x N points, choices: 20, 50, 80
        self.grid = None
        self.grid_lines = None
        self.mesh = None
        self.bbx =[[-1, -1, 0], [1, 1, 0]]
        self.bound = None
        self.Voronoi = None

        # populations, counts, and ratios
        self.POPULATION = 1000
        self.POPULATION_SIZE = 0.7
        self.WELLS = 15
        self.ratio_of_infected_wells = 0.2
        self.P1 = 0.8 # probability of choosing the closest well
        self.P2 = 0.15 # probability of choosing the second closest well
        self.P3 = 0.05 # probability of choosing the third closest well

        # logic, controllers    
        self.RANDOM_SELECTION = False
        self.DENSE_REGIONS = False
        self.EUCLIDEAN = False
        self.ELEVATION_DISTANCE = False
        self.elevation_distance_matrix = None # for check
        self.elevated_shortest_paths_matrix = None # for check
        # self.VORONOI_ACTIVE = False
        self.VORONOI_VISIBLE = False
        self.COMPUTE_WITH_VORONOI = False
        # debug
        self.debug_names = []
        self.wireframe_toggle = False
        self.show_path_start = None
        self.show_path_end = None
        self.uphill_indices_matrix = None
        self.downhill_indices_matrix = None

        # colors
        self.current_color_pointer = 11
        self.GRID_COLOR = Color.GREY
        self.HELPER_COLOR = Color.BLACK
        self.healthy_population_color = Color.BLUE
        self.infected_population_color = Color.YELLOW
        self.healthy_wells_color = Color.GREEN
        self.infected_wells_color = Color.RED

    def _print_instructions(self):
        self.print("> ENTER: reset scene & print instructions.")
        self.print("> BACKSPACE: print scenario parameters.")
        self.print("> UP: toggle trial and normal mode.")
        self.print("> RIGHT or LEFT: increase, decrease wells.")
        self.print("> M or N: increase, decrease population.")
        self.print("> 1 or 2: scenario version 1 or 2.")
        self.print("> W: toggle dense regions.")
        self.print("> E: toggle geodesic or euclidean")
        self.print("> G: consider elevation distances.")
        # self.print("> V: toggle Voronoi diagram.")
        # self.print("> SHIFT + V: use Voronoi diagram for computations.")
        self.print("> SHIFT + MOUSELEFT: add, remove a well.")
        self.print("> SHIFT + MOUSERIGHT: infect, disinfect a well.")
        self.print("> R: toggle between deterministic, stochastic scenario.")
        if self.RANDOM_SELECTION:
            self.print(">-> P: reduce probability of closest well.")
            self.print(">-> SHIFT + P: increase probability of closest well.")
        self.print("Debug: ----------------------------")
        self.print("> ALT + UP: show matrices of the grid. Default: uphills, downhills.")
        self.print("> ALT + LEFTMOUSE: set start point of path.")
        self.print("> ALT + RIGHTMOUSE: set end point of path.")
        self.print("> ALT + LEFT or RIGHT: change color of grid.")
        self.print("> ALT + C: toggle visibility mode of mesh.")
        self.print("> ALT + SPACE: clear debug shapes.")

        # print and not console_log because we want the instructions to appear either way
        print("--> Press ENTER to reset the scene & print instructions.")
        print("--> Press BACKSPACE to print the scenario parameters.")
        print("--> Press UP to toggle between trial mode and normal mode.")
        print("--> Press RIGHT or LEFT to increase or decrease the number of wells.")
        print("--> Press M or N to increase or decrease the population.")
        print("--> Press 1 or 2 to set the scenario to version 1 or 2.")
        print("--> Press W to toggle dense regions of the population.")
        print("--> Press E to toggle between geodesic and euclidean distances.")
        print("--> Press G to consider the elevation in distance calculations.")
        print("--> Press V to toggle the Voronoi diagram.")
        print("--> Press SHIFT + V to use the Voronoi diagram for computations.")
        print("--> Press SHIFT + LEFT MOUSE BUTTON to add or remove a well.")
        print("--> Press SHIFT + LEFT MOUSE BUTTON to infect or disinfect a well.")
        print("--> Press R to toggle between deterministic and stochastic scenario.")
        if self.RANDOM_SELECTION:
            print("-->---> Press P to reduce the probability of choosing the closest well.")
            print("-->---> Press SHIFT + P to increase the probability of choosing the closest well.")
        print("Debug: ----------------------------")
        print("--> Press ALT + UP to show matrices of the grid. Default: uphills, downhills.")
        print("--> Press ALT + LEFTMOUSE to set the start point of the path (euclidean, geodesic or elevation aware).")
        print("--> Press ALT + RIGHTMOUSE to set the end point of the path (euclidean, geodesic or elevation aware).")
        print("--> Press ALT + LEFT or RIGHT to change the color of the grid.")
        print("--> Press ALT + C to toggle the visibility mode of the mesh.")
        print("--> Press ALT + SPACE to clear the debug shapes.")

    def wipe_scene(self):
        '''Wipes the scene of all shapes.'''
        self.shapes = [self.population_pcd_name, self.wells_pcd_name, "bound", "infected_people", "Voronoi", "Voronoi Points"]
        for shape in self.shapes:
            self.removeShape(shape)
    
    def reset_scene(self):
        '''Reloads the scene '''
        self.terminal_log("====================================")
        self.wipe_scene()
        self.construct_scenario() if not self.TRIAL_MODE else self.construct_mini_scenario()
        if self.wells_pcd.points.size > 1:
            self.Voronoi = self.getVoronoi(self.wells_pcd.points)
        if self.VORONOI_VISIBLE:
                self.drawVoronoi()
        self.terminal_log("====================================")

    def construct_scenario(self):
        '''Constructs the scenario for the plague spread simulation.'''
        self.terminal_log("Constructing scenario...")

        # bounding box of the scenario
        self.bound = Cuboid3D(self.bbx[0], self.bbx[1], color=Color.BLACK)
        self.addShape(self.bound, "bound")

        # population point cloud
        console_log("Creating the population...")
        self.population_pcd_name = "Population"
        self.population_pcd = PointSet3D(color=self.healthy_population_color, size=self.POPULATION_SIZE)
        if not self.DENSE_REGIONS:
            self.population_pcd.createRandom(self.bound, self.POPULATION, 42, self.healthy_population_color) # dislikes seed of self.population_pcd_name
        else:
            # regions of interests
            rois = np.array([[-0.5, -0.5, 0], [0.76215255, 0.58612746, 0]])
            if self.POPULATION <= 1000:
                weights = np.array([0.6, 0.4])
                rois_radii = np.array([0.3, 0.2])
                decrease_factor = 0.5
            elif self.POPULATION < 10000:
                weights = np.array([0.5, 0.5])
                rois_radii = np.array([0.3, 0.2])
                decrease_factor = 0.8
            elif self.POPULATION < 30000 or self.POPULATION >= 30000:
                weights = np.array([0.7, 0.7])
                rois_radii = np.array([0.3, 0.2])
                decrease_factor = 2
            self.population_pcd.createRandomWeighted(self.bound, self.POPULATION, 42, self.healthy_population_color, rois, rois_radii, weights, decrease_factor)
        console_log(f"Adjusting the height of the population points...")
        self.move_from_impassable_areas(self.population_pcd) 
        self.adjust_height_of_points(self.population_pcd)
        self.addShape(self.population_pcd, self.population_pcd_name)

        # wells point cloud
        console_log("Creating the wells...")
        self.wells_pcd_name = "Wells"
        self.wells_pcd = PointSet3D(color=self.healthy_wells_color, size=2.5)
        self.wells_pcd.createRandom(self.bound, self.WELLS, 42, self.healthy_wells_color)
        self.move_from_impassable_areas(self.wells_pcd)
        self.adjust_height_of_points(self.wells_pcd)
        self.addShape(self.wells_pcd, self.wells_pcd_name)
        # initialize the kd-tree for well selection
        console_log("Building the KD-Tree for wells...")
        self.kd_wells_root = KdNode.build_kd_node(self.wells_pcd.points)
        console_log("KD-Tree for wells built.")

        self.terminal_log(f"Wells point cloud is {len(self.wells_pcd.points)} points")

        self.infect_wells(self.ratio_of_infected_wells)

        (self.find_infected_people() if not self.RANDOM_SELECTION else self.find_infected_people_stochastic())

    def construct_mini_scenario(self):
        '''Constructs a mini scenario for testing purposes.'''
        self.terminal_log("Constructing mini scenario...")

        # bounding box of the scenario
        self.bound = Cuboid3D(self.bbx[0], self.bbx[1], color=Color.BLACK)
        self.addShape(self.bound, "bound")

        # population point cloud
        self.population_pcd_name = "Mini Population"
        self.population_pcd = PointSet3D(color=self.healthy_population_color, size=self.POPULATION_SIZE)
        if not self.DENSE_REGIONS:
            self.population_pcd.createRandom(self.bound, self.POPULATION, 42, self.healthy_population_color) # dislikes seed of self.population_pcd_name
        else:
            # regions of interests
            rois = np.array([[-0.5, -0.5, 0], [0.76215255, 0.58612746, 0]])
            if self.POPULATION <= 5:
                weights = np.array([0.6, 0.4])
                rois_radii = np.array([0.3, 0.2])
                decrease_factor = 0.5
            elif self.POPULATION < 10:
                weights = np.array([0.5, 0.5])
                rois_radii = np.array([0.3, 0.2])
                decrease_factor = 0.8
            elif self.POPULATION < 15 or self.POPULATION >= 15: # for version 4, this changes
                weights = np.array([0.7, 0.7])
                rois_radii = np.array([0.3, 0.2])
                decrease_factor = 2
            self.population_pcd.createRandomWeighted(self.bound, self.POPULATION, 42, self.healthy_population_color, rois, rois_radii, weights, decrease_factor)
        self.move_from_impassable_areas(self.population_pcd)
        self.adjust_height_of_points(self.population_pcd)
        self.addShape(self.population_pcd, self.population_pcd_name)

        # wells point cloud
        self.wells_pcd_name = "Mini Wells"
        self.wells_pcd = PointSet3D(color=self.healthy_wells_color, size=2.5)
        self.wells_pcd.createRandom(self.bound, self.WELLS, 42, self.healthy_wells_color)
        self.move_from_impassable_areas(self.wells_pcd)
        self.adjust_height_of_points(self.wells_pcd)
        self.addShape(self.wells_pcd, self.wells_pcd_name)
        # initialize the kd-tree for well selection
        self.kd_wells_root = KdNode.build_kd_node(self.wells_pcd.points)

        self.terminal_log(f"Wells point cloud is {len(self.wells_pcd.points)} points")

        self.infect_wells(self.ratio_of_infected_wells)

        (self.find_infected_people() if not self.RANDOM_SELECTION else self.find_infected_people_stochastic()) #if self.DEBUG else None

    def adjust_height_of_points(self, pointset:PointSet3D|np.ndarray|list|tuple):
        '''Adjusts the height of the points in the pointset using interpolation.'''
        if isinstance(pointset, PointSet3D):
            points_nparray = np.array(pointset.points)
        if isinstance(pointset, (np.ndarray, list, tuple)):
            points_nparray = np.array(pointset)
        
        # points_nparray = np.array(pointset.points)
        # for i in range(len(points_nparray)):
        #     z = barycentric_interpolate_height_grid(points_nparray[i], self.grid.points[:, 2], self.GRID_SIZE, self.bound.x_min, self.bound.x_max)
        #     points_nparray[i][2] = z

        # pointset.points = points_nparray 
        
        # vectorized call to the interpolation function
        heights = barycentric_interpolate_height(points_nparray[:, :2], self.triangle_indices, self.mesh.vertices)
        
        points_nparray[:, 2] = heights

        if isinstance(pointset, PointSet3D):
            pointset.points = points_nparray
        return points_nparray

    def move_from_impassable_areas(self, pointset:PointSet3D):
        '''Moves the points in the pointset away from impassable areas, as
        determined by the self.mountainous_area point cloud.'''
        points_nparray = np.array(pointset.points)
        mountainous_points = np.array(self.mountainous_area.points)

        def order_points(points):
            '''Orders points in a counterclockwise manner around their centroid.'''
            center = np.mean(points, axis=0)
            angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
            return points[np.argsort(angles)]

        
        ch = CH_quickhull(mountainous_points)
        ch = order_points(ch)

        lines = []
        for i in range(len(ch)):
            lines.append((i, (i+1)%len(ch)))

        # ch_pointset = PointSet3D(points=ch, color=Color.BLACK, size=2)
        ch_lineset = LineSet3D(ch, lines, width=1, color=Color.BLACK)
        # self.addShape(ch_pointset, "ch_pointset")
        self.removeShape("ch_lineset")
        self.addShape(ch_lineset, "ch_lineset")

        # for each point in the pointset, check if it is within the convex hull
        points_inside = is_inside_polygon_2d(points_nparray[:, :2], ch[:, :2]) # we can use the 2D function
                                                # because the z-coordinate is not important here
        # if a point is inside the convex hull, move it a random distance away away from the hull
        points_nparray[points_inside] = np.array([random.uniform(self.bound.x_min, self.bound.x_max), random.uniform(self.bound.y_min, self.bound.y_max), 0])

        # re-adjust the height of the points
        # points_nparray = self.adjust_height_of_points(points_nparray)
        pointset.points = points_nparray
            
    def infect_wells(self, ratio:float|None = 0.2, hard_number:int|None = None):
        ''' Infects a certain number of wells with the plague.
        Args:
            ratio: The ratio of wells to infect. If None, use hard_number.
            hard_number: The number of wells to infect. If None, use ratio.
        '''
        self.terminal_log(f"Entering infect_wells with Ratio: {ratio}, Hard number: {hard_number}")

        # infected_wells_indices is a list of indices of the infected wells from the wells_pcd.points
        self.infected_wells_indices = []
        wells_nparray = np.array(self.wells_pcd.points)
        wells_color_nparray = np.array(self.wells_pcd.colors)
        
        # ratio has priority over hard_number
        num_of_infected_wells = 0
        if ratio:
            num_of_infected_wells = int(ratio * len(wells_nparray))
            if num_of_infected_wells == 0:
                # infect at least one well
                num_of_infected_wells = 1
            elif num_of_infected_wells == 1:
                # infect at least two wells
                num_of_infected_wells = 2
        elif hard_number:
            num_of_infected_wells = hard_number

        # select num_of_infected_wells random wells from the wells_nparray variable
        infected_wells_indices = random.sample(range(len(wells_nparray)), num_of_infected_wells)
        for i in infected_wells_indices:
            # change the color of the infected wells to yellow
            wells_color_nparray[i] = self.infected_wells_color
            # store the indices of the infected wells
            self.infected_wells_indices.append(i)
        # update the colors of the wells_pcd
        self.wells_pcd.colors = wells_color_nparray

        # show the infected wells
        self.updateShape(self.wells_pcd_name)

        self.terminal_log(f"Infected number of wells {num_of_infected_wells}, with indices {self.infected_wells_indices}")

    def infect_single_well(self, index:int):
        ''' Infects a single well with the plague.
        Args:
            index: The index of the well to infect.
        Returns:
            The Point3D object of the infected well.
        '''
        console_log(f"Entering infect_single_well with index {index}")

        index = int(index) # make sure the index is an integer, please typehinting at getPointAt --> numpy.int64

        wells_color_nparray = np.array(self.wells_pcd.colors)
        # if the well is not already infected, infect it
        if not np.array_equal(wells_color_nparray[index], self.infected_wells_color):

            wells_color_nparray[index] = self.infected_wells_color
            # store the index of the infected well
            self.infected_wells_indices.append(index)
            # change the colors of the wells_pcd
            self.wells_pcd.colors = wells_color_nparray
            # update the colors of the wells_pcd
            self.updateShape(self.wells_pcd_name)
        console_log(f"Infected well with Point3D object {self.wells_pcd[index]}, value {self.wells_pcd.points[index]}, index {index}")
        self.print(f"Infected well with index {index}")
        return self.wells_pcd[index]
    
    def disinfect_single_well(self, index:int):
        ''' Disinfects a single well.
        Args:
            index: The index of the well to disinfect.
        '''
        console_log(f"Entering disinfect_single_well with index {index}")

        index = int(index) # make sure the index is an integer, please typehinting at getPointAt --> numpy.int64

        wells_color_nparray = np.array(self.wells_pcd.colors)
        # if the well is infected, disinfect it
        if np.array_equal(wells_color_nparray[index], self.infected_wells_color):
            wells_color_nparray[index] = self.healthy_wells_color
            # remove the index of the infected well
            self.infected_wells_indices.remove(index)
            # change the colors of the wells_pcd
            self.wells_pcd.colors = wells_color_nparray
            # update the colors of the wells_pcd
            self.updateShape(self.wells_pcd_name)

        console_log(f"Disinfected well with Point2D object {self.wells_pcd[index]}, value {self.wells_pcd.points[index]}, index {index}")
        self.print(f"Disinfected well with index {index}")
        return self.wells_pcd[index]

    def add_single_well(self, x:int, y:int):
        '''Adds a single well to the scene.
        Args:
            x: The x-coordinate of the well.
            y: The y-coordinate of the well.
        '''
        console_log(f"Entering add_single_well with x: {x}, y: {y}")

        new_well = np.array([[x, y, 0]])
        # find the height of the well
        new_well[0][2] = barycentric_interpolate_height(new_well[:, :2], self.triangle_indices, self.mesh.vertices)[0]
        new_well = new_well.flatten()
        console_log(f"Adjusted height of the new well: {new_well[2]}")
        # add the well to the wells_pcd
        self.WELLS += 1
        self.wells_pcd.points = np.vstack((self.wells_pcd.points, new_well))
        self.wells_pcd.colors = np.vstack((self.wells_pcd.colors, self.healthy_wells_color))
        # add the well to the kd-tree
        self.kd_wells_root = KdNode.build_kd_node(self.wells_pcd.points)
        # update the shape
        self.updateShape(self.wells_pcd_name)
        self.terminal_log(f"New well added at {new_well}")
        return new_well
        
    def remove_single_well(self, index:int):
        '''Removes a single well from the scene.
        Args:
            index: The index of the well to remove.
        '''
        console_log(f"Entering remove_single_well with index {index}")

        index = int(index) # make sure the index is an integer, please typehinting at getPointAt --> numpy.int64

        # un-infect the well if it is infected
        console_log(f"Removing well at index {index} from infected wells indices: {self.infected_wells_indices}")
        self.infected_wells_indices = [i for i in self.infected_wells_indices if i != index]
        self.terminal_log(f"Removed well at index {index} from infected wells indices: {self.infected_wells_indices}")
        # update the rest of the indices to point to the correct wells
        for i in range(len(self.infected_wells_indices)):
            if self.infected_wells_indices[i] > index:
                self.infected_wells_indices[i] -= 1
        
        # remove the well
        self.WELLS -= 1
        self.wells_pcd.points = np.delete(self.wells_pcd.points, index, axis=0)
        self.wells_pcd.colors = np.delete(self.wells_pcd.colors, index, axis=0)
        # rebuild the kd-tree
        self.kd_wells_root = KdNode.build_kd_node(self.wells_pcd.points)
        self.updateShape(self.wells_pcd_name)
        self.terminal_log(f"Removed a well at index {index}")

    def _geodesic_distance_naive(self, start:Point3D|np.ndarray|list|tuple, end:Point3D|np.ndarray|list|tuple):
        '''Calculates the geodesic distance between two points on the grid naively, by identifying the triangles in
        which the start and end points are located, and searching for the index linearly in the triangle_indices list.''' 
        if isinstance(start, Point3D):
            start = np.array([start.x, start.y, start.z])
        if isinstance(end, Point3D):
            end = np.array([end.x, end.y, end.z])
        if isinstance(start, (list, tuple)):
            start = np.array(start)
        if isinstance(end, (list, tuple)):
            end = np.array(end)
    
        # get the triangle in which the start point is located
        starting_triangle_indices = get_triangle_of_point(start[:, :2], self.triangle_indices, self.mesh.vertices)
        starting_triangle_vertices = self.mesh.vertices[starting_triangle_indices]
        # get the triangle in which the end point is located
        ending_triangle_vertices = get_triangle_of_point(end[:, :2], self.triangle_indices, self.mesh.vertices)
        ending_triangle_vertices = self.mesh.vertices[ending_triangle_vertices]
        
        if np.array_equal(starting_triangle_vertices, ending_triangle_vertices):
            # if the start and end points are in the same triangle, return the euclidean distance between them
            return np.linalg.norm(start - end)
        
        # find the index of the starting and ending triangle in the triangle_indices list
        starting_triangle_found, ending_triangle_found = False, False
        for i, triangle_index in enumerate(self.triangle_indices):
            grid_points = self.grid.points[triangle_index]
            # console_log(f"Grid points: {grid_points}")
            if np.array_equal(grid_points, starting_triangle_vertices):
                starting_triangle_idx = i
                starting_triangle_found = True
            if np.array_equal(grid_points, ending_triangle_vertices):
                ending_triangle_idx = i
                ending_triangle_found = True
            if starting_triangle_found and ending_triangle_found:
                break
        if not starting_triangle_found and ending_triangle_found:
            console_log(f"Starting triangle not found for vertices {starting_triangle_vertices}")
            console_log("================>", ending_triangle_idx)
        if not ending_triangle_found and starting_triangle_found:
            console_log(f"Ending triangle not found for vertices {ending_triangle_vertices}")
            console_log("================>", starting_triangle_idx)
        if not starting_triangle_found and not ending_triangle_found:
            console_log(f"Starting and ending triangles not found for vertices {starting_triangle_vertices} and {ending_triangle_vertices}")
        if not starting_triangle_found or not ending_triangle_found:
            raise ValueError("Starting or ending triangle not found.")
        # else:
        #     console_log(f"Passed")
        
        starting_centroid = self.centroids[starting_triangle_idx]
        # get the euclidean distance between the start point and the centroid of the triangle
        start_distance = np.linalg.norm(start - starting_centroid)

        ending_centroid = self.centroids[ending_triangle_idx]
        # get the euclidean distance between the end point and the centroid of the triangle
        end_distance = np.linalg.norm(end - ending_centroid)

        # get the geodesic distance between the starting_centroid and the ending_centroid
        geodesic_distance = self.shortest_paths_matrix[starting_triangle_idx][ending_triangle_idx]    #self.dijkstra.get_distance_from_to(starting_triangle_idx, ending_triangle_idx)
        # add the euclidean distances to the geodesic distance
        total_distance = geodesic_distance + start_distance + end_distance
        return total_distance
    
    def geodesic_distance(self, start:Point3D|np.ndarray|list|tuple, end:Point3D|np.ndarray|list|tuple):
        '''Calculates the geodesic distance between two points on the grid using the centroid k-d tree.'''
        if isinstance(start, Point3D):
            start = np.array([start.x, start.y, start.z])
        if isinstance(end, Point3D):
            end = np.array([end.x, end.y, end.z])
        if isinstance(start, (list, tuple)):
            start = np.array(start)
        if isinstance(end, (list, tuple)):
            end = np.array(end)

        # get the nearest centroid to the start point
        # Since we're dealing with a grid and smooth perlin noise, 
        # we can assume that the nearest centroid to the start point is the centroid of the triangle in which the start point is located
        start_centroid_node = KdNode.nearestNeighbor(start, self.kd_centroid_root)
        start_centroid = start_centroid_node.point
        # start_triangle_index_from_map = self.centroid_to_triangle_index_map.get(tuple(start_centroid))
        start_triangle_index = np.where(np.all(self.centroids == start_centroid, axis=1))[0][0]
        # get the nearest centroid to the end point
        end_centroid = KdNode.nearestNeighbor(end, self.kd_centroid_root)
        end_centroid = end_centroid.point
        # end_triangle_index = self.centroid_to_triangle_index_map.get(tuple(end_centroid))
        end_triangle_index = np.where(np.all(self.centroids == end_centroid, axis=1))[0][0]

        # get the euclidean distance between the start point and the centroid of the triangle
        start_distance = np.linalg.norm(start - start_centroid)
        # get the euclidean distance between the end point and the centroid of the triangle
        end_distance = np.linalg.norm(end - end_centroid)

        # get the geodesic distance between the starting_centroid and the ending_centroid
        geodesic_distance = self.shortest_paths_matrix[start_triangle_index][end_triangle_index]    #self.dijkstra.get_distance_from_to(starting_triangle_idx, ending_triangle_idx)
        # add the euclidean distances to the geodesic distance
        total_distance = geodesic_distance + start_distance + end_distance
        return total_distance
    
    def geodesic_distance_to(self, start, end, start_centroid, start_triangle_index):
        '''Calculates the geodesic distance between two points on the grid using the centroid k-d tree,
        assuming the start centroid and triangle index are already known.'''
        if isinstance(end, Point3D):
            end = np.array([end.x, end.y, end.z])
        elif isinstance(end, (list, tuple)):
            end = np.array(end)

        # get the nearest centroid and triangle index for the end point
        # Since we're dealing with a grid and smooth perlin noise, 
        # we can assume that the nearest centroid to the start point is the centroid of the triangle in which the start point is located
        end_centroid_node = KdNode.nearestNeighbor(end, self.kd_centroid_root)
        end_centroid = end_centroid_node.point
        end_triangle_index = np.where(np.all(self.centroids == end_centroid, axis=1))[0][0]

        start_distance = np.linalg.norm(start - start_centroid)
        end_distance = np.linalg.norm(end - end_centroid)
        geodesic_distance = self.shortest_paths_matrix[start_triangle_index][end_triangle_index]

        total_distance = geodesic_distance + start_distance + end_distance
        return total_distance
        
    def find_closest_well_index_geodesic(self, person:Point3D|np.ndarray|list|tuple, wells:np.ndarray, wells_triangle_indices:np.ndarray):
        '''Finds the index of the closest well to the person using geodesic distance.'''
        if isinstance(person, Point3D):
            person = np.array([person.x, person.y, person.z])
        elif isinstance(person, (list, tuple)):
            person = np.array(person)

        person_centroid_node = KdNode.nearestNeighbor(person, self.kd_centroid_root)
        person_centroid = person_centroid_node.point
        person_triangle_index = np.where(np.all(self.centroids == person_centroid, axis=1))[0][0]

        # get the euclidean distance between the person and the centroid of the triangle
        person_to_centroid_distance = np.linalg.norm(person - person_centroid)

        # min_distance = np.inf
        # closest_well_index = -1

        # for j, well in enumerate(wells):
        #     distance = self.geodesic_distance_to(person, well, person_centroid, person_triangle_index)
        #     if distance < min_distance:
        #         min_distance = distance
        #         closest_well_index = j

        # for all the wells, calculate the euclidean distance between the well and the centroid of the triangle
        # get the centroids of the triangles in which the wells are located
        wells_centroids = self.centroids[wells_triangle_indices]
        # calculate the euclidean distances between the wells and the centroids of the triangles
        wells_to_centroids_distances = np.linalg.norm(wells - wells_centroids, axis=1)
        # get the shortest distances between the person and the wells
        if not self.ELEVATION_DISTANCE:
            shortest_distances_to_wells = self.shortest_paths_matrix[person_triangle_index][wells_triangle_indices]
        else:
            shortest_distances_to_wells = self.elevated_shortest_paths_matrix[person_triangle_index][wells_triangle_indices]
        # calculate the total distances
        total_distances = person_to_centroid_distance + shortest_distances_to_wells + wells_to_centroids_distances

        # we only need the closest well
        closest_well_index = np.argsort(total_distances)[0]
        
        return closest_well_index
    
    def find_closest_well_index_kd_tree(self, person:Point3D|np.ndarray|list|tuple, wells:np.ndarray):
        '''Finds the index of the closest well to the person using a k-d tree, euclidean distance.'''
        # get the nearest well to the person
        closest_well_node = KdNode.nearestNeighbor(person, self.kd_wells_root)
        closest_well_index = np.where(np.all(wells == closest_well_node.point, axis=1))[0][0]
        return closest_well_index
        
    def find_infected_people_with_voronoi(self):
        '''Finds the people infected by the wells, using Voronoi diagram.'''
        ...

    def find_infected_people(self):
        '''Finds the people infected by the wells, using geodesic or euclidean distances.'''

        if len(self.infected_wells_indices) == 0:
            return # no need to infect anyone if there are no infected wells
        if len(self.population_pcd.points) == 0:
            return # no need to infect anyone if there are no people

        # infected_people_indices is a list of indices of the infected people from the population_pcd.points
        self.infected_people_indices = []

        population_nparray = np.array(self.population_pcd.points)
        population_color_nparray = np.array(self.population_pcd.colors)
        wells_nparray = np.array(self.wells_pcd.points)
        
        if not self.COMPUTE_WITH_VORONOI:
             # for all the wells, get the index of the nearest centroid
            wells_triangle_index = []
            for well in wells_nparray:
                well_centroid_node = KdNode.nearestNeighbor(well, self.kd_centroid_root)
                wells_centroid = well_centroid_node.point
                well_triangle_index = np.where(np.all(self.centroids == wells_centroid, axis=1))[0][0]
                wells_triangle_index.append(well_triangle_index)
    
            # for every person in the population, check if the closest well to them is infected
            for i, person in enumerate(tqdm(population_nparray, desc="For all people, infect:")):
                if self.EUCLIDEAN and not self.ELEVATION_DISTANCE:
                    closest_well_index = self.find_closest_well_index_kd_tree(person, wells_nparray) 
                else:
                    closest_well_index = self.find_closest_well_index_geodesic(person, wells_nparray, wells_triangle_index) 
                # if the closest well is infected, infect the person
                if closest_well_index in self.infected_wells_indices:
                    population_color_nparray[i] = self.infected_population_color
                    self.infected_people_indices.append(i)
                else:
                    population_color_nparray[i] = self.healthy_population_color

            # update the colors of the population_pcd
            self.population_pcd.colors = population_color_nparray
            self.updateShape(self.population_pcd_name)

        elif self.COMPUTE_WITH_VORONOI:
            self.find_infected_people_with_voronoi()

        self.terminal_log(f"Infected number of people {len(self.infected_people_indices)}") #, with indices {self.infected_people_indices}")
        self.terminal_log(f"Percentage of infected people: {len(self.infected_people_indices) / self.POPULATION * 100}%")

    def get_geodesic_distances_from_to_many_naive(self, person, wells, wells_triangle_indices):
        '''Calculates the geodesic distances between a person and many wells naively, by identifying the triangle in
        which the start point is located, and searching for the index linearly in the triangle_indices list'''

        # get the triangle in which the person is located
        starting_triangle_indices = get_triangle_of_point(person[:, :2], self.triangle_indices, self.mesh.vertices)
        starting_triangle_vertices = self.mesh.vertices[starting_triangle_indices]
        for i, triangle_index in enumerate(self.triangle_indices):
            grid_points = self.grid.points[triangle_index]
            if np.array_equal(grid_points, starting_triangle_vertices):
                starting_triangle_idx = i
                break

        # get the euclidean distance between the person and the centroid of the triangle
        starting_centroid = self.centroids[starting_triangle_idx]
        person_to_centroid_distance = np.linalg.norm(person - starting_centroid)

        # for all the wells, calculate the euclidean distance between the well and the centroid of the triangle
        wells_to_centroids_distances = []
        for i, well in enumerate(wells):
            ending_centroid = self.centroids[wells_triangle_indices[i]]
            well_to_centroid_distance = np.linalg.norm(well - ending_centroid)
            wells_to_centroids_distances.append(well_to_centroid_distance)

        shortest_distances_to_wells = self.shortest_paths_matrix[starting_triangle_idx][wells_triangle_indices]

        # for all the wells, calculate the total distance
        total_distances = np.zeros(len(wells))
        for i, distance in enumerate(shortest_distances_to_wells):
            total_distances[i] = person_to_centroid_distance + distance + wells_to_centroids_distances[i]

        # sort the wells by the total distances
        sorted_wells = np.argsort(total_distances)
        # get the 3 closest wells
        closest_well_index = sorted_wells[0]
        second_closest_well_index = sorted_wells[1]
        third_closest_well_index = sorted_wells[2]

        return closest_well_index, second_closest_well_index, third_closest_well_index
    
    def find_stochastic_infected_linear(self, population_nparray, population_color_nparray, wells_nparray):
        '''Infects people by the wells in a stochastic manner, by linearly searching for the triangles.'''
        # for all wells, get the triangle in which the well is located
        wells_triangles_indices = []
        for well in wells_nparray:
            triangle_indices = get_triangle_of_point(well[:,:2], self.triangle_indices, self.mesh.vertices)
            triangle_vertices = self.mesh.vertices[triangle_indices]
            triangle_vertices = np.array(triangle_vertices)
            for i, triangle_index in enumerate(self.triangle_indices):
                grid_points = self.grid.points[triangle_index]
                if np.array_equal(grid_points, triangle_vertices):
                    wells_triangles_indices.append(i)
                    break
        if len(wells_triangles_indices) != len(wells_nparray):
            raise ValueError("Not all wells have a corresponding triangle index.")            

        # for every person in the population, check if the closest well to them is infected
        for i, person in enumerate(tqdm(population_nparray, desc="For all people, infect:")):
            # get the geodesic distances between the person and all the wells
            closest_wells = self.get_geodesic_distances_from_to_many_naive(person, wells_nparray, wells_triangles_indices)
            choice = np.random.choice(closest_wells, p=[self.P1, self.P2, self.P3])
            if choice in self.infected_wells_indices:
                self.infected_people_indices.append(i)
                population_color_nparray[i] = self.infected_population_color
            else:
                population_color_nparray[i] = self.healthy_population_color

        return population_color_nparray
    
    def get_geodesic_distances_from_to_many(self, person, wells, wells_triangle_indices):
        '''Calculates the geodesic distances between a person and many wells using the centroid k-d tree.'''
        
        # get the nearest centroid to the person
        person_centroid_node = KdNode.nearestNeighbor(person, self.kd_centroid_root)
        person_centroid = person_centroid_node.point
        person_triangle_index = np.where(np.all(self.centroids == person_centroid, axis=1))[0][0]

        # get the euclidean distance between the person and the centroid of the triangle
        person_to_centroid_distance = np.linalg.norm(person - person_centroid)

        # for all the wells, calculate the euclidean distance between the well and the centroid of the triangle
        # get the centroids of the triangles in which the wells are located
        wells_centroids = self.centroids[wells_triangle_indices]
        # calculate the euclidean distances between the wells and the centroids of the triangles
        wells_to_centroids_distances = np.linalg.norm(wells - wells_centroids, axis=1)
        # get the shortest distances between the person and the wells
        if not self.ELEVATION_DISTANCE:
            shortest_distances_to_wells = self.shortest_paths_matrix[person_triangle_index][wells_triangle_indices]
        else:
            shortest_distances_to_wells = self.elevated_shortest_paths_matrix[person_triangle_index][wells_triangle_indices]
        # calculate the total distances
        total_distances = person_to_centroid_distance + shortest_distances_to_wells + wells_to_centroids_distances

        # we only need the 3 closest wells
        closest_wells_indices = np.argsort(total_distances)[:3]

        return closest_wells_indices

    def find_stochastic_infected_geodesic(self, population_nparray, population_color_nparray, wells_nparray):
        '''Infects people by the wells in a stochastic manner geodesically, using the centroids k-d tree.'''
        # for all the wells, get the index of the nearest centroid
        wells_triangle_index = []
        for well in wells_nparray:
            well_centroid_node = KdNode.nearestNeighbor(well, self.kd_centroid_root)
            wells_centroid = well_centroid_node.point
            well_triangle_index = np.where(np.all(self.centroids == wells_centroid, axis=1))[0][0]
            wells_triangle_index.append(well_triangle_index)

        # for every person in the population, find the 3 closest wells to them
        for i, person in enumerate(tqdm(population_nparray, desc="For all infected people")):
            closest_wells = self.get_geodesic_distances_from_to_many(person, wells_nparray, wells_triangle_index)
            choice = np.random.choice(closest_wells, p=[self.P1, self.P2, self.P3])
            if choice in self.infected_wells_indices:
                self.infected_people_indices.append(i)
                population_color_nparray[i] = self.infected_population_color
            else:
                population_color_nparray[i] = self.healthy_population_color

        return population_color_nparray
    
    def find_stochastic_infected_euclidean(self, population_nparray, population_color_nparray, wells_nparray):
        '''Infects people by the wells in a stochastic manner, using the nearest neighbor well k-d tree.'''
        # for every person in the population, find the 3 closest wells to them
        for i, person in enumerate(tqdm(population_nparray, desc="For all infected people")):
            # find the 3 closest wells to the person
            closest_wells = KdNode.nearestK(person, self.kd_wells_root, 3)
            # transform the nodes to numpy arrays, in reverse order to get the closest well first
            closest_wells = np.array([closest_wells[2].point,
                                        closest_wells[1].point,
                                        closest_wells[0].point])
            closest_wells_indices = []
            for well in closest_wells:
                closest_well_index = np.where(np.all(wells_nparray == well, axis=1))[0][0]
                closest_wells_indices.append(closest_well_index)

            choice = np.random.choice(closest_wells_indices, p=[self.P1, self.P2, self.P3])
            if choice in self.infected_wells_indices:
                self.infected_people_indices.append(i)
                population_color_nparray[i] = self.infected_population_color
            else:
                population_color_nparray[i] = self.healthy_population_color

        return population_color_nparray
        
    def find_infected_people_stochastic(self):
        '''Finds the people infected by the wells in a stochastic manner.'''

        if len(self.infected_wells_indices) == 0:
            return # no need to infect anyone if there are no infected wells
        if len(self.population_pcd.points) == 0:
            return # no need to infect anyone if there are no people

        # infected_people_indices is a list of indices of the infected people from the population_pcd.points
        self.infected_people_indices = []

        population_nparray = np.array(self.population_pcd.points)
        population_color_nparray = np.array(self.population_pcd.colors)
        wells_nparray = np.array(self.wells_pcd.points)

        # slow
        # population_color_nparray = self.find_stochastic_infected_linear(population_nparray, population_color_nparray, wells_nparray)

        if self.EUCLIDEAN and not self.ELEVATION_DISTANCE:
            population_color_nparray = self.find_stochastic_infected_euclidean(population_nparray, population_color_nparray, wells_nparray)
        else:
            population_color_nparray = self.find_stochastic_infected_geodesic(population_nparray, population_color_nparray, wells_nparray)

        self.population_pcd.colors = population_color_nparray
        self.updateShape(self.population_pcd_name)

        self.terminal_log(f"Infected number of people {len(self.infected_people_indices)}") #, with indices {self.infected_people_indices}")
        self.terminal_log(f"Percentage of infected people: {len(self.infected_people_indices) / self.POPULATION * 100}%")

    def getVoronoi(self, points):
        '''Returns the Voronoi diagram of the points.'''
        pass

    def drawVoronoi(self):
        '''Draws the Voronoi diagram on the scene.'''
        pass

    def resetVoronoi(self):
        '''Resets the Voronoi diagram.'''
        pass
    
    def terminal_log(self, *args):
        '''Print message both to console and the scene.'''
        console_log(*args)
        self.print(*args)
        

def console_log(*args):
    '''Prints the arguments to the console if CONSOLE_TALK is True.'''
    if CONSOLE_TALK:
        print(*args)   

if __name__ == "__main__":
    app = PlagueSpread3DTerrain(1400, 800)
    app.mainLoop()