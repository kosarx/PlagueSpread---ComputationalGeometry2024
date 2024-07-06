'''Contains utility classes and functions for geometry operations.
- LineEquation2D: Represents a 2D line equation.
- isInsidePolygon: Returns True if the point is inside the polygon.
- barycentric_interpolate_height: Interpolates the height of a point using barycentric interpolation given a grid.'''
import os

if __name__ == "__main__":
    current_path = os.path.dirname(os.path.realpath(__file__))

    import sys
    sys.path.append(os.path.join(current_path, "..", ".."))

# standard imports
import numpy as np

# vvrpywork imports
from vvrpywork.shapes import Line2D, Point2D

# custom import for kd tree
from plagueSpread.KDTree import KdNode

class LineEquation2D:
    def __init__(self, line:Line2D):

        self.line = line
        self.vertical = False
        self.a = 0
        self.b = 0

        # y = ax + b // ax+by +c = 0 => b=0 kai ax = -c => x = -c/x
        # y - y0 = a(x - x0) + b
        pointFrom = line.getPointFrom()
        pointTo = line.getPointTo()

        if pointFrom.x == pointTo.x: #is vertical
            self.vertical = True
            self.a = self.b = pointFrom.x
        else:    #is not vertical
            self.a = (pointTo.y - pointFrom.y) / (pointTo.x - pointFrom.x)
            self.b = pointFrom.y - self.a * pointFrom.x
        
    @staticmethod
    def lineIntersection(line1:"LineEquation2D", line2:"LineEquation2D") -> Point2D|None:

        if line1.a != line2.b:
            if not line1.vertical and not line2.vertical:
                # y = a1x + b1, y2 = a2x + b2 => a1x + b1 = a2x + b2 => (a1-a2)x = b2-b1 =>
                # x = (b2-b1)/(a1-a2) and y = a1x + b1 => y 
                px = (line2.b - line1.b) / (line1.a - line2.a)
                py = line1.a * px + line1.b
            elif line1.vertical and not line2.vertical:
                # x = a1 and y = a2*x + b1
                px = line1.a
                py = line2.a*px + line2.b
            elif line2.vertical and not line1.vertical:
                # x = a2 and y = a1*x + b1
                px = line2.a
                py = line1.a * px + line1.b
            elif line1.vertical and line2.vertical:
                return None
            return Point2D((px, py), 1.5)
        
        return None
    
    @staticmethod
    def lineSegmentContainsPoint(line:Line2D, point:Point2D) -> bool:

        d1 = line.getPointFrom().distanceSq(point)
        d2 = line.getPointTo().distanceSq(point)
        d = line.length() ** 2

        return d1 + d2 - d < 0

def isInsidePolygon2D(point, polygon):  
    '''Returns True if the point is inside the polygon.'''
    n = len(polygon)
    inside = False
    # check if the point is inside the polygon
    p1x, p1y = polygon[0]
    # iterate over the edges of the polygon
    for i in range(n + 1):
        # get the coordinates of the next vertex
        p2x, p2y = polygon[i % n]
        # check if the point is inside the polygon
        if point[1] > min(p1y, p2y):
            # check if the point is below the edge
            if point[1] <= max(p1y, p2y):
                # check if the point is to the left of the edge
                if point[0] <= max(p1x, p2x):
                    # check if the edge is not vertical
                    if p1y != p2y:
                        # calculate the x-coordinate of the intersection point
                        xinters = (point[1] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    # check if the point is to the left of the intersection point
                    if p1x == p2x or point[0] <= xinters:
                        inside = not inside
        # move to the next vertex
        p1x, p1y = p2x, p2y
    return inside

# for 3D scene

# def get_triangle_of_grid_point(point, z_values, SIZE, x_min, x_max):
#     x, y = point[0], point[1]
#     cell_size = (x_max - x_min)/(SIZE - 1)

#     col = np.floor((x+1) / cell_size)
#     row = np.floor((y+1) / cell_size)

#     # grid cell corners
#     x0, y0 = x_min + col * cell_size, x_min + row * cell_size
#     x1, y1 = x_min + (col + 1) * cell_size, x_min + (row + 1) * cell_size

#     # cell vertices
#     A = np.array([x0, y0, z_values[int(row * SIZE + col)]])
#     B = np.array([x1, y0, z_values[int(row * SIZE + col + 1)]])
#     D = np.array([x0, y1, z_values[int((row + 1) * SIZE + col)]])
#     C = np.array([x1, y1, z_values[int((row + 1) * SIZE + col + 1)]])
#     # print(f"A: {A}, B: {B}, C: {C}, D: {D}")

#     # determine in which triangle of the cell the point is
#     if (x - x0) * (y1 - y0) > (y - y0) * (x1 - x0):
#         # Triangle ABC
#         v0, v1, v2 = A, B, C
#         # print("ABC")
#     else:
#         # Triangle BCD
#         v0, v1, v2 = D, A, C
#         # print("DAC")
#     return v0, v1, v2
def get_triangles_of_grid_points(points, z_values, SIZE, x_min, x_max):
    x = points[:, 0]
    y = points[:, 1]
    cell_size = (x_max - x_min) / (SIZE - 1)

    col = np.floor((x + 1) / cell_size).astype(int)
    row = np.floor((y + 1) / cell_size).astype(int)

    x0 = x_min + col * cell_size
    y0 = x_min + row * cell_size
    x1 = x_min + (col + 1) * cell_size
    y1 = x_min + (row + 1) * cell_size

    A = np.stack([x0, y0, z_values[row * SIZE + col]], axis=-1)
    B = np.stack([x1, y0, z_values[row * SIZE + col + 1]], axis=-1)
    D = np.stack([x0, y1, z_values[(row + 1) * SIZE + col]], axis=-1)
    C = np.stack([x1, y1, z_values[(row + 1) * SIZE + col + 1]], axis=-1)

    condition = (x - x0) * (y1 - y0) > (y - y0) * (x1 - x0)
    v0 = np.where(condition[:, None], A, D)
    v1 = np.where(condition[:, None], B, A)
    v2 = np.where(condition[:, None], C, C)

    return v0, v1, v2

# def barycentric_interpolate_height(point, z_values, SIZE, x_min, x_max):
#     '''Interpolates the height of a point using barycentric interpolation.'''
#     x,y = point[0], point[1]
#     v0, v1, v2 = get_triangle_of_grid_point(point, z_values, SIZE, x_min, x_max)
    
#     # compute the barycentric coordinates
#     l1 = ((v1[1] - v2[1]) * (x - v2[0]) + (v2[0] - v1[0]) * (y - v2[1])) / ((v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1]))
#     l2 = ((v2[1] - v0[1]) * (x - v2[0]) + (v0[0] - v2[0]) * (y - v2[1])) / ((v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1]))
#     l3 = 1 - l1 - l2

#     # interpolate the height of the point
#     z = l1 * v0[2] + l2 * v1[2] + l3 * v2[2]
#     return z
def barycentric_interpolate_height_grid(points, z_values, SIZE, x_min, x_max):
    '''Interpolates the height of points using barycentric interpolation.'''
    x = points[:, 0]
    y = points[:, 1]
    v0, v1, v2 = get_triangles_of_grid_points(points, z_values, SIZE, x_min, x_max)

    denom = ((v1[:, 1] - v2[:, 1]) * (v0[:, 0] - v2[:, 0]) + (v2[:, 0] - v1[:, 0]) * (v0[:, 1] - v2[:, 1]))
    l1 = ((v1[:, 1] - v2[:, 1]) * (x - v2[:, 0]) + (v2[:, 0] - v1[:, 0]) * (y - v2[:, 1])) / denom
    l2 = ((v2[:, 1] - v0[:, 1]) * (x - v2[:, 0]) + (v0[:, 0] - v2[:, 0]) * (y - v2[:, 1])) / denom
    l3 = 1 - l1 - l2

    z = l1 * v0[:, 2] + l2 * v1[:, 2] + l3 * v2[:, 2]
    return z

def calculate_triangle_centroids(triangle_indices, grid_points):
    '''Calculates the centroids of multiple triangles given their vertices.'''
    triangles = grid_points[triangle_indices]  # Extract the vertices of all triangles
    centroids = np.mean(triangles, axis=1)     # Calculate the centroid of each triangle
    return centroids


# 3D scene terrain
def compute_barycentric_coordinates(p, a, b, c):
    """
    Compute the barycentric coordinates of point p with respect to triangle (a, b, c).
    """
    denom = ((b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]))
    u = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / denom
    v = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / denom
    w = 1 - u - v
    return u, v, w

def get_triangle_of_point(point, triangles, vertices):
    """
    Determine which triangle a point lies in.
    
    Parameters:
    point (ndarray): 1x2 array representing the point (x, y).
    triangles (ndarray): Mx3 array of indices into the vertices array, defining the triangles.
    vertices (ndarray): Px3 array of vertex positions (x, y, z).
    
    Returns:
    int: The index of the triangle that contains the point, or -1 if the point is not in any triangle.
    tuple: The barycentric coordinates (u, v, w) of the point with respect to the found triangle.
    """
    for i, triangle in enumerate(triangles):
        a, b, c = vertices[triangle]
        u, v, w = compute_barycentric_coordinates(point[:2], a[:2], b[:2], c[:2])
        if u >= 0 and v >= 0 and w >= 0:  # Point is inside the triangle
            return i, (u, v, w)
    return -1, (None, None, None)

def barycentric_interpolate_height_serial(points, triangles, vertices):
    """
    Interpolate the height of points using barycentric interpolation.
    
    Parameters:
    points (ndarray): Nx2 array of points to interpolate.
    triangles (ndarray): Mx3 array of indices into the vertices array, defining the triangles.
    vertices (ndarray): Px3 array of vertex positions (x, y, z).
    
    Returns:
    ndarray: Nx1 array of interpolated heights.
    """
    interpolated_heights = np.zeros(len(points))
    
    for i, point in enumerate(points):
        triangle_index, bary_coords = get_triangle_of_point(point, triangles, vertices)
        if triangle_index != -1:
            u, v, w = bary_coords
            a, b, c = vertices[triangles[triangle_index]]
            interpolated_heights[i] = u * a[2] + v * b[2] + w * c[2]
        else:
            interpolated_heights[i] = np.nan  # or handle it as you see fit (e.g., extrapolate)
                
    return interpolated_heights

from scipy.spatial import KDTree
def barycentric_interpolate_height(points, triangles, vertices):
    """
    Interpolate the height of points using barycentric interpolation.

    Parameters:
    points (ndarray): Nx2 array of points to interpolate.
    triangles (ndarray): Mx3 array of indices into the vertices array, defining the triangles.
    vertices (ndarray): Px3 array of vertex positions (x, y, z).

    Returns:
    ndarray: Nx1 array of interpolated heights.
    """
    # Separate the vertices into x, y, z coordinates
    vert_x = vertices[:, 0]
    vert_y = vertices[:, 1]
    vert_z = vertices[:, 2]
    
    # Precompute barycentric coordinates
    tri_points = vertices[triangles]
    tri_x = tri_points[:, :, 0]
    tri_y = tri_points[:, :, 1]
    tri_z = tri_points[:, :, 2]

    # Vectors from vertex 0 to vertex 1 and vertex 0 to vertex 2
    v0 = tri_points[:, 1] - tri_points[:, 0]
    v1 = tri_points[:, 2] - tri_points[:, 0]
    
    # Compute the determinant of the triangle's matrix (for barycentric coordinates)
    denom = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]

    # KDTree for efficient point location
    kdtree = KDTree(vertices[:, :2])

    # Allocate array for interpolated heights
    interpolated_heights = np.zeros(len(points))

    for i, point in enumerate(points):
        # Find the closest vertex in the KDTree to get an initial guess of the triangle
        dist, idx = kdtree.query(point)
        triangle_indices = np.where(np.any(triangles == idx, axis=1))[0]

        for tri_idx in triangle_indices:
            v2 = point - tri_points[tri_idx, 0, :2]

            # Compute barycentric coordinates
            u = (v2[0] * v1[tri_idx, 1] - v1[tri_idx, 0] * v2[1]) / denom[tri_idx]
            v = (v0[tri_idx, 0] * v2[1] - v2[0] * v0[tri_idx, 1]) / denom[tri_idx]
            w = 1 - u - v

            # Check if the point is inside the triangle
            if u >= 0 and v >= 0 and w >= 0:
                a, b, c = tri_z[tri_idx]
                interpolated_heights[i] = u * a + v * b + w * c
                break
        else:
            # snap to the closest vertex
            interpolated_heights[i] = vert_z[idx]

    return interpolated_heights

if __name__ == "__main__":
    # test the functions
    point = (0, 0)
    polygon = [(0, 0), (0, 1), (1, 1), (1, 0)]
    print(isInsidePolygon2D(point, polygon))

    point = (0.5, 0.5)
    polygon = [(0, 0), (0, 1), (1, 1), (1, 0)]
    print(isInsidePolygon2D(point, polygon))

    point = np.array([[0.5, 0.5]])
    z_values = np.random.rand(100)
    SIZE = 10
    x_min, x_max = -1, 1
    print(barycentric_interpolate_height_grid(point, z_values, SIZE, x_min, x_max))
    
    ###
    # def test_get_triangle_of_point_given_kd():
    #     # Define vertices of the triangles
    #     vertices = np.array([
    #             [0, 0, 0], [1, 0, 0], [0, 1, 0],  # Triangle 0
    #             [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Triangle 1
    #             [1, 0, 0], [2, 0, 0], [1, 1, 0],  # Triangle 2
    #             [2, 0, 0], [2, 1, 0], [1, 1, 0]   # Triangle 3
    #         ])

    #     # Define triangles (by indices into the vertices array)
    #     triangles = np.array([
    #         [0, 1, 2],
    #         [3, 4, 5],
    #         [6, 7, 8],
    #         [9, 10, 11]
    #     ])

    #     # Calculate centroids of the triangles
    #     centroids = np.array([
    #         np.mean(vertices[tri], axis=0) for tri in triangles
    #     ])
    #     print(f"Centroids: {centroids}")
    #     # Build the k-d tree from centroids
    #     kd_tree = KdNode.build_kd_node(centroids)

    #     # Define a point that is inside one of the triangles
    #     point = [0.50, 0.50, 0]

    #     # Call the function to find the triangle containing the point
    #     triangle_index = get_triangle_of_point_given_kd(point, kd_tree, triangles, vertices, centroids)

    #     print(f"Triangle index: {triangle_index}")
    #     print(f"Containing triangle: {triangles[triangle_index]}")
    #     print(f"Containing triangle vertices: {vertices[triangles[triangle_index]]}")

    #     # Interpolate height using barycentric interpolation
    #     # heights = barycentric_interpolate_height(np.array([point]), triangles, vertices, centroids, kd_tree)
    #     # print(f"Interpolated height: {heights[0]}")
    #     # print(f"Final point: {point} with height: {heights[0]}")

    # # Run the test
    # test_get_triangle_of_point_given_kd()

    points = np.array([[0.25, 0.25], [0.75, 0.75], [0.5, 0.5], [1.5, 1.5]])
    triangles = np.array([[0, 1, 2], [2, 1, 3]])
    vertices = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 2]])

    heights = barycentric_interpolate_height(points, triangles, vertices)
    print(heights)
    final_points = np.hstack((points, heights[:, np.newaxis]))
    print(final_points)
    def compute_barycentric_coordinates_vectorized(p, a, b, c):
        """
        Vectorized computation of barycentric coordinates for multiple points and triangles.

        Args:
            p: A numpy array of shape (n, 2) representing n points in 2D space.
            a: A numpy array of shape (n, 2) representing n triangle vertices a.
            b: A numpy array of shape (n, 2) representing n triangle vertices b.
            c: A numpy array of shape (n, 2) representing n triangle vertices c.

        Returns:
            A tuple of numpy arrays of shape (n, 3) representing the barycentric coordinates (u, v, w) for each point-triangle pair.
        """

        # Calculate the denominator (area of the triangle)
        denom = ((b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1]))


        # Compute intermediate terms for efficiency
        v0 = b[:, 1] - c[:, 1]
        v1 = c[:, 0] - b[:, 0]
        v2 = a[:, 1] - c[:, 1]
        v3 = a[:, 0] - c[:, 0]

        # Calculate u using vectorized operations
        u = (v0 * (p[:, 0] - c[:, 0]) + v1 * (p[:, 1] - c[:, 1])) / denom

        # Calculate v using vectorized operations
        v = (v2 * (p[:, 0] - c[:, 0]) + v3 * (p[:, 1] - c[:, 1])) / denom
        # Calculate w = 1 - u - v
        w = 1 - u - v

        return u, v, w
    # Example usage
    p = np.array([[1, 2], [3, 4], [5, 6]])
    a = np.array([[0, 0], [1, 0], [0, 1]])
    b = np.array([[1, 1], [2, 1], [1, 2]])
    c = np.array([[2, 0], [0, 2], [2, 2]])

    u, v, w = compute_barycentric_coordinates_vectorized(p, a, b, c)

    print(u)
    print(v)
    print(w)
    