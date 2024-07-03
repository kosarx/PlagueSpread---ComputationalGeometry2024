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