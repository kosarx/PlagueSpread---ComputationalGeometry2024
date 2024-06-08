'''WORK IN PROGRESS'''
'''It doesn't produce the correct Voronoi diagram!'''

''' A Fortune's algorithm implementation of the Voronoi diagram
It uses a disorganized version of a binary search tree to store the parabolas
and a priority queue to store the events.'''

'''As described in the book "Computational Geometry: Algorithms and Applications" by Mark de Berg et al.'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import typing as tp

from VQueue import VQueue
from VPoint import VPoint
from VEvent import VEvent
from VCell import VCell
from VParabola import VParabola
from VEdge import VEdge

class Voronoi:
    def __init__(self) -> None:
        self.seeds = []
        self.edges = []
        self.cells = []
        self.queue = VQueue()

        self.width = 0
        self.height = 0
        self.root = None
        self.sweep_line = 0
        self.last_y = 0
        self.first_point = None

    def adjust_coordinate_system(self, points):
        return np.array([self.transform_point(point) for point in points])

    def transform_point(self, point):
        return np.array([point[0] + self.width / 2, -point[1] + self.height / 2])

    def revert_coordinate_system(self, points):
        return np.array([self.inverse_transform_point(point) for point in points])

    def inverse_transform_point(self, point):
        return np.array([point[0] - self.width / 2, -point[1] + self.height / 2])
    
    def generate(self, seeds: np.array, width: int, height: int) -> None:
        if (len(seeds) < 2):
            return []
        
        self.root = None
        self.seeds = VPoint.processMultiplePoints(seeds)
        self.edges = []
        self.cells = []
        self.width = width
        self.height = height
        # Adjust the coordinate system so that the origin is top-left
        seeds = self.adjust_coordinate_system(seeds)


        self.queue.clear()

        for i in range(len(seeds)):
            # disturb the seed at (0,0) to avoid degenerate cases
            if seeds[i][0] == 0 and seeds[i][1] == 0:
                seeds[i][0] += 0.001
                seeds[i][1] += 0.001
            # Create a new site event for each seed
            site_event = VEvent(seeds[i][0], seeds[i][1], self.seeds[i], False)
            # Create a new cell for each seed
            cell = VCell()
            self.seeds[i].cell = cell
            # Add the cell to the list of cells
            self.cells.append(cell)
            # Add the site event to the queue
            self.queue.enqueue(site_event)

        while not self.queue.empty():
            event = self.queue.dequeue()
            self.sweep_line = event.coords[1] # Update the sweep line to the y-coordinate of the event

            # print intermidiate plot with the beach line
            self.intermidiate_display(seeds, event)
            if event.isCircleEvent:
                self.handleCircleEvent(event)
            else:
                self.handleSiteEvent(event)
            self.intermidiate_display(seeds, event)

            self.last_y = event.coords[1]

        # this.FinishEdge(this.root);
        self.finishEdge(self.root)

        # Update the edges to have the correct start and end points
        for i in range(len(self.edges)):
            if self.edges[i].neighbour:
                self.edges[i].start = self.edges[i].neighbour.end

        # self.inner_display()
        # self.outer_display(seeds)
        # Return the coordinate system to the center
        # self.seeds = self.revert_coordinate_system(seeds)
        # for cell in self.cells:
        #     for i in range(len(cell.vertices)):
        #         cell.vertices[i] = VPoint(*self.inverse_transform_point(cell.vertices[i].coords))
        #     if cell.first_point and cell.last_point:
        #         cell.first_point = self.inverse_transform_point(cell.first_point.coords)
        #         cell.last_point = self.inverse_transform_point(cell.last_point.coords)

        # for edge in self.edges:
        #     inversed_start = self.inverse_transform_point(edge.start.coords)
        #     edge.start = VPoint(inversed_start[0], inversed_start[1])
        #     inversed_end = self.inverse_transform_point(edge.end.coords)
        #     edge.end = VPoint(inversed_end[0], inversed_end[1])

    def getEdges(self) -> np.array:
        return self.edges
    
    def getCells(self) -> np.array:
        return self.cells
    
    def intermidiate_display(self, seeds, event):
        plt.figure(figsize=(10, 10))
        plt.scatter(*zip(*seeds), color='blue', marker='o', label='Sites')
        plt.axhline(y=self.sweep_line, color='red', linestyle='--', label='Sweep Line')

        if hasattr(event, 'coords'):
            plt.scatter(event.coords[0], event.coords[1], color='green', marker='x', label='Event')

        # Display the edges (optional, for intermediate steps)
        for edge in self.edges:
            if edge.end:
                plt.plot([edge.start.x, edge.end.x], [edge.start.y, edge.end.y], 'black')
            else:
                plt.plot([edge.start.x, self.width], [edge.start.y, edge.f * self.width + edge.g], 'black')

        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        plt.gca().invert_yaxis()  # Invert y-axis to match the top-left origin coordinate system
        plt.legend()
        plt.title('Voronoi Diagram - Intermediate State')
        plt.show()

    def inner_display(self):
        '''Use matplotlib to display the Voronoi diagram
        with the inner coordinate system'''
        fig, ax = plt.subplots()
        for edge in self.edges:
            ax.plot([edge.start.x, edge.end.x], [edge.start.y, edge.end.y], color='black')
        for cell in self.cells:
            for vertex in cell.vertices:
                ax.plot(vertex.x, vertex.y, 'ro')
        # plot the seeds
        for seed in self.seeds:
            ax.plot(seed.x, seed.y, 'bo')
        # set the window size
        ax.set_xlim(-self.width/2, self.width/2)
        ax.set_ylim(-self.height/2, self.height/2)
        plt.show()

    def outer_display(self, seeds):
        '''Use matplotlib to display the Voronoi diagram
        with the given coordinate system'''
        # Revert the coordinate system
        # print(f"{type(self.seeds)}, {type(self.seeds[0])}")
        # temp_seed = self.revert_coordinate_system(self.seeds)
            # for i in range(len(self.seeds)):
            #     seeds[i] = self.seeds[i].coords
        temp_edges = self.edges
        temp_cells = self.cells
        for cell in temp_cells:
            for i in range(len(cell.vertices)):
                cell.vertices[i] = VPoint(*self.inverse_transform_point(cell.vertices[i].coords))
            if cell.first_point and cell.last_point:
                cell.first_point = self.inverse_transform_point(cell.first_point.coords)
                cell.last_point = self.inverse_transform_point(cell.last_point.coords)

        for edge in temp_edges:
            inversed_start = self.inverse_transform_point(edge.start.coords)
            edge.start = VPoint(inversed_start[0], inversed_start[1])
            inversed_end = self.inverse_transform_point(edge.end.coords)
            edge.end = VPoint(inversed_end[0], inversed_end[1])
        # plot the Voronoi diagram
        fig, ax = plt.subplots()
        for edge in temp_edges:
            ax.plot([edge.start.x, edge.end.x], [edge.start.y, edge.end.y], color='black')
        for cell in temp_cells:
            for vertex in cell.vertices:
                ax.plot(vertex.x, vertex.y, 'ro')
        # plot the seeds
        for seed in seeds:
            ax.plot(seed[0], seed[1], 'bo')
        # set the window size
        ax.set_xlim(-self.width/2, self.width/2)
        ax.set_ylim(-self.height/2, self.height/2)
        plt.show()


    def handleSiteEvent(self, event: VEvent) -> None:
        if self.root is None:
            # If the root is None, set the root to the event's point and return
            self.root = VParabola(event.point)
            self.first_point = event.point
            return
        
        # Find the parabola above the site event
        ## 
        print(f"seed y: {self.root.seed.coords[1]} event y: {event.coords[1]}")
        if self.root.is_leaf and self.root.seed.coords[1] - event.coords[1] < 0.01:  # degenerate case - first two places at the same height
            print("ENTERED")
            self.root.is_leaf = False
            self.root.left = VParabola(self.first_point) # set the left child to the first point
            self.root.right = VParabola(event.point) # set the right child to the event
            s = VPoint((event.coords[0] + self.first_point.coords[0]) / 2, self.height) # create a new point at the midpoint between the first point and the event
            if event.coords[0] > self.first_point.coords[0]: # if the event is to the right of the first point
                self.root.edge = VEdge(s, self.first_point, event) # create a new edge with the new point, the first point, and the event
            else: # if the event is to the left of the first point
                self.root.edge = VEdge(s, event, self.first_point) # create a new edge with the new point, the event, and the first point
            self.edges.append(self.root.edge) # add the new edge to the list of edges
            return
        
        #
        parabola = self.getParabolaByX(event.coords[0])

        if parabola.circle_event:
            # Remove the circle event from the queue
            self.queue.remove(parabola.circle_event)
            parabola.circle_event = None

        start = VPoint(event.coords[0], self.getY(parabola.seed, event.coords[0]))

        # Create the twin edges
        edge_left = VEdge(start, parabola.seed, event.point)
        edge_right = VEdge(start, event.point, parabola.seed)

        edge_left.neighbour = edge_right
        self.edges.append(edge_left)

        
        parabola.edge = edge_right
        parabola.is_leaf = False

        # Create 3 new parabolas
        parabola_0 = VParabola(parabola.seed)
        parabola_1 = VParabola(event.point)
        parabola_2 = VParabola(parabola.seed)

        parabola.right = parabola_2 
        parabola.left = VParabola(None) 
        parabola.left.edge = edge_left

        parabola.left.left = parabola_0 # left child of the left child of the parabola
        parabola.left.right = parabola_1 # right child of the left child of the parabola

        self.checkForCircleEvent(parabola_0)
        self.checkForCircleEvent(parabola_2) 

    def handleCircleEvent(self, event: VEvent) -> None:
        parabola = event.arch

        left_parent = self.getLeftParent(parabola)
        left_child_of_left_parent = self.getLeftChild(left_parent)

        right_parent = self.getRightParent(parabola)
        right_child_of_right_parent = self.getRightChild(right_parent)

        # Check if the circle event is still valid, if yes then remove it
        if left_child_of_left_parent.circle_event:
            self.queue.remove(left_child_of_left_parent.circle_event)
            left_child_of_left_parent.circle_event = None
        if right_child_of_right_parent.circle_event:
            self.queue.remove(right_child_of_right_parent.circle_event)
            right_child_of_right_parent.circle_event = None

        p = VPoint(event.coords[0], self.getY(parabola.seed, event.coords[0]))
        print(f"p: {p.coords}")
        input()
        if left_child_of_left_parent.seed.cell.last_point and right_child_of_right_parent.seed.cell.first_point:
            if left_child_of_left_parent.seed.cell.last_point == parabola.seed.cell.first_point:
                parabola.seed.cell.addPointLeft(p)
            else:
                parabola.seed.cell.addPointRight(p)
        left_child_of_left_parent.seed.cell.addPointRight(p)
        right_child_of_right_parent.seed.cell.addPointLeft(p) 

        self.last_y = event.coords[1]

        left_parent.edge.end = p
        right_parent.edge.end = p

        higher = None
        par = parabola
        # Loop through the parabola's ancestors to find the higher parabola
        while par != self.root:
            par = par.parent
            if par == left_parent:
                higher = left_parent
            if par == right_parent:
                higher = right_parent
        higher.edge = VEdge(p, left_child_of_left_parent.seed, right_child_of_right_parent.seed)
        self.edges.append(higher.edge)

        grandparent = parabola.parent.parent
        # If the current parabola is the left child of its parent
        if parabola.parent.left == parabola:
            # If the parent of the current parabola is the left child of the grandparent, set the left child of the grandparent to the right child of the parent
            if grandparent.left == parabola.parent:
                grandparent.left = parabola.parent.right
            # Otherwise, set the right child of the grandparent to the right child of the parent
            else:
                parabola.parent.parent.right = parabola.parent.right
        # If the current parabola is the right child of its parent
        else:
            if grandparent.left == parabola.parent:
                grandparent.left = parabola.parent.left
            else:
                grandparent.right = parabola.parent.left

        # Finally, check for circle events at the left and right children of the parent of the current parabola
        self.checkForCircleEvent(left_child_of_left_parent)
        self.checkForCircleEvent(right_child_of_right_parent)

    def finishEdge(self, parabola: VParabola) -> None:
        max_x = 0
        if parabola.edge.direction.x > 0:
            max_x = np.max([self.width, parabola.edge.start.x + 10])
        else:
            max_x = np.min([0, parabola.edge.start.x - 10])

        parabola.edge.end = VPoint(max_x, parabola.edge.f * max_x + parabola.edge.g)

        if not parabola.left.is_leaf:
            self.finishEdge(parabola.left)
        if not parabola.right.is_leaf:
            self.finishEdge(parabola.right)

#------------------- METHODS FOR WORKING WITHIN THE TREE -------------------#
    def getXOfEdge(self, parabola: VParabola, yy: float) -> float:
        '''Get the x-coordinate of the intersection of the parabola with the sweep line'''
        left = self.getLeftChild(parabola)
        right = self.getRightChild(parabola)

        p = left.seed
        q = right.seed

        # Calculate coefficients for the first parabola
        dp1 = 2 * (p.y - yy)
        a1 = 1 / dp1
        b1 = -2 * p.x / dp1
        c1 = yy + dp1 * 0.25 + p.x * p.x / dp1

        # Calculate coefficients for the second parabola
        dp2 = 2 * (q.y - yy)
        a2 = 1 / dp1
        b2 = -2 * q.x / dp2
        c2 = yy + dp2 * 0.25 + q.x * q.x / dp2

        # Combine the coefficients to form a quadratic equation
        a = a1 - a2
        b = b1 - b2
        c = c1 - c2

        # Handle the case when the quadratic term (a) is zero
        if a == 0:
            return -c / b

        # Calculate the discriminant
        discriminant = b * b - 4 * a * c
        assert discriminant >= 0, f"Discriminant is negative: {discriminant}"

        # Calculate the two possible x-coordinates
        x1 = (-b + np.sqrt(discriminant)) / (2 * a)
        x2 = (-b - np.sqrt(discriminant)) / (2 * a)

        # Return the correct x-coordinate based on the y-coordinates of the sites
        if p.y < q.y:
            return max(x1, x2)
        else:
            return min(x1, x2)

    def getParabolaByX(self, xx: float) -> VParabola:
        parabola = self.root
        x = 0

        while not parabola.is_leaf:
            x = self.getXOfEdge(parabola, self.sweep_line)

            if x > xx:
                parabola = parabola.left
            else:
                parabola = parabola.right

        return parabola

    def getY(self, point: VPoint, xx: float) -> float:
        '''Get the y-coordinate of the point given the x-coordinate
        using the equation of the parabola y = (x^2)/(2*(y0-y)) + (y0+y)/2'''
        dp = 2 * (point.y - self.sweep_line) # the distance between the point and the sweep line
        b1 = -2 * point.x/dp # coefficient b1 in the equation of the parabola
        c1 = self.sweep_line + dp/4 + point.x*point.x/dp # constant c1 in the equation of the parabola

        return xx*xx/dp + b1*xx + c1
    
    def getLeftChild(self, parabola: VParabola) -> VParabola:
        '''Get the left child of the given parabola'''
        if parabola is None:
            return None

        node = parabola.left
        while not node.is_leaf:
            node = node.right

        return node
    
    def getRightChild(self, parabola: VParabola) -> VParabola:
        '''Get the right child of the given parabola'''
        if parabola is None:
            return None

        node = parabola.right
        while not node.is_leaf:
            node = node.left

        return node

    def getLeftParent(self, parabola: VParabola) -> VParabola:
        '''Get the left parent of the given parabola'''
        parent = parabola.parent
        last = parabola

        # Move up the tree while the parent's left child is the last parabola
        while parent.left == last:
            # If the parent is None, return None
            if parent.parent is None:
                return None
            # Move up the tree
            last = parent
            parent = parent.parent

        return parent
    
    def getRightParent(self, parabola: VParabola) -> VParabola:
        '''Get the right parent of the given parabola'''
        parent = parabola.parent
        last = parabola

        # Move up the tree while the parent's right child is the last parabola
        while parent.right == last:
            # If the parent is None, return None
            if parent.parent is None:
                return None
            # Move up the tree
            last = parent
            parent = parent.parent

        return parent

    def getLineIntersection(self, point1: VPoint, B1: VPoint, point2: VPoint, B2: VPoint) -> VPoint:
        '''Get the intersection point of two lines'''
        # Calculate the differences
        dax = point1.x - B1.x
        day = point1.y - B1.y
        dbx = point2.x - B2.x
        dby = point2.y - B2.y

        # Calculate the denominator
        denominator = dax * dby - day * dbx
        if denominator == 0:
            return None # The lines are parallel
        
        # Calculate the constants
        A = point1.x * B1.y - point1.y * B1.x
        B = point2.x * B2.y - point2.y * B2.x

        # Calculate the intersection point, using Cramer's rule
        intersection = VPoint((A * dbx - dax * B) / denominator, (A * dby - day * B) / denominator)
        return intersection

    def getEdgeIntersection(self, edge1: VEdge, edge2: VEdge) -> VPoint:
        '''Get the intersection point of two edges'''
        intersection = self.getLineIntersection(edge1.start, edge1.B, edge2.start, edge2.B)
        
        # wrong direction of edge 
        wrong_direction = (intersection.x - edge1.start.x)*edge1.direction.x < 0 or (intersection.y - edge1.start.y)*edge1.direction.y < 0 or (intersection.x - edge2.start.x)*edge2.direction.x < 0 or (intersection.y - edge2.start.y)*edge2.direction.y < 0
        # input(f"Intersection: {intersection.coords}, wrong direction: {wrong_direction}")
        if wrong_direction:
            return None
        # if outside of the window
        # if intersection.x > self.width or intersection.x < 0 or intersection.y > self.height or intersection.y < 0:
        #     input("Outside of the window")
        #     return None
        return intersection
    
    def checkForCircleEvent(self, parabola: VParabola) -> None:
        '''Check for a circle event at the given parabola'''
        
        # Get the left
        left_parent = self.getLeftParent(parabola)
        left_child_of_left_parent = self.getLeftChild(left_parent)
        
        right_parent = self.getRightParent(parabola)
        right_child_of_right_parent = self.getRightChild(right_parent)

        if left_child_of_left_parent is None or right_child_of_right_parent is None or left_child_of_left_parent.seed == right_child_of_right_parent.seed:
            return
        
        intersection = self.getEdgeIntersection(left_parent.edge, right_parent.edge)
        if not intersection:
            return
        
        # Calculate the radius of the circle
        def distance(point1, point2):
            return np.sqrt(np.power((point1.x - point2.x),2) + np.power((point1.y - point2.y), 2))
               
        d = distance(left_child_of_left_parent.seed, intersection)
        # If the intersection point is below the sweep line, return
        if intersection.y - d >= self.sweep_line:
            return

        # Create the circle event
        new_point = VPoint(intersection.x, intersection.y - d) # The point of the circle event is the intersection point minus the radius
        circle_event = VEvent(new_point.x, new_point.y, new_point, True) 

        parabola.circle_event = circle_event 
        circle_event.arch = parabola
        self.queue.enqueue(circle_event)
        


def center_of_circle(points):
    """
    This function takes three points as input and returns the center of the circle they define.

    Args:
        p1 (numpy.ndarray): A numpy array representing the first point (x, y).
        p2 (numpy.ndarray): A numpy array representing the second point (x, y).
        p3 (numpy.ndarray): A numpy array representing the third point (x, y).

    Returns:
        numpy.ndarray: A numpy array representing the center of the circle (x, y).
    """
    assert len(points) == 3, "The function requires exactly three points."
    p1 = points[0]
    p2 = points[1]
    p3 = points[2]

    # Calculate the side lengths of the triangle formed by the three points
    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p1 - p3)

    # Check if the three points form a valid triangle (triangle inequality)
    if (a + b <= c) or (a + c <= b) or (b + c <= a):
        raise ValueError("The three points do not form a valid triangle.")

    # Calculate the area of the triangle
    s = (a + b + c) / 2
    A = np.sqrt(s * (s - a) * (s - b) * (s - c))

    # Calculate the circumradius (radius of the circle circumscribing the triangle)
    R = A / (s * 0.5)

    # Calculate the vector pointing from p1 to the center of the circle
    v12 = (p2 - p1) / np.linalg.norm(p2 - p1)

    # Apply the Law of Cosines to find the center coordinates
    center_x = p1[0] + R * v12[0]
    center_y = p1[1] + R * v12[1]

    return np.array([center_x, center_y])

if __name__ == "__main__":
    # Test the Voronoi class
    voronoi = Voronoi()
    # TODO: What if the point [0, 0] is included?
    # TODO: cells is empty, why? If it's not important, why doesn't the wells voronoi work?
    # seeds = np.array([[0, 0], [10, 10],[10, 0], [-20, 0]])
    # seeds = np.array([[-0.756, 0.6], [0.34, 0.2], [-0.22, -0.447], [0.5, -0.5], [0.5, 0.5]])

    # seeds = np.array([[-0.5, 0.5], [0.5, 0.5001], [0.3,0.3], [-1.0002, -0.4995]])
    # seeds = np.array([[-0.5, 0.5], [0.5, 0.5001], [0,0], [-1.0002, -0.4995], [0.2, 1.5]]) # breaks at over 4 points and if we make the points larger
    # seeds = np.array([[0.3, 0.3], [0.5, 0.5], [0.1, 0.9], [0.8, 0.2]])
    # seeds = np.array([[0.1, 0.1], [1, 1]])
    # seeds = np.array([[800, 100], [200, 200], [600, 650]])
    seeds = np.array([[39, 255], [-561, 155], [-161, -295]])
    center = center_of_circle(seeds)
    print(f"Center: {center}")
    width = 1522
    height = 710
    voronoi.generate(seeds, width, height )
    print(f"Original {seeds}")
    print(f"Adjusted {voronoi.adjust_coordinate_system(seeds)}")
    print(f"Reverted {voronoi.revert_coordinate_system(voronoi.adjust_coordinate_system(seeds))}")
    # print(f"Testing Revert {voronoi.revert_coordinate_system(seeds)}")
    edges = voronoi.getEdges()
    cells = voronoi.getCells()
    # print the edges
    print("Edges:")
    for edge in edges:
        print(edge.start.coords, edge.end.coords)
    # print the cells
    print("Cells:")
    for cell in cells:
        for vertex in cell.vertices:
            print(vertex.coords)
    voronoi.inner_display()
    voronoi.outer_display(seeds)
    # make a plot of this class's Voronoi diagram
    # fig, ax = plt.subplots()
    # for edge in edges:
    #     ax.plot([edge.start.x, edge.end.x], [edge.start.y, edge.end.y], color='black')
    # for cell in cells:
    #     for vertex in cell.vertices:
    #         ax.plot(vertex.x, vertex.y, 'ro')
    # # plot the seeds
    # for seed in seeds:
    #     ax.plot(seed[0], seed[1], 'bo')
    # # set the window size
    # ax.set_xlim(-width/2, width/2)
    # ax.set_ylim(-height/2, height)
    # plt.show()
    # confirm with scipy.spatial.Voronoi
    print("Scipy Voronoi:")
    vor = sp.spatial.Voronoi(seeds)
    # get the voronoi vertices
    print("Vertices:")
    for vertex in vor.vertices:
        print(vertex)
    fig = sp.spatial.voronoi_plot_2d(vor)
    plt.show()

