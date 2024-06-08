'''This file contains the VPoint class, which represents a point in the Voronoi diagram.
Created so that a seed of the Voronoi diagram can hold its coordinates and the cell it belongs to.'''

import numpy as np

class VPoint:
    def __init__(self, x, y):
        '''If x and y are integers, creates a new point with the given coordinates.'''
        # first way to access the x and y coordinates
        self.x = x
        self.y = y
        # second way to access the x and y coordinates
        self.coords = np.array([x, y])
        self.cell = None

    def __eq__(self, other):
        return np.array_equal(self.coords, other.coords)
    
    def __lt__(self, other):
        return np.less(self.coords, other.coords).any()
    
    def __gt__(self, other):
        return np.greater(self.coords, other.coords).any()
    
    def __le__(self, other):
        return np.less_equal(self.coords, other.coords).any()
    
    def __ge__(self, other):
        return np.greater_equal(self.coords, other.coords).any()
    
    def __ne__(self, other):
        return not self.__eq__(other)

    def distanceFrom(self, point):
        return np.linalg.norm(self.coords - point.coords)

    @staticmethod
    def processMultiplePoints(points):
        '''Returns a list of VPoints with the given coordinates.'''
        return [VPoint(point[0], point[1]) for point in points]