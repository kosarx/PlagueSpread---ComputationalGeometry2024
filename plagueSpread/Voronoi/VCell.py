import numpy as np

class VCell:
    def __init__(self):
        '''Initializes an empty cell.'''
        self.vertices = []  # 2D points
        self.size = 0
        self.first_point = None
        self.last_point = None

    def addPointRight(self, point):
        '''Adds a point to the right end of the vertices array.'''
        self.vertices.append(point)
        self.size += 1
        self.last_point = point
        if self.size == 1:
            self.first_point = point

    def addPointLeft(self, point):
        '''Adds a point to the left end of the vertices array.'''
        # self.vertices.insert(0, point)
        # self.size += 1
        # self.first_point = point
        # if self.size == 1:
        #     self.last_point = point
        vertices = self.vertices
        self.vertices = [point]
        for v in vertices:
            self.vertices.append(v)

        self.size += 1
        self.first_point = point
        if self.size == 1:
            self.last_point = point