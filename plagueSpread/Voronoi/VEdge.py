
from VPoint import VPoint

class VEdge:
    def __init__(self, start, left, right):
        self. start = start # The start of the edge
        self.end = None # The end of the edge, which is not known yet

        self.left = left # The left seed point
        self.right = right # The right seed point

        #TODO: Duality?
        assert(left is not None) 
        assert(right is not None)
        assert(left.y != right.y)
        self.f =  (right.x - left.x)/ (left.y - right.y) 
        self.g = start.y - self.f * start.x 
        self.direction = VPoint(right.y - left.y, -(right.x - left.x)) # 
        self.B = VPoint(start.x + self.direction.x, start.y + self.direction.y) # The second point of the line

        self.intersected = False # Whether the edge has been intersected
        self.isCounted = False # Whether the edge has been counted

        self.neighbour = None



