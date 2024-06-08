import typing as tp
from VPoint import VPoint

class VParabola:
    def __init__(self, seed):
        assert(isinstance(seed, VPoint) or seed is None)
        self.seed = seed
        self.circle_event = None
        self.parent = None
        self.edge = None
        self._left = None
        self._right = None

        # If the seed is None, then this is not a leaf node
        # otherwise, it is a leaf node
        self.is_leaf = seed is not None 

    # Getters and setters
    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, value):
        self._left = value
        value.parent = self

    @property
    def right(self):
        return self._right
    
    @right.setter
    def right(self, value):
        self._right = value
        value.parent = self