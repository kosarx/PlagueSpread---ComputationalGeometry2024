import numpy as np

class VEvent:
    def __init__(self, x, y, point, isCircleEvent):
        '''Creates a new event with the given coordinates.'''
        # first way to access the x and y coordinates
        self.x = x
        self.y = y
        # second way to access the x and y coordinates
        self.coords = np.array([x, y])
        # third way, through VPoint
        self.point = point
        self.isCircleEvent = isCircleEvent

        self.arch = None
        self.value = 0

        # Generate unique ID for the event
        self.ID = hash((x, y, isCircleEvent))

    # def __eq__(self, other):
    #     return np.array_equal(self.coords, other.coords)

    def __lt__(self, other):
        return self.y < other.y
    # def __lt__(self, other):
    #     return self.coords[1] < other.coords[1]

    # def __eq__(self, other):
    #     return self.coords[1] == other.coords[1]