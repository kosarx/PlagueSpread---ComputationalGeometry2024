# TODO
import typing as tp

from VEvent import VEvent

class VQueue:
    def __init__(self):
        '''Initializes an empty queue.'''
        self.q: tp.List[VEvent] = []

    def sort_on_y(self):
        '''Sorts the queue based on the y attribute of VEvent objects.'''
        self.queue.sort(key=lambda event: event.y, reverse=True)

    def enqueue(self, item: VEvent):
        '''Adds an item to the queue.'''
        self.queue.append(item)

    def dequeue(self) -> tp.Optional[VEvent]:
        '''Returns the item with the smallest y value from the queue.'''
        if not self.empty():
            self.sort_on_y()
            # for i in range(len(self.queue)):
                # print(self.queue[i].coords)
            # print(f"Will dequeue {self.queue[-1].coords}, isCircleEvent: {self.queue[-1].isCircleEvent}")
            # input()
            return self.queue.pop()
        return None

    def remove(self, item: VEvent):
        '''Removes the specified item from the queue.'''
        try:
            self.queue.remove(item)
        except ValueError:
            pass  # Item not found in the queue

    def empty(self) -> bool:
        '''Returns True if the queue is empty, False otherwise.'''
        return len(self.queue) == 0
    
    def clear(self):
        '''Clears the queue.'''
        self.queue = []


# class VQueue:
#     def __init__(self):
#         '''Initializes an empty queue.'''
#         self.queue: tp.List[VEvent] = []
#         self.index = 0

#     def sort_on_y(self):
#         '''Sorts the queue based on the y-coordinate of the events.'''
#         self.queue.sort(key=lambda event: event.y)

#     def push(self, item: VEvent):
#         '''Adds an item to the queue.'''
#         self.queue.append(item)

#     def pop(self) -> VEvent|None:
#         '''Returns the next item in the queue, or None if the queue is empty.'''
#         if self.index < len(self.queue):
#             item = self.queue[self.index]
#             self.index += 1
#             return item
#         else:
#             return None

#     def remove(self, item: VEvent):
#         '''Removes the specified item from the queue.'''
#         try:
#             self.queue.remove(item)
#         except ValueError:
#             pass # item not in queue

#     def empty(self) -> bool:
#         '''Returns True if the queue is empty, False otherwise.'''
#         return self.index >= len(self.queue)
    
#     def clear(self):
#         '''Clears the queue.'''
#         self.queue = []
#         self.index = 0