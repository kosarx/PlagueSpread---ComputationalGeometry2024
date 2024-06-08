import os
current_path = os.path.dirname(os.path.realpath(__file__))

import sys
sys.path.append(f"{current_path}/plagueSpread/Voronoi/")

import typing as tp

# VVR imports
from vvrpywork.constants import Key, Mouse, Color
from vvrpywork.scene import Scene2D
from vvrpywork.shapes import (
    Point2D, Line2D, Triangle2D, Circle2D, Rectangle2D,
    PointSet2D, LineSet2D, Polygon2D
)

# Plague Spread imports
from plagueSpread.Voronoi.Voronoi import Voronoi # Voronoi is a class from the plagueSpread package\ 
#                                                 Needs debugging, doesn't work!

# Standard imports
import random
import numpy as np
from scipy.spatial import Voronoi as SciVoronoi
import matplotlib.pyplot as plt


# WIDTH = 1024
# HEIGHT = 768
WIDTH = 800
HEIGHT = 800

DEBUG = False # False
CONSOLE_TALK = True # False
TRIAL_MODE = False # False

class MainClass(Scene2D):
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "Plague Spread", resizable=False)
        # initialize the MainClass scenario mode variables
        self._scenario_mode_init()

        # initialize the scenario

        # scenario encapsulation
        # rect = Rectangle2D((-0.8, -0.8), (0.8, 0.8))
        # self.addShape(rect, "encapsulation")
        
        # self.Voronoi = Voronoi()
        self.scenario_parameters_init()
        self.construct_scenario() if not self.TRIAL_MODE else self.construct_mini_scenario()

        self._print_instructions()
        self.my_mouse_pos = Point2D((0, 0))
        self.addShape(self.my_mouse_pos, "mouse")

    def _scenario_mode_init(self):
        self.DEBUG = DEBUG
        self.CONSOLE_TALK = CONSOLE_TALK
        self.TRIAL_MODE = TRIAL_MODE
    
    def _console_log_scenario(self):
        console_log("---")
        console_log(f"DEBUG: {self.DEBUG}, CONSOLE_TALK: {self.CONSOLE_TALK}, TRIAL_MODE: {self.TRIAL_MODE}")
        console_log(f"Population: {self.POPULATION}, Wells: {self.WELLS}, Number of infected wells: {len(self.infected_wells_indices)}, Infected wells indices: {self.infected_wells_indices}")
        console_log(f"RANDOM_SELECTION: {self.RANDOM_SELECTION}")
        console_log(f"Chances of choosing the closest well: {self.P1}, Chances of choosing the second closest well: {self.P2}, Chances of choosing the third closest well: {self.P3}") if self.RANDOM_SELECTION else None
        console_log(f"Number of infected people: {len(self.infected_people_indices)}")
        console_log("---")

    def on_mouse_press(self, x, y, button, modifiers):
        self.my_mouse_pos.x = x
        self.my_mouse_pos.y = y
        self.my_mouse_pos.color = Color.MAGENTA
        self.my_mouse_pos.size = 1
        self.updateShape("mouse")

    def on_mouse_release(self, x, y, button, modifiers):
        self.my_mouse_pos.x = x
        self.my_mouse_pos.y = y
        # self.my_mouse_pos.color = Color.WHITE
        self.my_mouse_pos.color = [1, 1, 1, 0]

        # if the mouse is released within the bound...
        if self.within_bound(x, y):
            # if the right mouse button was released...
            if button == Mouse.MOUSE2:
                # find the closest well to the mouse position
                closest_well_index = np.argmin(np.linalg.norm(np.array(self.wells_pcd.points) - np.array([x, y]), axis=1))
                # if its within a certain distance...
                if np.linalg.norm(np.array(self.wells_pcd.points[closest_well_index]) - np.array([x, y])) < 0.05:
                    # check if the closest well is not already infected
                    if closest_well_index not in self.infected_wells_indices:
                        # infect the closest well
                        self.infect_single_well(closest_well_index)
                        self.find_infected_people() if not self.RANDOM_SELECTION else self.find_infected_people_stochastic()
                    else:
                        # disenfect the closest well
                        self.disinfect_single_well(closest_well_index)
                        self.find_infected_people() if not self.RANDOM_SELECTION else self.find_infected_people_stochastic()
            # else, if the left mouse button was released...
            elif button == Mouse.MOUSE1:
                # find the closest well to the mouse position
                closest_well_index = np.argmin(np.linalg.norm(np.array(self.wells_pcd.points) - np.array([x, y]), axis=1))
                # if its within a certain distance...
                if np.linalg.norm(np.array(self.wells_pcd.points[closest_well_index]) - np.array([x, y])) < 0.05:
                    # remove the closest well
                    self.remove_single_well(closest_well_index)
                else:
                    # add a new well at the mouse position
                    self.add_single_well(x, y)
                self.find_infected_people() if not self.RANDOM_SELECTION else self.find_infected_people_stochastic()
                self.resetVoronoi()
                
        self.updateShape("mouse")

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.on_mouse_press(x, y, buttons, modifiers)

    def within_bound(self, x, y):
        '''Checks if the point (x, y) is within the bounding box.'''
        return x >= self.bbx[0][0] and x <= self.bbx[1][0] and y >= self.bbx[0][1] and y <= self.bbx[1][1]

    def on_key_press(self, symbol, modifiers):

        def version_1():
            self.POPULATION = 1000 if not self.TRIAL_MODE else 5
            self.WELLS = 15 if not self.TRIAL_MODE else 3
            self.reset_scene()

        def version_2():
            self.POPULATION = 10000 if not self.TRIAL_MODE else 10
            self.WELLS = 30 if not self.TRIAL_MODE else 5
            self.reset_scene()
        
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
            self.POPULATION += 10
            self.reset_scene()
        if symbol == Key.N:
            self.POPULATION -= 10
            self.reset_scene()
        # toggle between random selection and stochastic selection
        if symbol == Key.R:
            self.RANDOM_SELECTION = not self.RANDOM_SELECTION
            self.P1 = 0.8
            self.P2 = 0.15
            self.P3 = 0.05
            self.reset_scene()
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

            self.reset_scene()
        if symbol == Key.V:
            self.VORONOI_ACTIVE = not self.VORONOI_ACTIVE
            # self.Voronoi.generate(self.wells_pcd.points, WIDTH, HEIGHT)
            # edges = self.Voronoi.getEdges()
            if self.VORONOI_ACTIVE:
                self.Voronoi = self.getVoronoi(self.wells_pcd.points)
                self.drawVoronoi()
            else:
                self.removeShape("Voronoi")
                self.removeShape("Voronoi Points")
        # set the scenario to version 1 or 2
        if symbol == Key._1:
            version_1()
        if symbol == Key._2:
            version_2()
    
    def scenario_parameters_init(self):
        self.bbx =[[-0.9, -0.9], [0.9, 0.9]]
        self.Voronoi = None

        # populations, counts, and ratios
        self.POPULATION = 1000
        self.WELLS = 15
        self.ratio_of_infected_wells = 0.2
        self.P1 = 0.8 # probability of choosing the closest well
        self.P2 = 0.15 # probability of choosing the second closest well
        self.P3 = 0.05 # probability of choosing the third closest well

        # logic, controllers
        self.RANDOM_SELECTION = False
        self.VORONOI_ACTIVE = False

        # colors
        self.healthy_population_color = Color.BLUE
        self.infected_population_color = Color.YELLOW
        self.healthy_wells_color = Color.GREEN
        self.infected_wells_color = Color.RED

    def _print_instructions(self):
        print("--> Press ENTER to reset the scene & print instructions.")
        print("--> Press BACKSPACE to print the scenario parameters.")
        print("--> Press UP to toggle between trial mode and normal mode.")
        print("--> Press RIGHT or LEFT to increase or decrease the number of wells.")
        print("--> Press M or N to increase or decrease the population.")
        print("--> Press 1 or 2 to set the scenario to version 1 or 2.")
        print("--> Press V to toggle the Voronoi diagram.")
        print("--> Press LEFT MOUSE BUTTON to add or remove a well.")
        print("--> Press RIGHT MOUSE BUTTON to infect or disinfect a well.")
        print("--> Press R to toggle between deterministic and stochastic scenario.")
        if self.RANDOM_SELECTION:
            print("-->---> Press P to reduce the probability of choosing the closest well.")
            print("-->---> Press SHIFT, and then P to increase the probability of choosing the closest well.")

    
    def wipe_scene(self):
        '''Wipes the scene of all shapes.'''
        self.shapes = [self.population_pcd_name, self.wells_pcd_name, "bound", "infected_people", "Voronoi", "Voronoi Points"]
        for shape in self.shapes:
            self.removeShape(shape)
    
    def reset_scene(self):
        '''Reloads the scene '''
        console_log("=============================================")
        self.wipe_scene()
        self.construct_scenario() if not self.TRIAL_MODE else self.construct_mini_scenario()
        if self.VORONOI_ACTIVE:
            self.Voronoi = self.getVoronoi(self.wells_pcd.points)
            self.drawVoronoi()
        console_log("=============================================")


    def construct_scenario(self):
        '''Constructs the scenario for the plague spread simulation.'''
        console_log("Constructing scenario...")

        # self.my_mouse_pos = Point2D((0, 0))
        # self.addShape(self.my_mouse_pos, "mouse")

        # bounding box of the scenario
        bound = Rectangle2D(self.bbx[0], self.bbx[1])
        self.addShape(bound, "bound")
        
        # population point cloud
        self.population_pcd_name = "Population" 
        self.population_pcd = PointSet2D(color=self.healthy_population_color, size=0.7)
        self.population_pcd.createRandom(bound, self.POPULATION, self.population_pcd_name, self.healthy_population_color)
        self.addShape(self.population_pcd, self.population_pcd_name)
        
        console_log(f"Population point cloud is {len(self.population_pcd.points)} points")

        # wells point cloud
        self.wells_pcd_name = "Wells"
        self.wells_pcd = PointSet2D(color=self.healthy_wells_color)
        self.wells_pcd.createRandom(bound, self.WELLS, self.wells_pcd_name, self.healthy_wells_color)
        self.addShape(self.wells_pcd, self.wells_pcd_name)
        
        console_log(f"Wells point cloud is {len(self.wells_pcd.points)} points")

        self.infect_wells(self.ratio_of_infected_wells)

        self.find_infected_people() if not self.RANDOM_SELECTION else self.find_infected_people_stochastic()

    def construct_mini_scenario(self):
        '''Constructs a mini scenario for testing purposes.'''
        console_log("Constructing mini scenario...")

        # bound = Rectangle2D((-0.9, -0.9), (0.9, 0.9))
        bound = Rectangle2D(self.bbx[0], self.bbx[1])
        self.addShape(bound, "bound")

        self.population_pcd_name = "Mini Population"
        self.population_pcd = PointSet2D(color=self.healthy_population_color)
        self.population_pcd.createRandom(bound, self.POPULATION, self.population_pcd_name, self.healthy_population_color)
        self.addShape(self.population_pcd, self.population_pcd_name)
        
        console_log(f"Population point cloud is {len(self.population_pcd.points)} points")

        self.wells_pcd_name = "Mini Wells"
        self.wells_pcd = PointSet2D(color=self.healthy_wells_color)
        self.wells_pcd.createRandom(bound, self.WELLS, self.wells_pcd_name, self.healthy_wells_color)
        self.addShape(self.wells_pcd, self.wells_pcd_name)
    
        console_log(f"Wells point cloud is {len(self.wells_pcd.points)} points")

        self.infect_wells(self.ratio_of_infected_wells)

        self.find_infected_people() if not self.RANDOM_SELECTION else self.find_infected_people_stochastic()
    
    def infect_wells(self, ratio:float|None = 0.2, hard_number:int|None = None):
        ''' Infects a certain number of wells with the plague.
        Args:
            ratio: The ratio of wells to infect. If None, use hard_number.
            hard_number: The number of wells to infect. If None, use ratio.
        '''
        console_log(f"Entering infect_wells with Ratio: {ratio}, Hard number: {hard_number}")

        # infected_wells_indices is a list of indices of the infected wells from the wells_pcd.points
        self.infected_wells_indices = []
        wells_nparray = np.array(self.wells_pcd.points)
        wells_color_nparray = np.array(self.wells_pcd.colors)
        
        # ratio has priority over hard_number
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

        console_log(f"Infected number of wells {num_of_infected_wells}, with indices {self.infected_wells_indices}")
    
    def infect_single_well(self, index:int):
        ''' Infects a single well with the plague.
        Args:
            index: The index of the well to infect.
        Returns:
            The Point2D object of the infected well.
        '''
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
        console_log(f"Infected well with Point2D object {self.wells_pcd[index]}, value {self.wells_pcd.points[index]}, index {index}")
        return self.wells_pcd[index]
    
    def disinfect_single_well(self, index:int):
        ''' Disinfects a single well.
        Args:
            index: The index of the well to disinfect.
        '''
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
        return self.wells_pcd[index]

    def add_single_well(self, x:int, y:int):
        '''Adds a single well to the scene.
        Args:
            x: The x-coordinate of the well.
            y: The y-coordinate of the well.
        '''
        self.wells_pcd.points = np.vstack([self.wells_pcd.points, [x, y]])
        self.wells_pcd.colors = np.vstack([self.wells_pcd.colors, self.healthy_wells_color])
        self.updateShape(self.wells_pcd_name)
        console_log(f"Added a well at ({x}, {y})")
        return self.wells_pcd[-1]

    def remove_single_well(self, index:int):
        '''Removes a single well from the scene.
        Args:
            index: The index of the well to remove.
        '''
        index = int(index) # make sure the index is an integer, please typehinting at getPointAt --> numpy.int64

        # un-infect the well if it is infected
        self.infected_wells_indices = [i for i in self.infected_wells_indices if i != index]
        
        # remove the well
        self.wells_pcd.points = np.delete(self.wells_pcd.points, index, axis=0)
        self.wells_pcd.colors = np.delete(self.wells_pcd.colors, index, axis=0)
        self.updateShape(self.wells_pcd_name)
        console_log(f"Removed a well at index {index}")


    def find_infected_people(self):
        '''Finds the people infected by the wells.'''

        # infected_people_indices is a list of indices of the infected people from the population_pcd.points
        self.infected_people_indices = []

        population_nparray = np.array(self.population_pcd.points)
        population_color_nparray = np.array(self.population_pcd.colors)
        wells_nparray = np.array(self.wells_pcd.points)

        # for every person in the population, check if the closest well to them is infected
        for i, person in enumerate(population_nparray):
            closest_well = np.argmin(np.linalg.norm(wells_nparray - person, axis=1))
            if closest_well in self.infected_wells_indices:
                self.infected_people_indices.append(i)
                population_color_nparray[i] = self.infected_population_color
            else:
                population_color_nparray[i] = self.healthy_population_color
        
        self.population_pcd.colors = population_color_nparray
        self.updateShape(self.population_pcd_name)

        console_log(f"Infected number of people {len(self.infected_people_indices)}") #, with indices {self.infected_people_indices}")

    def find_infected_people_stochastic(self):
        '''Finds the people infected by the wells in a stochastic manner.'''
        self.infected_people_indices = []

        population_nparray = np.array(self.population_pcd.points)
        population_color_nparray = np.array(self.population_pcd.colors)
        wells_nparray = np.array(self.wells_pcd.points)

        console_log(f"Chances: {self.P1}, {self.P2}, {self.P3}")
        # for every person in the population, check if the closest well to them is infected
        for i, person in enumerate(population_nparray):
            # find the 3 closest wells to the person
            closest_wells = np.argsort(np.linalg.norm(wells_nparray - person, axis=1))[:3]
            '''the person chooses one of the 3 closest wells with varying probabilities
            the person becomes infected if the chosen well is infected
            the person remains healthy if the chosen well is healthy
            if the person is already infected, they remain infected'''
            choice = np.random.choice(closest_wells, p=[self.P1, self.P2, self.P3])
            if choice in self.infected_wells_indices:
                self.infected_people_indices.append(i)
                population_color_nparray[i] = self.infected_population_color
            else:
                population_color_nparray[i] = self.healthy_population_color

        self.population_pcd.colors = population_color_nparray
        self.updateShape(self.population_pcd_name)

        console_log(f"Infected number of people {len(self.infected_people_indices)}") #, with indices {self.infected_people_indices}")

    def resetVoronoi(self):
        '''Resets the Voronoi diagram.'''
        self.Voronoi = None
        self.removeShape("Voronoi")
        self.removeShape("Voronoi Points")
        if self.VORONOI_ACTIVE:    
            self.Voronoi = self.getVoronoi(self.wells_pcd.points)
            self.drawVoronoi()

    def getVoronoi(self, points):
        '''Returns the Voronoi diagram of the points.'''
        # add outer-bounding box points to the numpy array of points to get a better Voronoi diagram
        points = np.concatenate((points, np.array([[-2, -2], [-2, 2], [2, -2], [2, 2]])))

        # placeholder, use the scipy.spatial.Voronoi class to get the Voronoi diagram
        # vor = Voronoi(points) # Voronoi is a class from the plagueSpread package
        vor = SciVoronoi(points)
        return vor

    def drawVoronoi(self):
        '''Draws the Voronoi diagram on the scene.'''
        assert self.Voronoi is not None, "Voronoi diagram is not initialized."

        # draw the Voronoi diagram
        vor = self.Voronoi
        edges_indexes = []
        for region in vor.regions:
            if not -1 in region and len(region) > 0:
                for i in range(len(region) - 1):
                    edges_indexes.append((region[i], region[i+1]))
                edges_indexes.append((region[-1], region[0]))
                
        lineset = LineSet2D(vor.vertices, edges_indexes,  color=Color.BLACK)
        
        self.addShape(lineset, "Voronoi")
        self.addShape(PointSet2D(vor.vertices, color=Color.ORANGE), "Voronoi Points")

            

def console_log(*args):
    '''Prints the arguments to the console if CONSOLE_TALK is True.'''
    if CONSOLE_TALK:
        print(*args)

if __name__ == "__main__":
    app = MainClass()
    app.mainLoop()