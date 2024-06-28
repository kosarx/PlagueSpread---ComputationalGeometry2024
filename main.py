import os
current_path = os.path.dirname(os.path.realpath(__file__))

import sys
sys.path.append(os.path.join(current_path, "plagueSpread", "Voronoi"))
# sys.path.append(os.path.join(current_path, "plagueSpread", "utils"))
# sys.path.append(os.path.join(current_path, "plagueSpread", "resources"))


import typing as tp

# VVR imports
from vvrpywork.constants import Key, Mouse, Color
from vvrpywork.scene import Scene2D
from vvrpywork.shapes import (
    Point2D, Line2D, Rectangle2D,
    PointSet2D, LineSet2D, 
)

# Plague Spread imports
from plagueSpread.Voronoi.Voronoi import Voronoi # Voronoi is a class from the plagueSpread package\ 
#                                                 Needs debugging, doesn't work!
from plagueSpread.utils.GeometryUtils import LineEquation2D
from plagueSpread.utils.GeometryUtils import isInsidePolygon2D

# Standard imports
import random
import numpy as np
from scipy.spatial import Voronoi as SciVoronoi
import matplotlib.pyplot as plt


# WIDTH = 1024
# HEIGHT = 768
WIDTH_2D = 800
HEIGHT_2D = 800

WIDTH_3D = 1400
HEIGHT_3D = 800

DEBUG = False # False
CONSOLE_TALK = True # False
TRIAL_MODE = False # False

class MainClass:
    def __init__(self, start_with = None, end_immediately = False):
        print("================= PLAGUE SPREAD =================")
        self.option = start_with
        self.end_immediately = end_immediately
        self.get_input()
    
    def get_input(self):
        
        while True:
            if not self.option:
                print("> Press 1 to run the 2D Plague Spread simulation.")
                print("> Press 2 to run the 3D Plague Spread simulation.")
                print("> Press -1 or q to exit.")
                self.option = input("> ")
            selection = self.option
            if selection == "1" or selection == 1:
                self.plagueSpread2D = PlagueSpread2D(WIDTH_2D, HEIGHT_2D)
                self.plagueSpread2D.mainLoop()
                if self.end_immediately:
                    break
            elif selection == "2" or selection == 2:
                self.plagueSpread3D = PlagueSpread3D(WIDTH_3D, HEIGHT_3D)
                self.plagueSpread3D.mainLoop()
                if self.end_immediately:
                    break
            elif selection == "q" or selection == "-1" or selection == "exit":
                break
            self.option = ""
        print("Goodbye...")

class PlagueSpread2D(Scene2D):
    def __init__(self, WIDTH, HEIGHT):
        super().__init__(WIDTH, HEIGHT, "Plague Spread 2D", resizable=True)
        # initialize the PlagueSpread2D scenario mode variables
        self._scenario_mode_init()

        # initialize the scenario

        # self.Voronoi = Voronoi()
        self.scenario_parameters_init()
        self.construct_scenario() if not self.TRIAL_MODE else self.construct_mini_scenario()
        if self.wells_pcd.points.size > 1:
            self.Voronoi = self.getVoronoi(self.wells_pcd.points)


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
        if symbol == Key.V and not modifiers & Key.MOD_SHIFT:
            self.VORONOI_VISIBLE = not self.VORONOI_VISIBLE
            # self.Voronoi.generate(self.wells_pcd.points, WIDTH, HEIGHT)
            # edges = self.Voronoi.getEdges()
            if self.Voronoi is not None and self.VORONOI_VISIBLE:
                # self.Voronoi = self.getVoronoi(self.wells_pcd.points)
                self.drawVoronoi()
            else:
                self.removeShape("Voronoi")
                self.removeShape("Voronoi Points")
        if symbol == Key.V and modifiers & Key.MOD_SHIFT:
            self.COMPUTE_WITH_VORONOI = not self.COMPUTE_WITH_VORONOI
            if self.COMPUTE_WITH_VORONOI:
                self.Voronoi = self.getVoronoi(self.wells_pcd.points)
            self.reset_scene()
        # set the scenario to version 1 or 2
        if symbol == Key._1:
            version_1()
        if symbol == Key._2:
            version_2()
    
    def scenario_parameters_init(self):
        self.bbx =[[-0.9, -0.9], [0.9, 0.9]]
        self.bound = None
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
        # self.VORONOI_ACTIVE = False
        self.VORONOI_VISIBLE = False
        self.COMPUTE_WITH_VORONOI = False

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
        print("--> Press SHIFT + V to use the Voronoi diagram for computations.")
        print("--> Press LEFT MOUSE BUTTON to add or remove a well.")
        print("--> Press RIGHT MOUSE BUTTON to infect or disinfect a well.")
        print("--> Press R to toggle between deterministic and stochastic scenario.")
        if self.RANDOM_SELECTION:
            print("-->---> Press P to reduce the probability of choosing the closest well.")
            print("-->---> Press SHIFT + P to increase the probability of choosing the closest well.")
 
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
        if self.wells_pcd.points.size > 1:
            self.Voronoi = self.getVoronoi(self.wells_pcd.points)
        if self.VORONOI_VISIBLE:
                self.drawVoronoi()
        console_log("=============================================")

    def construct_scenario(self):
        '''Constructs the scenario for the plague spread simulation.'''
        console_log("Constructing scenario...")

        # self.my_mouse_pos = Point2D((0, 0))
        # self.addShape(self.my_mouse_pos, "mouse")

        # bounding box of the scenario
        self.bound = Rectangle2D(self.bbx[0], self.bbx[1])
        self.addShape(self.bound, "bound")
        
        # population point cloud
        self.population_pcd_name = "Population" 
        self.population_pcd = PointSet2D(color=self.healthy_population_color, size=0.7)
        self.population_pcd.createRandom(self.bound, self.POPULATION, self.population_pcd_name, self.healthy_population_color)
        self.addShape(self.population_pcd, self.population_pcd_name)
        
        console_log(f"Population point cloud is {len(self.population_pcd.points)} points")

        # wells point cloud
        self.wells_pcd_name = "Wells"
        self.wells_pcd = PointSet2D(color=self.healthy_wells_color)
        self.wells_pcd.createRandom(self.bound, self.WELLS, self.wells_pcd_name, self.healthy_wells_color)
        self.addShape(self.wells_pcd, self.wells_pcd_name)
        
        console_log(f"Wells point cloud is {len(self.wells_pcd.points)} points")

        self.infect_wells(self.ratio_of_infected_wells)

        self.find_infected_people() if not self.RANDOM_SELECTION else self.find_infected_people_stochastic()

    def construct_mini_scenario(self):
        '''Constructs a mini scenario for testing purposes.'''
        console_log("Constructing mini scenario...")

        # bound = Rectangle2D((-0.9, -0.9), (0.9, 0.9))
        self.bound = Rectangle2D(self.bbx[0], self.bbx[1])
        self.addShape(self.bound, "bound")

        self.population_pcd_name = "Mini Population"
        self.population_pcd = PointSet2D(color=self.healthy_population_color, size=0.7)
        self.population_pcd.createRandom(self.bound, self.POPULATION, self.population_pcd_name, self.healthy_population_color)
        self.addShape(self.population_pcd, self.population_pcd_name)
        
        console_log(f"Population point cloud is {len(self.population_pcd.points)} points")

        self.wells_pcd_name = "Mini Wells"
        self.wells_pcd = PointSet2D(color=self.healthy_wells_color)
        self.wells_pcd.createRandom(self.bound, self.WELLS, self.wells_pcd_name, self.healthy_wells_color)
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
        self.WELLS += 1
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
        console_log(f"Removing well at index {index} from infected wells indices: {self.infected_wells_indices}")
        self.infected_wells_indices = [i for i in self.infected_wells_indices if i != index]
        console_log(f"Removed well at index {index} from infected wells indices: {self.infected_wells_indices}")
        # update the rest of the indices to point to the correct wells
        for i in range(len(self.infected_wells_indices)):
            if self.infected_wells_indices[i] > index:
                self.infected_wells_indices[i] -= 1
        
        # remove the well
        self.WELLS -= 1
        self.wells_pcd.points = np.delete(self.wells_pcd.points, index, axis=0)
        self.wells_pcd.colors = np.delete(self.wells_pcd.colors, index, axis=0)
        self.updateShape(self.wells_pcd_name)
        console_log(f"Removed a well at index {index}")

    def find_infected_people_with_voronoi(self):
        '''Finds the people infected by the wells using the Voronoi diagram.'''
        assert self.Voronoi is not None, "Voronoi diagram is not initialized."

        # infected_people_indices is a list of indices of the infected people from the population_pcd.points
        self.infected_people_indices = []

        population_nparray = np.array(self.population_pcd.points)
        population_color_nparray = np.array(self.population_pcd.colors)

        vor = self.Voronoi

        visited_people = set()
        # for every person in the population, check if the closest well to them is infected
        for i, person in enumerate(population_nparray):
            # find the index of the Voronoi cell that contains the person
            region_idx = self.getVoronoiCell(person)
            # get the index of the well that is in the region of the person
            for point_idx, _region_idx in enumerate(vor.point_region):
                if _region_idx == region_idx:
                    break
            # if the well is infected, infect the person
            if point_idx in self.infected_wells_indices:
                self.infected_people_indices.append(i)
                population_color_nparray[i] = self.infected_population_color
            else:
                population_color_nparray[i] = self.healthy_population_color

        self.population_pcd.colors = population_color_nparray
        self.updateShape(self.population_pcd_name)

    def find_infected_people(self):
        '''Finds the people infected by the wells.'''

        # infected_people_indices is a list of indices of the infected people from the population_pcd.points
        self.infected_people_indices = []

        population_nparray = np.array(self.population_pcd.points)
        population_color_nparray = np.array(self.population_pcd.colors)
        wells_nparray = np.array(self.wells_pcd.points)
        
        if not self.COMPUTE_WITH_VORONOI:
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
        elif self.COMPUTE_WITH_VORONOI:
            self.find_infected_people_with_voronoi()

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
        if self.wells_pcd.points.size > 1:
            self.Voronoi = self.getVoronoi(self.wells_pcd.points)
        if self.VORONOI_VISIBLE:    
            self.drawVoronoi()

    def getVoronoi(self, points):
        '''Returns the Voronoi diagram of the points.'''
        # add outer-bounding box points to the numpy array of points to get a better Voronoi diagram
        points = np.concatenate((points, np.array([[-100, -100], [-100, 100], [100, -100], [100, 100]])))

        # placeholder, use the scipy.spatial.Voronoi class to get the Voronoi diagram
        # vor = Voronoi(points) # Voronoi is a class from the plagueSpread package
        vor = SciVoronoi(points)

        # get the edges of the Voronoi diagram
        edges_indexes = []
        for region in vor.regions:
            if not -1 in region and len(region) > 0:
                for i in range(len(region) - 1):
                    edges_indexes.append((region[i], region[i+1]))
                edges_indexes.append((region[-1], region[0]))
        
        edges_indexes = self.finishBBoxEdges(edges_indexes, vor)
        self.voronoi_edges_indexes = edges_indexes

        return vor

    def drawVoronoi(self):
        '''Draws the Voronoi diagram on the scene.'''
        assert self.Voronoi is not None, "Voronoi diagram is not initialized."

        # draw the Voronoi diagram
        vor = self.Voronoi
        to_draw_vertices = []
        for vertex in vor.vertices:
            if self.within_bound(vertex[0], vertex[1]):
                to_draw_vertices.append(vertex)
        edges_indexes = self.voronoi_edges_indexes
        
        lineset = LineSet2D(vor.vertices, edges_indexes,  color=Color.BLACK)
        
        self.addShape(lineset, "Voronoi")
        self.addShape(PointSet2D(to_draw_vertices, color=Color.ORANGE), "Voronoi Points")

    def getVoronoiCell(self, point):
        '''Returns the Voronoi cell of the point.'''
        assert self.Voronoi is not None, "Voronoi diagram is not initialized."
        
        vor = self.Voronoi
        # for every region in the Voronoi diagram, 
        for i, region in enumerate(vor.regions):
            if len(region) == 0:
                continue
            # retrieve the vertices that make up the region
            vertices = [vor.vertices[j] for j in region if j != -1]
            # if the point is within the region, return the index of the region
            if isInsidePolygon2D(point, vertices):
                return i
        return -1
    
    def finishBBoxEdges(self, edges_indexes, vor):
        """ Extend infinite edges of the Voronoi diagram to a finite distance. """
        # get the points of the edges
        original_voronoi_vertices = np.copy(vor.vertices)
        final_voronoi_vertices = np.copy(vor.vertices)
        final_edge_indexes= []
        for edge_index in edges_indexes:
            start = original_voronoi_vertices[edge_index[0]]
            end = original_voronoi_vertices[edge_index[1]]
            if self.within_bound(start[0], start[1]) and self.within_bound(end[0], end[1]):
                final_edge_indexes.append(edge_index)
                continue # both points are within the bounding box
            else:
                # at least one of the points is outside the bounding box,
                # define the bounding box's 4 LineEquation2D objects
                top_left = Point2D([self.bound.x_min, self.bound.y_max])
                top_right = Point2D([self.bound.x_max, self.bound.y_max])
                bottom_left = Point2D([self.bound.x_min, self.bound.y_min])
                bottom_right = Point2D([self.bound.x_max, self.bound.y_min])

                top_line = Line2D(top_left, top_right)
                right_line = Line2D(top_right, bottom_right)
                bottom_line = Line2D(bottom_right, bottom_left)
                left_line = Line2D(bottom_left, top_left)

                top_line_eq = LineEquation2D(top_line)
                right_line_eq = LineEquation2D(right_line)
                bottom_line_eq = LineEquation2D(bottom_line)
                left_line_eq = LineEquation2D(left_line)

                bbox_line_eqs = [top_line_eq, right_line_eq, bottom_line_eq, left_line_eq]

                line = Line2D(Point2D(start), Point2D(end))
                line_eq = LineEquation2D(line)
                seg_intersect = False
                for bbox_line_eq in bbox_line_eqs:
                    if line_eq.a != bbox_line_eq.a:
                        i = LineEquation2D.lineIntersection(line_eq, bbox_line_eq)
                        seg_intersect = LineEquation2D.lineSegmentContainsPoint(line_eq.line, i) and LineEquation2D.lineSegmentContainsPoint(bbox_line_eq.line, i)
                        if seg_intersect:
                            if self.within_bound(start[0], start[1]) and not self.within_bound(end[0], end[1]):
                                # start is within the bounding box, end is not
                                # get the intersection point of the edge with the bounding box using LineEquation2D
                                # add the intersection point to the final_voronoi_vertices
                                final_voronoi_vertices = np.append(final_voronoi_vertices, [[i.x, i.y]], axis=0)
                                n_edge_index = (edge_index[0], len(final_voronoi_vertices) - 1)
                                final_edge_indexes.append(n_edge_index)
                            elif not self.within_bound(start[0], start[1]) and self.within_bound(end[0], end[1]):
                                # start is not within the bounding box, end is
                                # get the intersection point of the edge with the bounding box using LineEquation2D
                                # add the intersection point to the final_voronoi_vertices
                                final_voronoi_vertices = np.append(final_voronoi_vertices, [[i.x, i.y]], axis=0)
                                n_edge_index = (len(final_voronoi_vertices) - 1, edge_index[1])
                                final_edge_indexes.append(n_edge_index)
                        else:
                            continue # the edge does not intersect with the specific bounding box line

        # check if all vertices are mentioned in the final_edge_indexes
        for edge_index in final_edge_indexes:
            if edge_index[0] not in final_voronoi_vertices:
                edge_index = (edge_index[1], edge_index[1])
            if edge_index[1] not in final_voronoi_vertices:
                edge_index = (edge_index[0], edge_index[0])

        # return edges_indexes
        vor.vertices = final_voronoi_vertices
        return final_edge_indexes

##=======================================================================================================================
##=======================================================================================================================
##=======================================================================================================================

from plagueSpread.utils.GeometryUtils import (
    get_triangle_of_grid_point, barycentric_interpolate_height, calculate_triangle_centroid 
)
from plagueSpread.utils.DijkstraAlgorithm import Dijkstra
from plagueSpread.KDTree import KdNode

from vvrpywork.scene import Scene3D, get_rotation_matrix, world_space
from vvrpywork.shapes import (
    Point3D, Line3D, Arrow3D, Sphere3D, Cuboid3D, Cuboid3DGeneralized,
    PointSet3D, LineSet3D, Mesh3D
)

from noise import pnoise2
from time import time
from tqdm import tqdm

class PlagueSpread3D(Scene3D):
    def __init__(self, WIDTH, HEIGHT):
        super().__init__(WIDTH, HEIGHT, "Lab5", output=True, n_sliders=5)
        self._scenario_mode_init()

        self.scenario_parameters_init()

        # setting up grid essentials
        self.create_grid()
        self.triangulate_grid(self.grid.points, self.GRID_SIZE, -1, 1)
        centroids_need_update, dist_need_update, adj_need_update, short_paths_need_update = self.perform_file_checks()
        self.calculate_centroids(centroids_need_update)
        self.create_adjacency_matrix(adj_need_update)
        self.create_distances_matrix(dist_need_update) # centroid distances matrix is the graph for Dijkstra
        self.create_shortest_paths_matrix(short_paths_need_update)
        console_log("Grid set up\n-----------------")

        self.construct_scenario() if not self.TRIAL_MODE else self.construct_mini_scenario()
        if self.wells_pcd.points.size > 1:
            self.Voronoi = self.getVoronoi(self.wells_pcd.points)


        self._print_instructions()
        self.my_mouse_pos = Point3D((0, 0, 0))
        self.addShape(self.my_mouse_pos, "mouse")

        # self.DEBUG = True
        # self._check_grid_mismatch(0) if self.DEBUG else None
        # self._check_grid_mismatch(1) if self.DEBUG else None
        # self._check_grid_mismatch(2) if self.DEBUG else None
        # self._check_grid_mismatch(20) if self.DEBUG else None
        # self._check_grid_mismatch(21) if self.DEBUG else None
        # self._check_grid_mismatch(22) if self.DEBUG else None
        # self._check_grid_mismatch(680) if self.DEBUG else None
        # self._check_grid_mismatch(681) if self.DEBUG else None
        # self._check_grid_mismatch(682) if self.DEBUG else None
        # self._check_grid_mismatch(685) if self.DEBUG else None
        # self._check_grid_mismatch(686) if self.DEBUG else None
        # self._check_grid_mismatch(117) if self.DEBUG else None
        

    def _check_grid_mismatch(self, idx):
        ## ---- PLACE THIS WHEN TRY EXCEPT FAILS IN GEODESIC DISTANCE ---- ##
        ######################################################################
        '''Check if get_triangle_of_grid_point and self.triangle_indices are consistent.'''
        console_log(f"{idx} triangle indices: {self.triangle_indices[idx]}")
        console_log(f"{idx} triangle {self.grid.points[self.triangle_indices[idx][0]]}, {self.grid.points[self.triangle_indices[idx][1]]}, {self.grid.points[self.triangle_indices[idx][2]]}")
        p1 = self.grid.points[self.triangle_indices[idx][0]]
        p2 = self.grid.points[self.triangle_indices[idx][1]]
        p3 = self.grid.points[self.triangle_indices[idx][2]]

        # self.addShape(Point3D(p1, size=1, color=Color.RED), f"p1_{idx}")
        # self.addShape(Point3D(p2, size=1, color=Color.RED), f"p2_{idx}")
        # self.addShape(Point3D(p3, size=1, color=Color.RED), f"p3_{idx}")

        # get the centroid of the triangle
        centroid = self.centroids[idx]
        console_log(f"Centroid: {centroid}")
        # self.addShape(Point3D(centroid, size=1, color=Color.DARKGREEN), f"centroid_{idx}") 

        function_triangle = get_triangle_of_grid_point(centroid, self.grid.points[:, 2], self.GRID_SIZE, -1, 1)
        console_log(f"Function triangle: {function_triangle}")
        f1, f2, f3 = function_triangle
        # self.addShape(Point3D(f1, size=1, color=Color.BLUE), f"f1_{idx}")
        # self.addShape(Point3D(f2, size=1, color=Color.BLUE), f"f2_{idx}")
        # self.addShape(Point3D(f3, size=1, color=Color.BLUE), f"f3_{idx}")

        # are they equal?
        if not np.array_equal(function_triangle, self.grid.points[self.triangle_indices[idx]]):
            console_log(f"Triangle indices {idx} mismatch!")
            console_log(f"Function triangle: {function_triangle}")
            console_log(f"Self triangle: {self.triangle_indices[idx]}")

        # debug triangle
        pp = np.array([[0.15789474, -0.78947368, -0.1404871],
              [0.26315789, -0.68421053, -0.19282421],
              [0.15789474, -0.68421053, -0.10920157]])
        self.addShape(Point3D(pp[0], size=1, color=Color.YELLOW), f"pp1_{idx}")
        self.addShape(Point3D(pp[1], size=1, color=Color.YELLOW), f"pp2_{idx}")
        self.addShape(Point3D(pp[2], size=1, color=Color.YELLOW), f"pp3_{idx}")

        self.addShape(Point3D([ 0.19731697, -0.68796272, -0.14163439], size=0.2, color=Color.ORANGE), f"p_{idx}")


    def create_grid(self):
        '''Creates a 3D grid on the z=0 plane'''
        # create a grid of evenly-spaced points
        grid = np.linspace(-1, 1, self.GRID_SIZE)
        grid_x, grid_y = np.meshgrid(grid, grid)
        grid_x_flat = grid_x.ravel()
        grid_y_flat = grid_y.ravel()
        
        # Generate Perlin noise values for the grid points
        z_values = np.array([pnoise2(x, y, octaves=1, persistence=0.5, lacunarity=2.0, repeatx=self.GRID_SIZE, repeaty=self.GRID_SIZE, base=0)
                            for x, y in zip(grid_x_flat, grid_y_flat)])

        # Combine x, y, and z coordinates
        grid = np.column_stack([grid_x_flat, grid_y_flat, z_values])
        grid = PointSet3D(grid, size=1, color=Color.GRAY)
        self.grid = grid
        self.addShape(self.grid, "grid")

    def triangulate_grid(self, grid, size, x_min, x_max):
        '''Triangulates the grid to form a mesh.'''
        list_of_indexed_triangles = []
        for i, point in enumerate(grid):
            x, y, z = point
            next = i + 1
            upper = i - size
            diagonal = i + size + 1
            if next % size != 0:
                if upper >= 0:
                    list_of_indexed_triangles.append(np.array([i, upper, next]))
                if diagonal < len(grid):
                    list_of_indexed_triangles.append(np.array([i, next, diagonal]))
        
        # store the indices of the triangles
        self.triangle_indices = list_of_indexed_triangles

        line_indices = []
        for i, triangle_index in enumerate(list_of_indexed_triangles):
            for j in range(3):
                line_indices.append((triangle_index[j], triangle_index[(j + 1) % 3]))

        lineset = LineSet3D(grid, line_indices, color=Color.GRAY)
        # add the lineset to the scene
        self.addShape(lineset, "grid_lines")

        self.triangles_lineset =lineset
    
    def perform_file_checks(self):
        update_centroids = True
        update_distances = True
        update_adjacency = True
        update_shortest_paths = True

        # file paths
        path = os.path.join(os.path.dirname(__file__), "plagueSpread", "resources")
        grid_file_path = os.path.join(path, "grid.npy")

        # check if grid.npy exists
        console_log("Checking if the grid file exists...")
        grid_file_check = os.path.exists(grid_file_path)
        if not grid_file_check:
            console_log("Grid file does not exist. Saving grid...\n---------")
            start_time = time()
            np.save(grid_file_path, self.grid.points)
            end_time = time()
            console_log(f"Grid saved in {end_time - start_time} seconds.")
        
        # check if the instance grid is the same as the saved grid
        if grid_file_check:
            console_log("Checking if the grid is the same as the stored grid...")
            start_time = time()
            saved_grid = np.load(grid_file_path)
            end_time = time()
            console_log(f"Stored grid loaded in {end_time - start_time} seconds.")
            is_same_grid = np.array_equal(saved_grid, self.grid.points)
            console_log(f"The instance grid is the same as the stored grid? {is_same_grid}")

            # if the grids are not the same, store the new grid, both need updating
            if not is_same_grid:
                console_log("Grid has changed, storing new one...\n---------")
                np.save(grid_file_path, self.grid.points)
                return update_centroids, update_distances, update_adjacency, update_shortest_paths
        # else, if the grid file didn't exist, we've already assured that the grid is the same as the saved grid
        else:
            is_same_grid = True

        console_log("Checking if the centroids exist...")
        centroids_file_path = os.path.join(path, "centroids.npy")

        # if the grid is the same as the saved grid and we have the centroids saved, load them
        if os.path.exists(centroids_file_path) and is_same_grid:
            console_log("Centroids exist, grid hasn't changed.")
            update_centroids = False
        # else, if the grid is the same as the saved grid but we don't have the centroids saved, calculate them, save them, and load them
        elif not os.path.exists(centroids_file_path) and is_same_grid:
            console_log("Centroids do not exist, grid hasn't changed.")
            update_centroids = True

        console_log("Checking if the adjacency matrix exists...")
        adj_file_path = os.path.join(path, "adjacency.npy")

        if os.path.exists(adj_file_path) and is_same_grid:
            console_log("Adjacency matrix exists, grid hasn't changed.")
            update_adjacency = False
        elif not os.path.exists(adj_file_path) and is_same_grid:
            console_log("Adjacency matrix does not exist, grid hasn't changed.")
            update_adjacency = True

        console_log("Checking if the distances matrix exists...")
        distances_file_path = os.path.join(path, "centroid_distances.npy")
    
        # if the grid is the same as the saved grid and we have the distances matrix saved, load it
        if os.path.exists(distances_file_path) and is_same_grid:
            console_log("Distances matrix exists, grid hasn't changed.")
            update_distances = False
        # else, if the grid is the same as the saved grid but we don't have the distances matrix saved, calculate it, save it, and load it
        elif not os.path.exists(distances_file_path) and is_same_grid:
            console_log("Distances matrix does not exist, grid hasn't changed.")
            update_distances = True
        
        console_log("Checking if the shortest paths exist...")
        shortest_paths_file_path = os.path.join(path, "shortest_paths.npy")
        # if the grid is the same as the saved grid, and we have the shortest paths saved, load them
        if os.path.exists(shortest_paths_file_path) and is_same_grid:
            console_log("Shortest paths exist, grid hasn't changed.")
            update_shortest_paths = False
        # else, if the grid is the same as the saved grid but we don't have the shortest paths saved, calculate them, save them, and load them
        elif not os.path.exists(shortest_paths_file_path) and is_same_grid:
            console_log("Shortest paths do not exist, grid hasn't changed.")
            update_shortest_paths = True
        
        return update_centroids, update_distances, update_adjacency, update_shortest_paths
    
    def calculate_centroids(self, update:bool=False):
        '''Calculates the centroids of the triangles.'''

        path = os.path.join(os.path.dirname(__file__), "plagueSpread", "resources")
        centroids_file_path = os.path.join(path, "centroids.npy")
        if not update:
            console_log("Centroids exist.")
            console_log("Loading the centroids...")
            start_time = time()
            centroids = np.load(centroids_file_path)
            end_time = time()
            console_log(f"Centroids loaded in {end_time - start_time} seconds.")
        else:
            centroids = []
            for triangle_index in tqdm(self.triangle_indices, desc="Calculating centroids"):
                triangle = [self.grid.points[triangle_index[0]], self.grid.points[triangle_index[1]], self.grid.points[triangle_index[2]]]
                centroid = calculate_triangle_centroid(triangle)
                centroids.append(centroid)
            centroids = np.array(centroids)
            console_log("Saving the centroids...")
            start_time = time()
            np.save(centroids_file_path, centroids)
            end_time = time()
            console_log(f"Centroids saved in {end_time - start_time} seconds.")

        console_log(f"Shape of the centroids array: {centroids.shape}")
        self.centroids = centroids
        return centroids

    def calculate_adjacency_matrix(self):
        '''Calculates the adjacency matrix for the grid.'''
        console_log("Calculating the adjacency matrix...")
        adjacency_matrix = np.zeros((len(self.triangle_indices), len(self.triangle_indices)))
        for i, triangle in enumerate(tqdm(self.triangle_indices, desc="Calculating adjacency matrix")):
            # for every triangle, we consider its adjacent triangles only if they share an edge
            for j, other_triangle in enumerate(self.triangle_indices):
                if i == j:
                    continue
                # if the triangles share an edge, they are adjacent
                if len(set(triangle) & set(other_triangle)) == 2:
                    adjacency_matrix[i, j] = 1
        console_log(f"Shape of the adjacency matrix: {adjacency_matrix.shape}")
        return adjacency_matrix

    def create_adjacency_matrix(self, update:bool=False):
        '''Creates an adjacency matrix for the grid.'''
        adjacency_matrix = None

        # file paths
        path = os.path.join(os.path.dirname(__file__), "plagueSpread", "resources")
        adj_file_path = os.path.join(path, "adjacency.npy")
        if not update:
            console_log("Adjacency matrix exists.")
            
            console_log("Loading the adjacency matrix...")
            start_time = time()
            adjacency_matrix = np.load(adj_file_path)
            end_time = time()
            console_log(f"Adjacency matrix loaded in {end_time - start_time} seconds.")
        else:
            console_log("Adjacency matrix: calculating and storing it...")

            # calculate the adjacency matrix
            adjacency_matrix = self.calculate_adjacency_matrix()
            console_log("Saving the adjacency matrix...")
            start_time = time()
            np.save(adj_file_path, adjacency_matrix)
            end_time = time()
            console_log(f"Adjacency matrix saved in {end_time - start_time} seconds.")

        self.adjacency_matrix = adjacency_matrix
        return adjacency_matrix

    def calculate_distances_matrix(self):
        '''Calculates a matrix of distances between the centroids of the triangles.'''

        # for all the centroids, calculate the distances between them
        centroids = self.centroids
        distances_matrix = np.zeros((len(centroids), len(centroids)))
        for i, centroid1 in enumerate(tqdm(centroids, desc="Calculating distances", leave=False)):
            for j, centroid2 in enumerate(centroids):
                # if the triangle centroids are adjacent, calculate the distance between them, otherwise 0 (same triangle or not adjacent)
                if self.adjacency_matrix[i, j] == 1:
                    distances_matrix[i, j] = np.linalg.norm(centroid1 - centroid2)
                else:
                    distances_matrix[i, j] = 0
        # shape of the distances matrix for 80x80 grid should be (80, 80)
        console_log(f"Shape of the distances matrix: {distances_matrix.shape}")
        return distances_matrix
        
    def create_distances_matrix(self, update:bool=False):
        '''Creates a matrix of distances between the centroids of the triangles.'''
        centroid_distances_matrix = None
        path = os.path.join(os.path.dirname(__file__), "plagueSpread", "resources")
        distances_file_path = os.path.join(path, "centroid_distances.npy")
        
        if not update:
            console_log("Loading the distances matrix...")
            start_time = time()
            centroid_distances_matrix = np.load(distances_file_path)
            end_time = time()
            console_log(f"Distances matrix loaded in {end_time - start_time} seconds.")
        else:
            console_log("Need to calculate the distances matrix...")
            start_time = time()
            centroid_distances_matrix = self.calculate_distances_matrix()
            end_time = time()
            console_log(f"Distances matrix calculated in {end_time - start_time} seconds.")
            console_log("Saving the distances matrix...")
            start_time = time()
            np.save(distances_file_path, centroid_distances_matrix)
            end_time = time()
            console_log(f"Distances matrix saved in {end_time - start_time} seconds.")
        
        self.centroid_distances_matrix = centroid_distances_matrix
        return centroid_distances_matrix
    
    def create_shortest_paths_matrix(self, update:bool=False):
        '''Creates a matrix of shortest paths between the centroids of the triangles.'''
        shortest_paths_matrix = None
        path = os.path.join(os.path.dirname(__file__), "plagueSpread", "resources")
        shortest_paths_file_path = os.path.join(path, "shortest_paths.npy")
        if not update:
            console_log("Loading the shortest paths matrix...")
            start_time = time()
            shortest_paths_matrix = np.load(shortest_paths_file_path, allow_pickle=True)
            end_time = time()
            console_log(f"Shortest paths matrix loaded in {end_time - start_time} seconds.")
        else:
            console_log("Calculating the shortest paths matrix...")
            start_time = time()
            self.dijkstra = Dijkstra(self.centroid_distances_matrix) # Dijkstra object for finding shortest paths between centroids
            self.dijkstra.calculate_all_shortest_paths()
            shortest_paths_matrix = self.dijkstra.get_distances()
            end_time = time()
            console_log(f"Shortest paths matrix calculated in {end_time - start_time} seconds.")
            console_log("Saving the shortest paths matrix...")
            start_time = time()
            np.save(shortest_paths_file_path, shortest_paths_matrix)
            end_time = time()
            console_log(f"Shortest paths matrix saved in {end_time - start_time} seconds.")
        
        console_log("Shape of the shortest paths matrix: ", shortest_paths_matrix.shape)
        self.shortest_paths_matrix = shortest_paths_matrix
        return shortest_paths_matrix
    
    def _scenario_mode_init(self):
        self.DEBUG = DEBUG
        self.CONSOLE_TALK = CONSOLE_TALK
        self.TRIAL_MODE = TRIAL_MODE

    def _console_log_scenario(self):
        self.terminal_log("---")
        self.terminal_log(f"DEBUG: {self.DEBUG}, CONSOLE_TALK: {self.CONSOLE_TALK}, TRIAL_MODE: {self.TRIAL_MODE}")
        self.terminal_log(f"Population: {self.POPULATION}, Wells: {self.WELLS}, Number of infected wells: {len(self.infected_wells_indices)}, Infected wells indices: {self.infected_wells_indices}")
        self.terminal_log(f"RANDOM_SELECTION: {self.RANDOM_SELECTION}")
        self.terminal_log(f"Chances of choosing the closest well: {self.P1}, Chances of choosing the second closest well: {self.P2}, Chances of choosing the third closest well: {self.P3}") if self.RANDOM_SELECTION else None
        self.terminal_log(f"Number of infected people: {len(self.infected_people_indices)}")
        self.terminal_log("---")

        # console_log("---")
        # console_log(f"DEBUG: {self.DEBUG}, CONSOLE_TALK: {self.CONSOLE_TALK}, TRIAL_MODE: {self.TRIAL_MODE}")
        # console_log(f"Population: {self.POPULATION}, Wells: {self.WELLS}, Number of infected wells: {len(self.infected_wells_indices)}, Infected wells indices: {self.infected_wells_indices}")
        # console_log(f"RANDOM_SELECTION: {self.RANDOM_SELECTION}")
        # console_log(f"Chances of choosing the closest well: {self.P1}, Chances of choosing the second closest well: {self.P2}, Chances of choosing the third closest well: {self.P3}") if self.RANDOM_SELECTION else None
        # console_log(f"Number of infected people: {len(self.infected_people_indices)}")
        # console_log("---")

    @world_space
    def on_mouse_press(self, x, y, button, modifiers):
        self.my_mouse_pos.x = x
        self.my_mouse_pos.y = y
        self.my_mouse_pos.color = Color.MAGENTA
        self.my_mouse_pos.size = 1
        self.updateShape("mouse")

    @world_space
    def on_mouse_release(self, x, y, button, modifiers):
        self.my_mouse_pos.x = x
        self.my_mouse_pos.y = y
        # self.my_mouse_pos.color = Color.WHITE
        self.my_mouse_pos.color = [1, 1, 1, 0]

        # if the mouse is released within the bound...
        if self.within_bound(x, y):
            # if the right mouse button was released...
            if button == Mouse.MOUSE2 and modifiers & Key.MOD_SHIFT:
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
            elif button == Mouse.MOUSE1 and modifiers & Key.MOD_SHIFT:
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

    @world_space
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
        if symbol == Key.V and not modifiers & Key.MOD_SHIFT:
            self.VORONOI_VISIBLE = not self.VORONOI_VISIBLE
            # self.Voronoi.generate(self.wells_pcd.points, WIDTH, HEIGHT)
            # edges = self.Voronoi.getEdges()
            if self.Voronoi is not None and self.VORONOI_VISIBLE:
                # self.Voronoi = self.getVoronoi(self.wells_pcd.points)
                self.drawVoronoi()
            else:
                self.removeShape("Voronoi")
                self.removeShape("Voronoi Points")
        if symbol == Key.V and modifiers & Key.MOD_SHIFT:
            self.COMPUTE_WITH_VORONOI = not self.COMPUTE_WITH_VORONOI
            if self.COMPUTE_WITH_VORONOI:
                self.Voronoi = self.getVoronoi(self.wells_pcd.points)
            self.reset_scene()
        # set the scenario to version 1 or 2
        if symbol == Key._1:
            version_1()
        if symbol == Key._2:
            version_2()

    def scenario_parameters_init(self):
        self.GRID_SIZE = 20 # will create a grid of N x N points, choices: 20, 80
        self.grid = None
        self.bbx =[[-1, -1, 0], [1, 1, 0]]
        self.bound = None
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
        # self.VORONOI_ACTIVE = False
        self.VORONOI_VISIBLE = False
        self.COMPUTE_WITH_VORONOI = False

        # colors
        self.healthy_population_color = Color.BLUE
        self.infected_population_color = Color.YELLOW
        self.healthy_wells_color = Color.GREEN
        self.infected_wells_color = Color.RED

    def _print_instructions(self):
        self.print("> ENTER: reset scene & print instructions.")
        self.print("> BACKSPACE: print scenario parameters.")
        self.print("> UP: toggle trial and normal mode.")
        self.print("> RIGHT or LEFT: increase, decrease wells.")
        self.print("> M or N: increase, decrease population.")
        self.print("> 1 or 2: scenario version 1 or 2.")
        self.print("> V: toggle Voronoi diagram.")
        self.print("> SHIFT + V: use Voronoi diagram for computations.")
        self.print("> LEFT MOUSE BUTTON: add, remove a well.")
        self.print("> RIGHT MOUSE BUTTON: infect, disinfect a well.")
        self.print("> R: toggle between deterministic, stochastic scenario.")
        if self.RANDOM_SELECTION:
            self.print(">-> P: reduce probability of closest well.")
            self.print(">-> SHIFT + P: increase probability of closest well.")

        print("--> Press ENTER to reset the scene & print instructions.")
        print("--> Press BACKSPACE to print the scenario parameters.")
        print("--> Press UP to toggle between trial mode and normal mode.")
        print("--> Press RIGHT or LEFT to increase or decrease the number of wells.")
        print("--> Press M or N to increase or decrease the population.")
        print("--> Press 1 or 2 to set the scenario to version 1 or 2.")
        print("--> Press V to toggle the Voronoi diagram.")
        print("--> Press SHIFT + V to use the Voronoi diagram for computations.")
        print("--> Press LEFT MOUSE BUTTON to add or remove a well.")
        print("--> Press RIGHT MOUSE BUTTON to infect or disinfect a well.")
        print("--> Press R to toggle between deterministic and stochastic scenario.")
        if self.RANDOM_SELECTION:
            print("-->---> Press P to reduce the probability of choosing the closest well.")
            print("-->---> Press SHIFT + P to increase the probability of choosing the closest well.")

    def wipe_scene(self):
        '''Wipes the scene of all shapes.'''
        self.shapes = [self.population_pcd_name, self.wells_pcd_name, "bound", "infected_people", "Voronoi", "Voronoi Points"]
        for shape in self.shapes:
            self.removeShape(shape)
    
    def reset_scene(self):
        '''Reloads the scene '''
        self.terminal_log("=============================================")
        self.wipe_scene()
        self.construct_scenario() if not self.TRIAL_MODE else self.construct_mini_scenario()
        if self.wells_pcd.points.size > 1:
            self.Voronoi = self.getVoronoi(self.wells_pcd.points)
        if self.VORONOI_VISIBLE:
                self.drawVoronoi()
        self.terminal_log("=============================================")

    def construct_scenario(self):
        '''Constructs the scenario for the plague spread simulation.'''
        self.terminal_log("Constructing scenario...")

        # bounding box of the scenario
        self.bound = Cuboid3D(self.bbx[0], self.bbx[1], color=Color.BLACK)
        self.addShape(self.bound, "bound")

        # population point cloud
        self.population_pcd_name = "Population"
        self.population_pcd = PointSet3D(color=self.healthy_population_color, size=0.7)
        self.population_pcd.createRandom(self.bound, self.POPULATION, 42, self.healthy_population_color) # dislikes seed of self.population_pcd_name
        self.adjust_height_of_points(self.population_pcd)
        self.addShape(self.population_pcd, self.population_pcd_name)

        # wells point cloud
        self.wells_pcd_name = "Wells"
        self.wells_pcd = PointSet3D(color=self.healthy_wells_color)
        self.wells_pcd.createRandom(self.bound, self.WELLS, 42, self.healthy_wells_color)
        self.adjust_height_of_points(self.wells_pcd)
        self.addShape(self.wells_pcd, self.wells_pcd_name)

        self.infect_wells(self.ratio_of_infected_wells)

        (self.find_infected_people() if not self.RANDOM_SELECTION else self.find_infected_people_stochastic()) #if not self.DEBUG else None

    def construct_mini_scenario(self):
        '''Constructs a mini scenario for testing purposes.'''
        self.terminal_log("Constructing mini scenario...")

        # bounding box of the scenario
        self.bound = Cuboid3D(self.bbx[0], self.bbx[1], color=Color.BLACK)
        self.addShape(self.bound, "bound")

        # population point cloud
        self.population_pcd_name = "Mini Population"
        self.population_pcd = PointSet3D(color=self.healthy_population_color, size=0.7)
        self.population_pcd.createRandom(self.bound, self.POPULATION, 42, self.healthy_population_color) # dislikes seed of self.population_pcd_name
        self.adjust_height_of_points(self.population_pcd)
        self.addShape(self.population_pcd, self.population_pcd_name)

        # wells point cloud
        self.wells_pcd_name = "Mini Wells"
        self.wells_pcd = PointSet3D(color=self.healthy_wells_color)
        self.wells_pcd.createRandom(self.bound, self.WELLS, 42, self.healthy_wells_color)
        self.adjust_height_of_points(self.wells_pcd)
        self.addShape(self.wells_pcd, self.wells_pcd_name)

        self.infect_wells(self.ratio_of_infected_wells)

        (self.find_infected_people() if not self.RANDOM_SELECTION else self.find_infected_people_stochastic()) #if self.DEBUG else None


    def adjust_height_of_points(self, pointset:PointSet3D):
        '''Adjusts the height of the points in the pointset using interpolation.'''
        
        points_nparray = np.array(pointset.points)
        for i in range(len(points_nparray)):
            z = barycentric_interpolate_height(points_nparray[i], self.grid.points[:, 2], self.GRID_SIZE, self.bound.x_min, self.bound.x_max)
            points_nparray[i][2] = z

        pointset.points = points_nparray        

    def infect_wells(self, ratio:float|None = 0.2, hard_number:int|None = None):
        ''' Infects a certain number of wells with the plague.
        Args:
            ratio: The ratio of wells to infect. If None, use hard_number.
            hard_number: The number of wells to infect. If None, use ratio.
        '''
        self.terminal_log(f"Entering infect_wells with Ratio: {ratio}, Hard number: {hard_number}")

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
            The Point3D object of the infected well.
        '''
        console_log(f"Entering infect_single_well with index {index}")
    
    def disinfect_single_well(self, index:int):
        ''' Disinfects a single well.
        Args:
            index: The index of the well to disinfect.
        '''
        console_log(f"Entering disinfect_single_well with index {index}")

    def add_single_well(self, x:int, y:int):
        '''Adds a single well to the scene.
        Args:
            x: The x-coordinate of the well.
            y: The y-coordinate of the well.
        '''
        console_log(f"Entering add_single_well with x: {x}, y: {y}")

    def remove_single_well(self, index:int):
        '''Removes a single well from the scene.
        Args:
            index: The index of the well to remove.
        '''
        console_log(f"Entering remove_single_well with index {index}")

    def geodesic_distance(self, start:Point3D|np.ndarray|list|tuple, end:Point3D|np.ndarray|list|tuple):
        '''Calculates the geodesic distance between two points on the grid.'''
        if isinstance(start, Point3D):
            start = np.array([start.x, start.y, start.z])
        if isinstance(end, Point3D):
            end = np.array([end.x, end.y, end.z])
        if isinstance(start, (list, tuple)):
            start = np.array(start)
        if isinstance(end, (list, tuple)):
            end = np.array(end)
    
        # get the triangle in which the start point is located
        starting_triangle_vertices = get_triangle_of_grid_point(start, self.grid.points[:, 2], self.GRID_SIZE, self.bound.x_min, self.bound.x_max)
        starting_triangle_vertices = np.array(starting_triangle_vertices)
        # get the triangle in which the end point is located
        ending_triangle_vertices = get_triangle_of_grid_point(end, self.grid.points[:, 2], self.GRID_SIZE, self.bound.x_min, self.bound.x_max)
        ending_triangle_vertices = np.array(ending_triangle_vertices)
        
        if np.array_equal(starting_triangle_vertices, ending_triangle_vertices):
            # if the start and end points are in the same triangle, return the euclidean distance between them
            return np.linalg.norm(start - end)
        # console_log(f"starting point {start}, ending point {end}")
        # console_log(f"Starting triangle vertices: {starting_triangle_vertices}, Ending triangle vertices: {ending_triangle_vertices}")
        # find the index of the starting and ending triangle in the triangle_indices list
        starting_triangle_found, ending_triangle_found = False, False
        for i, triangle_index in enumerate(self.triangle_indices):
            grid_points = self.grid.points[triangle_index]
            # console_log(f"Grid points: {grid_points}")
            if np.array_equal(grid_points, starting_triangle_vertices):
                starting_triangle_idx = i
                starting_triangle_found = True
            if np.array_equal(grid_points, ending_triangle_vertices):
                ending_triangle_idx = i
                ending_triangle_found = True
            if starting_triangle_found and ending_triangle_found:
                break
        if not starting_triangle_found and ending_triangle_found:
            console_log(f"Starting triangle not found for vertices {starting_triangle_vertices}")
            console_log("================>", ending_triangle_idx)
        if not ending_triangle_found and starting_triangle_found:
            console_log(f"Ending triangle not found for vertices {ending_triangle_vertices}")
            console_log("================>", starting_triangle_idx)
        if not starting_triangle_found and not ending_triangle_found:
            console_log(f"Starting and ending triangles not found for vertices {starting_triangle_vertices} and {ending_triangle_vertices}")
        if not starting_triangle_found or not ending_triangle_found:
            raise ValueError("Starting or ending triangle not found.")
        # else:
        #     console_log(f"Passed")
        
        starting_centroid = self.centroids[starting_triangle_idx]
        # get the euclidean distance between the start point and the centroid of the triangle
        start_distance = np.linalg.norm(start - starting_centroid)

        ending_centroid = self.centroids[ending_triangle_idx]
        # get the euclidean distance between the end point and the centroid of the triangle
        end_distance = np.linalg.norm(end - ending_centroid)

        # get the geodesic distance between the starting_centroid and the ending_centroid
        geodesic_distance = self.shortest_paths_matrix[starting_triangle_idx][ending_triangle_idx]    #self.dijkstra.get_distance_from_to(starting_triangle_idx, ending_triangle_idx)
        # add the euclidean distances to the geodesic distance
        total_distance = geodesic_distance + start_distance + end_distance
        return total_distance
    
    def find_infected_people_with_voronoi(self):
        '''Finds the people infected by the wells, using Voronoi diagram.'''
        ...

    def find_infected_people(self):
        '''Finds the people infected by the wells, using geodesic distances.'''

        # infected_people_indices is a list of indices of the infected people from the population_pcd.points
        self.infected_people_indices = []

        population_nparray = np.array(self.population_pcd.points)
        population_color_nparray = np.array(self.population_pcd.colors)
        wells_nparray = np.array(self.wells_pcd.points)
        
        if not self.COMPUTE_WITH_VORONOI:
            # for every person in the population, check if the closest well to them is infected
            for i, person in enumerate(tqdm(population_nparray, desc="For all infected people")):
                min_distance = np.inf
                for j, well in enumerate(wells_nparray):
                    # get the geodesic distance between the person and the well
                    distance = self.geodesic_distance(person, well)
                    # if the distance is less than the minimum distance, update the minimum distance
                    if distance < min_distance:
                        min_distance = distance
                        closest_well_index = j
                # if the closest well is infected, infect the person
                if closest_well_index in self.infected_wells_indices:
                    population_color_nparray[i] = self.infected_population_color
                    self.infected_people_indices.append(i)

            # update the colors of the population_pcd
            self.population_pcd.colors = population_color_nparray
            self.updateShape(self.population_pcd_name)

        elif self.COMPUTE_WITH_VORONOI:
            self.find_infected_people_with_voronoi()

        self.terminal_log(f"Infected number of people {len(self.infected_people_indices)}") #, with indices {self.infected_people_indices}")
    
    def get_geodesic_distances_from_to_many(self, person, wells, wells_triangle_indices):
        '''Calculates the geodesic distances between a person and many wells.'''

        # get the triangle in which the person is located
        starting_triangle_vertices = get_triangle_of_grid_point(person, self.grid.points[:, 2], self.GRID_SIZE, self.bound.x_min, self.bound.x_max)
        for i, triangle_index in enumerate(self.triangle_indices):
            grid_points = self.grid.points[triangle_index]
            if np.array_equal(grid_points, self.grid.points[starting_triangle_vertices]):
                starting_triangle_idx = i
                break

        # get the euclidean distance between the person and the centroid of the triangle
        starting_centroid = self.centroids[starting_triangle_idx]
        person_to_centroid_distance = np.linalg.norm(person - starting_centroid)

        # for all the wells, calculate the euclidean distance between the well and the centroid of the triangle
        wells_to_centroids_distances = []
        for i, well in enumerate(wells):
            ending_centroid = self.centroids[wells_triangle_indices[i]]
            well_to_centroid_distance = np.linalg.norm(well - ending_centroid)
            wells_to_centroids_distances.append(well_to_centroid_distance)

        self.dijkstra.calculate_shortest_paths_from_vertex(starting_triangle_idx)
        shortest_distances_to_all = self.dijkstra.get_distances()
        shortest_distances_to_wells = shortest_distances_to_all[wells_triangle_indices]

        # for all the wells, calculate the total distance
        total_distances = np.zeros(len(wells))
        for i, distance in enumerate(shortest_distances_to_wells):
            total_distances[i] = person_to_centroid_distance + distance + wells_to_centroids_distances[i]

        # sort the wells by the total distances
        sorted_wells = np.argsort(total_distances)
        # get the 3 closest wells
        closest_well_index = sorted_wells[0]
        second_closest_well_index = sorted_wells[1]
        third_closest_well_index = sorted_wells[2]

        return closest_well_index, second_closest_well_index, third_closest_well_index



    def find_infected_people_stochastic(self):
        '''Finds the people infected by the wells in a stochastic manner.'''
        
        # infected_people_indices is a list of indices of the infected people from the population_pcd.points
        self.infected_people_indices = []

        population_nparray = np.array(self.population_pcd.points)
        population_color_nparray = np.array(self.population_pcd.colors)
        wells_nparray = np.array(self.wells_pcd.points)

        # for all wells, get the triangle in which the well is located
        wells_triangles_indices = []
        for well in wells_nparray:
            triangle_vertices = get_triangle_of_grid_point(well, self.grid.points[:, 2], self.GRID_SIZE, self.bound.x_min, self.bound.x_max)
            for i, triangle_index in enumerate(self.triangle_indices):
                if np.array_equal(triangle_index, triangle_vertices):
                    wells_triangles_indices.append(i)
                    break            

        # for every person in the population, check if the closest well to them is infected
        for i, person in enumerate(population_nparray):
            # get the geodesic distances between the person and all the wells
            closest_wells = self.get_geodesic_distances_from_to_many(person, wells_nparray, wells_triangles_indices)
            choice = np.random.choice(closest_wells, p=[self.P1, self.P2, self.P3])
            if choice in self.infected_wells_indices:
                self.infected_people_indices.append(i)
                population_color_nparray[i] = self.infected_population_color
            else:
                population_color_nparray[i] = self.healthy_population_color

        self.population_pcd.colors = population_color_nparray
        self.updateShape(self.population_pcd_name)

        console_log(f"Infected number of people {len(self.infected_people_indices)}") #, with indices {self.infected_people_indices}")

    def getVoronoi(self, points):
        '''Returns the Voronoi diagram of the points.'''
        pass

    def drawVoronoi(self):
        '''Draws the Voronoi diagram on the scene.'''
        pass

    def resetVoronoi(self):
        '''Resets the Voronoi diagram.'''
        pass
    
    def terminal_log(self, *args):
        '''Print message both to console and the scene.'''
        console_log(*args)
        self.print(*args)
        

def console_log(*args):
    '''Prints the arguments to the console if CONSOLE_TALK is True.'''
    if CONSOLE_TALK:
        print(*args)   

if __name__ == "__main__":
    app = MainClass(start_with = 2, end_immediately = True)