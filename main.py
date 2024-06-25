import os
current_path = os.path.dirname(os.path.realpath(__file__))

import sys
sys.path.append(f"{current_path}/plagueSpread/Voronoi/")
sys.path.append(f"{current_path}/plagueSpread/utils/")

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
from plagueSpread.utils.GeometryUtils import isInsidePolygon, barycentric_interpolate_height

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
    def __init__(self, option = None):
        console_log("================= PLAGUE SPREAD =================")
        print("> Press 1 to run the 2D Plague Spread simulation.")
        print("> Press 2 to run the 3D Plague Spread simulation.")
        self.option = option
        self.get_input()
    
    def get_input(self):
        if not self.option:
            self.option = input("> ")
        selection = self.option
        if selection == "1" or selection == 1:
            self.plagueSpread2D = PlagueSpread2D(WIDTH_2D, HEIGHT_2D)
            self.plagueSpread2D.mainLoop()
        elif selection == "2" or selection == 2:
            self.plagueSpread3D = PlagueSpread3D(WIDTH_3D, HEIGHT_3D)
            self.plagueSpread3D.mainLoop()

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
            if isInsidePolygon(point, vertices):
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

from vvrpywork.scene import Scene3D, get_rotation_matrix, world_space
from vvrpywork.shapes import (
    Point3D, Line3D, Arrow3D, Sphere3D, Cuboid3D, Cuboid3DGeneralized,
    PointSet3D, LineSet3D, Mesh3D
)

from noise import pnoise2

class PlagueSpread3D(Scene3D):
    def __init__(self, WIDTH, HEIGHT):
        super().__init__(WIDTH, HEIGHT, "Lab5", output=True, n_sliders=5)
        self._scenario_mode_init()

        self.scenario_parameters_init()
        self.create_grid()

        self.construct_scenario() if not self.TRIAL_MODE else self.construct_mini_scenario()
        if self.wells_pcd.points.size > 1:
            self.Voronoi = self.getVoronoi(self.wells_pcd.points)


        self._print_instructions()
        self.my_mouse_pos = Point3D((0, 0, 0))
        self.addShape(self.my_mouse_pos, "mouse")

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
        grid = PointSet3D(grid, size=1, color=Color.BLACK)
        self.grid = grid
        self.addShape(self.grid, "grid")

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
        self.GRID_SIZE = 80 # will create a grid of N x N points
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

        console_log("--> Press ENTER to reset the scene & print instructions.")
        console_log("--> Press BACKSPACE to print the scenario parameters.")
        console_log("--> Press UP to toggle between trial mode and normal mode.")
        console_log("--> Press RIGHT or LEFT to increase or decrease the number of wells.")
        console_log("--> Press M or N to increase or decrease the population.")
        console_log("--> Press 1 or 2 to set the scenario to version 1 or 2.")
        console_log("--> Press V to toggle the Voronoi diagram.")
        console_log("--> Press SHIFT + V to use the Voronoi diagram for computations.")
        console_log("--> Press LEFT MOUSE BUTTON to add or remove a well.")
        console_log("--> Press RIGHT MOUSE BUTTON to infect or disinfect a well.")
        console_log("--> Press R to toggle between deterministic and stochastic scenario.")
        if self.RANDOM_SELECTION:
            console_log("-->---> Press P to reduce the probability of choosing the closest well.")
            console_log("-->---> Press SHIFT + P to increase the probability of choosing the closest well.")

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

        self.find_infected_people() if not self.RANDOM_SELECTION else self.find_infected_people_stochastic()

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

        self.find_infected_people() if not self.RANDOM_SELECTION else self.find_infected_people_stochastic()


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

    def find_infected_people(self):
        '''Finds the people infected by the wells.'''
        ...
    
    def find_infected_people_stochastic(self):
        '''Finds the people infected by the wells in a stochastic manner.'''
        ...

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
    app = MainClass(2)