if __name__ == "__main__":
    import os
    current_path = os.path.dirname(os.path.realpath(__file__))

    import sys
    sys.path.append(os.path.join(current_path, "Voronoi"))
    sys.path.append(os.path.join(current_path, ".."))


# Standard imports
import random
random.seed(42)
import numpy as np
from scipy.spatial import Voronoi as SciVoronoi
import matplotlib.pyplot as plt

# Plague Spread imports
from Voronoi import Voronoi # Voronoi is a class from the plagueSpread package\ 
#                                                 Needs debugging, doesn't work!
from plagueSpread.utils.GeometryUtils import LineEquation2D
from plagueSpread.utils.GeometryUtils import is_inside_polygon_2d#isInsidePolygon2D

# VVR imports
from vvrpywork.constants import Key, Mouse, Color
from vvrpywork.scene import Scene2D
from vvrpywork.shapes import (
    Point2D, Line2D, Rectangle2D,
    PointSet2D, LineSet2D, 
)

DEBUG = False # False
CONSOLE_TALK = True # False
TRIAL_MODE = False # False

class PlagueSpread2D(Scene2D):
    def __init__(self, WIDTH, HEIGHT, faux_run = False):
        if faux_run:
            super().__init__(WIDTH, HEIGHT, "Plague Spread 2D", resizable=False)
            super().on_key_press(Key.ESCAPE, None)
            return
        super().__init__(WIDTH, HEIGHT, "Plague Spread 2D", resizable=False)
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
        console_log(f"Percentage of infected people: {len(self.infected_people_indices) / self.POPULATION * 100}%")
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
                        ## infect the closest well
                        # get the current infected percentage
                        infected_percentage = len(self.infected_people_indices) / self.POPULATION
                        self.infect_single_well(closest_well_index)
                        self.find_infected_people() if not self.RANDOM_SELECTION else self.find_infected_people_stochastic()
                        # get the new infected percentage
                        new_infected_percentage = len(self.infected_people_indices) / self.POPULATION
                        # print the percentage increase
                        console_log(f"Percentage impact: {(new_infected_percentage - infected_percentage)*100}")
                    else:
                        ## disenfect the closest well
                        # get the current infected percentage
                        infected_percentage = len(self.infected_people_indices) / self.POPULATION
                        self.disinfect_single_well(closest_well_index)
                        self.find_infected_people() if not self.RANDOM_SELECTION else self.find_infected_people_stochastic()
                        # get the new infected percentage
                        new_infected_percentage = len(self.infected_people_indices) / self.POPULATION
                        # print the percentage decrease
                        console_log(f"Percentage impact: {(new_infected_percentage - infected_percentage)*100}")
            # else, if the left mouse button was released...
            elif button == Mouse.MOUSE1:
                infected_percentage = len(self.infected_people_indices) / self.POPULATION
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
                new_infected_percentage = len(self.infected_people_indices) / self.POPULATION
                console_log(f"Percentage impact: {(new_infected_percentage - infected_percentage)*100}")
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
            self.POPULATION_SIZE = 0.7 if not self.TRIAL_MODE else 0.7
            self.WELLS = 15 if not self.TRIAL_MODE else 3
            self.reset_scene()

        def version_2():
            self.POPULATION = 10000 if not self.TRIAL_MODE else 10
            self.POPULATION_SIZE = 0.7 if not self.TRIAL_MODE else 0.7
            self.WELLS = 30 if not self.TRIAL_MODE else 5
            self.reset_scene()

        def version_3():
            self.POPULATION = 30000 if not self.TRIAL_MODE else 15
            self.POPULATION_SIZE = 0.5 if not self.TRIAL_MODE else 0.7
            self.WELLS = 45 if not self.TRIAL_MODE else 7
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
            self.POPULATION += 100 if not self.TRIAL_MODE else 10
            self.reset_scene()
        if symbol == Key.N:
            self.POPULATION -= 100 if not self.TRIAL_MODE else 10
            self.reset_scene()
        # toggle between dense regions of the population
        if symbol == Key.W:
            self.DENSE_REGIONS = not self.DENSE_REGIONS
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
        if symbol == Key._3:
            version_3()
    
    def scenario_parameters_init(self):
        self.bbx =[[-0.9, -0.9], [0.9, 0.9]]
        self.bound = None
        self.Voronoi = None

        # populations, counts, and ratios
        self.POPULATION = 1000
        self.POPULATION_SIZE = 0.7
        self.WELLS = 15
        self.ratio_of_infected_wells = 0.2
        self.P1 = 0.8 # probability of choosing the closest well
        self.P2 = 0.15 # probability of choosing the second closest well
        self.P3 = 0.05 # probability of choosing the third closest well

        # logic, controllers    
        self.RANDOM_SELECTION = False
        self.DENSE_REGIONS = False
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
        print("--> Press 1 or 2 or 3 to set the scenario to version 1 or 2 or 3.")
        print("--> Press W to toggle dense regions of the population.")
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
        self.population_pcd = PointSet2D(color=self.healthy_population_color, size=self.POPULATION_SIZE)
        if not self.DENSE_REGIONS:
            self.population_pcd.createRandom(self.bound, self.POPULATION, self.population_pcd_name, self.healthy_population_color)
        else:
            # regions of interests
            rois = np.array([[-0.5, -0.5], [0.5, 0.5]])
            if self.POPULATION <= 1000:
                weights = np.array([0.6, 0.4])
                rois_radii = np.array([0.3, 0.2])
                decrease_factor = 0.5
            elif self.POPULATION < 10000:
                weights = np.array([0.5, 0.5])
                rois_radii = np.array([0.3, 0.2])
                decrease_factor = 0.8
            elif self.POPULATION < 30000 or self.POPULATION >= 30000:
                weights = np.array([0.7, 0.7])
                rois_radii = np.array([0.3, 0.4])
                decrease_factor = 2
            self.population_pcd.createRandomWeighted(self.bound, self.POPULATION, self.population_pcd_name, self.healthy_population_color, rois, rois_radii, weights, decrease_factor)
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

        self.bound = Rectangle2D(self.bbx[0], self.bbx[1])
        self.addShape(self.bound, "bound")

        self.population_pcd_name = "Mini Population"
        self.population_pcd = PointSet2D(color=self.healthy_population_color, size=self.POPULATION_SIZE)
        if not self.DENSE_REGIONS:
            self.population_pcd.createRandom(self.bound, self.POPULATION, self.population_pcd_name, self.healthy_population_color)
        else:
            # regions of interests
            rois = np.array([[-0.5, -0.5], [0.5, 0.5]])
            if self.POPULATION <= 5:
                weights = np.array([0.6, 0.4])
                rois_radii = np.array([0.3, 0.2])
                decrease_factor = 0.5
            elif self.POPULATION < 10:
                weights = np.array([0.5, 0.5])
                rois_radii = np.array([0.3, 0.2])
                decrease_factor = 0.8
            elif self.POPULATION < 15 or self.POPULATION >= 15:
                weights = np.array([3, 0.7])
                rois_radii = np.array([0.3, 0.4])
                decrease_factor = 2
            self.population_pcd.createRandomWeighted(self.bound, self.POPULATION, self.population_pcd_name, self.healthy_population_color, rois, rois_radii, weights, decrease_factor)
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
        console_log(f"Percentage of infected people: {len(self.infected_people_indices) / self.POPULATION * 100}%")

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
        console_log(f"Percentage of infected people: {len(self.infected_people_indices) / self.POPULATION * 100}%")

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
            if is_inside_polygon_2d(point, vertices):
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
    

def console_log(*args):
    '''Prints the arguments to the console if CONSOLE_TALK is True.'''
    if CONSOLE_TALK:
        print(*args)   

if __name__ == "__main__":
    # create the scene
    scene = PlagueSpread2D(800, 800)
    # run the scene
    scene.mainLoop()