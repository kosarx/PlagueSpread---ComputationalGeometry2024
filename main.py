import os
current_path = os.path.dirname(os.path.realpath(__file__))

import sys
sys.path.append(os.path.join(current_path, "plagueSpread", "Voronoi"))

from plagueSpread.PlagueSpread2D import PlagueSpread2D
from plagueSpread.PlagueSpread3D import PlagueSpread3D

# WIDTH_2D = 1024
# HEIGHT_2D = 768
WIDTH_2D = 800
HEIGHT_2D = 800

WIDTH_3D = 1400
HEIGHT_3D = 800

class MainClass:
    def __init__(self, start_with = None, end_immediately = False):
        print("================= PLAGUE SPREAD =================")
        self.option = start_with
        self.end_immediately = end_immediately
        self.get_input()
    
    def get_input(self):
        
        _has_been_run = False
        while True:
            if not self.option:
                print("> Press 1 to run the 2D Plague Spread simulation.")
                print("> Press 2 to run the 3D Plague Spread simulation.")
                print("> Press -1 or q to exit.")
                self.option = input("> ")
            selection = self.option
            if selection == "1" or selection == 1:
                try:
                    _has_been_run = True
                    self.plagueSpread2D = PlagueSpread2D(WIDTH_2D, HEIGHT_2D)
                    self.plagueSpread2D.mainLoop()
                except Exception as e:
                    print(f"An error occurred: {e}")
                    print("Please try running the 2D simulation first, then the 3D.")
                    break
                if self.end_immediately:
                    break
            elif selection == "2" or selection == 2:
                try:
                    self.plagueSpread3D = PlagueSpread3D(WIDTH_3D, HEIGHT_3D)
                    self.plagueSpread3D.mainLoop()
                except Exception as e:
                    print(f"An error occurred: {e}")
                    print("Please try running the 2D simulation first, then the 3D.")
                    break
                if self.end_immediately:
                    break
            elif selection == "q" or selection == "-1" or selection == "exit":
                break
            self.option = ""
        print("Goodbye...")

if __name__ == "__main__":
    app = MainClass(start_with = None, end_immediately = None)