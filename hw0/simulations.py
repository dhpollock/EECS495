##Author: D. Harmon Pollock
##Date: Oct '14
##Class: EECS 495 ML & AI for Robotics
##Professor: Brenna Argall
##Assignment: hw0
## File Name: simulations.py

#Questions Functions for HW0

import numpy as np
import matplotlib.pyplot as plt
from models import *

def runControlSimulator(initialConditions, commands):
	path = [initialConditions]
	currentLocation = initialConditions
	for command in commands:
		currentLocation = simulatedControllerEstimate(currentLocation, command[0], command[1], command[2])
		path.append(currentLocation)

	plotX = []
	plotY = []
	for point in path:
		plotX.append(point[0])
		plotY.append(point[1])

	plt.scatter(plotX, plotY)
	plt.show()

	return currentLocation

