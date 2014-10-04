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
from helperFunctions import *

def runControlSimulator(initialConditions, commands, obstacleBool):
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

	# plt.scatter(plotX, plotY)
	if(obstacleBool):
		plt.plot(plotX, plotY, 'b-', obstacle1.x, obstacle1.y, 'r-o', obstacle2.x, obstacle2.y, 'r-o', obstacle3.x, obstacle3.y, 'r-o')
	else:
		plt.plot(plotX, plotY, 'b-')
	plt.xlabel('X Position (meters)')
	plt.ylabel('Y Position (meters)')
	plt.title("State Estimate Based on Command Input")
	plt.show()

	return currentLocation

