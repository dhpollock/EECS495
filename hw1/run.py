
#!/usr/bin/env python

##Author: D. Harmon Pollock
##Date: Oct '14
##Class: EECS 495 ML & AI for Robotics
##Professor: Brenna Argall
##Assignment: hw1
##File Name: run.py

##Note: In order to run, the following datafiles must be located in the same folder as this file:
##-->ds1_Barcodes.dat
##-->ds1_Groundtruth.dat
##-->ds1_Landmark_Groundtruth.dat
##-->ds1_Odometry.dat
##-->ds1_Measurement.dat

##Import all the things
import sys
from helperFunctions import *
from models import *
import grid
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import time
from astar import *

blue_line = mlines.Line2D([],[], color = 'blue')
obstacle_line = mlines.Line2D([],[], color = 'red', marker = 'o', markersize = 5)
obstacle_cell = mpatches.Patch(color = 'red', alpha = .5, label = "Obstacle Cell")
expanded_cell = mpatches.Patch(color = 'blue', alpha = .5, label = "Expanded Cell")
path_cell = mpatches.Patch(color = 'green', alpha = .5, label = "Path Cell")

#====Plots for Question 1 ====#

def q1():

	##define gridspace and cell size
	myGrid = grid.GridSpace([-2 , 5], [-6, 6], 1)

	## Simple Obstacles for no walls, complex with walls
	myGrid.initObstaclesSimple([obstacle1,obstacle2,obstacle3])
	# myGrid.initObstaclesComplex([obstacle1,obstacle2,obstacle3])
 
	fig = plt.figure()

	ax = fig.add_subplot(111)

	ax.plot(obstacle1.x, obstacle1.y, 'ro', obstacle2.x, obstacle2.y, 'ro', obstacle3.x, obstacle3.y, 'ro')
	ax.legend([obstacle_line, obstacle_cell], ['True Obstacle', 'Obstacle Cell'])
	ax.set_xlabel('X Position (meters)')
	ax.set_ylabel('Y Position (meters)')
	ax.set_title("Grid Space with Cell Size 1.0m")

	gridPatches = myGrid.paintGrid()


	for patch in gridPatches:
		ax.add_patch(patch)

	ax.set_xticks(myGrid.getXTicks())
	ax.set_yticks(myGrid.getYTicks())
	ax.grid(True)

	plt.show()



def q3():

	myGrid = grid.GridSpace([-2 , 5], [-6, 6], 1)
	myGrid.initObstaclesSimple([obstacle1,obstacle2,obstacle3])
	# myGrid.initObstaclesComplex([obstacle1,obstacle2,obstacle3])


	myAstar = AStar(myGrid)

	setA = [[0.5, -1.5], [0.5, 1.5]]
	setB = [[4.5, 3.5], [4.5, -1.5]]
	setC = [[-0.5, 5.5], [1.5,-3.5]]

	mySets = [setA, setB, setC]
	mySetNames = ["A", "B", "C"]

	n = 0
	for sets in mySets:

		solutionGrid = myAstar.gridGoalSpace(sets[0], sets[1])[0]

	 
		fig = plt.figure()

		ax = fig.add_subplot(111)

		ax.plot(obstacle1.x, obstacle1.y, 'ro', obstacle2.x, obstacle2.y, 'ro', obstacle3.x, obstacle3.y, 'ro')
		ax.legend([obstacle_line, obstacle_cell, expanded_cell, path_cell], ['True Obstacle', 'Obstacle Cell', "Expanded Cell", "Path Cell"])
		ax.set_xlabel('X Position (meters)')
		ax.set_ylabel('Y Position (meters)')
		ax.set_title("Path for Set " + mySetNames[n])
		ax.annotate("start", xy = (sets[0][0],sets[0][1]), xytext = (sets[0][0]+1.0, sets[0][1]), arrowprops = dict(facecolor = 'gray',shrink = 0.05, alpha = 0.5))
		ax.annotate("goal", xy = (sets[1][0],sets[1][1]), xytext = (sets[1][0]+1.0, sets[1][1]), arrowprops = dict(facecolor = 'gray',shrink = 0.05, alpha = 0.5))

		gridPatches = solutionGrid.paintGrid()


		for patch in gridPatches:
			ax.add_patch(patch)

		ax.set_xticks(myGrid.getXTicks())
		ax.set_yticks(myGrid.getYTicks())
		ax.grid(True)

		plt.show()

		n = n+1

def q4():

	##define gridspace and cell size
	myGrid = grid.GridSpace([-2 , 5], [-6, 6], .1)

	## Simple Obstacles for no walls, complex with walls
	myGrid.initObstaclesSimple([obstacle1,obstacle2,obstacle3], expandDist = 0.3)
	# myGrid.initObstaclesComplex([obstacle1,obstacle2,obstacle3])

 
	fig = plt.figure()

	ax = fig.add_subplot(111)

	ax.plot(obstacle1.x, obstacle1.y, 'ro', obstacle2.x, obstacle2.y, 'ro', obstacle3.x, obstacle3.y, 'ro')
	ax.legend([obstacle_line, obstacle_cell], ['True Obstacle', 'Obstacle Cell'])
	ax.set_xlabel('X Position (meters)')
	ax.set_ylabel('Y Position (meters)')
	ax.set_title("Grid Space with Cell Size 0.1 m")

	gridPatches = myGrid.paintGrid()


	for patch in gridPatches:
		ax.add_patch(patch)

	ax.set_xticks(myGrid.getXTicks())
	ax.set_yticks(myGrid.getYTicks())
	ax.grid(True)

	ax.get_xaxis().set_ticklabels([])
	ax.get_yaxis().set_ticklabels([])

	plt.show()

def q5():

	myGrid = grid.GridSpace([-2 , 5], [-6, 6], 0.1)
	myGrid.initObstaclesSimple([obstacle1,obstacle2,obstacle3], expandDist = 0.3)
	# myGrid.initObstaclesComplex([obstacle1,obstacle2,obstacle3], expandDist = 0.3)


	myAstar = AStar(myGrid)

	setA = [[2.45, -3.55], [0.95, -1.55]]
	setB = [[4.95, -0.05], [2.45, 0.25]]
	setC = [[-0.55, 1.45], [1.95, 3.95]]

	mySets = [setA, setB, setC]
	mySetNames = ["A", "B", "C"]

	n = 0
	for sets in mySets:

		solutionGrid = myAstar.gridGoalSpace(sets[0], sets[1])[0]

	 
		fig = plt.figure()

		ax = fig.add_subplot(111)

		ax.plot(obstacle1.x, obstacle1.y, 'ro', obstacle2.x, obstacle2.y, 'ro', obstacle3.x, obstacle3.y, 'ro')
		ax.legend([obstacle_line, obstacle_cell, expanded_cell, path_cell], ['True Obstacle', 'Obstacle Cell', "Expanded Cell", "Path Cell"])
		ax.set_xlabel('X Position (meters)')
		ax.set_ylabel('Y Position (meters)')
		ax.set_title("Path for Set " + mySetNames[n])

		gridPatches = solutionGrid.paintGrid()


		for patch in gridPatches:
			ax.add_patch(patch)
		ax.annotate("start", xy = (sets[0][0],sets[0][1]), xytext = (sets[0][0]+1.0, sets[0][1]), arrowprops = dict(facecolor = 'gray',shrink = 0.05, alpha = 0.5))
		ax.annotate("goal", xy = (sets[1][0],sets[1][1]), xytext = (sets[1][0]+1.0, sets[1][1]), arrowprops = dict(facecolor = 'gray',shrink = 0.05, alpha = 0.5))

		# ax.set_xticks(myGrid.getXTicks())
		# ax.set_yticks(myGrid.getYTicks())
		# ax.grid(True)

		plt.show()
		n = n +1


def q7():

	myGrid = grid.GridSpace([-2 , 5], [-6, 6], 0.1)
	myGrid.initObstaclesSimple([obstacle1,obstacle2,obstacle3], expandDist = 0.3)
	# myGrid.initObstaclesComplex([obstacle1,obstacle2,obstacle3], expandDist = 0.3)

	myAstar = AStar(myGrid)

	setA = [[2.45, -3.55], [0.95, -1.55]]
	setB = [[4.95, -0.05], [2.45, 0.25]]
	setC = [[-0.55, 1.45], [1.95, 3.95]]

	mySets = [setA, setB, setC]
	mySetNames = ["A", "B", "C"]

	n = 0
	for sets in mySets:

		soln = myAstar.gridGoalSpace(sets[0], sets[1])
		solutionGrid = soln[0]
		path = soln[1]
		# print("paths =" + str(path))

		#generate commands and drive them

		pathX = []
		pathY = []
		state = [sets[0][0],sets[0][1],-np.pi/2]
		quivers = np.array([[state[0], state[1], np.cos(state[2]), np.sin(state[2])]])
		for node in path:

			commands = outputDriveControls(state, node, 0.1)

			for command in commands:
				state = simulatedControllerEstimate(state, command[0], command[1], 0.1)
				pathX.append(state[0])
				pathY.append(state[1])
				quivers = np.concatenate((quivers, np.array([[state[0], state[1], np.cos(state[2]), np.sin(state[2])]])), axis = 0)

		qX,qY,qU,qV = zip(*quivers)



		fig = plt.figure()

		ax = fig.add_subplot(111)

		ax.plot(pathX, pathY, 'b-', obstacle1.x, obstacle1.y, 'ro', obstacle2.x, obstacle2.y, 'ro', obstacle3.x, obstacle3.y, 'ro')
		ax.legend([obstacle_line, obstacle_cell, expanded_cell, path_cell, blue_line], ['True Obstacle', 'Obstacle Cell', "Expanded Cell", "Path Cell", "Robot Path"])
		ax.set_xlabel('X Position (meters)')
		ax.set_ylabel('Y Position (meters)')
		ax.set_title("Robot Path through Set " + mySetNames[n])
		ax.annotate("start", xy = (sets[0][0],sets[0][1]), xytext = (sets[0][0]+1.0, sets[0][1]), arrowprops = dict(facecolor = 'gray',shrink = 0.05, alpha = 0.5))
		ax.annotate("goal", xy = (sets[1][0],sets[1][1]), xytext = (sets[1][0]+1.0, sets[1][1]), arrowprops = dict(facecolor = 'gray',shrink = 0.05, alpha = 0.5))
		ax.quiver(qX,qY,qU,qV,angles='xy', scale_units='xy',scale=1)
		gridPatches = solutionGrid.paintGrid()


		for patch in gridPatches:
			ax.add_patch(patch)

		plt.show()
		n = n +1

def q8():

	myGrid = grid.GridSpace([-2 , 5], [-6, 6], 0.1)
	myGrid.initObstaclesSimple([obstacle1,obstacle2,obstacle3], expandDist = 0.3)

	myAstar = AStar(myGrid)

	setA = [[2.45, -3.55], [0.95, -1.55]]
	setB = [[4.95, -0.05], [2.45, 0.25]]
	setC = [[-0.55, 1.45], [1.95, 3.95]]

	mySets = [setA, setB, setC]
	mySetNames = ["A", "B", "C"]

	n = 0
	for sets in mySets:
		for s in range(5,25,5):
			soln = myAstar.driveGridGoalSpace(sets[0], sets[1], [sets[0][0], sets[0][1], -np.pi/2.0],s)
			solutionGrid = soln[0]
			pathX = soln[1]
			pathY = soln[2]
			quivers = soln[3]
			#generate commands and drive them
			
			qX,qY,qU,qV = zip(*quivers)


			fig = plt.figure()

			ax = fig.add_subplot(111)

			ax.plot(pathX, pathY, 'b-', obstacle1.x, obstacle1.y, 'ro', obstacle2.x, obstacle2.y, 'ro', obstacle3.x, obstacle3.y, 'ro')
			ax.legend([obstacle_line, obstacle_cell, expanded_cell, path_cell, blue_line], ['True Obstacle', 'Obstacle Cell', "Expanded Cell", "Path Cell", "Robot Path"])
			ax.set_xlabel('X Position (meters)')
			ax.set_ylabel('Y Position (meters)')
			ax.set_title("Robot Path through Set " + mySetNames[n])
			ax.annotate("start", xy = (sets[0][0],sets[0][1]), xytext = (sets[0][0]+1.0, sets[0][1]), arrowprops = dict(facecolor = 'gray',shrink = 0.05, alpha = 0.5))
			ax.annotate("goal", xy = (sets[1][0],sets[1][1]), xytext = (sets[1][0]+1.0, sets[1][1]), arrowprops = dict(facecolor = 'gray',shrink = 0.05, alpha = 0.5))
			ax.quiver(qX,qY,qU,qV,angles='xy', scale_units='xy',scale=1)


			gridPatches = solutionGrid.paintGrid()


			for patch in gridPatches:
				ax.add_patch(patch)

			plt.show()
		n = n+1

def q9():

	myGrid = grid.GridSpace([-2 , 5], [-6, 6], 1)
	myGrid.initObstaclesSimple([obstacle1,obstacle2,obstacle3])
	# myGrid.initObstaclesComplex([obstacle1,obstacle2,obstacle3])


	myAstar = AStar(myGrid)

	setA = [[0.5, -1.5], [0.5, 1.5]]
	setB = [[4.5, 3.5], [4.5, -1.5]]
	setC = [[-0.5, 5.5], [1.5,-3.5]]

	mySets = [setA, setB, setC]
	mySetNames = ["A", "B", "C"]

	n = 0
	for sets in mySets:

		soln = myAstar.driveGridGoalSpace(sets[0], sets[1], [sets[0][0], sets[0][1], -np.pi/2.0])
		solutionGrid = soln[0]
		pathX = soln[1]
		pathY = soln[2]
		quivers = soln[3]		
		qX,qY,qU,qV = zip(*quivers)
		#generate commands and drive them
		fig = plt.figure()

		ax = fig.add_subplot(111)

		ax.plot(pathX, pathY, 'b-', obstacle1.x, obstacle1.y, 'ro', obstacle2.x, obstacle2.y, 'ro', obstacle3.x, obstacle3.y, 'ro')
		ax.legend([obstacle_line, obstacle_cell, expanded_cell, path_cell, blue_line], ['True Obstacle', 'Obstacle Cell', "Expanded Cell", "Path Cell", "Robot Path"])
		ax.set_xlabel('X Position (meters)')
		ax.set_ylabel('Y Position (meters)')
		ax.set_title("Robot Path through Set " + mySetNames[n])
		ax.annotate("start", xy = (sets[0][0],sets[0][1]), xytext = (sets[0][0]+1.0, sets[0][1]), arrowprops = dict(facecolor = 'gray',shrink = 0.05, alpha = 0.5))
		ax.annotate("goal", xy = (sets[1][0],sets[1][1]), xytext = (sets[1][0]+1.0, sets[1][1]), arrowprops = dict(facecolor = 'gray',shrink = 0.05, alpha = 0.5))
		ax.quiver(qX,qY,qU,qV,angles='xy', scale_units='xy',scale=1)


		gridPatches = solutionGrid.paintGrid()


		for patch in gridPatches:
			ax.add_patch(patch)

		ax.set_xticks(myGrid.getXTicks())
		ax.set_yticks(myGrid.getYTicks())
		ax.grid(True)

		plt.show()
		n = n+1

def main():

	##Load the files, else return error
	print("Trying to load data files...")

	try:
		#Load Data Files Here
		print("Data files loaded.")
	except:
		print("Error loading data files, make sure they are in the same directory as this file")
		time.sleep(3.0)
		sys.exit()

	print("")

	##Print out types of commands and wait for input

	print("The following commands create graphical outputs for the following questions:")
	print('"1" -- Outputs graph for question 1')
	print('"3" -- Outputs graph for question 3')
	print('"4" -- Outputs graph for question 4')
	print('"5" -- Outputs graph for question 5')
	print('"7" -- Outputs graph for question 7')
	print('"8" -- Outputs graph for question 8')
	print('"9" -- Outputs graph for question 9')
	print("")
	print('"exit" or "quit" -- to exit the program')

	while(True):

		command = raw_input("Command: ")

		if(command == '1'):
			q1()
		elif(command == '3'):
			q3()
		elif(command == '4'):
			q4()
		elif(command == '5'):
			q5()
		elif(command == '7'):
			q7()
		elif(command == '8'):
			q8()
		elif(command == '9'):
			q9()
		elif(command == 'exit' or command == 'quit'):
			sys.exit()





if __name__ == '__main__':

    main()