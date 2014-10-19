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
import numpy as np
import time
from astar import *


#====Plots for Question 2 ====#

def q1():

	myGrid = grid.GridSpace([-2 , 5], [-6, 6], 1)
	# myGrid.initObstaclesSimple([obstacle1,obstacle2,obstacle3])
	myGrid.initObstaclesComplex([obstacle1,obstacle2,obstacle3])

	# blue_line = mlines.Line2D([],[], color = 'blue')
	# green_line = mlines.Line2D([],[], color = 'green')
	# yellow_line = mlines.Line2D([],[], color = 'yellow', linestyle = '--')
	obstacle_line = mlines.Line2D([],[], color = 'red', marker = 'o', markersize = 5)
 
	fig = plt.figure()

	ax = fig.add_subplot(111)

	ax.plot(obstacle1.x, obstacle1.y, 'r-o', obstacle2.x, obstacle2.y, 'r-o', obstacle3.x, obstacle3.y, 'r-o')
	ax.legend([obstacle_line], ['Obstacles'])
	ax.set_xlabel('X Position (meters)')
	ax.set_ylabel('Y Position (meters)')
	ax.set_title("Title")

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

	for sets in mySets:

		solutionGrid = myAstar.gridGoalSpace(sets[0], sets[1])



		# blue_line = mlines.Line2D([],[], color = 'blue')
		# green_line = mlines.Line2D([],[], color = 'green')
		# yellow_line = mlines.Line2D([],[], color = 'yellow', linestyle = '--')
		obstacle_line = mlines.Line2D([],[], color = 'red', marker = 'o', markersize = 5)
	 
		fig = plt.figure()

		ax = fig.add_subplot(111)

		ax.plot(obstacle1.x, obstacle1.y, 'r-o', obstacle2.x, obstacle2.y, 'r-o', obstacle3.x, obstacle3.y, 'r-o')
		ax.legend([obstacle_line], ['Obstacles'])
		ax.set_xlabel('X Position (meters)')
		ax.set_ylabel('Y Position (meters)')
		ax.set_title("Title")

		gridPatches = solutionGrid.paintGrid()


		for patch in gridPatches:
			ax.add_patch(patch)

		ax.set_xticks(myGrid.getXTicks())
		ax.set_yticks(myGrid.getYTicks())
		ax.grid(True)

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

	for sets in mySets:

		solutionGrid = myAstar.gridGoalSpace(sets[0], sets[1])



		# blue_line = mlines.Line2D([],[], color = 'blue')
		# green_line = mlines.Line2D([],[], color = 'green')
		# yellow_line = mlines.Line2D([],[], color = 'yellow', linestyle = '--')
		obstacle_line = mlines.Line2D([],[], color = 'red', marker = 'o', markersize = 5)
	 
		fig = plt.figure()

		ax = fig.add_subplot(111)

		ax.plot(obstacle1.x, obstacle1.y, 'r-o', obstacle2.x, obstacle2.y, 'r-o', obstacle3.x, obstacle3.y, 'r-o')
		ax.legend([obstacle_line], ['Obstacles'])
		ax.set_xlabel('X Position (meters)')
		ax.set_ylabel('Y Position (meters)')
		ax.set_title("Title")

		gridPatches = solutionGrid.paintGrid()


		for patch in gridPatches:
			ax.add_patch(patch)

		ax.set_xticks(myGrid.getXTicks())
		ax.set_yticks(myGrid.getYTicks())
		ax.grid(True)

		plt.show()


## The Question 9 function is where most of the heavy lifting is done in terms of creating
## a number of different scenarios by easily commenting out differnt section of this code
## more comments to follow...
def q7():



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
	print('"2" -- Outputs graph for question 2')
	print('"3" -- Outputs graph for question 3')
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
		elif(command == '5'):
			q5()
		elif(command == '7'):
			q7()
		elif(command == 'exit' or command == 'quit'):
			sys.exit()





if __name__ == '__main__':

    main()