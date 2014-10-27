#!/usr/bin/env python

##Author: D. Harmon Pollock
##Date: Oct '14
##Class: EECS 495 ML & AI for Robotics
##Professor: Brenna Argall
##Assignment: hw1
##File Name: astar.py

from helperFunctions import *
from copy import deepcopy
from models import *


##===AStar Class===##
## Class used for doing A* search on a given GridSpace class

class AStar:

	##AStar Class __init__ function
	##
	##Input:
	##	GridObject -- an instance of a grid for seraching
	##Output:
	##	N/A
	##
	def __init__(self, GridObject):
		self.environment = GridObject

	##AStar Class gridGoalSpace function
	##
	## This is an A* search the returns a gridSpace object with data and a completable path 
	##
	##Input:
	##	startVector -- a x,y coordinate vector from where to start search
	##	endVector -- x,y coordinate vector for goal location
	##Output:
	##	tempGoalSpace -- copy of self.enivronment containing updated information about the space
	##	pathX -- a list of x-coordinates of how to reach the goal location from start location
	##	pathY -- a list of y-coordinates of how to reach the goal location from start location
	##	quivers -- an np.array of quivers that allows for plotting the robots vector for each command step

	def gridGoalSpace(self, startVector, endVector):

		## Copy the goal space in case we want to do more with it later
		tempGoalSpace = deepcopy(self.environment)

		##Initalize the start node in the grid space
		expandedNode = tempGoalSpace.getCell(startVector[0], startVector[1])
		expandedNode.visited = True
		
		##Initalize the goal node in the grid space
		goalNode = tempGoalSpace.getCell(endVector[0], endVector[1])
		goalNode.cost = 0

		## Path on how to get to goal from start
		path = []

		##Build path and update gride space until we find the goal
		while(expandedNode != goalNode):

			##List of neighbors to evaluate
			nodesToOpen = []

			##Cost of visiting those neighbors
			costs = []
			
			##Visit all 8 neighbors 1 space away, including diagonals
			rowMax = expandedNode.row + 1
			rowMin = expandedNode.row - 1
			colMax = expandedNode.column +1
			colMin = expandedNode.column -1

			##Account for edge and corner cases
			if(expandedNode.row <= 0):
				rowMin = expandedNode.row
			elif(expandedNode.row >= tempGoalSpace.xMaxIndex-1):
				rowMax = expandedNode.row
			if(expandedNode.column <= 0):
				colMin = expandedNode.column
			elif(expandedNode.column >= tempGoalSpace.yMaxIndex-1):
				colMax = expandedNode.column

			##Iterate for each neighbor 
			for i in range(rowMin, rowMax+1):
				for j in range(colMin, colMax+1):

					##Get the cell node at that neighbor location
					tempNode = tempGoalSpace.spaceArray[i][j]
					##Mark that we have calculated its heuristic, for plotting purposes
					tempNode.occupied = True

					##Make sure its not the node we are currently at
					if(expandedNode != tempNode):

						##Add to neighbors we have visited
						nodesToOpen.append(tempNode)
						##Calculate the cost of each node, heuristic of cost to visit + euclidian distance
						costs.append(tempNode.cost + distance([goalNode.x, goalNode.y],[tempNode.x, tempNode.y]))

			##Find the min cost neighbor
			minCostNodeIndex = costs.index(min(costs))

			##Double check to make sure its not where are currently are
			if(nodesToOpen[minCostNodeIndex] != expandedNode):

				##Visit that neighbor and track it as our path
				nodesToOpen[minCostNodeIndex].visited = True
				path.append([nodesToOpen[minCostNodeIndex].x, nodesToOpen[minCostNodeIndex].y])

				##Reset node to expand to our new min node and iterate
				expandedNode = nodesToOpen[minCostNodeIndex]
			else:
				print("Failure, unable to reach goal");
				break;



		return [tempGoalSpace, path]

	##AStar Class driveGridGoalSpace function
	##
	## Same A* search as before, but now with added driving capabilities
	##
	##Input:
	##	startVector -- a x,y coordinate vector from where to start search
	##	endVector -- x,y coordinate vector for goal location
	##	robotInitVector -- a x,y,theta coordinate for the robots starting position
	##	steps -- an optional variable for outputing certain steps of the search
	##Output:
	##	tempGoalSpace -- copy of self.enivronment containing updated information about the space
	##	path -- a list of [x,y] coordinates of how to reach the goal location from start location


	def driveGridGoalSpace(self, startVector, endVector, robotInitVector, steps=-1):

		## Copy the goal space in case we want to do more with it later
		tempGoalSpace = deepcopy(self.environment)

		##Initalize the start node in the grid space
		expandedNode = tempGoalSpace.getCell(startVector[0], startVector[1])
		expandedNode.visited = True

		##Initalize the goal node in the grid space
		goalNode = tempGoalSpace.getCell(endVector[0], endVector[1])
		goalNode.cost = 0

		path = []

		## Path on how to get to goal from start
		pathX = []
		pathY = []

		##Track the robot positions
		state = robotInitVector
		quivers = np.array([[state[0], state[1], np.cos(state[2]), np.sin(state[2])]])
		
		n = 0

		##Build path and update gride space until we find the goal
		while(expandedNode != goalNode):
			
			##Stop if we are only looking to execute a certain number of steps
			if(steps != -1):
				if(n > steps):
					break

			##List of neighbors to evaluate
			nodesToOpen = []

			##Cost of visiting those neighbors
			costs = []
			
			##Visit all 8 neighbors 1 space away, including diagonals
			rowMax = expandedNode.row + 1
			rowMin = expandedNode.row - 1
			colMax = expandedNode.column +1
			colMin = expandedNode.column -1

			##Account for edge and corner cases
			if(expandedNode.row <= 0):
				rowMin = expandedNode.row
			elif(expandedNode.row >= tempGoalSpace.xMaxIndex-1):
				rowMax = expandedNode.row
			if(expandedNode.column <= 0):
				colMin = expandedNode.column
			elif(expandedNode.column >= tempGoalSpace.yMaxIndex-1):
				colMax = expandedNode.column

			##Iterate for each neighbor 
			for i in range(rowMin, rowMax+1):
				for j in range(colMin, colMax+1):

					##Get the cell node at that neighbor location
					tempNode = tempGoalSpace.spaceArray[i][j]
					##Mark that we have calculated its heuristic, for plotting purposes
					tempNode.occupied = True

					##Make sure its not the node we are currently at
					if(expandedNode != tempNode):

						##Add to neighbors we have visited
						nodesToOpen.append(tempNode)
						##Calculate the cost of each node, heuristic of cost to visit + euclidian distance
						costs.append(tempNode.cost + distance([goalNode.x, goalNode.y],[tempNode.x, tempNode.y]))
		
			##Find the min cost neighbor
			minCostNodeIndex = costs.index(min(costs))
			
			##Double check to make sure its not where are currently are
			if(nodesToOpen[minCostNodeIndex] != expandedNode):

				##Visit that neighbor and track it as our path
				nodesToOpen[minCostNodeIndex].visited = True
				path.append([nodesToOpen[minCostNodeIndex].x, nodesToOpen[minCostNodeIndex].y])

				##Reset node to expand to our new min node and iterate
				expandedNode = nodesToOpen[minCostNodeIndex]

				##Create commands through the IK controller get fet from our current state to the goal location
				commands = outputDriveControls(state, [nodesToOpen[minCostNodeIndex].x, nodesToOpen[minCostNodeIndex].y], 0.1)
				
				##Step through each command and track the state changes through path and quivers
				for command in commands:
					state = simulatedControllerEstimate(state, command[0], command[1], 0.1)
					pathX.append(state[0])
					pathY.append(state[1])
					quivers = np.concatenate((quivers, np.array([[state[0], state[1], np.cos(state[2]), np.sin(state[2])]])), axis = 0)
			
			else:
				print("Failure, unable to reach goal")
				break
			n=n+1

		return [tempGoalSpace, pathX, pathY, quivers]