##Author: D. Harmon Pollock
##Date: Oct '14
##Class: EECS 495 ML & AI for Robotics
##Professor: Brenna Argall
##Assignment: hw1
##File Name: astar.py

from helperFunctions import *
from copy import deepcopy
from models import *

class AStar:
	def __init__(self, GridObject):
		self.environment = GridObject

	def gridGoalSpace(self, startVector, endVector):

		tempGoalSpace = deepcopy(self.environment)

		expandedNode = tempGoalSpace.getCell(startVector[0], startVector[1])
		# expandedNode.cost = 0
		expandedNode.visited = True
		goalNode = tempGoalSpace.getCell(endVector[0], endVector[1])
		goalNode.cost = 0

		path = []

		while(expandedNode != goalNode):
			
			# if(expandedNode == goalNode):
			# 	print("found goal")
			# 	break;

			nodesToOpen = []
			costs = []
			

			rowMax = expandedNode.row + 1
			rowMin = expandedNode.row - 1
			colMax = expandedNode.column +1
			colMin = expandedNode.column -1
			if(expandedNode.row <= 0):
				rowMin = expandedNode.row
			elif(expandedNode.row >= tempGoalSpace.xMaxIndex-1):
				rowMax = expandedNode.row
			if(expandedNode.column <= 0):
				colMin = expandedNode.column
			elif(expandedNode.column >= tempGoalSpace.yMaxIndex-1):
				colMax = expandedNode.column

			for i in range(rowMin, rowMax+1):
				for j in range(colMin, colMax+1):
					tempNode = tempGoalSpace.spaceArray[i][j]
					tempNode.occupied = True

					if(expandedNode != tempNode):
						nodesToOpen.append(tempNode)
						costs.append(tempNode.cost + distance([goalNode.x, goalNode.y],[tempNode.x, tempNode.y]))

			minCostNodeIndex = costs.index(min(costs))
			##maybe can't do this if statement, subject to minimam
			if(nodesToOpen[minCostNodeIndex] != expandedNode):
				nodesToOpen[minCostNodeIndex].visited = True
				path.append([nodesToOpen[minCostNodeIndex].x, nodesToOpen[minCostNodeIndex].y])
				expandedNode = nodesToOpen[minCostNodeIndex]
			else:
				print("Failure, unable to reach goal");
				break;



		return [tempGoalSpace, path]

	def driveGridGoalSpace(self, startVector, endVector, robotInitVector):

		tempGoalSpace = deepcopy(self.environment)

		expandedNode = tempGoalSpace.getCell(startVector[0], startVector[1])
		# expandedNode.cost = 0
		expandedNode.visited = True
		goalNode = tempGoalSpace.getCell(endVector[0], endVector[1])
		goalNode.cost = 0

		path = []

		#robot path
		pathX = []
		pathY = []
		state = robotInitVector

		while(expandedNode != goalNode):
			
			# if(expandedNode == goalNode):
			# 	print("found goal")
			# 	break;

			nodesToOpen = []
			costs = []
			

			rowMax = expandedNode.row + 1
			rowMin = expandedNode.row - 1
			colMax = expandedNode.column +1
			colMin = expandedNode.column -1
			if(expandedNode.row <= 0):
				rowMin = expandedNode.row
			elif(expandedNode.row >= tempGoalSpace.xMaxIndex-1):
				rowMax = expandedNode.row
			if(expandedNode.column <= 0):
				colMin = expandedNode.column
			elif(expandedNode.column >= tempGoalSpace.yMaxIndex-1):
				colMax = expandedNode.column

			for i in range(rowMin, rowMax+1):
				for j in range(colMin, colMax+1):
					tempNode = tempGoalSpace.spaceArray[i][j]
					tempNode.occupied = True

					if(expandedNode != tempNode):
						nodesToOpen.append(tempNode)
						costs.append(tempNode.cost + distance([goalNode.x, goalNode.y],[tempNode.x, tempNode.y]))

			minCostNodeIndex = costs.index(min(costs))
			##maybe can't do this if statement, subject to minimam
			if(nodesToOpen[minCostNodeIndex] != expandedNode):
				nodesToOpen[minCostNodeIndex].visited = True
				path.append([nodesToOpen[minCostNodeIndex].x, nodesToOpen[minCostNodeIndex].y])
				expandedNode = nodesToOpen[minCostNodeIndex]



				commands = outputDriveControls(state, [nodesToOpen[minCostNodeIndex].x, nodesToOpen[minCostNodeIndex].y], 0.1)
				print commands
				for command in commands:
					state = simulatedControllerEstimate(state, command[0], command[1], 0.1)
					pathX.append(state[0])
					pathY.append(state[1])

			else:
				print("Failure, unable to reach goal");
				break;


		return [tempGoalSpace, pathX, pathY]