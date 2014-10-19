##Author: D. Harmon Pollock
##Date: Oct '14
##Class: EECS 495 ML & AI for Robotics
##Professor: Brenna Argall
##Assignment: hw1
##File Name: astar.py

from helperFunctions import *
from copy import deepcopy

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
				expandedNode = nodesToOpen[minCostNodeIndex]
			else:
				print("Failure, unable to reach goal");
				break;



		return tempGoalSpace