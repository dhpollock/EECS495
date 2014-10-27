#!/usr/bin/env python

##Author: D. Harmon Pollock
##Date: Oct '14
##Class: EECS 495 ML & AI for Robotics
##Professor: Brenna Argall
##Assignment: hw1
##File Name: grid.py

import math
import matplotlib.pyplot as plt
from helperFunctions import *
import numpy as np
from astar import *

##===GridSpace Class===##
## Class used for creating the environment in which the robot operates
## Each grid cell contains a node instance which holds the data for each cell

class GridSpace:

	##Grid Space Class __init__ function
	##
	##Input:
	##	xRange -- a vector of [num, num] containin the min and max x state bounds
	##  yRange -- a vector of [num, num] containin the min and max y state bounds
	##	resolution -- determines the resolution of grid decompisition
	##Output:
	##	N/A
	##

	def __init__(self, xRange, yRange, resolution):
		self.xMin = xRange[0]
		self.xMax = xRange[1]

		self.yMin = yRange[0]
		self.yMax = yRange[1]

		self.cellSize = resolution

		self.xMaxIndex = int(math.floor((self.xMax - self.xMin)/self.cellSize))
		self.yMaxIndex = int(math.floor((self.yMax - self.yMin)/self.cellSize))



		self.spaceArray = []
		self.initSpace()

	##Grid Space Class initSpace function
	##
	## This creates new grid nodes for the grid space defined in the __init__
	##
	##Input:
	##	N/A
	##Output:
	##	N/A
	##
	def initSpace(self):
		#Create the rows
		for i in range(self.xMaxIndex):
			row = []
			for j in range(self.yMaxIndex):
				row.append(GridNode(i, j, self.xMin + i*self.cellSize + self.cellSize/2.0, self.yMin + j*self.cellSize + self.cellSize/2.0))
			self.spaceArray.append(row)


	##Grid Space Class initObstaclesSimple function
	##
	## This modifies the grid nodes to reflect obstacles in the space
	## Simple implies that it does NOT account for connectedness amongst the obstacle nodes (ie Walls)
	##
	##Input:
	##	listOfObstacles -- takes in a list of obstacle objects (defined in helper functions)
	##	expandDist -- optional paramenter, if > 0 then it expands the obstacle into nearby nodes
	##Output:
	##	N/A
	##
	def initObstaclesSimple(self, listOfObstacles, expandDist = 0):
		for obstacle in listOfObstacles:
			for i in range(len(obstacle.obstacleLandmarks)):
				tempCell = self.getCell(obstacle.obstacleLandmarks[i].x, obstacle.obstacleLandmarks[i].y)
				tempCell.obstacle = True
				tempCell.cost = 1000

				if(expandDist > 0):
					for theta in range(0, 360, 15):
						res = self.cellSize/5.0
						delta = 0
						angle = theta*np.pi/180.0
						while(abs(delta) < expandDist):
							tempCell = self.getCell(obstacle.obstacleLandmarks[i].x + np.cos(angle)*delta, obstacle.obstacleLandmarks[i].y + np.sin(angle)*delta)
							if(tempCell.obstacle == False):
								tempCell.obstacle = True
								tempCell.cost = 1000
							delta = delta + res

	##Grid Space Class initObstaclesComplex function
	##
	## This modifies the grid nodes to reflect obstacles in the space
	## Complex implies that it DOES account for connectedness amongst the obstacle nodes (ie Walls)
	##
	##Input:
	##	listOfObstacles -- takes in a list of obstacle objects (defined in helper functions)
	##	expandDist -- optional paramenter, if > 0 then it expands the obstacle into nearby nodes
	##Output:
	##	N/A
	##
	def initObstaclesComplex(self, listOfObstacles, expandDist = 0):
		for obstacle in listOfObstacles:
			for i in range(len(obstacle.obstacleLandmarks)-1):
				res = self.cellSize/5.0
				delta = 0
				distX = obstacle.obstacleLandmarks[i+1].x - obstacle.obstacleLandmarks[i].x
				distY = obstacle.obstacleLandmarks[i+1].y - obstacle.obstacleLandmarks[i].y
				dist = distance([obstacle.obstacleLandmarks[i+1].x, obstacle.obstacleLandmarks[i+1].y], [obstacle.obstacleLandmarks[i].x, obstacle.obstacleLandmarks[i].y])
				angle = np.arctan2(distY, distX)
				while(abs(delta) < dist):
					pointX = obstacle.obstacleLandmarks[i].x + np.cos(angle)*delta
					pointY = obstacle.obstacleLandmarks[i].y + np.sin(angle)*delta
					tempCell = self.getCell(pointX, pointY)
					if(tempCell.obstacle == False):
						tempCell.obstacle = True
						tempCell.cost = 1000
					delta = delta + res

					if(expandDist > 0):
						for theta in range(0, 360, 15):
							res = self.cellSize/5.0
							expandDelta = 0
							expandAngle = theta*np.pi/180.0
							while(abs(expandDelta) < expandDist):
								tempCell = self.getCell(pointX + np.cos(expandAngle)*expandDelta, pointY + np.sin(expandAngle)*expandDelta)
								if(tempCell.obstacle == False):
									tempCell.obstacle = True
									tempCell.cost = 1000
								expandDelta = expandDelta + res


	##Grid Space Class getCell function
	##
	## Gets a node object of a cell given a position
	##
	##Input:
	##	x -- num representing an x coordinate
	##	y -- num representing a y coordinate
	##Output:
	##	a node object for the supplied coordinates
	##
	def getCell(self, x, y):
		if(x >= self.xMax or y >= self.yMax or x < self.xMin or y < self.yMin):
			print("Error: getCell call out of bounds")
			print(x)
			print(y)
			return GridNode(0,0,0,0)
		xindex = int(math.floor((x - self.xMin)/self.cellSize))
		yindex = int(math.floor((y - self.yMin)/self.cellSize))
		return self.spaceArray[xindex][yindex]


	##Grid Space Class paintGrid function
	##
	## Returns a list of matlibplot patches that can be used for plotting
	##
	##Input:
	##	N/A
	##Output:
	##	a list of matlibplot patch objects
	##
	def paintGrid(self):
		patches = []
		for row in self.spaceArray:
			for cell in row:
				if(cell.obstacle):
					patches.append(plt.Rectangle((cell.x - self.cellSize/2.0, cell.y - self.cellSize/2.0),self.cellSize,self.cellSize, alpha = .5, color = 'red'))
				elif(cell.visited):
					patches.append(plt.Rectangle((cell.x- self.cellSize/2.0, cell.y- self.cellSize/2.0),self.cellSize,self.cellSize, alpha = .5, color = 'green'))
				elif(cell.occupied):
					patches.append(plt.Rectangle((cell.x- self.cellSize/2.0, cell.y- self.cellSize/2.0),self.cellSize,self.cellSize, alpha = .5, color = 'blue'))
		return patches

	##Grid Space Class getXTicks function
	##
	## Returns a list of x-coordinates for the start of each cell
	## Can be used for displaying the grid space on a plot
	##
	##Input:
	##	N/A
	##Output:
	##	a list of nums
	##
	def getXTicks(self):
		xticks = [self.xMin]

		for i in range(int(math.ceil((self.xMax - self.xMin)/self.cellSize))):
			xticks.append(xticks[-1] + self.cellSize)

		return xticks

	##Grid Space Class getYTicks function
	##
	## Returns a list of y-coordinates for the start of each cell
	## Can be used for displaying the grid space on a plot
	##
	##Input:
	##	N/A
	##Output:
	##	a list of nums
	##
	def getYTicks(self):
		yticks = [self.yMin]

		for i in range(int(math.ceil((self.yMax - self.yMin)/self.cellSize))):
			yticks.append(yticks[-1] + self.cellSize)

		return yticks


##===GridNode Class===##
## Class used for creating GridNodes used in each cell of the GridSpace

class GridNode:
	##Grid Node Class __init__ function
	##
	## A data storage class
	##
	##Input:
	##	row -- num representing the row in the gridspace array
	##	column -- num representing the column in the gridspace array
	##	x -- num, the x-coordinate of the cells center
	##	y -- num, the y-coordinate of the cells center 
	##Output:
	##	N/A
	def __init__(self, row, column, x, y):
		self.cost = 1
		self.row = row
		self.column = column
		self.x  = x
		self.y = y
		self.obstacle = False
		self.visited = False
		self.occupied = False