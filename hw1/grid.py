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

class GridSpace:

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

	def initSpace(self):
		#Create the rows
		for i in range(self.xMaxIndex):
			row = []
			for j in range(self.yMaxIndex):
				row.append(GridNode(i, j, self.xMin + i*self.cellSize + self.cellSize/2.0, self.yMin + j*self.cellSize + self.cellSize/2.0))
			self.spaceArray.append(row)

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

	# def expandObstacles(self, expandDist, simpleBool):
		# if(simpleBool):



	def getCell(self, x, y):
		if(x >= self.xMax or y >= self.yMax or x < self.xMin or y < self.yMin):
			print("Error: getCell call out of bounds")
			print(x)
			print(y)
			return GridNode(0,0,0,0)
		xindex = int(math.floor((x - self.xMin)/self.cellSize))
		yindex = int(math.floor((y - self.yMin)/self.cellSize))
		return self.spaceArray[xindex][yindex]



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

	def getXTicks(self):
		xticks = [self.xMin]

		for i in range(int(math.ceil((self.xMax - self.xMin)/self.cellSize))):
			xticks.append(xticks[-1] + self.cellSize)

		return xticks


	def getYTicks(self):
		yticks = [self.yMin]

		for i in range(int(math.ceil((self.yMax - self.yMin)/self.cellSize))):
			yticks.append(yticks[-1] + self.cellSize)

		return yticks



class GridNode:

	def __init__(self, row, column, x, y):
		self.cost = 1
		self.row = row
		self.column = column
		self.x  = x
		self.y = y
		self.obstacle = False
		self.visited = False
		self.occupied = False