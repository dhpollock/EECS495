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



class ParticleFilter:
	def __init__(self, initialStateVector, initialStateProbabiltyFunctionList, probFunctionArgs, numberOfParticles):
		self.m = numberOfParticles
		self.X = []
		self.newX = []
		self.xMean = initialStateVector[0]
		self.yMean = initialStateVector[1]
		self.rotMean = initialStateVector[2]
		
		if(len(initialStateVector) != len(initialStateProbabiltyFunctionList)):
			return "Error, length of state vector and prob functino list are not equal"
		else:
			for i in range(self.m):
				posVector = []
				posiF = 0
				for j in range(len(initialStateVector)):
					output = initialStateProbabiltyFunctionList[j](initialStateVector[j], probFunctionArgs[j])
					posVector.append(output[0])
					posiF = posiF + output[1]
				self.X.append([posVector,posiF/3.0])

			# print(self.X[0])


	def updateStep(self, command, measurement):
		
		for i in range(self.m):
			entry = []
			#fill in newX with commandstep
			point = simulatedControllerEstimate(self.X[i][0], command[0], command[1], command[3])
			iF = 0
			n = 0
			if(len(measurement) > 0):
				for timeStep in measurement:
					for landmarkMeasure in timeStep.landmarkMeasurements:
						if(landmarkMeasure[0] > 5):
							expected = expectedMeasurement(point, landmarkMeasure[0])
							# iF = iF + getImportanceFactor(expected[0], landmarkMeasure[2])
							# iF = iF + getImportanceFactor(expected[1], landmarkMeasure[3])
							iF = iF + getImportanceFactor(expected, [landmarkMeasure[2], landmarkMeasure[3]])
						# else:
						# 	iF = self.X[i][1]
							n = n+1.0
				if(iF != 0):
					iF = iF/n
				else:
					iF = self.X[i][1]
			else:
				iF = self.X[i][1]
			entry.append(point)
			entry.append(iF)
			self.newX.append(entry)
		self.X = self.newX
		self.newX = []

		# print(self.X[0])

	def getMean(self):
		xVals = [x for [[x,y,rot], iF] in self.X]
		# print(xVals)
		yVals = [y for [[x,y,rot], iF] in self.X]
		rotVals = [rot for [[x,y,rot], iF] in self.X]

		if(len(xVals) > 0 and len(yVals) > 0 and len(rotVals) > 0):
			self.xMean = sum(xVals)/len(xVals)
			self.yMean = sum(yVals)/len(yVals)
			self.rotMean = sum(rotVals)/len(rotVals)

			return [self.xMean, self.yMean, self.rotMean]
		else:
			return [0, 0 , 0]

	def getMaxKernalDesnity(self):
		pass

	def getHistogramMax(self):
		xVals = [x for [[x,y,rot], iF] in self.X]
		# print(xVals)
		yVals = [y for [[x,y,rot], iF] in self.X]
		rotVals = [rot for [[x,y,rot], iF] in self.X]
		pass

	def normalizeWeights(self):
		myWeights = [iF for [[x,y,rot], iF] in self.X]
		myWeightSum = sum(myWeights)
		if(myWeightSum != 0.0):
			for i in range(self.m):
				self.X[i][1] = self.X[i][1]/myWeightSum

	def resampleStep(self):
		myWeights = [iF for [[x,y,rot], iF] in self.X]
		myWeights = np.array(myWeights)
		weightCumSum = np.cumsum(myWeights)
		# print myWeights
		for i in range(self.m):
			filterNum = np.random.rand()
			# if(filterNum < 0.00005):
				# self.newX.append([[randomGaussianPointAndProb(self.xMean, 0.125)[0], randomGaussianPointAndProb(self.yMean, 0.125)[0], randomGaussianPointAndProb(self.rotMean, 2*np.pi/64.0)[0]], 1/(2*self.m)])
				# self.newX.append([[7*np.random.rand()-2, 12*np.random.rand()-6, 2*np.pi*np.random.rand()], 1/(1000*self.m)])
			if(filterNum >= 0 and filterNum < 0.2):
				self.newX.append([[randomGaussianPointAndProb(self.xMean, 3*0.125)[0], randomGaussianPointAndProb(self.yMean, 3*0.125)[0], randomGaussianPointAndProb(self.rotMean, 3*2*np.pi/64.0)[0]], 1/(10*self.m)])

			else:
				index = (np.array([],), )
				n = 0
				while(index[0].size < 1 and n < 10):
					index = np.where(weightCumSum >= np.random.rand())
					n = n+1
				# print index[0]
				if(index[0].size > 0):
					self.newX.append(self.X[index[0][0]])
				else:
					print("resampling not working")
					print(index)
					print(myWeights)
					print(weightCumSum)
					self.newX.append(self.X[i])
		self.X = self.newX
		self.newX = []


		pass
	def getStateProb(self):
		pass
	
