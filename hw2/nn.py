##Author: D. Harmon Pollock
##Date: Nov '14
##Class: EECS 495 ML & AI for Robotics
##Professor: Brenna Argall
##Assignment: hw2
##File Name: nn.py

import random
import numpy as np

class NeuralNetwork:

	def __init__(self, structureList):
		self.layers = []
		self.inputIndex = 0
		self.outputIndex = len(structureList)-1

		for i in range(self.outputIndex):
			curLayer = []
			for j in range(structureList[i]):
				curLayer.append(Preceptron(structureList[i+1]))
			self.layers.append(curLayer)
		
		outputLayer = []
		for i in range(structureList[self.outputIndex]):
			outputLayer.append(Preceptron(0))
		
		self.layers.append(outputLayer)


	def train(self, inputData, targetData, targetSSE = 0.1, lr = 0.00001, maxIter = 200):
		sse = 10000000
		interCount = 0
		while(sse > targetSSE and interCount < maxIter):
			tempSSE = 0
			self.resetValues()
			for i in range(len(inputData)):
				curOutput = self.computeOutput(inputData[i])
				deltaOutput = 0
				for j in range(len(curOutput)):
					deltaOutput = deltaOutput + (targetData[i][j] - curOutput[j])
					# print curOutput, targetData[i][j]
				tempSSE = tempSSE + deltaOutput*deltaOutput
				for j in range(len(self.layers)):
					for k in range(len(self.layers[j])):
						self.layers[j][k].updateDeltaWeights(lr, deltaOutput)
						# self.layers[j][k].updateStochasticWeights(lr, deltaOutput)
			self.updateWeights()
			if(tempSSE > sse - .05):
				break
			else:
				sse = tempSSE
				interCount = interCount + 1
			# print sse
		return sse

	def trainBP(self, inputData, targetData, targetSSE = 0.001, lr = 0.1, maxIter = 200):
		sse = 10000000
		interCount = 0
		while(sse > targetSSE and interCount < maxIter):
			tempSSE = 0
			for i in range(len(inputData)):
				curOutput = self.computeOutput(inputData[i])

				deltaOutput = 0
				for j in range(len(curOutput)):
					deltaOutput = deltaOutput + (targetData[i][j] - curOutput[j])
					self.layers[self.outputIndex][j].delta = curOutput[j]*(1 - curOutput[j])*(targetData[i][j] - curOutput[j])
					# print curOutput, targetData[i][j]
				tempSSE = tempSSE + deltaOutput*deltaOutput

				for j in range(self.outputIndex-1, -1, -1):
					for k in range(len(self.layers[j])):
						# print j
						self.layers[j][k].updateBPDelta(self.layers[j+1])

				self.updateBPWeights(lr)
			if(tempSSE > sse - .05):
				break
			else:
				sse = tempSSE
				interCount = interCount + 1
			print sse
		return sse 

	def computeOutput(self, myInput):
		self.resetValues()
		for i in range(len(myInput)):
			self.layers[0][i].value = myInput[i]
		for i in range(1, len(self.layers)):
			for j in range(len(self.layers[i])):
				# print self.layers[i][j].value
				# self.layers[i][j].value = self.layers[i][j].value+1*self.layers[i][j].listofWeights[0]
				for weightNode in self.layers[i-1]:
					self.layers[i][j].value = self.layers[i][j].value + weightNode.value*weightNode.listofWeights[j+1]


		output = []
		for node in self.layers[self.outputIndex]:
			output.append(node.value)
		return output

	def resetValues(self):
		for i in range(len(self.layers)):
			for j in range(len(self.layers[i])):
				self.layers[i][j].value = 0
	def resetDeltaWeights(self):
		for i in range(len(self.layers)):
			for j in range(len(self.layers[i])):
				self.layers[i][j].resetDeltaWeights()
	def updateWeights(self):
		for i in range(len(self.layers)):
			for j in range(len(self.layers[i])):
				self.layers[i][j].updateWeights()
	def updateBPWeights(self, lr):
		for i in range(len(self.layers)-1):
			for j in range(len(self.layers[i])):
				self.layers[i][j].updateBPWeights(lr, self.layers[i+1])
	def resetBPDeltaValues(self):
		for i in range(len(self.layers)):
			for j in range(len(self.layers[i])):
				self.layers[i][j].delta = 0

class Preceptron:

	def __init__(self, connections):
		self.listofWeights = []
		self.value = 0
		self.listofDeltaWeights = []
		self.delta = 0 
		for i in range(connections+1):
			self.listofWeights.append(.01 + .05*np.random.random())
			self.listofDeltaWeights.append(0)

	def updateStochasticWeights(self, lr, deltaOutput):
		self.listofWeights[0] = self.listofWeights[0] + lr*deltaOutput*1
		for i in range(1, len(self.listofDeltaWeights)):
			self.listofWeights[i] = self.listofWeights[i] + lr*deltaOutput*self.value

	def updateDeltaWeights(self, lr, deltaOutput):
		self.listofDeltaWeights[0] = self.listofDeltaWeights[0] + lr*deltaOutput*1
		for i in range(1, len(self.listofDeltaWeights)):
			self.listofDeltaWeights[i] = self.listofDeltaWeights[i] + lr*deltaOutput*self.value
	def updateWeights(self):
		for i in range(len(self.listofDeltaWeights)):
			self.listofWeights[i] = self.listofWeights[i] + self.listofDeltaWeights[i]
			self.listofDeltaWeights[i] = 0
	def resetDeltaWeights(self):
		for weight in self.listofDeltaWeights:
			weight = 0

	def updateBPDelta(self, layer):
		sumError = 0
		for i in range(1,len(layer)+1):
			sumError = sumError + self.listofWeights[i]*layer[i-1].delta
		self.delta = self.value*(1-self.value)*sumError
	def updateBPWeights(self, lr, layer):
		# self.listofDeltaWeights[0] = self.listofDeltaWeights[0] + lr*self.delta*1
		for i in range(1, len(self.listofDeltaWeights)):
			self.listofWeights[i] = self.listofWeights[i] + lr*layer[i-1].delta*self.value