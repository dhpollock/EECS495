##Author: D. Harmon Pollock
##Date: Nov '14
##Class: EECS 495 ML & AI for Robotics
##Professor: Brenna Argall
##Assignment: hw2
##File Name: nn.py

import random
import numpy as np

# class NeuralNetwork:

# 	def __init__(self, structureList):
# 		self.layers = []
# 		self.inputIndex = 0
# 		self.outputIndex = len(structureList)-1

# 		for i in range(self.outputIndex):
# 			curLayer = []
# 			for j in range(structureList[i]):
# 				curLayer.append(Preceptron(structureList[i+1]))
# 			self.layers.append(curLayer)
		
# 		outputLayer = []
# 		for i in range(structureList[self.outputIndex]):
# 			outputLayer.append(Preceptron(0))
		
# 		self.layers.append(outputLayer)


# 	def train(self, inputData, targetData, targetSSE = 0.1, lr = 0.00001, maxIter = 500):
# 		sse = 10000000
# 		interCount = 0
# 		while(sse > targetSSE and interCount < maxIter):
# 			tempSSE = 0
# 			self.resetValues()
# 			for i in range(len(inputData)):
# 				curOutput = self.computeOutput(inputData[i])
# 				deltaOutput = 0
# 				for j in range(len(curOutput)):
# 					deltaOutput = deltaOutput + (targetData[i][j] - curOutput[j])
# 					# print curOutput, targetData[i][j]
# 				tempSSE = tempSSE + deltaOutput*deltaOutput
# 				for j in range(len(self.layers)):
# 					for k in range(len(self.layers[j])):
# 						self.layers[j][k].updateDeltaWeights(lr, deltaOutput)
# 						# self.layers[j][k].updateStochasticWeights(lr, deltaOutput)
# 			self.updateWeights()
# 			if(tempSSE > sse - .005):
# 				break
# 			else:
# 				sse = tempSSE
# 				interCount = interCount + 1
# 			# print sse
# 		return sse

# 	def trainBP(self, inputData, targetData, targetSSE = 1000, lr = 0.1, maxIter = 500):
# 		sse = 10000000
# 		interCount = 0
# 		tempCounter = 0
# 		while(sse > targetSSE and interCount < maxIter):
# 			tempSSE = 0
			
# 			for i in range(len(inputData)):
# 				curOutput = self.computeOutput(inputData[i])
# 				deltaOutput = 0
# 				for j in range(len(curOutput)):
# 					deltaOutput = deltaOutput + (targetData[i][j] - curOutput[j])
# 					self.layers[self.outputIndex][j].delta = curOutput[j]*(1 - curOutput[j])*(targetData[i][j] - curOutput[j])
# 				tempSSE = tempSSE + deltaOutput*deltaOutput

# 				self.updateBPWeights(lr, self.layers[self.outputIndex-1], self.layers[self.outputIndex])

# 				for j in range(self.outputIndex-2, -1, -1):
# 					for k in range(len(self.layers[j])):
# 						self.layers[j][k].updateBPDelta(self.layers[j+1])

# 						self.layers[j][k].updateBPWeights(lr, self.layers[j+1])

# 			if(interCount == maxIter-1):
# 				print("passed maxIter")
# 			if(tempSSE < targetSSE):
# 				print("reached goal")
# 			if(tempSSE > sse):
# 				tempCounter = 0
# 			if(tempSSE > sse+1000):
# 				print("breaking --- run away")
# 				break
# 			if(sse - tempSSE < targetSSE/2000.0):
# 				tempCounter = tempCounter+1
# 				# if(tempSSE > sse):
# 				# 	tempCounter = 0
# 				if(tempCounter > 5):
# 					print "End Condition -- breaking"
# 					break
# 			sse = tempSSE
# 			interCount = interCount + 1

# 			if(interCount % 10 == 0):
# 				print sse
# 		return sse 

# 	def computeOutput(self, myInput):
# 		self.resetValues()
# 		for i in range(len(myInput)):
# 			self.layers[0][i].value = myInput[i]
# 		for i in range(1, len(self.layers)):
# 			for j in range(len(self.layers[i])):
# 				# if(self.layers[i][j].value != 0):
# 				# 	print self.layers[i][j].value
# 				self.layers[i][j].value = self.layers[i][j].value+1*self.layers[i][j].listofWeights[0]
# 				for weightNode in self.layers[i-1]:
# 					self.layers[i][j].value = self.layers[i][j].value + weightNode.value*weightNode.listofWeights[j+1]
# 					# if(self.layers[i][j].value > 100000000000):
# 					# 	self.layers[i][j].value =100000000000
# 					# elif(self.layers[i][j].value < -100000000000):
# 					# 	self.layers[i][j].value = -100000000000
# 					if(abs(self.layers[i][j].value) < .00000000001):
# 						self.layers[i][j].value  = 0
# 					# print weightNode.value, weightNode.listofWeights[j+1]
# 				self.layers[i][j].value = 1/(1+ np.exp(-1*self.layers[i][j].value)) 


# 		output = []
# 		for node in self.layers[self.outputIndex]:
# 			output.append(node.value)
# 		return output

# 	def resetValues(self):
# 		for i in range(len(self.layers)):
# 			for j in range(len(self.layers[i])):
# 				self.layers[i][j].value = 0
# 	def resetDeltaWeights(self):
# 		for i in range(len(self.layers)):
# 			for j in range(len(self.layers[i])):
# 				self.layers[i][j].resetDeltaWeights()
# 	def updateWeights(self):
# 		for i in range(len(self.layers)):
# 			for j in range(len(self.layers[i])):
# 				self.layers[i][j].updateWeights()
# 	def updateBPWeights(self, lr, layer, nextLayer):
# 			for j in range(len(layer)):
# 				layer[j].updateBPWeights(lr, nextLayer)
# 	def resetBPDeltaValues(self):
# 		for i in range(len(self.layers)):
# 			for j in range(len(self.layers[i])):
# 				self.layers[i][j].delta = 0

# class Preceptron:

# 	def __init__(self, connections):
# 		self.listofWeights = []
# 		self.value = 0
# 		self.listofDeltaWeights = []
# 		self.delta = 0 
# 		for i in range(connections+1):
# 			self.listofWeights.append(.01 + .05*np.random.random())
# 			self.listofDeltaWeights.append(0)

# 	def updateStochasticWeights(self, lr, deltaOutput):
# 		self.listofWeights[0] = self.listofWeights[0] + lr*deltaOutput*1
# 		for i in range(1, len(self.listofDeltaWeights)):
# 			self.listofWeights[i] = self.listofWeights[i] + lr*deltaOutput*self.value

# 	def updateDeltaWeights(self, lr, deltaOutput):
# 		self.listofDeltaWeights[0] = self.listofDeltaWeights[0] + lr*deltaOutput*1
# 		for i in range(1, len(self.listofDeltaWeights)):
# 			self.listofDeltaWeights[i] = self.listofDeltaWeights[i] + lr*deltaOutput*self.value
# 	def updateWeights(self):
# 		for i in range(len(self.listofDeltaWeights)):
# 			self.listofWeights[i] = self.listofWeights[i] + self.listofDeltaWeights[i]
# 			self.listofDeltaWeights[i] = 0
# 	def resetDeltaWeights(self):
# 		for weight in self.listofDeltaWeights:
# 			weight = 0

# 	def updateBPDelta(self, layer):
# 		sumError = 0
# 		# sumError = self.listofWeights[0]*1
# 		for i in range(1,len(layer)+1):
# 			sumError = sumError + self.listofWeights[i]*layer[i-1].delta
# 		self.delta = self.value*(1-self.value)*sumError
# 	def updateBPWeights(self, lr, layer):
# 		self.listofWeights[0] = self.listofWeights[0] + lr*self.delta
# 		for i in range(1, len(self.listofDeltaWeights)):
# 			self.listofWeights[i] = self.listofWeights[i] + lr*layer[i-1].delta*self.value
# 			if(abs(self.listofWeights[i]) < .0000000001):
# 				self.listofWeights[i] = 0
# 			# elif(abs(self.listofWeights[i]) < -100000000000):
# 			# 	self.listofWeights[i] = -100000000000
# 			# elif(abs(self.listofWeights[i]) >10000000000):
# 			# 	self.listofWeights[i] = 100000000000

######===========

# 	def trainBP(self, inputData, targetData, targetSSE = 1000, lr = 0.1, maxIter = 500):
# 		sse = 10000000
# 		interCount = 0
# 		tempCounter = 0
# 		while(sse > targetSSE and interCount < maxIter):
# 			tempSSE = 0
			
# 			for i in range(len(inputData)):
# 				curOutput = self.computeOutput(inputData[i])
# 				deltaOutput = 0
# 				for j in range(len(curOutput)):
# 					deltaOutput = deltaOutput + (targetData[i][j] - curOutput[j])
# 					self.layers[self.outputIndex][j].delta = float(curOutput[j])*(1 - float(curOutput[j]))*(float(targetData[i][j]) - float(curOutput[j]))
# 				tempSSE = tempSSE + deltaOutput*deltaOutput

# 				for j in range(self.outputIndex-1, -1, -1):
# 					for k in range(len(self.layers[j])):
# 						self.layers[j][k].updateBPDelta(self.layers[j+1])

# 				self.updateBPWeights(lr)

# 			if(interCount == maxIter-1):
# 				print("passed maxIter")
# 			if(tempSSE < targetSSE):
# 				print("reached goal")
# 			if(tempSSE > sse):
# 				tempCounter = 0
# 			if(tempSSE > sse+1000):
# 				print("breaking --- run away")
# 				break
# 			if(sse - tempSSE < targetSSE/200.0):
# 				tempCounter = tempCounter+1
# 				# if(tempSSE > sse):
# 				# 	tempCounter = 0
# 				if(tempCounter > 5):
# 					print "End Condition -- breaking"
# 					break
# 			sse = tempSSE
# 			interCount = interCount + 1

# 			if(interCount % 10 == 0):
# 				print sse
# 		return sse 

# 	def computeOutput(self, myInput):
# 		self.resetValues()
# 		for i in range(len(myInput)):
# 			self.layers[0][i].value = myInput[i]
# 		for i in range(1, len(self.layers)):
# 			for j in range(len(self.layers[i])):
# 				# if(self.layers[i][j].value != 0):
# 				# 	print self.layers[i][j].value
# 				self.layers[i][j].value = self.layers[i][j].value+1*self.layers[i][j].listofWeights[0]
# 				for weightNode in self.layers[i-1]:
# 					self.layers[i][j].value = self.layers[i][j].value + weightNode.value*weightNode.listofWeights[j+1]
# 					if(self.layers[i][j].value > 100000000000):
# 						self.layers[i][j].value =100000000000
# 					elif(self.layers[i][j].value < -100000000000):
# 						self.layers[i][j].value = -100000000000
# 					if(abs(self.layers[i][j].value) < .00000000001):
# 						self.layers[i][j].value  = 0
# 					# print weightNode.value, weightNode.listofWeights[j+1]


# 		output = []
# 		for node in self.layers[self.outputIndex]:
# 			output.append(node.value)
# 		return output

# 	def resetValues(self):
# 		for i in range(len(self.layers)):
# 			for j in range(len(self.layers[i])):
# 				self.layers[i][j].value = 0
# 	def resetDeltaWeights(self):
# 		for i in range(len(self.layers)):
# 			for j in range(len(self.layers[i])):
# 				self.layers[i][j].resetDeltaWeights()
# 	def updateWeights(self):
# 		for i in range(len(self.layers)):
# 			for j in range(len(self.layers[i])):
# 				self.layers[i][j].updateWeights()
# 	def updateBPWeights(self, lr):
# 		for i in range(len(self.layers)-1):
# 			for j in range(len(self.layers[i])):
# 				self.layers[i][j].updateBPWeights(lr, self.layers[i+1])
# 	def resetBPDeltaValues(self):
# 		for i in range(len(self.layers)):
# 			for j in range(len(self.layers[i])):
# 				self.layers[i][j].delta = 0

# class Preceptron:

# 	def __init__(self, connections):
# 		self.listofWeights = []
# 		self.value = 0
# 		self.listofDeltaWeights = []
# 		self.delta = 0 
# 		for i in range(connections+1):
# 			self.listofWeights.append(.01 + .05*np.random.random())
# 			self.listofDeltaWeights.append(0)

# 	def updateStochasticWeights(self, lr, deltaOutput):
# 		self.listofWeights[0] = self.listofWeights[0] + lr*deltaOutput*1
# 		for i in range(1, len(self.listofDeltaWeights)):
# 			self.listofWeights[i] = self.listofWeights[i] + lr*deltaOutput*self.value

# 	def updateDeltaWeights(self, lr, deltaOutput):
# 		self.listofDeltaWeights[0] = self.listofDeltaWeights[0] + lr*deltaOutput*1
# 		for i in range(1, len(self.listofDeltaWeights)):
# 			self.listofDeltaWeights[i] = self.listofDeltaWeights[i] + lr*deltaOutput*self.value
# 	def updateWeights(self):
# 		for i in range(len(self.listofDeltaWeights)):
# 			self.listofWeights[i] = self.listofWeights[i] + self.listofDeltaWeights[i]
# 			self.listofDeltaWeights[i] = 0
# 	def resetDeltaWeights(self):
# 		for weight in self.listofDeltaWeights:
# 			weight = 0

# 	def updateBPDelta(self, layer):
# 		sumError = 0
# 		# sumError = self.listofWeights[0]*1
# 		for i in range(1,len(layer)+1):
# 			sumError = sumError + self.listofWeights[i]*layer[i-1].delta
# 		self.delta = self.value*(1-self.value)*sumError
# 	def updateBPWeights(self, lr, layer):
# 		self.listofWeights[0] = self.listofWeights[0] + lr*self.delta
# 		for i in range(1, len(self.listofDeltaWeights)):
# 			self.listofWeights[i] = self.listofWeights[i] + lr*layer[i-1].delta*self.value
# 			if(abs(self.listofWeights[i]) < .0000000001):
# 				self.listofWeights[i] = 0
# 			elif(abs(self.listofWeights[i]) < -100000000000):
# 				self.listofWeights[i] = -100000000000
# 			elif(abs(self.listofWeights[i]) >10000000000):
# 				self.listofWeights[i] = 100000000000


class NN:
	def __init__(self, structureList):
		self.layers = []
		self.inputIndex = 0
		self.outputIndex = len(structureList)-1

		outputLayer = []
		for i in range(structureList[0]):
			outputLayer.append(BPPerceptron(0))
		self.layers.append(outputLayer)

		for i in range(1,len(structureList)):
			curLayer = []
			for j in range(structureList[i]):
				curLayer.append(BPPerceptron(structureList[i-1]))
			self.layers.append(curLayer)
	def computeOutput(self, myInput):
		self.resetValues()
		for i in range(len(myInput)):
			self.layers[0][i].value = myInput[i]
		for i in range(1, len(self.layers)):
			for j in range(len(self.layers[i])):
				mySum = self.layers[i][j].bias*1
				for k in range(len(self.layers[i][j].weights)):
					mySum += self.layers[i][j].weights[k]*self.layers[i-1][k].value

				self.layers[i][j].value = 1/(1+ np.exp(-1*mySum)) 
				# if(self.layers[i][j].value > 100000000000):
				# 	self.layers[i][j].value =100000000000
				# elif(self.layers[i][j].value < -100000000000):
				# 	self.layers[i][j].value = -100000000000
				# if(abs(self.layers[i][j].value) < .00000000001):
				# 	self.layers[i][j].value  = 0
		output = []
		for node in self.layers[self.outputIndex]:
			output.append(node.value)
		return output

	def resetValues(self):
		for i in range(len(self.layers)):
			for j in range(len(self.layers[i])):
				self.layers[i][j].value = 0


	def trainBP(self, myInput, myTarget, lr = 0.1, maxIter = 500, targetSSE = 1.0, show = 100):
		sse = 10000000
		interCount = 0
		tempCounter = 0
		printHelp = True
		while(sse > targetSSE and interCount < maxIter):

			for i in range(len(myInput)):
				curOutput = self.computeOutput(myInput[i])

				##Cacluate SSE for data point 
				for j in range(len(self.layers[self.outputIndex])):
						## Cacluate Errors on Output Neurons
						self.layers[self.outputIndex][j].delta = float(self.layers[self.outputIndex][j].value)*(1 - float(self.layers[self.outputIndex][j].value))*(float(myTarget[i][j]) - float(self.layers[self.outputIndex][j].value))
						##Update Weight for output Neurons

						# self.layers[self.outputIndex][j].bias += lr*self.layers[self.outputIndex][j].delta
						# for k in range(len(self.layers[self.outputIndex][j].weights)):
						# 	self.layers[self.outputIndex][j].weights[k] +=  lr*self.layers[self.outputIndex][j].delta*self.layers[self.outputIndex-1][k].value


				

				for l in range(self.outputIndex-1, 0, -1):
					##Update Hidden Layer Errors 
					for j in range(len(self.layers[l])):
						deltaSum = 0
						for k in range(len(self.layers[l+1])):
							deltaSum += self.layers[l+1][k].delta*self.layers[l+1][k].weights[j]
							# print deltaSum
						self.layers[l][j].delta = self.layers[l][j].value*(1-self.layers[l][j].value)*deltaSum
					##Update Hidden Layer Weights 
					# for j in range(len(self.layers[l])):
					# 	self.layers[l][j].bias += lr*self.layers[l][j].delta
					# 	for k in range(len(self.layers[l][j].weights)):
					# 		self.layers[l][j].weights[k] += lr*self.layers[l][j].delta*self.layers[l-1][k].value

				##UPDATE WEIGHTS AFTER?
				for j in range(len(curOutput)):
					self.layers[self.outputIndex][j].bias += lr*self.layers[self.outputIndex][j].delta
					for k in range(len(self.layers[self.outputIndex][j].weights)):
						self.layers[self.outputIndex][j].weights[k] +=  lr*self.layers[self.outputIndex][j].delta*self.layers[self.outputIndex-1][k].value

				for l in range(self.outputIndex-1, 0, -1):
					for j in range(len(self.layers[l])):
						self.layers[l][j].bias = self.layers[l][j].bias + lr*self.layers[l][j].delta
						# deltaW = 0
						for k in range(len(self.layers[l][j].weights)):
							# deltaW += 
							self.layers[l][j].weights[k] += lr*self.layers[l][j].delta*self.layers[l-1][k].value
						# if(printHelp):
				# 			print(self.layers[l][j].weights)
				# printHelp = False

			tempSSE = 0
			for i in range(len(myInput)):
				curOutput = self.computeOutput(myInput[i])
				for j in range(len(curOutput)):
					tempSSE += (myTarget[i][j] - curOutput[j])*(myTarget[i][j] - curOutput[j])
			sse = tempSSE

			interCount = interCount + 1
			if(interCount == maxIter-1):
				print("passed maxIter")
			if(tempSSE < targetSSE):
				print("reached goal")
			if(tempSSE > sse):
				tempCounter = 0
			if(tempSSE > sse+1000):
				print("breaking --- run away")
				break
			# if(sse - tempSSE < .0000000000000000000001):
			# 	tempCounter = tempCounter+1
			# 	if(tempCounter > 15):
			# 		print "End Condition -- breaking"
			# 		break
			sse = tempSSE
			interCount = interCount + 1

			if(interCount % show == 0):
				printHelp = True
				print sse
		return sse

class BPPerceptron:

	def __init__(self, connections):
		self.value = 0
		self.weights = []
		self.delta = 0
		self.bias = .0005*(.5-np.random.random())

		for i in range(connections):
			self.weights.append(.0005*(.5-np.random.random()))

