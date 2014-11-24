##Author: D. Harmon Pollock
##Date: Nov '14
##Class: EECS 495 ML & AI for Robotics
##Professor: Brenna Argall
##Assignment: hw2
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
import simulations
from nn import * 
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import time


def q1(barcodes, groundTruth, landmarkGroundtrush, odometry, measurement, removeOutlier = False):
	
	##Pull all the relevent data in to an array
	dataset = createSensorNoiseDataset(measurement, groundTruth, barcodes, randomize = True)
	
	##Create input and target arrays
	target = makeTargetArray(dataset)
	myInput = getTrainingData(dataset)

	##Find the min/max of each attribute of the sets in order to normalize
	minMaxDataset = getListMinMax(myInput)
	# targetMinMax = getListMinMax(target)
	
	# ##Normalize
	# targetNorm = normalize(target, targetMinMax)
	# myInputNorm = normalize(myInput, minMaxDataset)
	if(removeOutlier):
		count = 0
		for i in range(len(myInput)):
			if(target[i][1] < minMaxDataset[1][0]):
				dataset[i] = 0
				target[i] = 0
				myInput[i] = 0
				count = count+1
			elif(target[i][1] > minMaxDataset[1][1]):
				dataset[i] = 0
				target[i] = 0
				myInput[i] = 0
				count = count+1
		print "Outlier Count: ", count
		print "New Dataset Size w/o Outliers: ", len(myInput)
		dataset = [x for x in dataset if x !=0]
		myInput = [x for x in myInput if x !=0]
		target = [x for x in target if x !=0]


	np.savetxt("dataset.csv", dataset, delimiter = ',')
	np.savetxt("input.csv", myInput, delimiter = ',')
	np.savetxt("target.csv", target, delimiter = ',')

def q2(barcodes, groundTruth, landmarkGroundtrush, odometry, measurement, removeOutlier = False):
	# ##Pull all the relevent data in to an array
	# dataset = createSensorNoiseDataset(measurement, groundTruth, barcodes, randomize = True)
	
	# ##Create input and target arrays
	# target = makeTargetArray(dataset)
	# myInput = getTrainingData(dataset)

	# ##Find the min/max of each attribute of the sets in order to normalize
	# minMaxDataset = getListMinMax(myInput)
	# # targetMinMax = getListMinMax(target)
	
	# # ##Normalize
	# # targetNorm = normalize(target, targetMinMax)
	# # myInputNorm = normalize(myInput, minMaxDataset)
	# if(removeOutlier):
	# 	count = 0
	# 	for i in range(len(myInput)):
	# 		if(target[i][1] < minMaxDataset[1][0]):
	# 			dataset[i] = 0
	# 			target[i] = 0
	# 			myInput[i] = 0
	# 			count = count+1
	# 		elif(target[i][1] > minMaxDataset[1][1]):
	# 			dataset[i] = 0
	# 			target[i] = 0
	# 			myInput[i] = 0
	# 			count = count+1
	# 	print "Outlier Count: ", count
	# 	print "New Dataset Size w/o Outliers: ", len(myInput)
	# 	dataset = [x for x in dataset if x !=0]
	# 	myInput = [x for x in myInput if x !=0]
	# 	target = [x for x in target if x !=0]


	# [landmarkX, landmarkY] = getXYRangeLocations(dataset, 15)
	# [landmarkXgoal, landmarkYgoal] = getXYRangeLocations(dataset, 15, target)

 # 	##===Plotting Details===##
	# fig = plt.figure()
	# ax = fig.add_subplot(111)
	# ax.plot(obstacle1.x, obstacle1.y, 'ro', obstacle2.x, obstacle2.y, 'ro', obstacle3.x, obstacle3.y, 'ro', landmarkX, landmarkY, 'go', landmarkXgoal, landmarkYgoal, 'yo')
	# # ax.legend([obstacle_line, obstacle_cell], ['True Obstacle', 'Obstacle Cell'])
	# ax.set_xlabel('X Position (meters)')
	# ax.set_ylabel('Y Position (meters)')
	# ax.set_title("Sensor Reading")

	# plt.show()

	##Run NN on simple sine function
	myLen = 20
	x = np.linspace(0, 1,myLen)
	y = [i*i for i in x]
	# print y

	myInput = [[float(i)] for i in x]
	myTarget = [[float(i)] for i in y]
	
	myInputMinMax = getListMinMax(myInput)
	myTargetMinMax = getListMinMax(myTarget)

	# totalminMaxSet = []

	# for i in range(len(myInputMinMax)):
	# 	totalminMax  = []
	# 	if(myInputMinMax[i][0] < myTargetMinMax[i][0]):
	# 		totalminMax.append(myInputMinMax[i][0])
	# 	else:
	# 		totalminMax.append(myTargetMinMax[i][0])
	# 	if(myInputMinMax[i][1] < myTargetMinMax[i][1]):
	# 		totalminMax.append(myTargetMinMax[i][1])
	# 	else:
	# 		totalminMax.append(myInputMinMax[i][1])
	# 	totalminMaxSet.append(totalminMax)


	# myInputNorm = normalize(myInput,myInputMinMax)
	# myTargetNorm = normalize(myTarget, myTargetMinMax)

	# print myInputNorm
	# print myTargetNorm

	print myInput
	print myTarget

	net = NN([1, 10,5, 1])
	error = net.trainBP(myInput, myTarget, targetSSE= .01, lr = 1.0, maxIter = 10000)

	out = []

	for inputVal in myInput:
		out.append(net.computeOutput(inputVal))
	print out

	# out = rescale(out, myTargetMinMax)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(myInput, myTarget, 'ro', myInput, out, 'go')
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_title("X-Squared Learning")

	plt.show()
	sys.exit()

def q3(barcodes, groundTruth, landmarkGroundtrush, odometry, measurement):
	dataset = createSensorNoiseDataset(measurement, groundTruth, barcodes, randomize = True)
	# dataset = createDRDataset(odometry, groundTruth, randomize = True)
	target = makeTargetArray(dataset)
	# target = makeDeadTargetArray(dataset)
	myInput = getTrainingData(dataset)
	# myInput = getDeadTrainingData(dataset)
	minMaxDataset = getListMinMax(myInput)
	print minMaxDataset
	# print minMaxDataset


	count = 0
	for i in range(len(myInput)):
		if(target[i][1] < minMaxDataset[1][0]):
			dataset[i] = 0
			target[i] = 0
			myInput[i] = 0
			count = count+1
		elif(target[i][1] > minMaxDataset[1][1]):
			dataset[i] = 0
			target[i] = 0
			myInput[i] = 0
			count = count+1
	print "Outlier Count: ", count
	print "New Dataset Size w/o Outliers: ", len(myInput)
	dataset = [x for x in dataset if x !=0]
	myInput = [x for x in myInput if x !=0]
	target = [x for x in target if x !=0]

	print len(myInput)
	# ax.set_xlabel('X Position (meters)')
	# ax.set_ylabel('Y Position (meters)')

	# plt.show()


	targetMinMax = getListMinMax(target)
	print targetMinMax

	totalminMaxSet = minMaxDataset
	totalminMaxSet = []
	for i in range(len(minMaxDataset)):
		totalminMax  = []
		if(minMaxDataset[i][0] < targetMinMax[i][0]):
			totalminMax.append(minMaxDataset[i][0])
		else:
			totalminMax.append(targetMinMax[i][0])
		if(minMaxDataset[i][1] < targetMinMax[i][1]):
			totalminMax.append(targetMinMax[i][1])
		else:
			totalminMax.append(minMaxDataset[i][1])
		totalminMaxSet.append(totalminMax)

	targetNorm = normalize(target, totalminMaxSet)
	myInputNorm = normalize(myInput, totalminMaxSet)
	minMaxDatasetNorm = getListMinMax(myInputNorm)

	# print targetNorm

	# myInput = np.linspace(0,1,20)
	# target = myInput * 4.0

	# targetNormalize = target/4.0


	# myInput2 = myInput.reshape(len(myInput), 1)
	# target2 = targetNormalize.reshape(len(targetNormalize),1)


	# print myInput2
	# print target2

	

	# error1 = 0
	# error2 = 0
	# ##10fold cross validation
	# crossfoldNum = 3
	# inc = int(len(myInputNorm)/crossfoldNum)
	# outNet = NeuralNetwork([2,3,2])
	# for i in range(crossfoldNum):
	# 	testRange = myInputNorm[i*inc:(i+1)*inc]
	# 	inputRange = myInputNorm[0:i*inc] + myInputNorm[(i+1)*inc:]

	# 	testTargetRange = targetNorm[i*inc:(i+1)*inc]
	# 	inputTargetRange = targetNorm[0:i*inc] + targetNorm[(i+1)*inc:]
	# 	# net = NeuralNetwork([2,10,2])
	# 	net = NeuralNetwork([2, 10,  2])
	# 	error = net.trainBP(inputRange, inputTargetRange, targetSSE=180, lr = 1.0)
	# 	outNet = net
	# 	out = []

	# 	for testDataPt in testRange:
	# 		out.append(net.computeOutput(testDataPt))

	# # print out



	# 	# rescaleOutput = rescale(out, targetMinMax)

	# 	compError = sse([[row[0], row[1]] for row in testRange], testTargetRange)
	# 	error1 = error1 + compError

	# 	tempError = sse(out, testTargetRange)
	# 	error2 = error2 + tempError
		
	# 	print("==CrossCompleted==")
	# 	print compError, tempError


	# print("====DONE====")
	# print error1
	# print error2

	# print (1-error2/error1)*100

	# out2 = []
	# for datapt in myInput:
	# 	out2.append(outNet.computeOutput(datapt))

	# error3 = sse(myInput, target)
	# error4 = sse(out2, target)

	# print("==General SSE error on actual set==")
	# print error3
	# print error4
	# print (1-error4/error3)*100

	# out3 = []
	# for datapt in myInputNorm:
	# 	out3.append(outNet.computeOutput(datapt))
	# error5 = sse(myInputNorm, targetNorm)
	# error6 = sse(out3, targetNorm)

	# print("==Normalized then Rescaled SSE error on actual set==")
	# print error5
	# print error6
	# print (1-error6/error5)*100

	# print("==VECTOR WISE SSE NORMALIZED==")
	# out4 = []
	# for datapt in myInputNorm:
	# 	out4.append(outNet.computeOutput(datapt))
	# error7 = sseVector(myInputNorm, targetNorm)
	# error8 = sseVector(out3, targetNorm)

	# print("==Normalized then Rescaled SSE error on actual set==")
	# print error7
	# print error8

	# print("==VECTOR WISE SSE NORMALIZED==")
	# out5 = []
	# for datapt in myInput:
	# 	out5.append(outNet.computeOutput(datapt))
	# error9 = sseVector(myInput, target)
	# error10 = sseVector(out5, target)

	# print("==Normalized then Rescaled SSE error on actual set==")
	# print error9
	# print error10

	net = NN([2,10,5, 2])
	error = net.trainBP(myInputNorm, targetNorm, targetSSE=50, lr = 1.0, maxIter = 7000)
	out5 = []
	for datapt in myInputNorm:
		out5.append(net.computeOutput(datapt))

	out5 = rescale(out5, totalminMaxSet)

	for i in range(10, 21):
		[landmarkX, landmarkY] = getXYRangeLocations(dataset, i)
		[landmarkXgoal, landmarkYgoal] = getXYRangeLocations(dataset, i, target)
		[landmarkXlearned, landmarkYlearned] = getXYRangeLocations(dataset, i, out5)


	 	##===Plotting Details===##
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(obstacle1.x, obstacle1.y, 'ro', obstacle2.x, obstacle2.y, 'ro', obstacle3.x, obstacle3.y, 'ro', landmarkX, landmarkY, 'go', landmarkXlearned, landmarkYlearned, 'yo', landmarkXgoal, landmarkYgoal, 'bo')
		# ax.legend([obstacle_line, obstacle_cell], ['True Obstacle', 'Obstacle Cell'])
		ax.set_xlabel('X Position (meters)')
		ax.set_ylabel('Y Position (meters)')
		ax.set_title("Sensor Reading")

		plt.show()


def extest(barcodes, groundTruth, landmarkGroundtrush, odometry, measurement):

	try:
		import neurolab as nl
	except ImportError:
		print "Unable in import neurolab package"
		print "Try <pip install neurolab> to install package"
		return

	# myLen = 20
	# x = np.linspace(-1,1,myLen)
	# y = np.sin(x)

	# myInput = [[float(i)] for i in x]
	# myTarget = [[float(i)] for i in y]
	
	# myInputMinMax = getListMinMax(myInput)
	# myInputNorm = normalize(myInput,myInputMinMax)
	# myInputMinMaxNorm = getListMinMax(myInputNorm)

	# print myInput
	# print myTarget
	# print myInputNorm
	# print myInputMinMax


	# net = nl.net.newff(myInputMinMaxNorm, [5,2])
	# error = net.train(myInputNorm, myTarget,epochs=500, show = 10, goal=.1)
	# out = net.sim(testRange)

	# fig = plt.figure()
	# ax = fig.add_subplot(111)
	# ax.plot(myInput, myTarget, 'ro', myInput, out, 'go')
	# ax.set_xlabel('X')
	# ax.set_ylabel('Y')
	# ax.set_title("Sin Reading")

	# plt.show()
	# print out

	dataset = createSensorNoiseDataset(measurement, groundTruth, barcodes, randomize = True)

	target = makeTargetArray(dataset)

	myInput = getTrainingData(dataset)

	minMaxDataset = getListMinMax(myInput)
	print minMaxDataset


	count = 0
	for i in range(len(myInput)):
		if(target[i][1] < minMaxDataset[1][0]):
			dataset[i] = 0
			target[i] = 0
			myInput[i] = 0
			count = count+1
		elif(target[i][1] > minMaxDataset[1][1]):
			dataset[i] = 0
			target[i] = 0
			myInput[i] = 0
			count = count+1

	dataset = [x for x in dataset if x !=0]
	myInput = [x for x in myInput if x !=0]
	target = [x for x in target if x !=0]
	print "Outlier Count: ", count
	print "New Dataset Size w/o Outliers: ", len(myInput)

	targetMinMax = getListMinMax(target)
	print targetMinMax
	totalminMaxSet = []

	for i in range(len(minMaxDataset)):
		totalminMax  = []
		if(minMaxDataset[i][0] < targetMinMax[i][0]):
			totalminMax.append(minMaxDataset[i][0])
		else:
			totalminMax.append(targetMinMax[i][0])
		if(minMaxDataset[i][1] < targetMinMax[i][1]):
			totalminMax.append(targetMinMax[i][1])
		else:
			totalminMax.append(minMaxDataset[i][1])
		totalminMaxSet.append(totalminMax)

	targetNorm = normalize(target, totalminMaxSet)
	myInputNorm = normalize(myInput, totalminMaxSet)
	minMaxDatasetNorm = getListMinMax(myInputNorm)

	# dataset = createDRDataset(odometry, groundTruth, randomize = True)
	# target = makeDeadTargetArray(dataset)
	# myInput = getDeadTrainingData(dataset)
	# minMaxDataset = getListMinMax(myInput)
	# print minMaxDataset


	# targetMinMax = getListMinMax(target)
	# print targetMinMax

	# totalminMaxSet = minMaxDataset


	# targetNorm = normalize(target, totalminMaxSet[3:6])
	# myInputNorm = normalize(myInput, totalminMaxSet)
	# minMaxDatasetNorm = getListMinMax(myInputNorm)

	

	# error1 = 0
	# error2 = 0
	# crossfoldNum = 10
	# ##10fold cross validation
	# inc = int(len(myInputNorm)/crossfoldNum)
	# outNet = nl.net.newff(minMaxDatasetNorm, [10,2])
	# for i in range(crossfoldNum):
	# 	testRange = myInputNorm[i*inc:(i+1)*inc]
	# 	inputRange = myInputNorm[0:i*inc] + myInputNorm[(i+1)*inc:]

	# 	testTargetRange = targetNorm[i*inc:(i+1)*inc]
	# 	inputTargetRange = targetNorm[0:i*inc] + targetNorm[(i+1)*inc:]

	# 	net = nl.net.newff(minMaxDatasetNorm, [10,2])
	# 	error = net.train(inputRange, inputTargetRange,epochs=500, show = 10, goal=95.0)
	# 	outNet = net
	# 	out = net.sim(testRange)
	# # print out

	# 	rescaleOutput = rescale(out, targetMinMax)

	# 	compError = sse([[row[0], row[1]] for row in testRange], testTargetRange)
	# 	print compError
	# 	error1 = error1 + compError
	# 	tempError = sse(out, testTargetRange)
	# 	print tempError
	# 	error2 = error2 + tempError

	# # error1 = sse([[row[0], row[1]] for row in myInput], target)
	# print("==SUM of NN sses==")
	# print error1
	# print error2
	# print (1-error2/error1)*100
	
	# out2 = net.sim(myInput)
	# # print out
	# print("==sse of NN on all input data==")
	# error3 = sse(myInput, target)
	# error4 = sse(out2, target)
	# print error3
	# print error4
	# print (1-error4/error3)*100

	net = nl.net.newff(totalminMaxSet, [10,5,3,2])
	net.trainf = nl.train.train_bfgs
	error = net.train(myInputNorm, targetNorm,epochs=300, show = 10, goal=20.0)

	out2 = net.sim(myInputNorm)
	output =  out2.tolist()
	output = rescale(output, totalminMaxSet)
	inputRescaled = rescale(myInputNorm, totalminMaxSet)
	# output = [[i[0],-i[1]] for i in output]
	print dataset[0:10]
	print myInput[0:10]
	print inputRescaled[0:10]
	print target[0:10]
	print output[0:10]

	for i in range(5, 21):
		[landmarkX, landmarkY] = getXYRangeLocations(dataset, i, myInput)
		[landmarkXgoal, landmarkYgoal] = getXYRangeLocations(dataset, i, target)
		[landmarkXlearned, landmarkYlearned] = getXYRangeLocations(dataset, i, output)


	 	##===Plotting Details===##
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(obstacle1.x, obstacle1.y, 'ro', obstacle2.x, obstacle2.y, 'ro', obstacle3.x, obstacle3.y, 'ro', landmarkX, landmarkY, 'go', landmarkXlearned, landmarkYlearned, 'yo', landmarkXgoal, landmarkYgoal, 'bo')
		# ax.legend([obstacle_line, obstacle_cell], ['True Obstacle', 'Obstacle Cell'])
		ax.set_xlabel('X Position (meters)')
		ax.set_ylabel('Y Position (meters)')
		ax.set_title("Sensor Reading")

		plt.show()


	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot([i[0] for i in myInput], [i[0] for i in output], 'ro')
	# ax.legend([obstacle_line, obstacle_cell], ['True Obstacle', 'Obstacle Cell'])
	ax.set_xlabel('X Position (meters)')
	ax.set_ylabel('Y Position (meters)')
	ax.set_title("Sensor Reading")

	plt.show()

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot([i[0] for i in myInput], [i[0] for i in target], 'ro')
	# ax.legend([obstacle_line, obstacle_cell], ['True Obstacle', 'Obstacle Cell'])
	ax.set_xlabel('X Position (meters)')
	ax.set_ylabel('Y Position (meters)')
	ax.set_title("Sensor Reading")

	plt.show()


	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot([i[1] for i in myInput], [i[1] for i in output], 'ro')
	# ax.legend([obstacle_line, obstacle_cell], ['True Obstacle', 'Obstacle Cell'])
	ax.set_xlabel('X Position (meters)')
	ax.set_ylabel('Y Position (meters)')
	ax.set_title("Sensor Reading")

	plt.show()

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot([i[1] for i in myInput], [i[1] for i in target], 'ro')
	# ax.legend([obstacle_line, obstacle_cell], ['True Obstacle', 'Obstacle Cell'])
	ax.set_xlabel('X Position (meters)')
	ax.set_ylabel('Y Position (meters)')
	ax.set_title("Sensor Reading")

	plt.show()

	# Create train samples
	x = np.linspace(-7, 7, 20)
	y = np.sin(x) * 0.5

	size = len(x)

	inp = x.reshape(size,1)
	tar = y.reshape(size,1)

	# Create network with 2 layers and random initialized
	net = nl.net.newff([[-7, 7]],[10,5,3,1])
	net.trainf = nl.train.train_bfgs
	# Train network
	error = net.train(inp, tar, epochs=500, show=100, goal=0.02)

	# Simulate network
	out = net.sim(inp)

	# Plot result
	import pylab as pl
	pl.subplot(211)
	pl.plot(error)
	pl.xlabel('Epoch number')
	pl.ylabel('error (default SSE)')

	x2 = np.linspace(-6.0,6.0,150)
	y2 = net.sim(x2.reshape(x2.size,1)).reshape(x2.size)

	y3 = out.reshape(size)

	pl.subplot(212)
	pl.plot(x2, y2, '-',x , y, '.', x, y3, 'p')
	pl.legend(['train target', 'net output'])
	pl.show()

def main():

	##Load the files, else return error
	print("Trying to load data files...")

	try:
		barcodes = loadFileToLists("ds1_Barcodes.dat")
		groundTruth = loadFileToLists("ds1_Groundtruth.dat")
		landmarkGroundtrush = loadFileToLists("ds1_Landmark_Groundtruth.dat")
		odometry = loadFileToLists("ds1_Odometry.dat")
		measurement = loadFileToLists("ds1_Measurement.dat")

		print("Data files loaded.")
	except:
		print("Error loading data files, make sure they are in the same directory as this file")
		time.sleep(3.0)
		sys.exit()

	print("")

	##Print out types of commands and wait for input

	print("The following commands create graphical outputs for the following questions:")
	print('"1" -- Outputs graph for question 1')
	print('"2" -- Outputs graph for question 2')
	print('"3" -- Outputs graph for question 3')
	print('"extest" -- Outputs Results from NeuralLab NN')
	print("")
	print('"exit" or "quit" -- to exit the program')

	while(True):

		command = raw_input("Command: ")

		if(command == '1'):
			q1(barcodes, groundTruth, landmarkGroundtrush, odometry, measurement)
		elif(command == '2'):
			q2(barcodes, groundTruth, landmarkGroundtrush, odometry, measurement)
		elif(command == '3'):
			q3(barcodes, groundTruth, landmarkGroundtrush, odometry, measurement)
		elif(command == 'extest'):
			extest(barcodes, groundTruth, landmarkGroundtrush, odometry, measurement)
		elif(command == 'exit' or command == 'quit'):
			sys.exit()





if __name__ == '__main__':

    main()