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


	[landmarkX, landmarkY] = getXYRangeLocations(dataset, 7, myInput)
	[landmarkXgoal, landmarkYgoal] = getXYRangeLocations(dataset, 7, target)

 	##===Plotting Details===##
	fig = plt.figure()
	ax = fig.add_subplot(111)
	red_dot,red_dot,red_dot,green_dot,blue_dot = ax.plot(obstacle1.x, obstacle1.y, 'ro', obstacle2.x, obstacle2.y, 'ro', obstacle3.x, obstacle3.y, 'ro', landmarkX, landmarkY, 'go', landmarkXgoal, landmarkYgoal, 'bo')
	ax.legend([red_dot, green_dot, blue_dot], ['True Landmarks', 'Sensor Estimates of Selected Landmark', 'Selected Landmark'])
	ax.set_xlabel('X Position (meters)')
	ax.set_ylabel('Y Position (meters)')
	ax.set_title("Sensor Reading")

	plt.show()

	## Run NN on simple X^2 function
	myLen = 20
	x = np.linspace(0,1,myLen)
	y = [i*i for i in x]

	## Reformat the data to be compatible with NN
	myInput = [[float(i)] for i in x]
	myTarget = [[float(i)] for i in y]
	
	## Create a simple NN
	net = NN([1,5,1])

	## Train the NN!
	error = net.trainBP(myInput, myTarget, targetSSE= .01, lr = 1.0, maxIter = 7000, show=10)

	## Compute the Output
	out = []
	for inputVal in myInput:
		out.append(net.computeOutput(inputVal))
	print out

	##==Plotting Details==##
	fig = plt.figure()
	ax = fig.add_subplot(111)
	red_dot, green_dot = ax.plot(myInput, myTarget, 'ro', myInput, out, 'go')
	ax.legend([red_dot, green_dot], ['True X-Squared', 'NN Approxmination'])
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_title("X-Squared Learning")

	plt.show()

def q3(barcodes, groundTruth, landmarkGroundtrush, odometry, measurement, removeOutlier = False):

	dataset = createSensorNoiseDataset(measurement, groundTruth, barcodes, randomize = True)
	target = makeTargetArray(dataset)
	myInput = getTrainingData(dataset)
	minMaxDataset = getListMinMax(myInput)

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
		dataset = [x for x in dataset if x !=0]
		myInput = [x for x in myInput if x !=0]
		target = [x for x in target if x !=0]
		print "Outlier Count: ", count
		print "New Dataset Size w/o Outliers: ", len(myInput)


	## Normalize the Input and Target Datasets by the same factor
	targetMinMax = getListMinMax(target)

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




	resultsSSEXFold = [0,0]
	resultsMeanXFold = [[0,0],[0,0]]
	##====10fold cross validation====#
	crossfoldNum = 10
	inc = int(len(myInputNorm)/crossfoldNum)
	outNet = NN([2,3,2])
	for i in range(crossfoldNum):
		testRange = myInputNorm[i*inc:(i+1)*inc]
		inputRange = myInputNorm[0:i*inc] + myInputNorm[(i+1)*inc:]

		testTargetRange = targetNorm[i*inc:(i+1)*inc]
		inputTargetRange = targetNorm[0:i*inc] + targetNorm[(i+1)*inc:]
		net = NN([2,10,2])
		net.trainBP(inputRange, inputTargetRange, targetSSE=460.0s, lr = 1.0, maxIter = 7000, show = 10)
		outNet = net
		
		out = []
		for datapt in testRange:
			out.append(net.computeOutput(datapt))
		out = rescale(out, totalminMaxSet)
		
		resultsSSEXFold += sseVector(out, testTargetRange)
		resultsMeanXFold += meanNstddevVector(out, testTargetRange)




	##===CODE FROM RUNNING NON CROSSFOLD===##

	# net = NN([2,10, 2])
	# error = net.trainBP(myInputNorm, targetNorm, targetSSE=460.0, lr = 1.0, maxIter = 7000, show = 10)
	# output = []
	# for datapt in myInputNorm:
	# 	output.append(outNet.computeOutput(datapt))

	# output = rescale(output, totalminMaxSet)

	# for i in range(6, 21):
	# 	[landmarkX, landmarkY] = getXYRangeLocations(dataset, i, myInput)
	# 	[landmarkXgoal, landmarkYgoal] = getXYRangeLocations(dataset, i, target)
	# 	[landmarkXlearned, landmarkYlearned] = getXYRangeLocations(dataset, i, output)


	#  	##===Plotting Details===##
	# 	fig = plt.figure()
	# 	ax = fig.add_subplot(111)
	# 	red_dot,red_dot,red_dot,green_dot,yellow_dot, blue_dot = ax.plot(obstacle1.x, obstacle1.y, 'ro', obstacle2.x, obstacle2.y, 'ro', obstacle3.x, obstacle3.y, 'ro', landmarkX, landmarkY, 'go', landmarkXlearned, landmarkYlearned, 'yo', landmarkXgoal, landmarkYgoal, 'bo')
	# 	ax.legend([red_dot, green_dot, yellow_dot, blue_dot], ['True Landmarks', 'Sensor Estimates of Selected Landmark','NN of Selected Landmark', 'Selected Landmark'])
	# 	ax.set_xlabel('X Position (meters)')
	# 	ax.set_ylabel('Y Position (meters)')
	# 	ax.set_title("Sensor Reading")

	# 	plt.show()

	# print("==SSE For Actual Measuments vs Target Values==")
	# results = sseVector(myInput, target)
	# print "Range SSE: ", results[0], "Bearing SSE: ", results[1]
	# print("\n")

	# print("==SSE for NN Adjusted Measurements vs Target Values==")
	# results = sseVector(output, target)
	# print "Range SSE: ", results[0], "Bearing SSE: ", results[1]
	# print("\n")
	
	# print("==Average Error and Std Deviation For Actual Measuments vs Target Values==")
	# results = meanNstddevVector(myInput, target)
	# print "Range Mean: ", results[0][0],"  Range Std Dev: ", results[0][1]
	# print "Bearing Mean: ", results[1][0],"  Bearing Std Dev: ", results[1][1]
	# print("\n")

	# print("==Average Error and Std Deviation for NN Adjusted Measurements vs Target Values==")
	# results = meanNstddevVector(output, target)
	# print "Range Mean: ", results[0][0],"  Range Std Dev: ", results[0][1]
	# print "Bearing Mean: ", results[1][0],"  Bearing Std Dev: ", results[1][1]
	# print("\n")


def extest(barcodes, groundTruth, landmarkGroundtrush, odometry, measurement):

	try:
		import neurolab as nl
	except ImportError:
		print "Unable in import neurolab package"
		print "Try <pip install neurolab> to install package"
		return

	
	dataset = createSensorNoiseDataset(measurement, groundTruth, barcodes, randomize = True)
	target = makeTargetArray(dataset)
	myInput = getTrainingData(dataset)
	minMaxDataset = getListMinMax(myInput)

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
	error = net.train(myInputNorm, targetNorm,epochs=300, show = 10, goal=10.0)

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
		red_dot,red_dot,red_dot,green_dot,yellow_dot, blue_dot = ax.plot(obstacle1.x, obstacle1.y, 'ro', obstacle2.x, obstacle2.y, 'ro', obstacle3.x, obstacle3.y, 'ro', landmarkX, landmarkY, 'go', landmarkXlearned, landmarkYlearned, 'yo', landmarkXgoal, landmarkYgoal, 'bo')
		ax.legend([obstacle_line, obstacle_cell], ['True Obstacle', 'Obstacle Cell'])
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

	print("SSE For Actual Measuments vs Target Values")
	print sseVector(myInput, target)
	print("Average Error and Std Deviation For Actual Measuments vs Target Values")
	results = meanNstddevVector(myInput, target)
	print "Range Mean: ", results[0][0],"  Range Std Dev: ", results[0][1]
	print "Bearing Mean: ", results[1][0],"  Bearing Std Dev: ", results[1][1]
	
	print("SSE for NN Adjusted Measurements vs Target Values")
	print sseVector(output, target)
	print("Average Error and Std Deviation for NN Adjusted Measurements vs Target Values")
	results = meanNstddevVector(output, target)
	print "Range Mean: ", results[0][0],"  Range Std Dev: ", results[0][1]
	print "Bearing Mean: ", results[1][0],"  Bearing Std Dev: ", results[1][1]
	

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