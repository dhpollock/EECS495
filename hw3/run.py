#!/usr/bin/env python

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
from nn import * 
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import time
from sklearn.externals import joblib

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
	print("Dataset Generated!")

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
	error = net.trainBP(myInput, myTarget, targetSSE= .01, lr = 1.0, maxIter = 7000, show=1000)

	## Compute the Output
	out = []
	for inputVal in myInput:
		out.append(net.computeOutput(inputVal))

	##==Plotting Details==##
	fig = plt.figure()
	ax = fig.add_subplot(111)
	red_dot, green_dot = ax.plot(myInput, myTarget, 'ro', myInput, out, 'go')
	ax.legend([red_dot, green_dot], ['True X-Squared', 'NN Approxmination'])
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_title("X-Squared Learning")

	plt.show()

def q3(barcodes, groundTruth, landmarkGroundtrush, odometry, measurement, removeOutlier = True):

	masterDataset = np.load('alexs_dataset.pkl_01.npy')

	# for i in range(3,18):
	# 	print i
	# 	print masterDataset[:,[0,1,2, i]]

	# dataset = createSensorNoiseDataset(measurement, groundTruth, barcodes, randomize = True)
	# target = makeTargetArray(dataset)
	# myInput = getTrainingData(dataset)
	# minMaxDataset = getListMinMax(myInput)

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
	# 	dataset = [x for x in dataset if x !=0]
	# 	myInput = [x for x in myInput if x !=0]
	# 	target = [x for x in target if x !=0]
	# 	print "Outlier Count: ", count
	# 	print "New Dataset Size w/o Outliers: ", len(myInput)


	## Normalize the Input and Target Datasets by the same factor

	np.random.shuffle(masterDataset)

	myInput = masterDataset[:,[0,1,2]]
	myInput = myInput.tolist()

	minMaxDataset = getListMinMax(myInput)
	targetMinMax = [[-1,1]]

	# totalminMaxSet = minMaxDataset
	# totalminMaxSet = []
	# for i in range(len(minMaxDataset)):
	# 	totalminMax  = []
	# 	if(minMaxDataset[i][0] < targetMinMax[i][0]):
	# 		totalminMax.append(minMaxDataset[i][0])
	# 	else:
	# 		totalminMax.append(targetMinMax[i][0])
	# 	if(minMaxDataset[i][1] < targetMinMax[i][1]):
	# 		totalminMax.append(targetMinMax[i][1])
	# 	else:
	# 		totalminMax.append(minMaxDataset[i][1])
	# 	totalminMaxSet.append(totalminMax)

	# targetNorm = normalize(target, totalminMaxSet)
	myInputNorm = normalize(myInput, minMaxDataset)
	minMaxDatasetNorm = getListMinMax(myInputNorm)

	target = masterDataset[:,[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]]
	# target = masterDataset[:,[3]]
	target[target<0] = 0
	target = target.tolist()



##===CODE FROM RUNNING CROSSFOLD===##

	error = []

	myTargetSSE = 0

	crossfoldNum = 10
	inc = int(len(myInputNorm)/crossfoldNum)
	outNet = NN([3,15,15])
	for i in range(crossfoldNum):
		print"Computing CrossFold ", i+1
		testRange = myInputNorm[i*inc:(i+1)*inc]
		inputRange = myInput[0:i*inc] + myInput[(i+1)*inc:]

		testTargetRange = target[i*inc:(i+1)*inc]
		inputTargetRange = target[0:i*inc] + target[(i+1)*inc:]
		net = NN([3,10,30,20,15])
		
		tempError = net.trainBP(inputRange, inputTargetRange, targetSSE=100, lr = .10, maxIter = 1000, show = 10)
		outNet = net
		out = []
		for datapt in testRange:
			out.append(outNet.computeOutput(datapt))

		negCount = 0
		posCount = 0
		negCorrectCount = 0
		posCorrectCount = 0
		for i in range(len(out)):
			for j in range(len(out[i])):
				if(target[i][j] >0):
					posCount += 1
				else:
					negCount +=1

				if(out[i][j] >= 0.1):
					posCorrectCount += 1
				else:
					negCorrectCount += 1
		print "Positive Correct: ", posCorrectCount
		print posCorrectCount/(posCount+0.000001)
		print "Pos Count", posCount
		print "Negtive Correct: ", negCorrectCount
		print  negCorrectCount/(negCount+0.000001)
		print "Neg Count", negCount

 		error.append([posCorrectCount/(posCount+0.000001), negCorrectCount/(negCount+0.000001)])

	print "==Cross Fold Error=="
	print error



	# ##===CODE FROM RUNNING NON CROSSFOLD===##


	# net = NN([3,15, 15])
	# error = net.trainBP(myInputNorm, target, targetSSE=1.0, lr = 1.0, maxIter = 300, show = 10)
	# out = []
	# for datapt in myInputNorm:
	# 	out.append(net.computeOutput(datapt))
	# print out

	negCount = 0
	posCount = 0
	negCorrectCount = 0
	posCorrectCount = 0
	for i in range(len(out)):
		for j in range(len(out[i])):
			if(target[i][j] >0):
				posCount += 1
			else:
				negCount +=1

			if(out[i][j] >= 0.1):
				posCorrectCount += 1
			else:
				negCorrectCount += 1
	print "Positive Correct: ", posCorrectCount
	print posCorrectCount/(posCount+0.000001)
	print "Pos Count", posCount
	print "Negtive Correct: ", negCorrectCount
	print negCorrectCount/(negCount+0.000001)
	print "Neg Count", negCount


def extest(barcodes, groundTruth, landmarkGroundtrush, odometry, measurement, removeOutlier = True):

	try:
		import neurolab as nl
	except ImportError:
		print "Unable to import neurolab package"
		print "Try <pip install neurolab> to install package or <easy_install neurolab>"
		return

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


##===CODE FROM RUNNING CROSSFOLD===##

	# error = []

	# myTargetSSE = 0
	# if removeOutlier:
	# 	myTargetSSE = 420
	# else:
	# 	myTargetSSE = 245

	# crossfoldNum = 10
	# inc = int(len(myInputNorm)/crossfoldNum)
	# outNet = NN([2,3,2])
	# for i in range(crossfoldNum):
	# 	print"Computing CrossFold ", i+1
	# 	testRange = myInputNorm[i*inc:(i+1)*inc]
	# 	inputRange = myInputNorm[0:i*inc] + myInputNorm[(i+1)*inc:]

	# 	testTargetRange = targetNorm[i*inc:(i+1)*inc]
	# 	inputTargetRange = targetNorm[0:i*inc] + targetNorm[(i+1)*inc:]
	# 	net = NN([2,10,2])
		
	# 	tempError = net.trainBP(inputRange, inputTargetRange, targetSSE=myTargetSSE, lr = 1.0, maxIter = 7000, show = 10)
	# 	outNet = net
	# 	out = []
	# 	for datapt in testRange:
	# 		out.append(outNet.computeOutput(datapt))

	# 	out = rescale(out, totalminMaxSet)
	# 	targetRescale = rescale(testTargetRange, totalminMaxSet)
 # 		error.append(sse(out, targetRescale))

	# print "==Cross Fold Validation Completed:=="
	# print"Average SSE Error on Validation Sets:", np.mean(error), "Std Dev: ", np.std(error)
	# print"\n"



	# ##===CODE FROM RUNNING NON CROSSFOLD===##

	myTargetSSE = 0
	if removeOutlier:
		myTargetSSE = 20
	else:
		myTargetSSE = 105

	net = nl.net.newff(totalminMaxSet, [10,2])
	net.trainf = nl.train.train_bfgs
	error = net.train(myInputNorm, targetNorm,epochs=300, show = 10, goal=myTargetSSE)

	out = net.sim(myInputNorm)
	out =  out.tolist()
	output = rescale(out, totalminMaxSet)
	inputRescaled = rescale(myInputNorm, totalminMaxSet)


	for i in range(6, 21):
		[landmarkX, landmarkY] = getXYRangeLocations(dataset, i, myInput)
		[landmarkXgoal, landmarkYgoal] = getXYRangeLocations(dataset, i, target)
		[landmarkXlearned, landmarkYlearned] = getXYRangeLocations(dataset, i, output)


	 	##===Plotting Details===##
		fig = plt.figure()
		ax = fig.add_subplot(111)
		red_dot,red_dot,red_dot,green_dot,yellow_dot, blue_dot = ax.plot(obstacle1.x, obstacle1.y, 'ro', obstacle2.x, obstacle2.y, 'ro', obstacle3.x, obstacle3.y, 'ro', landmarkX, landmarkY, 'go', landmarkXlearned, landmarkYlearned, 'yo', landmarkXgoal, landmarkYgoal, 'bo')
		ax.legend([red_dot, green_dot, yellow_dot, blue_dot], ['True Landmarks', 'Sensor Estimates of Selected Landmark','NN of Selected Landmark', 'Selected Landmark'])
		ax.set_xlabel('X Position (meters)')
		ax.set_ylabel('Y Position (meters)')
		ax.set_title("Sensor Reading")

		plt.show()

	print("==SSE For Actual Measuments vs Target Values==")
	results = sseVector(myInput, target)
	print "Range SSE: ", results[0], "Bearing SSE: ", results[1]
	print("\n")

	print("==SSE for NN Adjusted Measurements vs Target Values==")
	results = sseVector(output, target)
	print "Range SSE: ", results[0], "Bearing SSE: ", results[1]
	print("\n")
	
	print("==Average Error and Std Deviation For Actual Measuments vs Target Values==")
	results = meanNstddevVector(myInput, target)
	print "Range Mean: ", results[0][0],"  Range Std Dev: ", results[0][1]
	print "Bearing Mean: ", results[1][0],"  Bearing Std Dev: ", results[1][1]
	print("\n")

	print("==Average Error and Std Deviation for NN Adjusted Measurements vs Target Values==")
	results = meanNstddevVector(output, target)
	print "Range Mean: ", results[0][0],"  Range Std Dev: ", results[0][1]
	print "Bearing Mean: ", results[1][0],"  Bearing Std Dev: ", results[1][1]
	print("\n")
	dataset = createSensorNoiseDataset(measurement, groundTruth, barcodes, randomize = True)
	target = makeTargetArray(dataset)
	myInput = getTrainingData(dataset)
	minMaxDataset = getListMinMax(myInput)



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