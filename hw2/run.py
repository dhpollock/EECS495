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
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import time
import neurolab as nl

## The Question 9 function is where most of the heavy lifting is done in terms of creating
## a number of different scenarios by easily commenting out differnt section of this code
## more comments to follow...
def q1(barcodes, groundTruth, landmarkGroundtrush, odometry, measurement):

	dataset = createSensorNoiseDataset(measurement, groundTruth, barcodes, randomize = True)

	target = makeTargetArray(dataset)

	train = getTrainingData(dataset)

	minMaxDataset = getListMinMax(train)
	print minMaxDataset

	target = normalize(target, minMaxDataset)
	train = normalize(train, minMaxDataset)
	print np.array(target)
	print np.array(train)
	net = nl.net.newff(minMaxDataset, [10,1])

	error = net.train(np.array(train), np.array(target),epochs=500, show = 10, goal=.05)
	print error
	out = net.sim(np.array(train))
	print out

	plt.plot(error)
	plt.show()

	# Create train samples
	# x = np.linspace(-7, 7, 20)
	# y = np.sin(x) * 0.5

	# # print x

	# size = len(x)

	# inp = x.reshape(size,1)
	# tar = y.reshape(size,1)

	# print inp
	# print tar

	# # Create network with 2 layers and random initialized
	# net = nl.net.newff([[-7, 7]],[5, 1])

	# # Train network
	# error = net.train(inp, tar, epochs=500, show=100, goal=0.02)

	# # Simulate network
	# out = net.sim(inp)

	# # Plot result
	# plt.subplot(211)
	# plt.plot(error)
	# plt.xlabel('Epoch number')
	# plt.ylabel('error (default SSE)')

	# x2 = np.linspace(-6.0,6.0,150)
	# y2 = net.sim(x2.reshape(x2.size,1)).reshape(x2.size)

	# y3 = out.reshape(size)

	# plt.subplot(212)
	# plt.plot(x2, y2, '-',x , y, '.', x, y3, 'p')
	# plt.legend(['train target', 'net output'])
	# plt.show()


	# ##parse the measurements in to measurement step objects
	# myMeasurements = parseMeasurements(measurement, barcodes)

	# ##parse the commands into p.f. format
	# myCommands = []
	# for i in range(len(odometry)-1):
	# 	myCommands.append([float(odometry[i][1]), float(odometry[i][2]), float(odometry[i][0]), float(odometry[i+1][0])-float(odometry[i][0])])

	# myGroundtruth = []
	# for i in range(len(groundTruth)):
	# 	myGroundtruth.append([groundTruth[i][0], groundTruth[i][1], groundTruth[i][2], groundTruth[i][3]])


	# 	timeMin = command[2] + .75*command[3]
	# 	timeMax = command[2] + 1.75*command[3]
		
	# 	[myMeasurements, curMeasurements] = getMeasurements(myMeasurements, timeMin, timeMax)

		
	# ##plotting legend etc...
	# blue_line = mlines.Line2D([],[], color = 'blue')
	# green_line = mlines.Line2D([],[], color = 'green')
	# yellow_line = mlines.Line2D([],[], color = 'yellow', linestyle = '--')
	# obstacle_line = mlines.Line2D([],[], color = 'red', marker = 'o', markersize = 5)

	# plt.plot(plotX, plotY, 'b-', plotXGT, plotYGT, 'g-', plotOdoX, plotOdoY, 'y--', obstacle1.x, obstacle1.y, 'r-o', obstacle2.x, obstacle2.y, 'r-o', obstacle3.x, obstacle3.y, 'r-o')

	# plt.xlabel('X Position (meters)')
	# plt.ylabel('Y Position (meters)')
	# plt.legend([blue_line, green_line, yellow_line, obstacle_line], ['P.F. Path', 'Ground Truth', 'Controller Estimate', 'Obstacles'])
	# plt.title("Mean PF Position Estimate")
	# plt.show()


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
	print('"2" -- Outputs graph for question 2')
	print('"3" -- Outputs graph for question 3')
	print('"8" -- Outputs graph for question 8')
	print('"9" -- Outputs graph for question 9')
	print("")
	print('"exit" or "quit" -- to exit the program')

	while(True):

		command = raw_input("Command: ")

		if(command == '1'):
			q1(barcodes, groundTruth, landmarkGroundtrush, odometry, measurement)
		elif(command == '3'):
			q3(barcodes, groundTruth, landmarkGroundtrush, odometry, measurement)
		elif(command == '8'):
			q8(barcodes, groundTruth, landmarkGroundtrush, odometry, measurement)
		elif(command == '9'):
			q9(barcodes, groundTruth, landmarkGroundtrush, odometry, measurement)
		elif(command == 'exit' or command == 'quit'):
			sys.exit()





if __name__ == '__main__':

    main()