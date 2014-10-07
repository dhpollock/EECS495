##Author: D. Harmon Pollock
##Date: Oct '14
##Class: EECS 495 ML & AI for Robotics
##Professor: Brenna Argall
##Assignment: hw0
##File Name: run.py

##Note: In order to run, the following datafiles must be located in the same folder as this file:
##-->ds1_Barcodes.dat
##-->ds1_Groundtruth.dat
##-->ds1_Landmark_Groundtruth.dat
##-->ds1_Odometry.dat
##-->ds1_Measurement.dat


import sys
from helperFunctions import *
from models import *
import simulations
import matplotlib.pyplot as plt
import numpy as np

def q2(barcodes, groundTruth, landmarkGroundtrush, odometry, measurement):

	#Execute Q2=================================================================================
	testCommands = [[.5,0,1.0],[0, -1.0/(2.0*np.pi), 1.0],[.5, 0.0 , 1.0],[0.0, 1.0/(2.0*np.pi), 1.0],[.5, 0.0, 1.0]]
	# testCommands = [[0, 1.0/8.0*np.pi, 1.0],[.5, 0.0 , 1.0],[0, 1.0/8.0*np.pi, 1.0],[.5, 0.0 , 1.0],[0, 1.0/8.0*np.pi, 1.0],[.5, 0.0 , 1.0],[0, 1.0/8.0*np.pi, 1.0],[.5, 0.0 , 1.0],[0, 1.0/8.0*np.pi, 1.0],[.5, 0.0 , 1.0],[0, 1.0/8.0*np.pi, 1.0],[.5, 0.0 , 1.0],[0, 1.0/8.0*np.pi, 1.0],[.5, 0.0 , 1.0],[0, 1.0/8.0*np.pi, 1.0],[.5, 0.0 , 1.0],[0, 1.0/8.0*np.pi, 1.0],[.5, 0.0 , 1.0],[0, 1.0/8.0*np.pi, 1.0],[.5, 0.0 , 1.0],[0, 1.0/8.0*np.pi, 1.0],[.5, 0.0 , 1.0],[0, 1.0/8.0*np.pi, 1.0],[.5, 0.0 , 1.0],[0, 1.0/8.0*np.pi, 1.0],[.5, 0.0 , 1.0],[0, 1.0/8.0*np.pi, 1.0],[.5, 0.0 , 1.0],[0, 1.0/8.0*np.pi, 1.0],[.5, 0.0 , 1.0],[0, 1.0/8.0*np.pi, 1.0],[.5, 0.0 , 1.0]]
	# testCommands = [[1.0,1.0/8.0*np.pi,1],[1.0,1.0/8.0*np.pi,1],[1.0,1.0/8.0*np.pi,1],[1.0,1.0/8.0*np.pi,1],[1.0,1.0/8.0*np.pi,1],[1.0,1.0/8.0*np.pi,1],[1.0,1.0/8.0*np.pi,1],[1.0,1.0/8.0*np.pi,1],[1.0,1.0/8.0*np.pi,1]]
	vector = simulations.runControlSimulator([0,0,0], testCommands, False)


def q3(barcodes, groundTruth, landmarkGroundtrush, odometry, measurement):

	#Execute Q3=================================================================================

	#parse commands into format useable with runControlSimulator
	odoCommands = []
	for i in range(len(odometry)-1):
		odoCommands.append([float(odometry[i][1]), float(odometry[i][2]), float(odometry[i+1][0])-float(odometry[i][0])])
	#Run the simulator and get resulting plot
	vector = simulations.runControlSimulator([0.98038490,-4.99232180,1.44849633], odoCommands, True)

	#Plot Ground Truth for Comparison
	plotX = []
	plotY = []
	for line in groundTruth:
		plotX.append(float(line[1]))
		plotY.append(float(line[2]))

	plt.plot(plotX, plotY, 'b-', obstacle1.x, obstacle1.y, 'r-o', obstacle2.x, obstacle2.y, 'r-o', obstacle3.x, obstacle3.y, 'r-o')

	plt.xlabel('X Position (meters)')
	plt.ylabel('Y Position (meters)')
	plt.title("Ground Truth Position")
	plt.show()	

def q4(barcodes, groundTruth, landmarkGroundtrush, odometry, measurement):

	#Execute Q4=================================================================================
	myMeasurements = parseMeasurements(measurement, barcodes)
	print(myMeasurements[0].landmarkMeasurements[0])
	# print(myMeasurements[1].landmarkMeasurements)
	print(simulations.expectedMeasurement([0.98038490,-4.99232180,1.44849633], myMeasurements[0].landmarkMeasurements[0][0]))

def q5(barcodes, groundTruth, landmarkGroundtrush, odometry, measurement):

	myPF = simulations.ParticleFilter([0.98038490,-4.99232180,1.44849633], [randomGaussianPointAndProb, randomGaussianPointAndProb, randomGaussianPointAndProb], [0.1, 0.1, 0.05], 500)
	myMeasurements = parseMeasurements(measurement, barcodes)
	myCommands = []
	for i in range(len(odometry)-1):
		myCommands.append([float(odometry[i][1]), float(odometry[i][2]), float(odometry[i][0]), float(odometry[i+1][0])-float(odometry[i][0])])


	plotX = []
	plotY = []
	for command in myCommands[0:1000]:
		timeMin = command[2]
		timeMax = command[2]+command[3]
		[myMeasurements, curMeasurements] = getMeasurements(myMeasurements, timeMin, timeMax)
		# print(command)
		myPF.updateStep(command, curMeasurements)
		myPF.normalizeWeights()
		myPF.resampleStep()
		[tempX, tempY, tempRot] = myPF.getMean()
		# print("=====new step=====")
		# print(command)
		# if(curMeasurements != []):
		# 	print(curMeasurements[0].landmarkMeasurements)
		# print(tempX, tempY)
		plotX.append(tempX)
		plotY.append(tempY)


	plt.plot(plotX, plotY, 'b-', obstacle1.x, obstacle1.y, 'r-o', obstacle2.x, obstacle2.y, 'r-o', obstacle3.x, obstacle3.y, 'r-o')

	plt.xlabel('X Position (meters)')
	plt.ylabel('Y Position (meters)')
	plt.title("Mean PF Position Estimate")
	plt.show()



def main():
	barcodes = loadFileToLists("ds1_Barcodes.dat")
	groundTruth = loadFileToLists("ds1_Groundtruth.dat")
	landmarkGroundtrush = loadFileToLists("ds1_Landmark_Groundtruth.dat")
	odometry = loadFileToLists("ds1_Odometry.dat")
	measurement = loadFileToLists("ds1_Measurement.dat")

	
	while(True):
		command = raw_input("Command: ")

		if(command == '2'):
			q2(barcodes, groundTruth, landmarkGroundtrush, odometry, measurement)
		elif(command == '3'):
			q3(barcodes, groundTruth, landmarkGroundtrush, odometry, measurement)
		elif(command == '4'):
			q4(barcodes, groundTruth, landmarkGroundtrush, odometry, measurement)
		elif(command == '5'):
			q5(barcodes, groundTruth, landmarkGroundtrush, odometry, measurement)
		elif(command == 'exit'):
			sys.exit()





if __name__ == '__main__':

    main()