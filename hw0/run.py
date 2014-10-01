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


def main():
	barcodes = loadFileToLists("ds1_Barcodes.dat")
	groundTruth = loadFileToLists("ds1_Groundtruth.dat")
	landmarkGroundtrush = loadFileToLists("ds1_Landmark_Groundtruth.dat")
	odometry = loadFileToLists("ds1_Odometry.dat")
	measurement = loadFileToLists("ds1_Measurement.dat")

	
	#Execute Q2
	testCommands = [[.5,0,1.0],[0, -1.0/(2.0*np.pi), 1.0],[.5, 0.0 , 1.0],[0.0, 1.0/(2.0*np.pi), 1.0],[.5, 0.0, 1.0]]
	# vector = simulations.runControlSimulator([0,0,0], testCommands)

	odoCommands = []
	for i in range(len(odometry)-1):
		odoCommands.append([float(odometry[i][1]), float(odometry[i][2]), float(odometry[i+1][0])-float(odometry[i][0])])

	vector = simulations.runControlSimulator([0,0,0], odoCommands)

	print vector

	plotX = []
	plotY = []
	for line in groundTruth:
		plotX.append(float(line[1]))
		plotY.append(float(line[2]))

	plt.scatter(plotX, plotY)
	plt.show()


if __name__ == '__main__':

    main()