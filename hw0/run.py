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



def main():
	barcodes = loadFileToLists("ds1_Barcodes.dat")
	groundTruth = loadFileToLists("ds1_Groundtruth.dat")
	landmarkGroundtrush = loadFileToLists("ds1_Landmark_Groundtruth.dat")
	odometry = loadFileToLists("ds1_Odometry.dat")
	measurement = loadFileToLists("ds1_Measurement.dat")
	
	print measurement

	



if __name__ == '__main__':

    main()