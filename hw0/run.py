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

##Import all the things
import sys
from helperFunctions import *
from models import *
import simulations
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import time

#====Plots for Question 2 ====#

def q2(barcodes, groundTruth, landmarkGroundtrush, odometry, measurement):

	##Various test commands to check the control simulator
	testCommands = [[.5,0,1.0],[0, -1.0/(2.0*np.pi), 1.0],[.5, 0.0 , 1.0],[0.0, 1.0/(2.0*np.pi), 1.0],[.5, 0.0, 1.0]]
	# testCommands = [[0, 1.0/8.0*np.pi, 1.0],[.5, 0.0 , 1.0],[0, 1.0/8.0*np.pi, 1.0],[.5, 0.0 , 1.0],[0, 1.0/8.0*np.pi, 1.0],[.5, 0.0 , 1.0],[0, 1.0/8.0*np.pi, 1.0],[.5, 0.0 , 1.0],[0, 1.0/8.0*np.pi, 1.0],[.5, 0.0 , 1.0],[0, 1.0/8.0*np.pi, 1.0],[.5, 0.0 , 1.0],[0, 1.0/8.0*np.pi, 1.0],[.5, 0.0 , 1.0],[0, 1.0/8.0*np.pi, 1.0],[.5, 0.0 , 1.0],[0, 1.0/8.0*np.pi, 1.0],[.5, 0.0 , 1.0],[0, 1.0/8.0*np.pi, 1.0],[.5, 0.0 , 1.0],[0, 1.0/8.0*np.pi, 1.0],[.5, 0.0 , 1.0],[0, 1.0/8.0*np.pi, 1.0],[.5, 0.0 , 1.0],[0, 1.0/8.0*np.pi, 1.0],[.5, 0.0 , 1.0],[0, 1.0/8.0*np.pi, 1.0],[.5, 0.0 , 1.0],[0, 1.0/8.0*np.pi, 1.0],[.5, 0.0 , 1.0],[0, 1.0/8.0*np.pi, 1.0],[.5, 0.0 , 1.0]]
	# testCommands = [[1.0,1.0/8.0*np.pi,1],[1.0,1.0/8.0*np.pi,1],[1.0,1.0/8.0*np.pi,1],[1.0,1.0/8.0*np.pi,1],[1.0,1.0/8.0*np.pi,1],[1.0,1.0/8.0*np.pi,1],[1.0,1.0/8.0*np.pi,1],[1.0,1.0/8.0*np.pi,1],[1.0,1.0/8.0*np.pi,1]]
	
	##run the control simulator!
	vector = simulations.runControlSimulator([0,0,0], testCommands, False)


def q3(barcodes, groundTruth, landmarkGroundtrush, odometry, measurement):

	##parse commands into format useable with runControlSimulator
	odoCommands = []
	for i in range(len(odometry)-1):
		odoCommands.append([float(odometry[i][1]), float(odometry[i][2]), float(odometry[i+1][0])-float(odometry[i][0])])
	
	##Run the simulator and get resulting plot
	vector = simulations.runControlSimulator([0.98038490,-4.99232180,1.44849633], odoCommands, True, groundTruth)


def q8(barcodes, groundTruth, landmarkGroundtrush, odometry, measurement):

	##======Part one with simple test vector========##

	##Reformated Test Vector
	testCommands = [[0.0,.5,0],[1.0, 0, -1.0/(2.0*np.pi)],[2.0,.5, 0.0],[3.0, 0.0, 1.0/(2.0*np.pi)],[4.0, .5, 0.0],[5.0, 0.0, 0.0]]
	##Create A simple p.f. with a known starting position and a gaussian around the position
	myPF = simulations.ParticleFilter([0.0,0.0,0.0], [randomGaussianPointAndProb, randomGaussianPointAndProb, randomGaussianPointAndProb], [0.001, 0.001, 0.005], 500)
	myCommands = []
	##parse the commands into p.f. form
	for i in range(len(testCommands)-1):
		myCommands.append([float(testCommands[i][1]), float(testCommands[i][2]), float(testCommands[i][0]), float(testCommands[i+1][0])-float(testCommands[i][0])])

	##track points for plotting
	plotX = []
	plotY = []
	plotOdoX = [0.0]
	plotOdoY = [0.0]
	plotOdoRot = [0.0]

	##Step through the commands using the p.f.!
	for command in myCommands:
		if(command[0] != 0 or command[1] != 0):
			myPF.updateStep(command, [])
			myPF.normalizeWeights()
			myPF.resampleStep()
		[tempX, tempY, tempRot] = myPF.getMeanSmoothed()

		plotX.append(tempX)
		plotY.append(tempY)

		##Also plot the control path
		vector = simulatedControllerEstimate([plotOdoX[len(plotOdoX)-1], plotOdoY[len(plotOdoY)-1], plotOdoRot[len(plotOdoRot)-1]], command[0], command[1], command[3])
		plotOdoX.append(vector[0])
		plotOdoY.append(vector[1])
		plotOdoRot.append(vector[2])


	##Generate the plot
	blue_line = mlines.Line2D([],[], color = 'blue')
	yellow_line = mlines.Line2D([],[], color = 'yellow', linestyle = '--')

	plt.plot(plotX, plotY, 'b-', plotOdoX, plotOdoY, 'y--')
	plt.legend([blue_line, yellow_line], ['P.F. No Measurement Path', 'Controller Estimate'])
	plt.xlabel('X Position (meters)')
	plt.ylabel('Y Position (meters)')
	plt.title("Mean PF Position Estimate With Zero Measurements")
	plt.show()

	##======Part two with odometry commands========##
	##Create a p.f. with the ground truth starting position and a gaussian around the position and 500 particles
	myPF = simulations.ParticleFilter([0.98038490,-4.99232180,1.44849633], [randomGaussianPointAndProb, randomGaussianPointAndProb, randomGaussianPointAndProb], [0.1, 0.1, 0.05], 500)
	
	##parse the commands into p.f. form
	myCommands = []
	for i in range(len(odometry)-1):
		myCommands.append([float(odometry[i][1]), float(odometry[i][2]), float(odometry[i][0]), float(odometry[i+1][0])-float(odometry[i][0])])

	##track points for plotting
	plotX = []
	plotY = []
	plotOdoX = [0.98038490]
	plotOdoY = [-4.99232180]
	plotOdoRot = [1.44849633]

	##Step through the commands using the p.f.!
	for command in myCommands:
		if(command[0] != 0 or command[1] != 0):
			myPF.updateStep(command, [])
			myPF.normalizeWeights() # they are here but will not do much good
			myPF.resampleStep()	#also will not do much good since there are no measurements
		[tempX, tempY, tempRot] = myPF.getMeanSmoothed()

		plotX.append(tempX)
		plotY.append(tempY)

		##keep track of simulated contoller estimate 
		vector = simulatedControllerEstimate([plotOdoX[len(plotOdoX)-1], plotOdoY[len(plotOdoY)-1], plotOdoRot[len(plotOdoRot)-1]], command[0], command[1], command[3])
		plotOdoX.append(vector[0])
		plotOdoY.append(vector[1])
		plotOdoRot.append(vector[2])

	##get the ground truth
	plotXGT = []
	plotYGT = []
	for line in groundTruth:
		plotXGT.append(float(line[1]))
		plotYGT.append(float(line[2]))

	##plot all three lines ontop of the obstacle space
	blue_line = mlines.Line2D([],[], color = 'blue')
	green_line = mlines.Line2D([],[], color = 'green')
	yellow_line = mlines.Line2D([],[], color = 'yellow', linestyle = '--')
	obstacle_line = mlines.Line2D([],[], color = 'red', marker = 'o', markersize = 5)

	plt.plot(plotX, plotY, 'b-', plotXGT, plotYGT, 'g-', plotOdoX, plotOdoY, 'y--', obstacle1.x, obstacle1.y, 'r-o', obstacle2.x, obstacle2.y, 'r-o', obstacle3.x, obstacle3.y, 'r-o')
	plt.legend([blue_line, green_line, yellow_line, obstacle_line], ['P.F. No Measurement Path', 'Ground Truth', 'Controller Estimate', 'Obstacles'])
	plt.xlabel('X Position (meters)')
	plt.ylabel('Y Position (meters)')
	plt.title("Mean PF Position Estimate With Zero Measurements")
	plt.show()



## The Question 9 function is where most of the heavy lifting is done in terms of creating
## a number of different scenarios by easily commenting out differnt section of this code
## more comments to follow...
def q9(barcodes, groundTruth, landmarkGroundtrush, odometry, measurement):

	## ===Create the particle filter instance with various start conditions: === #
	##This P.F starts at the known ground truth location and particle cloud based on a gaussian around it with respective weights.
	myPF = simulations.ParticleFilter([0.98038490,-4.99232180,1.44849633], [randomGaussianPointAndProb, randomGaussianPointAndProb, randomGaussianPointAndProb], [0.1, 0.1, 0.05], 500)

	##This P.F. starts with a unknown location with a random particle cloud and uniform weighting
	# myPF = simulations.ParticleFilter([0.0,0.0,0.0], [uniformProb, uniformProb, uniformProb], [500.0, 500.0, 500.0], 500)


	##parse the measurements in to measurement step objects
	myMeasurements = parseMeasurements(measurement, barcodes)

	##parse the commands into p.f. format
	myCommands = []
	for i in range(len(odometry)-1):
		myCommands.append([float(odometry[i][1]), float(odometry[i][2]), float(odometry[i][0]), float(odometry[i+1][0])-float(odometry[i][0])])


	##plot point tracking
	plotX = []
	plotY = []
	plotOdoX = [0.98038490]
	plotOdoY = [-4.99232180]
	plotOdoRot = [1.44849633]

	##Start the particle filter with commands, the range for which the p.f. runs 
	##can be adjusted here, but commenting out or change the range of myCommands
	for command in myCommands[0:2000]:
		
		##determine the current time step of the commands and find the range of acceptable measurements
		##for comparision
		timeMin = command[2] + .75*command[3]
		timeMax = command[2] + 1.75*command[3]
		#retrieve those measurements and remove them from the myMeasurements list
		[myMeasurements, curMeasurements] = getMeasurements(myMeasurements, timeMin, timeMax)

		##Only update the p.f. if there is a moving command, used for the sack of brevity.
		if(command[0] != 0 or command[1] != 0):
			myPF.updateStep(command, curMeasurements)

			##only normalize the weights of the particle cloud and resample it, 
			## if there is an actual measurement to take into account.
			if(curMeasurements != []):
				myPF.normalizeWeights()
				myPF.resampleStep()

		##==Here the method of state extraction can be specified.==##
		##Just averaging the components seperatly, and smoothed over previous states
		[tempX, tempY, tempRot] = myPF.getMeanSmoothed()
		##Take the max value of a histogram constucted off weights
		# [tempX, tempY, tempRot] = myPF.getHistogramMax()
		##Histogram method smoothed over previous states
		# [tempX, tempY, tempRot] = myPF.getHistogramMaxSmoothed()

		## save the temp state for plotting
		plotX.append(tempX)
		plotY.append(tempY)

		##also track the simulated controller for comparison
		vector = simulatedControllerEstimate([plotOdoX[len(plotOdoX)-1], plotOdoY[len(plotOdoY)-1], plotOdoRot[len(plotOdoRot)-1]], command[0], command[1], command[3])
		plotOdoX.append(vector[0])
		plotOdoY.append(vector[1])
		plotOdoRot.append(vector[2])

	##plot the ground truth as well
	plotXGT = []
	plotYGT = []
	for line in groundTruth[0:25000]:
		plotXGT.append(float(line[1]))
		plotYGT.append(float(line[2]))

	##plotting legend etc...
	blue_line = mlines.Line2D([],[], color = 'blue')
	green_line = mlines.Line2D([],[], color = 'green')
	yellow_line = mlines.Line2D([],[], color = 'yellow', linestyle = '--')
	obstacle_line = mlines.Line2D([],[], color = 'red', marker = 'o', markersize = 5)

	plt.plot(plotX, plotY, 'b-', plotXGT, plotYGT, 'g-', plotOdoX, plotOdoY, 'y--', obstacle1.x, obstacle1.y, 'r-o', obstacle2.x, obstacle2.y, 'r-o', obstacle3.x, obstacle3.y, 'r-o')

	plt.xlabel('X Position (meters)')
	plt.ylabel('Y Position (meters)')
	plt.legend([blue_line, green_line, yellow_line, obstacle_line], ['P.F. Path', 'Ground Truth', 'Controller Estimate', 'Obstacles'])
	plt.title("Mean PF Position Estimate -- R = 10.0")
	plt.show()


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

		if(command == '2'):
			q2(barcodes, groundTruth, landmarkGroundtrush, odometry, measurement)
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