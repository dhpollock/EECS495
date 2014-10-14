##Author: D. Harmon Pollock
##Date: Oct '14
##Class: EECS 495 ML & AI for Robotics
##Professor: Brenna Argall
##Assignment: hw0
## File Name: simulations.py

##Simulation Functions for HW0

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from models import *
from helperFunctions import *


##Simulated Controller
##Input: 
##		initial condition state vector
##  	commmand vector
##		obstacle boolean, if true -- display the obstacles on the graph
##		optional -- groundtruth list, if provided displays groundtruth on the graph
##Output:
##		returns a state vector of current location (final location)
def runControlSimulator(initialConditions, commands, obstacleBool,*groundTruth):
	##Maintain path of the simulated controller esitmate 
	path = [initialConditions]

	##Start with the inital conditions
	currentLocation = initialConditions

	##Iterate through the commands, passing them to the controller model
	for command in commands:
		currentLocation = simulatedControllerEstimate(currentLocation, command[0], command[1], command[2])
		path.append(currentLocation)

	##Create plottable lists for position
	plotX = []
	plotY = []
	for point in path:
		plotX.append(point[0])
		plotY.append(point[1])

	##Plot groundtruth if needed
	plotGTX = []
	plotGTY = []
	if(len(groundTruth) > 0):
		for line in groundTruth[0]:
			plotGTX.append(float(line[1]))
			plotGTY.append(float(line[2]))

	##Handle the plotting legend and various combinations of plotting
	blue_line = mlines.Line2D([],[], color = 'blue')
	green_line = mlines.Line2D([],[], color = 'green')
	obstacle_line = mlines.Line2D([],[], color = 'red', marker = 'o', markersize = 5)

	if(obstacleBool and len(plotGTX) > 0):
		plt.plot(plotX, plotY, 'b-', plotGTX, plotGTY, 'g-', obstacle1.x, obstacle1.y, 'r-o', obstacle2.x, obstacle2.y, 'r-o', obstacle3.x, obstacle3.y, 'r-o')
		plt.legend([blue_line, green_line, obstacle_line], ['Controller Path', 'Ground Truth', 'Obstacles'])
	elif(obstacleBool and len(plotGTX) < 1):
		plt.plot(plotX, plotY, 'b-', obstacle1.x, obstacle1.y, 'r-o', obstacle2.x, obstacle2.y, 'r-o', obstacle3.x, obstacle3.y, 'r-o')
	elif(not obstacleBool and len(plotGTX) > 0):
		plt.plot(plotX, plotY, 'b-', plotGTX, plotGTY, 'g-')
	else:
		plt.plot(plotX, plotY, 'b-')


	plt.xlabel('X Position (meters)')
	plt.ylabel('Y Position (meters)')
	plt.title("State Estimate Based on Command Input")
	plt.show()

	##Output the current, final, location
	return currentLocation


##===The BIG PARTICLE FILTER CLASS==##
class ParticleFilter:

	## P.F. __init__ function
	##Input: 
	##		initial condition state vector
	##  	a vector of functions, equal to the length of the state vector,
	##			these functions will be used to create an initial particle clouds so must return a
	##			point and weight
	##		arguments needed for the probabily functions above, e.g. the std for a gaussian distrobution
	##		the number of particles that the p.f. will track
	##Output:
	##		none
    def __init__(self, initialStateVector, initialStateProbabiltyFunctionList, probFunctionArgs, numberOfParticles):
        
        self.m = numberOfParticles
        
        ## List of particles and weights
        self.X = []
        self.newX = []

        ##State Estimates
        self.xMean = initialStateVector[0]
        self.yMean = initialStateVector[1]
        self.rotMean = initialStateVector[2]

        ##list of previous state estimates used for smooth
        self.xMeanList = [initialStateVector[0],]
        self.yMeanList = [initialStateVector[1],]
        self.rotMeanList = [initialStateVector[2],]

    	## Populate the initial particle cloud based off the initial prob functions provided
        if(len(initialStateVector) != len(initialStateProbabiltyFunctionList) or len(initialStateVector) != len(probFunctionArgs)):
			return "Error, length of state vector and prob functinon or function args list are not equal"
        else:
			for i in range(self.m):
				posVector = []
				posiF = 0
				for j in range(len(initialStateVector)):
					output = initialStateProbabiltyFunctionList[j](initialStateVector[j], probFunctionArgs[j])
					posVector.append(output[0])
					posiF = posiF + output[1]
				##divided the weight/importance factor by 3, since it was the sum of the x,y,rot weights
				##probably not too important since they will soon be normalized
				self.X.append([posVector,posiF/3.0])





	## P.F. updateStep function
	##
	## Update step, which updates the point from the provided command through the motion model
	## and updates the importance factor based off the measurement model
	##
	##Input: 
	##		the command vector for this timestep
	##  	a list of measurement time steps (there could be more then one timestep within valid
	##			 measurement period).-- each time step could have more then one landmark measurement
	##Output:
	##		none
    def updateStep(self, command, measurement):

    	## do this for every point
		for i in range(self.m):
			## create a new entry for the updated X matrix
			entry = []

			## get the updated point from the motion model
			point = simulatedControllerEstimate(self.X[i][0], command[0], command[1], command[3])
			
			## track the importance factor value, and how many measurements are incorporated into it
			iF = 0
			n = 0

			if(len(measurement) > 0):
				
				## There could be more more then time stamp for the valid measurement period
				for timeStep in measurement:
					
					## within that time step there could be more then one landmark measurement
					for landmarkMeasure in timeStep.landmarkMeasurements:
						
						## make sure the landmark is not a robot (id's 1-5)
						if(landmarkMeasure[0] >= 6 and landmarkMeasure[0]<=21):

							##get the expected measurement for the given landmark
							expected = expectedMeasurement(point, landmarkMeasure[0])
							##use this expected measurement and create a weight by compareing it to the actual
							## measurement.
							iF = iF + getImportanceFactor(expected, [landmarkMeasure[2], landmarkMeasure[3]])

							##track the number of measurements are used to impact the iF
							n = n+1.0

				## average out the importance factor
				if(iF != 0):
					iF = iF/n
				## if there were no measurements keep previous weight/iF
				else:
					iF = self.X[i][1]
			## if there were no measurements keep previous weight/iF
			else:
				iF = self.X[i][1]

			## finally create the entry and add it to the new X matrix
			entry.append(point)
			entry.append(iF)
			self.newX.append(entry)

		## update the X matrix with the new one
		self.X = self.newX
		self.newX = []

		# print(self.X[0])

	##======== State Extraction Methods============#

	## P.F. getMean function
	##
	## gets the mean value of each state independent of other states or weights
	##
	##Input: 
	##		none
	##Output:
	##		state vector estimate
    def getMean(self):
		xVals = [x for [[x,y,rot], iF] in self.X]
		yVals = [y for [[x,y,rot], iF] in self.X]
		rotVals = [rot for [[x,y,rot], iF] in self.X]

		if(len(xVals) > 0 and len(yVals) > 0 and len(rotVals) > 0):
			self.xMean = sum(xVals)/len(xVals)
			self.yMean = sum(yVals)/len(yVals)
			self.rotMean = sum(rotVals)/len(rotVals)

			return [self.xMean, self.yMean, self.rotMean]
		else:
			return [0, 0 , 0]

	## P.F. getMeanSmoothed function
	##
	## gets the mean value of each state independent of other states or weights
	##	and averages them over the past ten values
	##
	##Input: 
	##		none
	##Output:
	##		state vector estimate
    def getMeanSmoothed(self):
        xVals = [x for [[x,y,rot], iF] in self.X]
        yVals = [y for [[x,y,rot], iF] in self.X]
        rotVals = [rot for [[x,y,rot], iF] in self.X]

        ## divide by zero sanity check
        if(len(xVals) > 0 and len(yVals) > 0 and len(rotVals) > 0):
            xMean = sum(xVals)/len(xVals)
            yMean = sum(yVals)/len(yVals)
            rotMean = sum(rotVals)/len(rotVals)
            newEntry = [xMean, yMean, rotMean]

            if(len(self.xMeanList)> 10):
                del self.xMeanList[0]
                del self.yMeanList[0]
                del self.rotMeanList[0]

            self.xMeanList.append(xMean)
            self.yMeanList.append(yMean)
            self.rotMeanList.append(rotMean)

            self.xMean = sum(self.xMeanList)/len(self.xMeanList)
            self.yMean = sum(self.yMeanList)/len(self.yMeanList)
            self.rotMean = sum(self.rotMeanList)/len(self.rotMeanList)
            return [self.xMean, self.yMean, self.rotMean]
        else:
            return [0, 0 , 0]


	## P.F. getMaxKernalDesnity function
	##
	## pass
	##
	##Input: 
	##		none
	##Output:
	##		none
    def getMaxKernalDesnity(self):
		pass

	## P.F. getHistogramMax function
	##
	## this computes a histogram for each state based off the weight/importance factor
	##	the state is the determined by making it the average of the most weighted bin in the
	##	historgram.
	##
	##Input: 
	##		none
	##Output:
	##		state vector estimate	
    def getHistogramMax(self):

    	## break up the X values into various lists
        xVals = [x for [[x,y,rot], iF] in self.X]
        yVals = [y for [[x,y,rot], iF] in self.X]
        rotVals = [rot for [[x,y,rot], iF] in self.X]
        myWeights = [iF for [[x,y,rot], iF] in self.X]

        if(len(xVals) > 0 and len(yVals) > 0 and len(rotVals) > 0):

       		## x value histogram
            xhist, xbin_edges = np.histogram(xVals, bins = 20, normed = True, weights = myWeights)
            xi = np.argmax(xhist)
            self.xMean = xbin_edges[xi] + np.diff(xbin_edges)[xi]/2

            ## y value histogram
            yhist, ybin_edges = np.histogram(yVals, bins = 20, normed = True, weights = myWeights)
            yi = np.argmax(yhist)
            self.yMean = ybin_edges[yi] + np.diff(ybin_edges)[yi]/2

            ## rotation value histogram
            rothist, rotbin_edges = np.histogram(rotVals, bins = 20, normed = True, weights = myWeights)
            roti = np.argmax(rothist)
            self.rotMean = rotbin_edges[roti] + np.diff(rotbin_edges)[roti]/2

            return [self.xMean, self.yMean, self.rotMean]
        else:
            return [0, 0 , 0]

	## P.F. getHistogramMaxSmoothed function
	##
	## this computes a histogram for each state based off the weight/importance factor
	##	the state is then determined by making it the average of the most weighted bin in the
	##	historgram.  this is then averaged over the past ten state estimates
	##
	##Input: 
	##		none
	##Output:
	##		state vector estimate
    def getHistogramMaxSmoothed(self):

    	## break up the X values into various lists
        xVals = [x for [[x,y,rot], iF] in self.X]
        yVals = [y for [[x,y,rot], iF] in self.X]
        rotVals = [rot for [[x,y,rot], iF] in self.X]
        myWeights = [iF for [[x,y,rot], iF] in self.X]

        if(len(xVals) > 0 and len(yVals) > 0 and len(rotVals) > 0):

        	## x value histogram
            xhist, xbin_edges = np.histogram(xVals, bins = 20, normed = True, weights = myWeights)
            xi = np.argmax(xhist)
            xMean = xbin_edges[xi] + np.diff(xbin_edges)[xi]/2

			## y value histogram
            yhist, ybin_edges = np.histogram(yVals, bins = 20, normed = True, weights = myWeights)
            yi = np.argmax(yhist)
            yMean = ybin_edges[yi] + np.diff(ybin_edges)[yi]/2

            ## rotation value histogram
            rothist, rotbin_edges = np.histogram(rotVals, bins = 20, normed = True, weights = myWeights)
            roti = np.argmax(rothist)
            rotMean = rotbin_edges[roti] + np.diff(rotbin_edges)[roti]/2

            if(len(self.xMeanList)> 10):
                del self.xMeanList[0]
                del self.yMeanList[0]
                del self.rotMeanList[0]

            self.xMeanList.append(xMean)
            self.yMeanList.append(yMean)
            self.rotMeanList.append(rotMean)

            self.xMean = sum(self.xMeanList)/len(self.xMeanList)
            self.yMean = sum(self.yMeanList)/len(self.yMeanList)
            self.rotMean = sum(self.rotMeanList)/len(self.rotMeanList)
            return [self.xMean, self.yMean, self.rotMean]

        else:
            return [0, 0 , 0]

	## P.F. normalizeWeights function
	##
	## normalizes the weights calculated from update state such that they are probabilies and sum to 1.0
	##
	##Input: 
	##		none
	##Output:
	##		none
    def normalizeWeights(self):
		myWeights = [iF for [[x,y,rot], iF] in self.X]
		myWeightSum = sum(myWeights)
		if(myWeightSum != 0.0):
			for i in range(self.m):
				self.X[i][1] = self.X[i][1]/myWeightSum


	## P.F. resampleStep function
	##
	## this resamples the X matrix based off the weights/importance factor and introduces random particles
	##
	##Input: 
	##		none
	##Output:
	##		none
    def resampleStep(self):

    	##Get the weights of the X matrix and put them in a numpy array
        myWeights = [iF for [[x,y,rot], iF] in self.X]
        myWeights = np.array(myWeights)

        ## create a cummalitive sum array, so a randomly generated number will index 
        ## proportioanlly to the weights of the given points
        weightCumSum = np.cumsum(myWeights)


        for i in range(self.m):

        	## random filter number to decide if particle is resampled or randomly generated
			filterNum = np.random.rand()

			## for a very small number of particles introduce a completely random particle and 
			## give it a very small weight
			if(filterNum < 0.000000005):
				self.newX.append([[7*np.random.rand()-2, 12*np.random.rand()-6, 2*np.pi*np.random.rand()], 1.0/(10000000*self.m)])
			
			## for a larger sample of particles, 20%, put them within 3/8 of a meter 
			## within the calculated mean value
			elif(filterNum >= 0.000000005 and filterNum < 0.2):
				self.newX.append([[randomGaussianPointAndProb(self.xMean, 3*0.125)[0], randomGaussianPointAndProb(self.yMean, 3*0.125)[0], randomGaussianPointAndProb(self.rotMean, 3*2*np.pi/64.0)[0]], 1/(1000*self.m)])
			
			## for a larger sample of particles, 5%, put them within 1/8 of a meter 
			## within the calculated mean value
			elif(filterNum >= .2 and filterNum < .25):
				self.newX.append([[randomGaussianPointAndProb(self.xMean, 1*0.125)[0], randomGaussianPointAndProb(self.yMean, 1*0.125)[0], randomGaussianPointAndProb(self.rotMean, 1*2*np.pi/64.0)[0]], 1/(10*self.m)])
			
			## otherwise resample the particle from the given point cloud
			else:

				index = (np.array([],), )
				n = 0
				
				## make sure we find a particle, shouldnt be an issue relic from development
				while(index[0].size < 1 and n < 10):

					## find the index of the particle whose weight we are reference with our random number
					index = np.where(weightCumSum >= np.random.rand())
					n = n+1

				if(index[0].size > 0):
					##insert that particle from the provided sampled index
					self.newX.append(self.X[index[0][0]])
				## make sure we find a particle, shouldnt be an issue relic from development
				else:
					print("resampling not working")
					print(index)
					print(myWeights)
					print(weightCumSum)
					self.newX.append(self.X[i])
		##update X matrix with the new one
    	self.X = self.newX
    	self.newX = []


