##Author: D. Harmon Pollock
##Date: Oct '14
##Class: EECS 495 ML & AI for Robotics
##Professor: Brenna Argall
##Assignment: hw0
##File Name: helperFunctions.py

##Helper Functions for HW0

import math
import numpy as np
from scipy import integrate
import random


## loadFileToLists Function
##
##	helps load the files and pass them into lists, ignoring comments
##
##Input: 
##		a file name
##Output:
##		returns a list of lists of the file content, still in string format
def loadFileToLists(fileName):
	contents = []
	f = open(fileName, 'r')
	for line in f:
		if(line[0] != '#'):
			contents.append(line.split())

	return contents

## Landmark Class
##
##	holds the landmarks data
##
##Input: 
##		the init requires the landmark data
##			
##Output:
##		none
class Landmark:
	def __init__(self, subject, xval, yval, x_std_dev, y_std_dev):
		self.id = subject
		self.x = xval
		self.y = yval
		self.x_sd = x_std_dev
		self.y_sd = y_std_dev

## Obstacle Class
##
##	simple obstacle class, the assist with plotting
##
##Input: 
##		the init requires a list of landmarks in the connected order
##			
##Output:
##		none
class Obstacle:
	def __init__(self, listLandmarks):
		obstacleLandmarks = []
		for lm in listLandmarks:
			obstacleLandmarks.append(lm)

		self.x = [landmark.x for landmark in obstacleLandmarks]
		self.y = [landmark.y for landmark in obstacleLandmarks]

## Hardcode landmarks for plotting for now:
lm6 = Landmark(6,1.88032539,-5.57229508,0.00001974,0.00004067) 
lm7 = Landmark(7,1.77648406,-2.44386354,0.00002415,0.00003114)
lm8 = Landmark(8,4.42330143,-4.98170313,0.00010428,0.00010507)
lm9 = Landmark(9,-0.68768043,-5.11014717,0.00004077,0.00008785)
lm10 = Landmark(10,-0.85117881,-2.49223307,0.00005569,0.00004923)
lm11 = Landmark(11,4.42094946,-2.37103644,0.00006128,0.00009175)
lm12 = Landmark(12,4.34924478,0.25444762,0.00007713,0.00012118)
lm13 = Landmark(13,3.07964257,0.24942861,0.00003449,0.00005609)
lm14 = Landmark(14,0.46702834,0.18511889,0.00003942,0.00002907)
lm15 = Landmark(15,-1.00015496,0.17453779,0.00006536,0.00005926)
lm16 = Landmark(16,0.99953879,2.72607308,0.00003285,0.00002186)
lm17 = Landmark(17,-1.04151642,2.80020985,0.00012014,0.00005809)
lm18 = Landmark(18,0.34561556,5.02433367,0.00004452,0.00004957)
lm19 = Landmark(19,2.96594198,5.09583446,0.00006062,0.00008501)
lm20 = Landmark(20,4.30562926,2.86663299,0.00003748,0.00004206)

## getLandmark function
##
##	a simple landmark getter function
##
##Input: 
##		landmark id value
##			
##Output:
##		the landmark class object for that id
def getLandmark(id):
	if(id == 6): return lm6
	elif(id == 7): return lm7
	elif(id == 8): return lm8
	elif(id == 9): return lm9
	elif(id == 10): return lm10
	elif(id == 11): return lm11
	elif(id == 12): return lm12
	elif(id == 13): return lm13
	elif(id == 14): return lm14
	elif(id == 15): return lm15
	elif(id == 16): return lm16
	elif(id == 17): return lm17
	elif(id == 18): return lm18
	elif(id == 19): return lm19
	elif(id == 20): return lm20
	else: return None

## Hardcode the obstacle shapes for now:

##Obstacle Outside 17, 15, 10, 9, 6, 8, 11, 12, 20, 19, 18
obstacle1 = Obstacle([lm17, lm15, lm10, lm9, lm6, lm8, lm11, lm12, lm20, lm19, lm18, lm17])

## Obstacle 16,14,13
obstacle2 = Obstacle([lm16, lm14, lm13])

## Obstacle 10, 7
obstacle3 = Obstacle([lm10,lm7])


## distance function
##
##	distance formula for 2-d vectors, in the format of [x,y]
##
##Input: 
##		2d vector
##		2d vector
##			
##Output:
##		distance value
def distance(vectorOne, vectorTwo):
	return math.sqrt(math.pow(vectorTwo[0] - vectorOne[0], 2) + math.pow(vectorTwo[1] - vectorOne[1],2))



## Measurement Step Class
##
##	a helper class to bunch together measurements taken at the same timestep
##
##Input: 
##		the init requires a the time value for the step
##			
##Output:
##		none
class MeasurementStep:
	def __init__(self, myTime):
		self.time = myTime
		self.landmarkMeasurements = []

	## addLandmarkMeasurement function
	##
	##	adds a landmark measurement for the current time step
	##
	##Input: 
	##		time stamp
	##		subject id
	##		barcode id 
	##		range measurement reading
	##		bearing measurement reading
	##			
	##Output:
	##		none

	def addLandmarkMeasurement(self, time, subject, barcode, myRange, myBearing):
		if(time == self.time):
			self.landmarkMeasurements.append([subject, barcode, myRange, myBearing])
		else:
			print("Error, unable to add measurement, incorrect time stamp")
			return

## getMeasurements function
##
##	takes of list of measurement time steps and pops those within the provided time frame
##	and removes any extra readings below the time min
##
##Input: 
##		list of measurement step objects
##		min time value for timesteps to pop
##		man time value for timesteps to pop
##			
##Output:
##		a list of remeaining time steps for future list
##		a list of timesteps within the provided min/max bounds
def getMeasurements(measurementStepList, minTime, maxTime):
	##create modifyable list
	newMeasurementStepList = measurementStepList
	foundAll = False
	curMeasurements = []

	##go until we are above our current max timestamp
	while(foundAll == False):

		if(len(newMeasurementStepList)>0):
			
			## remove those measurements below our bounds
			if(newMeasurementStepList[0].time < minTime):
				del newMeasurementStepList[0]

			## found all the timestamples we needed
			elif(newMeasurementStepList[0].time >= maxTime):
				foundAll = True;
				return [newMeasurementStepList,curMeasurements]

			## if within bounds, pop the measurement step into our output list
			else:
				curMeasurements.append(newMeasurementStepList[0])
				del newMeasurementStepList[0]
		else:
			return[newMeasurementStepList, []]


## parseBarcode2Subject Function
##
##	creates a hashtable for converting barcodes (provided in measurement data, 
##	to subject/landmark values
##
##Input: 
##		barcode id
##			
##Output:
##		a hashtable to convert barcodes to subject ids
def parseBarcode2Subject(barcodes):
	myDict = {}
	for entry in barcodes:
		myDict[float(entry[1])] = float(entry[0])
	return myDict


## parseMeasurements Function
##
##	pareses the raw measurement lists into the MeasurementStep class for use.
##
##Input: 
##		a list of measurements
##		a list of barcodes
##			
##Output:
##		a list of MeasurementStep objects in chronological order

def parseMeasurements(measurementList, barcodes):
	
	## create a hashtable for converting barcode ids to subjects (for the landmarks)
	subjects = parseBarcode2Subject(barcodes)
	totalMeasurements = []
	
	## keep track of current time and timestep object
	time = float(measurementList[0][0])
	currentTimeMeasurements = MeasurementStep(time)
	
	for i in range(len(measurementList)):

		## if in the right time, add measurement to the MeasurementStep object
		if(float(measurementList[i][0]) == time):
			currentTimeMeasurements.addLandmarkMeasurement(float(measurementList[i][0]), subjects[float(measurementList[i][1])], float(measurementList[i][1]), float(measurementList[i][2]), float(measurementList[i][3]))
		
		## if we're not in the right time, log the current MeasurementStep object,
		## create a new one and update the time
		else:
			totalMeasurements.append(currentTimeMeasurements)
			time = float(measurementList[i][0])
			currentTimeMeasurements = MeasurementStep(time)
			currentTimeMeasurements.addLandmarkMeasurement(float(measurementList[i][0]), subjects[float(measurementList[i][1])], float(measurementList[i][1]), float(measurementList[i][2]), float(measurementList[i][3]))

	## return the lists of all the MeasurementStep objects
	return totalMeasurements



## randomGaussianPointAndProb Function
##
##	helper function for returning a random sample from a gaussian distrobution 
##	along with its probabilty value (aka weight)
##
##Input: 
##		a mean value
##		a standard deviation value
##			
##Output:
##		a list [ a random value generated on the defined normal distrobution, 
##			the prob density of the value occuring]
def randomGaussianPointAndProb(mean, std):
	point = np.random.normal(mean, std)
	prob = 1/(std * np.sqrt(2 * np.pi)) * np.exp( - (point - mean)**2 / (2 * std**2))
	return [point, prob]

## uniformProb Function
##
##	helper function for returning a random value between -10 and 10 and a 
##	prob uniform for a given size set
##
##Input: 
##		a pos value (not needed but allows function to act in randomGaussianPointAndProb format )
##		a size value for creating a uniform disto prob
##			
##Output:
##		a list [ a random value  from -10 to 10, 
##			a uniform prob for a size]
def uniformProb(pos, size):
	point = 20*np.random.rand() - 10.0
	return[point, 1.0/size]

## createSensorNoiseDataset Function
##
##	output dataset of entires with the following format, ignores landmarks ids < 6
##		Timestamp(of measurement)
##		LandmarkID(of measurement)
##		Range(of measurement)
##		Heading(of measurement)
##		GroundTruth X (nearest to timestamp)
##		GroundTruth Y (nearest to timestamp)
##		GroundTruth Orientation (nearest to timestamp)
##		Landmark True X
##		Landmark True Y
##		Landmark True X Std Dev (optional)
##		Landmark True Y Std Dev (optioanl)
##
##Input: 
##		a pos value (not needed but allows function to act in randomGaussianPointAndProb format )
##		a size value for creating a uniform disto prob
##			
##Output:
##		a list [ a random value  from -10 to 10, 
##			a uniform prob for a size]
def createSensorNoiseDataset(measurementList, groundTruthList, barcodes, randomize = False):
	dataset = []
	gt = groundTruthList
	landmarkIDs = parseBarcode2Subject(barcodes)
	for mEntry in measurementList:
		if(landmarkIDs[int(mEntry[1])] > 6):
			dEntry = []
			# dEntry.append(random.random())
			dEntry.append(float(mEntry[0]))
			dEntry.append(landmarkIDs[int(mEntry[1])])
			dEntry.append(float(mEntry[2]))
			dEntry.append(float(mEntry[3]))

			[myGroundTruth, gt] = getNearestGT(float(mEntry[0]), gt)
			dEntry.append(myGroundTruth[1])
			dEntry.append(myGroundTruth[2])
			dEntry.append(myGroundTruth[3])

			mylandmark = getLandmark(landmarkIDs[int(mEntry[1])])
			# print getLandmark(landmarkIDs[int(mEntry[1])])

			dEntry.append(mylandmark.x)
			dEntry.append(mylandmark.y)

			dataset.append(dEntry)
	if(randomize):
		random.shuffle(dataset)
		return dataset
	else:
		return dataset


def getNearestGT(time, groundTruth):
	minDelta = 100
	index = 0
	for i in range(len(groundTruth)):
		if((time - float(groundTruth[i][0])) < minDelta and (time - float(groundTruth[i][0])) > 0):
			minDelta = (time - float(groundTruth[i][0]))
			index = i
		elif((time - float(groundTruth[i][0])) < 0):
			return [[float(groundTruth[i][0]), float(groundTruth[i][1]), float(groundTruth[i][2]), float(groundTruth[i][3])], groundTruth[i:]]

## createSensorNoiseDataset Function
##
##	input dataset of entires with the following format, ignores landmarks ids < 6
##		0 Timestamp(of measurement)
##		1 LandmarkID(of measurement)
##		2 Range(of measurement)
##		3 Heading(of measurement)
##		4  GroundTruth X (nearest to timestamp)
##		5 GroundTruth Y (nearest to timestamp)
##		6 GroundTruth Orientation (nearest to timestamp)
##		7 Landmark True X
##		8 Landmark True Y
##		Landmark True X Std Dev (optional)
##		Landmark True Y Std Dev (optioanl)
##
##Input: 
##		dataset of above format
##			
##Output:
##		a list of target range/heading values

def makeTargetArray(dataset):
	target = []
	for entry in dataset:
		# targetRange = entry[2]*5
		targetRange = distance([entry[4], entry[5]], [entry[7], entry[8]])
		tempOrin = entry[6]
		if(tempOrin > np.pi):
			tempOrin = tempOrin - 2*np.pi
		elif(tempOrin < -np.pi):
			tempOrin = tempOrin + 2*np.pi
		targetBearing = tempOrin - np.arctan2([entry[8]-entry[5]], [entry[7]-entry[4]])[0]

		if(targetBearing > np.pi):
			targetBearing = targetBearing - 2*np.pi
		elif(targetBearing < -np.pi):
			targetBearing = targetBearing + 2*np.pi
		# print(targetBearing)
		target.append([targetRange, targetBearing])
	return target

def rescale(dataset, minMaxList):
	newDataset= []
	scale = []
	for i in range(len(minMaxList)):
		if(abs(minMaxList[i][0]) > minMaxList[i][1]):
			scale.append(abs(minMaxList[i][0]))
		else:
			scale.append(minMaxList[i][1])
	for entry in dataset:
		newRow = []
		for i in range(len(entry)):
			newRow.append(entry[i]*scale[i])
		newDataset.append(newRow)
	return newDataset

def getXYRangeLocations(dataset, landmark, solution = False):
	xs = []
	ys = []
	for i in range(len(dataset)):
		if(dataset[i][1] == landmark):
			if(solution != False):
				x = dataset[i][4] + solution[i][0]*np.cos(dataset[i][6] - solution[i][1])
				y = dataset[i][5] + solution[i][0]*np.sin(dataset[i][6] - solution[i][1])
				xs.append(x)
				ys.append(y)
			else:
				x = dataset[i][4] + dataset[i][2]*np.cos(dataset[i][6] - dataset[i][3])
				y = dataset[i][5] + dataset[i][2]*np.sin(dataset[i][6] - dataset[i][3])
				xs.append(x)
				ys.append(y)

	return [xs, ys]

def getListMinMax(dataset):
	minMax = []
	for item in dataset[0]:
		minMax.append([item, item])
	for entry in dataset:
		for i in range(len(entry)):
			if(minMax[i][0] > entry[i]):
				minMax[i][0] = entry[i]
			elif(minMax[i][1] < entry[i]):
				minMax[i][1] = entry[i]
	return minMax

def getTrainingData(dataset):
	training = []
	for entry in dataset:
		training.append([entry[2], entry[3]])

	return training

def normalize(dataset, maxList):
	maxs = []
	normalizedList = []
	for i in range(len(maxList)):
		if(abs(maxList[i][0])> maxList[i][1]):
			maxs.append(abs(maxList[i][0]))
		else:
			maxs.append(maxList[i][1])
	for entry in dataset:
		normalizedEntry = []
		for i in range(len(entry)):
			normalizedEntry.append(entry[i]/maxs[i])
		normalizedList.append(normalizedEntry)
	return normalizedList

def sse(dataset1, dataset2):
	sumSSE = 0
	for i in range(len(dataset1)):
		sumTemp = 0
		for j in range(len(dataset1[i])):
			sumTemp = sumTemp + ((dataset1[i][j] - dataset2[i][j])*(dataset1[i][j] - dataset2[i][j]))
		sumSSE = sumSSE + sumTemp
	return sumSSE

def sseVector(dataset1, dataset2):
	sumSSE = []
	for j in range(len(dataset1[0])):
		sumSSE.append(0)
	for i in range(len(dataset1)):
		for j in range(len(dataset1[i])):
			sumSSE[j] = sumSSE[j] + ((dataset1[i][j] - dataset2[i][j])*(dataset1[i][j] - dataset2[i][j]))
	return sumSSE



##deadreckoning learning

def createDRDataset(commandList, groundTruthList, randomize = False):
	dataset = []
	gt = groundTruthList
	for i in range(len(commandList)-1):
		dEntry = []
		# dEntry.append(random.random())
		dEntry.append(float(commandList[i][0]))
		dEntry.append(float(commandList[i][1]))
		dEntry.append(float(commandList[i][2]))

		[myGroundTruth, gt] = getNearestGT2(float(commandList[i][0]), gt)
		dEntry.append(myGroundTruth[1])
		dEntry.append(myGroundTruth[2])
		dEntry.append(myGroundTruth[3])

		dEntry.append(float(commandList[i+1][0]) - float(commandList[i][0]))

		[myGroundTruth2, gt2] = getNearestGT2(float(commandList[i+1][0]), gt, remove = False)
		dEntry.append(myGroundTruth2[1])
		dEntry.append(myGroundTruth2[2])
		dEntry.append(myGroundTruth2[3])

		dataset.append(dEntry)
	if(randomize):
		random.shuffle(dataset)
		return dataset
	else:
		return dataset


def getNearestGT2(time, groundTruth, remove = True):
	minDelta = 100
	index = 0
	for i in range(len(groundTruth)):
		if((time - float(groundTruth[i][0])) < minDelta and (time - float(groundTruth[i][0])) > 0):
			minDelta = (time - float(groundTruth[i][0]))
			index = i
		elif((time - float(groundTruth[i][0])) < 0):
			if(remove == True):
				return [[float(groundTruth[i][0]), float(groundTruth[i][1]), float(groundTruth[i][2]), float(groundTruth[i][3])], groundTruth[i:]]
			else:
				return [[float(groundTruth[i][0]), float(groundTruth[i][1]), float(groundTruth[i][2]), float(groundTruth[i][3])], groundTruth]

## createSensorNoiseDataset Function
##
##	input dataset of entires with the following format, ignores landmarks ids < 6
##		0 Timestamp(of measurement)
##		1 LandmarkID(of measurement)
##		2 Range(of measurement)
##		3 Heading(of measurement)
##		4  GroundTruth X (nearest to timestamp)
##		5 GroundTruth Y (nearest to timestamp)
##		6 GroundTruth Orientation (nearest to timestamp)
##		7 Landmark True X
##		8 Landmark True Y
##		Landmark True X Std Dev (optional)
##		Landmark True Y Std Dev (optioanl)
##
##Input: 
##		dataset of above format
##			
##Output:
##		a list of target range/heading values

def makeDeadTargetArray(dataset):
	target = []
	for entry in dataset:
		target.append([entry[7], entry[8], entry[9]])
	return target

def getDeadTrainingData(dataset):
	training = []
	for entry in dataset:
		training.append([entry[1], entry[2], entry[3], entry[4], entry[5], entry[6]])

	return training
