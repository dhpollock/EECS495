##Author: D. Harmon Pollock
##Date: Oct '14
##Class: EECS 495 ML & AI for Robotics
##Professor: Brenna Argall
##Assignment: hw0
##File Name: helperFunctions.py

#Helper Functions for HW0

import math
import numpy as np
from scipy import integrate

#helps load the files and pass them into lists, ignoring comments
def loadFileToLists(fileName):
	contents = []
	f = open(fileName, 'r')
	for line in f:
		if(line[0] != '#'):
			contents.append(line.split())

	return contents

#simple landmark class
class Landmark:
	def __init__(self, subject, xval, yval, x_std_dev, y_std_dev):
		self.id = subject
		self.x = xval
		self.y = yval
		self.x_sd = x_std_dev
		self.y_sd = y_std_dev

#simple obstacle class, the assist with plotting
class Obstacle:
	def __init__(self, listLandmarks):
		obstacleLandmarks = []
		for lm in listLandmarks:
			obstacleLandmarks.append(lm)

		self.x = [landmark.x for landmark in obstacleLandmarks]
		self.y = [landmark.y for landmark in obstacleLandmarks]

#hardcode landmarks for plotting for now:
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

#landmark fetcher
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

#Obstacle Outside 17, 15, 10, 9, 6, 8, 11, 12, 20, 19, 18
obstacle1 = Obstacle([lm17, lm15, lm10, lm9, lm6, lm8, lm11, lm12, lm20, lm19, lm18, lm17])

# #Obstacle 16,14,13
obstacle2 = Obstacle([lm16, lm14, lm13])

# #Obstacle 10, 7
obstacle3 = Obstacle([lm10,lm7])

#distance formula for 2-d vectors, in the format of [x,y]
def distance(vectorOne, vectorTwo):
	return math.sqrt(math.pow(vectorTwo[0] - vectorOne[0], 2) + math.pow(vectorTwo[1] - vectorOne[1],2))

#a helper class to bunch together measurements taken at the same timestep
class MeasurementStep:
	def __init__(self, myTime):
		self.time = myTime
		self.landmarkMeasurements = []

	def addLandmarkMeasurement(self, time, subject, barcode, myRange, myBearing):
		if(time == self.time):
			self.landmarkMeasurements.append([subject, barcode, myRange, myBearing])
		else:
			print("Error, unable to add measurement, incorrect time stamp")
			return
def getMeasurements(measurementStepList, minTime, maxTime):
	newMeasurementStepList = measurementStepList
	foundAll = False
	curMeasurements = []
	while(foundAll == False):
		if(len(newMeasurementStepList)>0):
			if(newMeasurementStepList[0].time < minTime):
				del newMeasurementStepList[0]
			elif(newMeasurementStepList[0].time >= maxTime):
				foundAll = True;
				return [newMeasurementStepList,curMeasurements]
			else:
				curMeasurements.append(newMeasurementStepList[0])
				del newMeasurementStepList[0]
		else:
			return[newMeasurementStepList, []]

#creates a hashtable for converting barcodes (provided in measurement data, to subject/landmark values
def parseBarcode2Subject(barcodes):
	myDict = {}
	for entry in barcodes:
		myDict[float(entry[1])] = float(entry[0])
	return myDict

#pareses the raw measurement lists into the MeasurementStep class for use.
def parseMeasurements(measurementList, barcodes):
	subjects = parseBarcode2Subject(barcodes)
	totalMeasurements = []
	
	time = float(measurementList[0][0])
	currentTimeMeasurements = MeasurementStep(time)
	for i in range(len(measurementList)):
		if(float(measurementList[i][0]) == time):
			currentTimeMeasurements.addLandmarkMeasurement(float(measurementList[i][0]), subjects[float(measurementList[i][1])], float(measurementList[i][1]), float(measurementList[i][2]), float(measurementList[i][3]))
		else:
			totalMeasurements.append(currentTimeMeasurements)
			time = float(measurementList[i][0])
			currentTimeMeasurements = MeasurementStep(time)
			currentTimeMeasurements.addLandmarkMeasurement(float(measurementList[i][0]), subjects[float(measurementList[i][1])], float(measurementList[i][1]), float(measurementList[i][2]), float(measurementList[i][3]))

	return totalMeasurements

#helper function for returning a random sample from a gaussian distrobution along with its probabilty value (aka weight)
def randomGaussianPointAndProb(mean, std):
	point = np.random.normal(mean, std)
	prob = 1/(std * np.sqrt(2 * np.pi)) * np.exp( - (point - mean)**2 / (2 * std**2))
	return [point, prob]


