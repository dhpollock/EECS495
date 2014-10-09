##Author: D. Harmon Pollock
##Date: Oct '14
##Class: EECS 495 ML & AI for Robotics
##Professor: Brenna Argall
##Assignment: hw0
## File Name: models.py

#Model Functions for HW0

from scipy import integrate
import numpy as np
from helperFunctions import *
import math

#Simulated Controller To Estimate Position and Heading
#Input: [Prior x,y,rot Estimates], Translational Speed Command,
#		Rotational Speed Command, Time Duration
#Output: [x-position estimate, y-position estimate, rotational-position estimate]
# Model:
# x = x' + Integral(transSpeedCommand*cos(rot + rotSpeedCommand * t) * t, 0, timeDuration, dt)
# y = y' + Integral(transSpeedCommand*sin(rot + rotSpeedCommand * t) * t, 0, timeDuration, dt)
# rot = rot' + rotSpeedCommand * timeDuration

def simulatedControllerEstimate(posVector, tranSpeedCommand, rotSpeedCommand, timeDuration):
	x2 = lambda t: np.cos(posVector[2] + rotSpeedCommand * t)
	x = posVector[0] + tranSpeedCommand*integrate.quad(x2, 0.0, timeDuration)[0]

	y2 = lambda t: np.sin(posVector[2] + rotSpeedCommand * t)
	y = posVector[1] + tranSpeedCommand*integrate.quad(y2, 0.0, timeDuration)[0]

	rot2 = lambda t: rotSpeedCommand
	rot = posVector[2] + integrate.quad(rot2, 0.0, timeDuration)[0]

	return [x, y, rot]

#given a state positioin predict the measurement value of a given landmark
def expectedMeasurement(posVector, landmarkID):
	myLandmark = getLandmark(landmarkID)
	meaDistance  = distance([myLandmark.x, myLandmark.y], [posVector[0], posVector[1]])
	meaBearing = np.arctan2([myLandmark.y - posVector[1]],[myLandmark.x - posVector[0]])[0] - posVector[2]
	return[meaDistance, meaBearing]

#given a measurement value, provide a weight, assuming that measuredZ is on a gaussian distrobution
def getImportanceFactor(measuredZ, expectedZ):
	R = .25 #our covarience, let it be 0.1 for now
	# iF = 1/np.sqrt(2 * np.pi * R)*np.exp(-math.pow(expectedZ - measuredZ,2)/(2*R))
	dif = [expectedZ[0] - measuredZ[0], expectedZ[1] - measuredZ[1]]
	iF = 1/np.sqrt(2 * np.pi * R)*np.exp(-(dif[0]*dif[0]+ dif[1]*dif[1])/(2*R))

	return iF





