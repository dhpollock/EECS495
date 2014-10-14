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

def simulatedControllerEstimate(posVector, tranSpeedCommandInput, rotSpeedCommandInput, timeDuration):

	tranSpeedCommand = tranSpeedCommandInput# + .025 * (.5-np.random.randn())
	rotSpeedCommand = rotSpeedCommandInput# + 0.075*(.5 - np.random.randn())


	# x2 = lambda t: np.cos(posVector[2] + rotSpeedCommand*t)
	# x = posVector[0] + tranSpeedCommand * integrate.quad(x2, 0.0, timeDuration)[0]

	# y2 = lambda t: np.sin(posVector[2] + rotSpeedCommand * t)
	# y = posVector[1] + tranSpeedCommand*integrate.quad(y2, 0.0, timeDuration)[0]

	# rot2 = lambda t: rotSpeedCommand
	# rot = posVector[2] + integrate.quad(rot2, 0.0, timeDuration)[0]

	# if(rotSpeedCommand == 0):
	# 	x2 = lambda t: -tranSpeedCommand*np.sin(posVector[2]) + tranSpeedCommand * np.sin(posVector[2]+ rotSpeedCommand*t)
	# else:
	# 	x2 = lambda t: -tranSpeedCommand/rotSpeedCommand*np.sin(posVector[2]) + tranSpeedCommand/rotSpeedCommand * np.sin(posVector[2]+ rotSpeedCommand*t)
	# # x2 = lambda t: np.cos(posVector[2] + rotSpeedCommand * t)
	# x = posVector[0] + integrate.quad(x2, 0.0, timeDuration)[0]

	# if(rotSpeedCommand == 0):
	# 	y2 = lambda t: tranSpeedCommand*np.cos(posVector[2]) - tranSpeedCommand * np.cos(posVector[2]+ rotSpeedCommand*t)
	# else:
	# 	y2 = lambda t: tranSpeedCommand/rotSpeedCommand*np.cos(posVector[2]) - tranSpeedCommand/rotSpeedCommand * np.cos(posVector[2]+ rotSpeedCommand*t)
	# # x2 = lambda t: np.cos(posVector[2] + rotSpeedCommand * t)
	# y = posVector[1] + integrate.quad(y2, 0.0, timeDuration)[0]

	# rot2 = lambda t: rotSpeedCommand
	# rot = posVector[2] + integrate.quad(rot2, 0.0, timeDuration)[0]

	if(rotSpeedCommand == 0.0):
		x = posVector[0] + tranSpeedCommand * np.cos(posVector[2]+ rotSpeedCommand*timeDuration)*timeDuration
	else:
		x = posVector[0] - tranSpeedCommand/rotSpeedCommand*np.sin(posVector[2]) + tranSpeedCommand/rotSpeedCommand * np.sin(posVector[2]+ rotSpeedCommand*timeDuration)
	
	if(rotSpeedCommand == 0.0):
		y = posVector[1] +tranSpeedCommand * np.sin(posVector[2]+ rotSpeedCommand*timeDuration)*timeDuration
	else:
		y = posVector[1] + tranSpeedCommand/rotSpeedCommand*np.cos(posVector[2]) - tranSpeedCommand/rotSpeedCommand * np.cos(posVector[2]+ rotSpeedCommand*timeDuration)
	
	rot = posVector[2] + rotSpeedCommand*timeDuration

	return [x, y, rot]


#given a state positioin predict the measurement value of a given landmark
def expectedMeasurement(posVector, landmarkID):
	myLandmark = getLandmark(landmarkID)
	meaDistance  = distance([myLandmark.x, myLandmark.y], [posVector[0], posVector[1]])
	meaBearing = np.arctan2([myLandmark.y - posVector[1]],[myLandmark.x - posVector[0]])[0] - posVector[2]
	if(meaBearing > np.pi):
		meaBearing = meaBearing - 2*np.pi
	if(meaBearing < -np.pi):
		meaBearing = meaBearing + 2*np.pi
	return[[meaDistance, meaBearing], landmarkID]

#given a measurement value, provide a weight, assuming that measuredZ is on a gaussian distrobution
def getImportanceFactor(expectedZ, measuredZ):
	# myLandmark = getLandmark(expectedZ[1])

	# R = 1-((myLandmark.x_sd*myLandmark.x_sd + myLandmark.y_sd*myLandmark.y_sd)*10000000)
	# R = .1 #our covarience, let it be 0.1 for now
	# R = .0001
	# R = .90
	R = 10.0
	# if(measuredZ[0] > 3.0):
	# 	R = .15*(measuredZ[0]/3)

	# R = .15*(measuredZ[0]/3)

	# iF = 1/np.sqrt(2 * np.pi * R)*np.exp(-math.pow(expectedZ - measuredZ,2)/(2*R))
	difAngle = expectedZ[0][1] - measuredZ[1]
	while(difAngle > np.pi):
		difAngle = 2* np.pi - difAngle
	while(difAngle < -np.pi):
		difAngle = 2* np.pi + difAngle
	dif = [expectedZ[0][0] - measuredZ[0], difAngle]
	iF = 1/np.sqrt(2 * np.pi * R)*np.exp(-(dif[0]*dif[0]+ dif[1]*dif[1])/(2*R))

	return iF





