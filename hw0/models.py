##Author: D. Harmon Pollock
##Date: Oct '14
##Class: EECS 495 ML & AI for Robotics
##Professor: Brenna Argall
##Assignment: hw0
## File Name: models.py

##Model Functions for HW0

from scipy import integrate
import numpy as np
from helperFunctions import *
import math

##Simulated Controller To Estimate Position function 
##
##Input:
##	[Prior x,y,rot Estimates], Translational Speed Command,
##		Rotational Speed Command, Time Duration
##Output:
##	[x-position estimate, y-position estimate, rotational-position estimate]
##
## Model:
## x = x' + Integral(transSpeedCommand*cos(rot + rotSpeedCommand * t) * t, 0, timeDuration, dt)
## y = y' + Integral(transSpeedCommand*sin(rot + rotSpeedCommand * t) * t, 0, timeDuration, dt)
## rot = rot' + rotSpeedCommand * timeDuration

def simulatedControllerEstimate(posVector, tranSpeedCommandInput, rotSpeedCommandInput, timeDuration):

	## experimented with adding noise to the command inputs, not useful, keep the same
	tranSpeedCommand = tranSpeedCommandInput# + .025 * (.5-np.random.randn())
	rotSpeedCommand = rotSpeedCommandInput# + 0.075*(.5 - np.random.randn())

	##====State Quations====##
	## I had two sets of equations to define the motion model, in the end the functionally the same:

	## == Equations I came up with independently == ##

	x2 = lambda t: np.cos(posVector[2] + rotSpeedCommand*t)
	x = posVector[0] + tranSpeedCommand * integrate.quad(x2, 0.0, timeDuration)[0]

	y2 = lambda t: np.sin(posVector[2] + rotSpeedCommand * t)
	y = posVector[1] + tranSpeedCommand*integrate.quad(y2, 0.0, timeDuration)[0]

	rot2 = lambda t: rotSpeedCommand
	rot = posVector[2] + integrate.quad(rot2, 0.0, timeDuration)[0]

	## ==Equations sourced for Probalistic Robotics formaula 5.9, pg 101 == ##

	# if(rotSpeedCommand == 0.0):
	# 	x = posVector[0] + tranSpeedCommand * np.cos(posVector[2]+ rotSpeedCommand*timeDuration)*timeDuration
	# else:
	# 	x = posVector[0] - tranSpeedCommand/rotSpeedCommand*np.sin(posVector[2]) + tranSpeedCommand/rotSpeedCommand * np.sin(posVector[2]+ rotSpeedCommand*timeDuration)
	
	# if(rotSpeedCommand == 0.0):
	# 	y = posVector[1] +tranSpeedCommand * np.sin(posVector[2]+ rotSpeedCommand*timeDuration)*timeDuration
	# else:
	# 	y = posVector[1] + tranSpeedCommand/rotSpeedCommand*np.cos(posVector[2]) - tranSpeedCommand/rotSpeedCommand * np.cos(posVector[2]+ rotSpeedCommand*timeDuration)
	
	# rot = posVector[2] + rotSpeedCommand*timeDuration

	return [x, y, rot]




##Simulated Controller To Estimate Position function 
##
##given a state positioin predict the measurement value of a given landmark
##
##Input:
##		a position vector
##		a landmark id
##Output:
##		a list of [expected measurement range value given posVector, 
##			expected measurement bearing value given posVector]
##		the landmark ID
##
def expectedMeasurement(posVector, landmarkID):
	## get the landmark object for ease information access
	myLandmark = getLandmark(landmarkID)

	## calculated the distance and bearing
	meaDistance  = distance([myLandmark.x, myLandmark.y], [posVector[0], posVector[1]])
	meaBearing = np.arctan2([myLandmark.y - posVector[1]],[myLandmark.x - posVector[0]])[0] - posVector[2]

	## SUPER IMPORTANT, make sure the bearing measurement is within the correct range, adjust if not
	if(meaBearing > np.pi):
		meaBearing = meaBearing - 2*np.pi
	if(meaBearing < -np.pi):
		meaBearing = meaBearing + 2*np.pi

	return[[meaDistance, meaBearing], landmarkID]


## getImportanceFactor function 
##
##  a measurement value, provide a weight, assuming that measuredZ is on a gaussian distrobution
##
##Input:
##		the expectedMeasurement function output list
##		and measurement vector of [range, bearing]
##Output:
##		a weight/importance factor value
##
def getImportanceFactor(expectedZ, measuredZ):
	## Experimented with giving certain landmarks different variablity in measurement, not super useful
	# myLandmark = getLandmark(expectedZ[1])

	## === Various hardcoded values for R ===#

	R = .1 # the best covarience I found
	# R = .0001
	# R = .90
	# R = 10.0

	## === Experimented with more dynamic values, again not super useful === ##
	# R = 1-((myLandmark.x_sd*myLandmark.x_sd + myLandmark.y_sd*myLandmark.y_sd)*10000000)

	# if(measuredZ[0] > 3.0):
	# 	R = .15*(measuredZ[0]/3)

	# R = .15*(measuredZ[0]/3)

	## calculate the difference between the expected and measured bearings
	difAngle = expectedZ[0][1] - measuredZ[1]

	## again IMPORTANT, make sure the differencation angle is the smaller of the two posibilities
	while(difAngle > np.pi):
		difAngle = 2* np.pi - difAngle
	while(difAngle < -np.pi):
		difAngle = 2* np.pi + difAngle

	## the differential vector
	dif = [expectedZ[0][0] - measuredZ[0], difAngle]
	
	## the probabilty density value for a gauassian distro given R
	iF = 1/np.sqrt(2 * np.pi * R)*np.exp(-(dif[0]*dif[0]+ dif[1]*dif[1])/(2*R))

	return iF





