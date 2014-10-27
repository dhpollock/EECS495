#!/usr/bin/env python

##Author: D. Harmon Pollock
##Date: Oct '14
##Class: EECS 495 ML & AI for Robotics
##Professor: Brenna Argall
##Assignment: hw1
##File Name: models.py

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
	## I had two sets of equations to define the motion model, in the end the functionally the same,
	## but using the other formulas as they are computationally more efficient

	## == Equations I came up with independently == ##

	# x2 = lambda t: np.cos(posVector[2] + rotSpeedCommand*t)
	# x = posVector[0] + tranSpeedCommand * integrate.quad(x2, 0.0, timeDuration)[0]

	# y2 = lambda t: np.sin(posVector[2] + rotSpeedCommand * t)
	# y = posVector[1] + tranSpeedCommand*integrate.quad(y2, 0.0, timeDuration)[0]

	# rot2 = lambda t: rotSpeedCommand
	# rot = posVector[2] + integrate.quad(rot2, 0.0, timeDuration)[0]

	## ==Equations sourced for Probalistic Robotics formaula 5.9, pg 101 == ##

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


##Ouput Drive Controls function 
##
##Input:
##	[current x,y,rot], [goal x, y],
##		time incriment
##Output:
##	list of control steps([ forward speed v, rotational speed r], [v, r], ...)
##
## Model:
## x = x' + Integral(transSpeedCommand*cos(rot + rotSpeedCommand * t) * t, 0, timeDuration, dt)
## y = y' + Integral(transSpeedCommand*sin(rot + rotSpeedCommand * t) * t, 0, timeDuration, dt)

def outputDriveControls(currectState, goalState, timeIncrement):
	maxAccel = 0.288 #m/s2
	maxRotAccel = 5.579 #rad/s2
	vMax = .5
	steps = []

	#get to bearing to goal
	neededHeading = np.arctan2(goalState[1]-currectState[1], goalState[0]-currectState[0])

	adjustedheading = 0

	if(currectState[2] > np.pi):
		adjustedheading = currectState[2] - 2*np.pi
	elif (currectState[2] < -np.pi):
		adjustedheading = currectState[2] + 2*np.pi
	else:
		adjustedheading = currectState[2]
	deltaHeading = neededHeading - adjustedheading

	curHeading = adjustedheading
	w = 0
	while(abs(neededHeading - curHeading) > 0.005):

		if(abs(((neededHeading - curHeading) - w)/timeIncrement) > maxRotAccel):
			w = w + maxRotAccel*timeIncrement*math.copysign(1,neededHeading - curHeading)
		else:
			w = (neededHeading - curHeading)
		

		curHeading = curHeading + w*timeIncrement

		# print("rot = " + str(w))

		steps.append([0, w])


	dist = distance(goalState, [currectState[0], currectState[1]])
	curdist = 0
	v = 0 
	while(abs(dist - curdist) > 0.01):

		if(abs(((dist-curdist) - v)/timeIncrement) > maxAccel):
			if(v > vMax):
				v = v
			else:
				v = v + maxAccel*timeIncrement*math.copysign(1,dist-curdist)
		else:
			v = dist - curdist

		curdist = curdist + v*timeIncrement
		steps.append([v,0])
		# print("vel = " + str(v))

	return steps






