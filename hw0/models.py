##Author: D. Harmon Pollock
##Date: Oct '14
##Class: EECS 495 ML & AI for Robotics
##Professor: Brenna Argall
##Assignment: hw0
## File Name: models.py

#Model Functions for HW0

from scipy import integrate
import numpy as np

#Simulated Controller To Estimate Position and Heading
#Input: [Current x,y,rot Estimates], Translational Speed Command, 
#		Rotational Speed Command, Time Duration
#Output: [x-position estimate, y-position estimate, rotational-position estimate]
# Model:
# x = x' + Integral(transSpeedCommand*cos(rot + rotSpeedCommand * t) * t, 0, timeDuration, dt)
# y = y' + Integral(transSpeedCommand*sin(rot + rotSpeedCommand * t) * t, 0, timeDuration, dt)
# rot = rot' + rotSpeedCommand * timeDuration

def simulatedControllerEstimate(posVector, tranSpeedCommand, rotSpeedCommand, timeDuration):
	x2 = lambda t: tranSpeedCommand*np.cos(posVector[2] + rotSpeedCommand * t)
	x = posVector[0] + integrate.quad(x2, 0.0, timeDuration)[0]

	y2 = lambda t: tranSpeedCommand*np.sin(posVector[2] + rotSpeedCommand * t)
	y = posVector[1] + integrate.quad(y2, 0.0, timeDuration)[0]

	rot = posVector[2] + rotSpeedCommand*timeDuration
	
	return [x, y, rot]


