##Author: D. Harmon Pollock
##Date: Oct '14
##Class: EECS 495 ML & AI for Robotics
##Professor: Brenna Argall
##Assignment: hw0
##File Name: helperFunctions.py

#Helper Functions for HW0



def loadFileToLists(fileName):
	contents = []
	f = open(fileName, 'r')
	for line in f:
		if(line[0] != '#'):
			contents.append(line.split())

	return contents