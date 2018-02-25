#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 3 14:55:03 2018

@author: ziyicui
"""
################################################################################
#
#
# Program:   
# Author:     Ziyi (Echo) Cui
# AndrewID:   ziyic
# Data:       2/3/2018
###############################################################################
import sys
import csv
import math
###############################################################################
###testOpen()
# test whether the ifle can be opened

def testOpen(filepath):
    try:
        f = open(filepath, "r")
        f.close()
    except:
        print("Unable to open file" + filepath)
        sys.exit()
        
################################################################################       
### getRootEntropy()
## take a csvReader and calculate the root entropy
# ignore the first line in csvReader, which is the header line
# calculation on the last element in each line list
def getRootEntropy(InputFile):
    labels        = []
    labelPercent  = {}    
    rootEntropy   = 0.0
    rootError         = 0.0
    with open(InputFile, 'rb') as csvFile:
        csvReader = csv.reader(csvFile, delimiter = ',')
        next(csvReader, None) # skip the header
    
        for line in csvReader:
            labels.append(line[-1])	
    
    labelPercent  = {l:float(labels.count(l))/len(labels) for l in labels}
    rootError         = min(labelPercent.values())
    rootEntropy   = sum([(-i*math.log(i,2)) for i in labelPercent.values()])
    csvFile.close()
    return rootEntropy, rootError

################################################################################
### writeEntropyAndError()
# write the calcualted entropy and error for root label (based on majority vote)
def writeEntropyAndError(rootEntropy, rootError, OutputFile):
    content = open(OutputFile, "w+")
    content.write("entropy: %s\n" % rootEntropy)
    content.write("error: %s\n" % rootError)
    content.close()

################################################################################  
### Main program
# test whether each file can be opened    
testOpen(sys.argv[1])

rootEntropy, rootError = getRootEntropy(sys.argv[1])
writeEntropyAndError(rootEntropy, rootError, sys.argv[2])
