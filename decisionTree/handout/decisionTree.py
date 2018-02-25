#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 3 14:55:03 2018

@author: ziyicui
"""
################################################################################
#
#
# Program:    decisionTree.py   
# Author:     Ziyi (Echo) Cui
# AndrewID:   ziyic
# Data:       2/3/2018
################################################################################
import sys
import csv
import math
################################################################################
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
###extractTrainData()
# extract Data from the input .csv file
# return the fields(arttributes + label), and rows, containing all the data
def extractTrainData(trainInput):
    fields = []
    rows = []
    with open(trainInput, 'rb') as csvFile:
        csvReader = csv.reader(csvFile, delimiter = ',')
        fields = csvReader.next()
        for row in csvReader:
            rows.append(row)
    csvFile.close()
    return fields,rows

##############################################################################
# buildDecisionTree():
#
def buildDecisionTree(fields, rows, maxDepth, currentDepth1):  
    currentDepth = currentDepth1
    labelDict = {}
    labelList = []
    split = False
    bestAttrIdx = 0
    bestAttrName = ''
    attrValues = []
    tree = {}

    
    #labelList = [row[-1] for row in rows]
    #labelList,labelDict = countValue(fields[-1], fields, rows)    
    labelList = [row[-1] for row in rows]
    labelDict = {i: labelList.count(i) for i in labelList} 
       
    # if there is only one label
    if labelList.count(labelList[0]) == len(labelList): 
        return labelList[0] # now the tree does not grow, simply return the value
    # if there is no feature // or meet the maxDepth
    if (len(rows[0]) == 1) or (currentDepth >= maxDepth):
        return majorityVote(labelDict) # return the majority value
   

    # find the first attribute to split-- non-recursive
    split, bestAttrIdx = bestAttrToSplit(rows) # get the field idx to split
    #print fields
    if split == False:
        return majorityVote(labelDict)
    else:
        bestAttrName = fields[bestAttrIdx]

    # start grow tree by splitting the first attribute
    tree = {bestAttrName:{}}
    del(fields[bestAttrIdx]) # remove the selected attribute
    attrValues = [row[bestAttrIdx] for row in rows] # extract the column of the current attribute
    valueList = set(attrValues)
    currentDepth += 1
    for v in valueList:
        subFields = fields[:] # copy all of labels, so trees don't mess
        subRows = splitData(rows, bestAttrIdx, v)
        print '|' * currentDepth, bestAttrName, '=',v,':', countLabel(subRows) 
        tree[bestAttrName][v] = buildDecisionTree(subFields, subRows, maxDepth, currentDepth)
   
    return tree     

###############################################################################
###countValue():
# count the values of the label
def countLabel(rows): 
    labelList = []
    labelDict = {}
    for row in rows:
        labelList.append(row[-1]) 
    labelDict = {i:labelList.count(i) for i in labelList}
    return labelDict


###############################################################################
###majorteVote():
# return the majority value
def majorityVote(labelDict):
    labelList = []
    majVote = {}

    labelList = labelDict.keys()
    if labelDict[labelList[0]] >= labelDict[labelList[1]]:
        return labelList[0]
    else: 
        return labelList[1]

###############################################################################
###bestAttrToSplit():
# find the next attribute to split based on maximum information gain
def bestAttrToSplit(rows):
    attrNum = 0
    bestAttrIdx = - 1
    infoGainList = []
    maxInfoGain = 0.0
    split = False
      

    attrNum = len(rows[0]) - 1
    baseEntropy = entropy(rows)  # calculate the current entropy

    for i in range(attrNum):
        attrList = [row[i] for row in rows] # get each column
        valueList = set(attrList) # get all values this feature have
        condEntropy = 0.0
        for value in valueList:
            subRows = splitData(rows, i, value)
            pValue = float(len(subRows))/float(len(rows))
            condEntropy += pValue * entropy(subRows)
        infoGain = baseEntropy - condEntropy
        infoGainList.append(infoGain)
    
    maxInfoGain = max(infoGainList)
    
    # no infoGain is greater than 0
    if maxInfoGain <= 0:
        return split, bestAttrIdx
    # can find the best attribute to splite 
    split = True
    bestAttrIdx = infoGainList.index(maxInfoGain)
    return split, bestAttrIdx

###############################################################################
###entropy()
def entropy(rows):
    entropy   = 0.0
    instances = []
    pDict     = {} 
    
    for row in rows:
        instances.append(row[-1])
         
    pDict     = {k:float(instances.count(k))/len(instances) for k in instances}
    entropy   = sum([(-v*math.log(v,2)) for v in pDict.values()])
    
    return entropy

###############################################################################
###splitData():
# 
def splitData(rows, idx, value):
    subData = []
    choppedRow = []
    for row in rows:
        if row[idx] == value:
            choppedRow = row[:idx]
            choppedRow.extend(row[idx+1:])
            # choppedRow.remove[idx]
            subData.append(choppedRow)
    
    return subData

##############################################################################
###classifyOneRow():
# classify one row
def classifyOneRow(inputTree, fields, testRow):
    firstAttr = inputTree.keys()[0] # this list only has one element
    fistAttrIdx = fields.index(firstAttr) # subtree of first attr
    subTree = inputTree[firstAttr]
    testKey = testRow[fistAttrIdx] # the value of Attr for this testRow
    testValue = subTree[testKey]
    if isinstance(testValue, dict):
        classLabel = classifyOneRow(testValue, fields, testRow)
    else:
        classLabel = testValue
    return classLabel
 
##############################################################################
###classify():
def classify(inputTree, fields, testRows):
    classLabelList = []
    classLabel = ''
    if isinstance(inputTree, basestring):
        for row in testRows:
            classLabelList.append(inputTree)
    else: # when inputTree is a dict
        for row in testRows:
            classLabel = classifyOneRow(inputTree, fields, row)
            classLabelList.append(classLabel)
    
    return classLabelList
##############################################################################
###errorRate():
def errorRate(classLabelList, testRows):
    errors = 0
    pError = 0.0
    for i in range(len(classLabelList)):
        if classLabelList[i] != testRows[i][-1]:
            errors += 1

    pError = float(errors)/float(len(classLabelList))
    return pError
##############################################################################
###main program:

testOpen(sys.argv[1])
testOpen(sys.argv[2])

fieldsTrain, rowsTrain = extractTrainData(sys.argv[1])
fieldsTest, rowsTest = extractTrainData(sys.argv[2])
maxDepth = int(sys.argv[3])

# print label before tree
currentField = fieldsTrain[-1]
labelDict = countLabel(rowsTrain)    
# format print
print labelDict

# train the tree
fieldsTrain = fieldsTrain[:-1] 
tree = buildDecisionTree(fieldsTrain, rowsTrain, maxDepth, 0) 
print 'the tree:'
print tree

# classify use the tree
fieldsTest = fieldsTest[:-1]
classifyResult = classify(tree, fieldsTest, rowsTest)
#print 'classify results:'
#print classifyResult

# calculate the error rate
error = errorRate(classifyResult, rowsTest)
print error

