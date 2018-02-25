#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:56:37 2018

@author: ziyicui
"""
###############################################################################
# Program:    decisionTree.py   
# Author:     Ziyi Cui
# AndrewID:   ziyic
# Data:       2/23/2018
# Usage: python3 tagger.py <train input> <validation input> <test input>
#         <train out> <test out> <metrics out> <num epoch> <feature flag>
###############################################################################
import sys
import numpy as np
import csv
import math
###############################################################################
###############################################################################
### DATA PROCESSING
###############################################################################
##testOpen()
# Test whether the ifle can be opened.

def testOpen(filepath):
    try:
        f = open(filepath, "r")
        f.close()
    except:
        print("Unable to open file" + filepath)
        sys.exit()
################################################################################
## readFile()
# read the input file and store as a list of rows
def readFile(inputFile):
    data = []
    with open(inputFile, 'r') as tsvFile:
        tsvReader = csv.reader(tsvFile, delimiter = '\t')
        for row in tsvReader:
            if len(row) != 0:
                data.append(row)
            ###
            else:
                data.append('\n')
            ###
    tsvFile.close()
    return data

################################################################################
###############################################################################
### FEATURE ENGINEERING
###############################################################################
## featureEngineer_model1
#
def featureEngineer_model1 (data_train, data_validation, data_test):
    feature2Idx_train = featureName2Idx_1(data_train)
    label2Feature_train = featureVector_1(data_train, feature2Idx_train)
    labelList_train, label2FeatureWithoutEmptyLine_train = labelList(label2Feature_train)
    
    label2Feature_validation = featureVector_1(data_validation, feature2Idx_train)
    label2Feature_test = featureVector_1(data_test, feature2Idx_train)
    return feature2Idx_train, label2Feature_train, label2FeatureWithoutEmptyLine_train, labelList_train, label2Feature_validation, label2Feature_test
###############################################################################
## featureEngineer_model2
#
def featureEngineer_model2 (data_train, data_validation, data_test):
    feature2Idx_train = featureName2Idx_2(data_train)
    label2Feature_train = featureVector_2(data_train, feature2Idx_train)
    labelList_train, label2FeatureWithoutEmptyLine_train = labelList(label2Feature_train)
    
    label2Feature_validation = featureVector_2(data_validation, feature2Idx_train)
    label2Feature_test = featureVector_2(data_test, feature2Idx_train)
    return feature2Idx_train, label2Feature_train, label2FeatureWithoutEmptyLine_train, labelList_train, label2Feature_validation, label2Feature_test
################################################################################
##featureName2Idx_1()
# Construct shared mapping of feature strings to indices m from the input file
# for Model 1: use only the current word.
def featureName2Idx_1(data):
    idx = 0
    feature2Idx = {}
    
    # add bias as the first feature:
    feature2Idx['bias'] = idx
    idx = 1

    for row in data:
        if row != '\n':
        #if len(row) != 0:
            if row[0] not in feature2Idx:
                feature2Idx[row[0]] = idx
                idx += 1
   
    return feature2Idx

###############################################################################
## featureName2Idx_2()
# Construct shared mapping of feature strings to indices m from the input file
# for Model 2: use thecurrent word, the previous word and the next word.
def featureName2Idx_2(data):
    oriMapLength = 0
    oriIdx = 0
    mappingFeature2Idx_1 = {}
    mappingFeature2Idx_2 = {}
    
    mappingFeature2Idx_1 =featureName2Idx_1(data)
    oriMapLength = len(mappingFeature2Idx_1)
    
    mappingFeature2Idx_2 = {('curr:'+ k):v for k,v in mappingFeature2Idx_1.items() if v}
    mappingFeature2Idx_2['bias'] = 0
    mappingFeature2Idx_2['prev:BOS'] = oriMapLength
    mappingFeature2Idx_2['next:EOS'] = oriMapLength*3 - 1
    
    for feature in mappingFeature2Idx_1:
        if feature != 'bias':
            oriIdx = mappingFeature2Idx_1[feature]
            mappingFeature2Idx_2[('prev:'+ feature)] = oriIdx + oriMapLength
            mappingFeature2Idx_2[('next:'+ feature)] = oriIdx + 2*oriMapLength - 1

    #print(mappingFeature2Idx_2)
    return mappingFeature2Idx_2
###############################################################################
## featureVector_1()
# Constuct the feature vector, and results in a list of the indices m for which
# x_m is 1 for Model 1.
def featureVector_1(data, mappingfeature2Idx):
    featureIdx = []
    label = ''
    label2Feature = []

    for sample in data:
        if sample != '\n':
            label = sample[1]
            featureIdx = []
            featureIdx.append(0)
            featureIdx.append(mappingfeature2Idx[sample[0]])
            label2Feature.append({label:featureIdx})
        else:
            label2Feature.append({'':[]}) # indicator for empty line
    return label2Feature
###############################################################################
## featureVector_2()
# Constuct the feature vector, and results in a list of the indices m for which
# x_m is 1 for Model 2.
def featureVector_2(data, mappingFeature2Idx):
    label = ''
    featureIdx = []
    label2Feature = []
    
    for i, sample in enumerate(data):
        if sample != '\n':
            label = sample[1]
            featureIdx = []
            featureIdx.append(0)
    
            if i==0:
                featureIdx.append(mappingFeature2Idx['prev:BOS'])
            elif data[i-1] == '\n':
                featureIdx.append(mappingFeature2Idx['prev:BOS'])
            else:
                featureIdx.append(mappingFeature2Idx['prev:'+ data[i-1][0]])

            featureIdx.append(mappingFeature2Idx['curr:'+ sample[0]])
            
            if i ==len(data) - 1:
                featureIdx.append(mappingFeature2Idx['next:EOS'])
            elif data[i+1] == '\n':
                featureIdx.append(mappingFeature2Idx['next:EOS'])
            else:
                featureIdx.append(mappingFeature2Idx['next:'+ data[i+1][0]])
            label2Feature.append({label:featureIdx})
        else:
            label2Feature.append({'':[]}) # indicator for empty line
    return label2Feature
###############################################################################
## labelList()
# get a list of all labels, reference for labelList idx
def labelList(label2Feature):
    label2FeatureWithoutEmptyLine =[]
    labelList = []
    for i in label2Feature:
        if i != {'':[]}:
            label2FeatureWithoutEmptyLine.append(i)
    
    for sample in label2FeatureWithoutEmptyLine:
        label = list(sample.keys())[0]
        if label not in labelList:
            labelList.append(label)
    return labelList,label2FeatureWithoutEmptyLine

###############################################################################
###############################################################################
### DATA TRAINING
###############################################################################
## trainData()
#
def SGD(feature2Idx_train, label2FeatureWithoutEmptyLine_train, labelList_train, numEpoch):
    thetaList = []
    
    eta = 0.5
    m = len(feature2Idx_train)
    k = len(labelList_train)
    theta = np.zeros((k,m))
    
    for i in range(numEpoch):
        # each epoch
        for sample in label2FeatureWithoutEmptyLine_train:
            update = gradientMtx(theta,sample, labelList_train)
            theta = np.subtract(theta, eta*update)
        thetaList.append(theta)
    #print(len(thetaList))
    return thetaList

###############################################################################
## gradientMtx():
#
def gradientMtx(theta,sample, labelList): # sample {'O': [0,1]}
    labelIdx = 0
    denomiator = 0.0
    update = np.zeros_like(theta)
    
    # get the label and the feature vector x
    x = list(sample.values())[0]
    label = list(sample.keys())[0]

    denomiator = sum([math.exp(sum([k[i] for i in x]))for k in theta])
    
    for i,v in enumerate(labelList):
        if v == label:
            labelIdx = i
            break

    for i,k in enumerate(theta):
        numerator = math.exp(sum([k[j] for j in x]))
        if i == labelIdx:
            for j in x:
                update[i][j] = -(1 - numerator/denomiator)
        else:
            for j in x:
                update[i][j] = numerator/denomiator
    return update
###############################################################################
## reportResults()
#
def reportResults(thetaList,label2Feature_train, label2Feature_validation, labelList_train, outputFile):
    content = open(outputFile, "w+")

    for i, theta in enumerate(thetaList):
        likelihood_train = negLogLikelihood(theta, label2Feature_train, labelList_train)
        likelihood_validation = negLogLikelihood(theta, label2Feature_validation, labelList_train) #labelList_validation
   
        content.write("epoch=%s likelihood(train): %s\n" %((i+1), likelihood_train))
        content.write("epoch=%s likelihood(validation): %s\n" %((i+1), likelihood_validation))
    
    # prediction error
    content.close()

###############################################################################
## negLogLikelihood()
#
def negLogLikelihood(theta, label2Feature, labelList):
    labelIdx = 0
    denomiator = 0.0
    likelihood = 0.0
    label2FeatureWithoutEmptyLine =[]
    
    for i in label2Feature:
        if i != {'':[]}:
            label2FeatureWithoutEmptyLine.append(i)

    for sample in label2FeatureWithoutEmptyLine:
        x = list(sample.values())[0]
        label = list(sample.keys())[0]
        denomiator = sum([math.exp(sum([k[i] for i in x]))for k in theta])
        
        for i,v in enumerate(labelList):
            if v == label:
                labelIdx = i
                break
        
        for i,k in enumerate(theta):
            numerator = math.exp(sum([k[j] for j in x]))
            if i == labelIdx:
                indicator = 1.0
                likelihood += math.log(numerator/denomiator)

    likelihood = - likelihood / float(len(label2FeatureWithoutEmptyLine))
    #print(likelihood)
    return likelihood

###############################################################################
## predictionAndError()
#
def predictionAndError(labelList_train, theta, label2Feature,labelFile, mtxFile,dataType):
    labelContent = open(labelFile, "w+")
    mtxContent = open(mtxFile, "a")
    prob = 0.0
    maxProb = 0.0
    error = 0.0
    predictedLabel = ""
    label2FeatureWithoutEmptyLine =[]
    
    for i in label2Feature:
        if i != {'':[]}:
            label2FeatureWithoutEmptyLine.append(i)

    for sample in label2Feature:
        if sample != {'':[]}:
            probList = []
            candidates = []
            x = list(sample.values())[0]
            label = list(sample.keys())[0]
            denomiator = sum([math.exp(sum([k[i] for i in x]))for k in theta])
            for k in theta:
                numerator = math.exp(sum([k[j] for j in x]))
                prob = numerator/denomiator
                probList.append(prob)
            maxProb = max(probList)
            for i,p in enumerate(probList):
                if p == maxProb:
                    candidates.append(labelList_train[i])
                    predictedLabel = max(candidates)
            labelContent.write("%s\n" %predictedLabel)
            #print(predictedLabel)
            if predictedLabel != label:
                error += 1
        else:
            labelContent.write("\n")

    error = error /float(len(label2FeatureWithoutEmptyLine))
    mtxContent.write("error(%s): %s\n" %(dataType, error))
    labelContent.close()
    mtxContent.close()

###############################################################################
### main program
#
testOpen(sys.argv[1])
testOpen(sys.argv[2])
testOpen(sys.argv[3])

# read command line arguments
data_train = readFile(sys.argv[1])
data_validation = readFile(sys.argv[2])
data_test = readFile(sys.argv[3])
train_out = sys.argv[4]
test_out = sys.argv[5]
mtx_out = sys.argv[6]
numEpoch = int(sys.argv[7])
feature_flag = sys.argv[8]

# select which model to engineer the input features
func_arg = {"1":featureEngineer_model1, "2":featureEngineer_model2}

if __name__ == "__main__":
    feature2Idx_train, label2Feature_train, label2FeatureWithoutEmptyLine_train, labelList_train, label2Feature_validation, label2Feature_test = func_arg[sys.argv[8]](data_train, data_validation, data_test)

# train and predict
thetaList = SGD(feature2Idx_train, label2FeatureWithoutEmptyLine_train, labelList_train,numEpoch)
reportResults(thetaList,label2Feature_train, label2Feature_validation, labelList_train, mtx_out)

predictionAndError(labelList_train, thetaList[-1], label2Feature_train,train_out, mtx_out,'train')
predictionAndError(labelList_train, thetaList[-1], label2Feature_test,test_out, mtx_out,'test')


