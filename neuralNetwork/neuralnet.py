#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 09 14:56:37 2018

@author: ziyicui
"""
###############################################################################
# Program:    neuralnet.py
# Author:     Ziyi Cui
# AndrewID:   ziyic
# Data:       03/09/2018
# Usage: python3 neuralnet.py <train_input> <validation_input> <train_out>
#          <validation_out> <metrics_out> <num_epoch> <hidden_units><init_flag>
#          <learning_rate>
###############################################################################
import sys
import numpy as np
import csv
import math
###############################################################################
class Intermediate_quantities:
    def __init__(self, x, a, z, b, y_pred, J):
        self.x = x
        self.a = a
        self.z = z
        self.b = b
        self.y_pred = y_pred
        self.J = J
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
# Read the input file and store as a list of rows.
def readFile(inputFile):
    data = np.genfromtxt(inputFile, delimiter = ',')
    x_data = np.array([np.array(x[1:]) for x in data]) # 2d np.array
    labels = np.array([x[0] for x in data]) # np.array
    return x_data, labels
################################################################################
### INITIALIZATION
################################################################################
###init_random()
# The weights are initialized randomly from a uniform distribution from -0.1 to
# 0.1.
# The bias parameters are initialized to zero.
# input: feature_size, hidden_units, label_types
# output: alpha, beta
def init_random(M, D, K):
    alpha = np.random.uniform(-0.1,0.1,(D, M)) # DxM
    beta = np.random.uniform(-0.1,0.1, (K, D)) # KxD
    # set the bias parameters to 0
    alpha = np.insert(alpha, 0, 0, axis = 1) # Dx(M+1)
    beta = np.insert(beta, 0, 0, axis = 1) # Kx(D+1)
    return alpha, beta
################################################################################
###init_zero()
# All weights are initialized to 0.
# input: feature_size, hidden_units, label_types
# output: alpha, beta
def init_zero(M,D,K):
    alpha = np.zeros((D, M+1))
    beta = np.zeros((K, D+1))
    return alpha, beta
################################################################################
### one_hot_vector()
# input: labels: np.array of float64; K: type of labelList_train
# output: one hot vector
def one_hot_vector(labels, K):
    N = labels.shape[0]
    y_train = np.zeros((N, K))
    y_train[np.arange(N), np.int_(labels)] = 1
    return y_train
################################################################################
### DATA TRAINING
################################################################################
### SGD()
# Train the neural network.
# input: x_train, y_train,x_validation, y_validation, alpha, beta, num_epoch,
# learning_rate
# output: a list of mean cross entropy for train data, a list of mean cross
# entropy for validation data, final alpha, final beta
def SGD(x_train, y_train,x_validation, y_validation, alpha, beta, num_epoch, \
learning_rate):
    list_train = []
    list_validation = []
    for i in range(num_epoch):
        # each epoch
        for idx, x in enumerate(x_train):
            # compute nerual network layers
            intermediates = NN_forward(x, y_train[idx],alpha, beta)
            # compute gradient via backprop
            g_alpha, g_beta = NN_backward(y_train[idx], alpha, beta, intermediates)
            # update parameters
            alpha_update = learning_rate*g_alpha
            beta_update= learning_rate*g_beta
            alpha = np.array(np.subtract(alpha, alpha_update))
            beta = np.array(np.subtract(beta, beta_update))
        # evaluate training average cross entropy using the updated alpha, beta
        J_train = 0.0
        J_validation = 0.0
        for idx, x in enumerate(x_train):
            intermediates_train = NN_forward(x, y_train[idx],alpha, beta)
            J_train += intermediates_train.J
        J_train = J_train/len(x_train)
        list_train.append(J_train)
        # evaluate validation average cross entropy using the updated alpha,beta
        for idx, x in enumerate(x_validation):
            intermediates_val = NN_forward(x, y_validation[idx],alpha, beta)
            J_validation += intermediates_val.J
        J_validation = J_validation/len(x_validation)
        list_validation.append(J_validation)

    return list_train, list_validation, alpha, beta
################################################################################
### predict()
# Predict labels for given data with input alpha and beta.
# input: x_data, true_labels, alpha, beta
# output: predicted_labels, 0-1 loss error
def predict(x_data, true_labels, alpha, beta):
    predicted_labels = []
    error = 0.0
    for idx, x in enumerate(x_data):
        a = linear_forward(x, alpha)
        z = sigmoid_forward(a)
        z = np.insert(z, 0, 1)
        b = linear_forward(z, beta)
        y_pred = softmax_forward(b)
        label = np.argmax(y_pred)
        predicted_labels.append(label)

        # report prediction error:
        if label != true_labels[idx]:
            error += 1
    error = error / len(x_data)
    return predicted_labels, error

################################################################################
### MODULE-BASED AUTOMATIC DIFFERENTIATION (AD) FOR NEURAL NETWORK
################################################################################
### NN_forward()
# Forward computation for 1-hidden-layer neural network.
# computed for each sample passed in SGD
# input: x, one-hot vector for y, alpha, beta
# output: the intermediates that holds all forward results
def NN_forward(x, y_true, alpha, beta):
    a = linear_forward(x, alpha)
    z = sigmoid_forward(a)
    # add bias for z, fixed
    z = np.insert(z, 0, 1)
    b = linear_forward(z, beta) # beta: Kx(1+D) mul (1+D)x1 -> b: Kx1
    y_pred = softmax_forward(b)
    J = cross_entropy_forward(y_true, y_pred)
    intermidiates = Intermediate_quantities(x, a, z, b, y_pred, J)
    return intermidiates
################################################################################
### linear_forward()
# input:  x: array (M+1) -> matmul convert it into 1x(M+1); alpha: Dx(1+M)
# output: a = (alpha)x, Dx1
def linear_forward(x, alpha):
    # parameters need transpose
    # 1x(1+M), Dx(M+1) ->  D x 1
    a = np.matmul(x, alpha.T)
    return a
################################################################################
### sigmoid_forward()
# element-wise sigmoid function
# input: a: Dx1
# output: z: Dx1
def sigmoid_forward(a):
    z = 1.0/(1+np.exp(-a))
    return z
################################################################################
### softmax_forward()
# element-wise softmax
# input: b: Kx1
# output: y_pred: Kx1
def softmax_forward(b):
    exp = np.exp(b)
    s = sum(np.array(exp))
    y_pred = np.divide(np.exp(b), s)
    return y_pred
################################################################################
### cross_entropy_forward()
# input: y_true: Kx1; y_pred: Kx1
# output: J: scalar
def cross_entropy_forward(y_true, y_pred):
    log_y_pred = np.log(y_pred)
    index = np.argwhere(y_true)[0,0]
    J = -log_y_pred[index]
    return J
################################################################################
### NN_backward()
# Forward computation for 1-hidden-layer neural network.
# computed for each sample passed in SGD
# input: y_true, alpha, beta, intermediate
# output: d_J/d_alpha,d_J/d_beta
def NN_backward(y_true, alpha, beta, intermediates):
    x = intermediates.x # array (1+M)x1
    a = intermediates.a # array Dx1
    z = intermediates.z # array (D+1)x1
    b = intermediates.b # array Kx1
    y_pred = intermediates.y_pred # array Kx1
    J = intermediates.J # scalar

    g_J = 1 # base case
    g_y_pred = cross_entropy_backward(y_true, y_pred, J, g_J) # array Kx1
    g_b = softmax_backward(b, y_pred, g_y_pred) # matrix 1xK
    g_beta, g_z = linear_backward(z, beta, b, g_b) # matrix Kx(D+1),matrix (D+1)x1
    g_a = sigmoid_backward(a, z, g_z) # matrix 1x(D+1)
    # need to modify g_a from 1x(D+1) to 1xD
    g_a = np.delete(g_a, 0, axis = 1)
    g_alpha, _ = linear_backward(x, alpha, a, g_a) # matrix Dx(M+1)
    return g_alpha, g_beta
################################################################################
### cross_entropy_backward()
# input: y_true: Kx1; y_pred: Kx1; g_J = 1
# output: g_y_pred: array Kx1
def cross_entropy_backward(y_true, y_pred, J, g_J):
    g_y_pred = np.zeros_like(y_true)
    index = np.argwhere(y_true)[0][0]
    g_y_pred[index] = -g_J*y_true[index]/y_pred[index]
    return g_y_pred
################################################################################
### softmax_backward()
# input: b: array Kx1; y_pred: array Kx1; g_y_pred: array Kx1
# output: g_b: matrix 1xK
def softmax_backward(b, y_pred, g_y_pred):
    y_pred_diag = np.diag(y_pred)
    y_pred = np.asmatrix(y_pred)
    y_pred_T = np.asmatrix(y_pred.T)
    y_yT =  np.matmul(y_pred_T, y_pred)
    g_b = y_pred_diag - y_yT
    g_y_pred = np.asmatrix(g_y_pred)
    g_b = np.matmul(g_y_pred, g_b)
    return g_b
################################################################################
### linear_backward()
# input: g_b: matrix 1xK, z: array, (D+1)x1; beta: array Kx(D+1), b: array Kx1
# output: g_beta: matrix Kx(D+1); g_z: matrix (D+1)x1
# input: g_a: matrix 1xD, a: array Dx1; alpha: array Dx(M+1), x: array (M+1)x1
# output: g_alpha: matrix Dx(M+1); g_x: matrix (M+1)x1
def linear_backward(z, beta, b, g_b):
    z = np.asmatrix(z) # (D+1)x1 -> 1x(D+1)
    g_beta = np.matmul(g_b.T, z) # (Kx1) (1x(D+1)) -> Kx(D+1)
    g_z = np.matmul(beta.T, g_b.T) # ((D+1)xK) (Kx1) -> (D+1)x1
    return g_beta, g_z
################################################################################
### sigmoid_backward()
# input: a: array Dx1; z: array (D+1)x1; g_z: matrix (D+1)x1
# output: g_a: matrix 1x(D+1)
def sigmoid_backward(a, z, g_z):
    g_z = np.array(g_z)
    g_z = g_z.flatten() # convert into array (D+1)
    g_a = np.subtract(1, z)
    g_a = np.multiply(z, g_a)
    g_a = np.multiply(g_z, g_a)
    g_a = np.asmatrix(g_a)
    return g_a
################################################################################
### report_results()
# write the results to required files
# input: train_out, validation_out, metrics_out, num_epoch, list_train,
# list_validation, error_train, error_validation, predicted_train,
# predicted_validation)
# output: none
def report_results(train_out, validation_out, metrics_out, num_epoch, \
list_train,list_validation, error_train, error_validation, predicted_train, \
predicted_validation):
    train_content= open(train_out, "w+")
    validation_content = open(validation_out, "w+")
    metrics_content = open(metrics_out, "w+")
    for label in predicted_train:
        train_content.write("%s\n" %label)
    for label in predicted_validation:
        validation_content.write("%s\n"%label)

    for i in range(num_epoch):
        metrics_content.write("epoch=%s crossentropy(train): %s\n"\
         %((i+1), list_train[i]))
        metrics_content.write("epoch=%s crossentropy(validation): %s\n" \
         %((i+1), list_validation[i]))

    metrics_content.write("error(train): %s\n" %error_train)
    metrics_content.write("error(validation): %s" %error_validation)

    train_content.close()
    validation_content.close()
    metrics_content.close()

################################################################################
### main Program
##
testOpen(sys.argv[1])
testOpen(sys.argv[2])

## read command line arguments
x_train, labels_train = readFile(sys.argv[1])
x_validation, labels_validation = readFile(sys.argv[2])
train_out = sys.argv[3]
validation_out = sys.argv[4]
metrics_out = sys.argv[5]
num_epoch = int(sys.argv[6])
D = int(sys.argv[7]) # hidden_units
init_flag = sys.argv[8]
learning_rate = float(sys.argv[9])

## data pre-processing
M = len(x_train[0])
K = (len(set(labels_train)))
# two types of initilization for parameters alpha and beta
func_arg = {"1": init_random, "2":init_zero}
if __name__ == "__main__":
    alpha,beta = func_arg[init_flag](M, D, K)

# add bias x0 = 1, z0 = 1 into data
x_train = np.insert(x_train, 0, 1, axis = 1) # n x (1 + M)
x_validation = np.insert(x_validation, 0, 1, axis = 1)
# create one-hot vectors for labels
y_train = one_hot_vector(labels_train, K)
y_validation = one_hot_vector(labels_validation, K)

## train and predict
list_train, list_validation, alpha, beta = SGD(x_train, y_train, x_validation,\
 y_validation, alpha, beta, num_epoch, learning_rate)

predicted_train, error_train = predict(x_train, labels_train, alpha, beta)

predicted_validation, error_validation = predict(x_validation, \
labels_validation, alpha, beta)

report_results(train_out, validation_out, metrics_out, num_epoch, list_train, \
list_validation, error_train, error_validation, predicted_train, \
predicted_validation)
'''
# ------- for empirical Q ----
#content = open("forPlot.csv","a")
#content.write('%s,%s\n'%(D, list_validation[-1]))
#figure2
filename = sys.argv[10]
content = open(filename,"w+")
for i in range(len(list_train)):
    content.write('%s,train,%s,validation,%s\n'%(i+1, list_train[i],list_validation[i]))
'''
#-------------------- for written Q -----
'''
x = np.array([1,1,1,0,0,1,1]) # with bias 7
y_true = np.array([0,1,0])
# 4x7
alpha = np.array([[1,1,2,-3,0,1,-3],[1,3,1,2,1,0,2],[1,2,2,2,2,2,1], [1,1,0,2,1,-2,2]])
# 3x5
beta = np.array([[1,1,2,-2,1], [1,1,-1,1,2], [1,3,1,-1,1]])

a = linear_forward(x, alpha)
print('a')
print(a)
z = sigmoid_forward(a)
print('z')
print(z)
# add bias for z, fixed
z = np.insert(z, 0, 1)
b = linear_forward(z, beta) # beta: Kx(1+D) mul (1+D)x1 -> b: Kx1
print('b')
print(b)
y_pred = softmax_forward(b)
print('y_pred')
print(y_pred)
J = cross_entropy_forward(y_true, y_pred)
print('J')
print(J)
#intermidiates = Intermediate_quantities(x, a, z, b, y_pred, J)

g_J = 1 # base case
g_y_pred = cross_entropy_backward(y_true, y_pred, J, g_J) # array Kx1
print('g_y_pred')
print(g_y_pred)
g_b = softmax_backward(b, y_pred, g_y_pred) # matrix 1xK
print('g_b')
print(g_b)
g_beta, g_z = linear_backward(z, beta, b, g_b) # matrix Kx(D+1),matrix (D+1)x1
print('g_beta')
print(g_beta)
print('g_z')
print(g_z)
g_a = sigmoid_backward(a, z, g_z) # matrix 1x(D+1)
print('g_a')
print(g_a)
# need to modify g_a from 1x(D+1) to 1xD
g_a = np.delete(g_a, 0, axis = 1)
g_alpha, _ = linear_backward(x, alpha, a, g_a) # matrix Dx(M+1)
print('g_alpha')
print(g_alpha)

alpha = np.array(np.subtract(alpha, g_alpha))
print('updated_alpha')
print(alpha)
beta = np.array(np.subtract(beta, g_beta))
print('updated_beta')
print(beta)

a = linear_forward(x, alpha)
z = sigmoid_forward(a)
# add bias for z, fixed
z = np.insert(z, 0, 1)
b = linear_forward(z, beta) # beta: Kx(1+D) mul (1+D)x1 -> b: Kx1
y_pred = softmax_forward(b)
print('y_pred')
print(y_pred)
'''
