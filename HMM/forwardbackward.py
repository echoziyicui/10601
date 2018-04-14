#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri April 13 14:56:37 2018

@author: ziyicui
"""
################################################################################
# Program:    forwardbackward.py
# Author:     Ziyi (Echo) Cui
# AndrewID:   ziyic
# Data:       4/13/2018
# Usage:      python forwardbackward.py <test input> <index to word>
#            <index to tag> <hmmprior> <hmmemit> <hmmtrans> <predicted file>
################################################################################
import sys
import copy
################################################################################
###test_open()
# test whether the ifle can be opened

def test_open(file):
    try:
        f = open(file, "r")
        f.close()
    except:
        print("Unable to open file" + filepath)
        sys.exit()
################################################################################
###read_as_matrix()
#
def read_as_matrix(file):
    matrix = []
    with open (file, 'r') as f:
        for line in f.readlines():
            if line != '\n':
                line = line.rstrip().split()
                matrix.append(line)
    f.close()
    return matrix
################################################################################
###read_as_matrix_float()
#
def read_as_matrix_float(file):
    matrix = []
    with open (file, 'r') as f:
        for line in f.readlines():
            line = line.rstrip().split()
            matrix.append(list(map(float, line)))
    f.close()
    return matrix
###############################################################################
###read_as_list()
#
def read_as_list(file):
    list = []
    with open (file, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            list.append(line)
    f.close()
    return list
################################################################################
###convert_2_index()
def convert_2_index(train_set, index_2_word, index_2_tag):
    train_set_with_indices = copy.deepcopy(train_set)
    for i, set in enumerate(train_set):
        for j, word_tag in enumerate(set):
            indexw_indext = []
            word_tag = word_tag.split('_')
            indexw_indext.append(index_2_word.index(word_tag[0]))
            indexw_indext.append(index_2_tag.index(word_tag[1]))
            train_set_with_indices[i][j] = indexw_indext
    return train_set_with_indices
################################################################################
### prediction()
#
def prediction(test_set_with_indices, k, A, B, C, output, index_2_word, index_2_tag):
    f = open(output, 'w')
    for sequence in test_set_with_indices:
        predicted_tag = predict_each_sequence(sequence, k, A, B, C)
        for i, token in enumerate(sequence):
            word_idx = token[0]
            predicted_tag_idx = predicted_tag[i]
            word_tag = index_2_word[word_idx] +'_'+ index_2_tag[predicted_tag_idx]
            f.write('%s ' %word_tag)
        f.write('\n')
    f.close()
################################################################################
### compute_prob_matrix()
# compute matrix for each sequence

def predict_each_sequence(sequence, k, A, B, C):
    predicted_tag = []
    T = len(sequence)
    tmp1 = 0
    tmp2 = 0
    # matrix : T x k, each entry = alpha * beta
    alpha_matrix = [[0]*k for i in range(T)]
    beta_matrix = [[0]*k for i in range(T)]
    matrix = [[0]*k for i in range(T)]

    # forward
    for t in range(T):
        word = sequence[t][0]
        if t == 0:
            for j in range(k):
                alpha_matrix[t][j] = C[j]*(B[j][word])
        else:
            for j in range(k):
                alpha_matrix[t][j] = (B[j][word])*tmp1
        tmp1 = sum([alpha_matrix[t][i]*A[i][j] for i in range(k)])

    # backward
    for t in (reversed(range(T))):
        word = sequence[t][0]
        if t == T-1:
            for j in range(k):
                beta_matrix[t][j] = 1
        else:
            for j in range(k):
                beta_matrix[t][j] = tmp2
        tmp2 = sum([B[i][word]*beta_matrix[t][i]*A[j][i] for i in range(k)])

    for t in range(T):
        for j in range(k):
            matrix[t][j] = alpha_matrix[t][j] * beta_matrix[t][j]

    predicted_tag = [seq.index(max(seq)) for seq in matrix]
    return predicted_tag
################################################################################
###main_program:
#

### read files
#<test input> <index to word> <index to tag> <hmmprior> <hmmemit> <hmmtrans>
#<predicted file>
# state: tag
# observation: word
test_set = read_as_matrix(sys.argv[1])
index_2_word = read_as_list(sys.argv[2])
index_2_tag = read_as_list(sys.argv[3])

# A: transition matrix, B: emission_matrix, C: initial prior
C = list(map(float, read_as_list(sys.argv[4])))
A = read_as_matrix_float(sys.argv[6])
B = read_as_matrix_float(sys.argv[5])
# each: [observation, state]
test_set_with_indices = convert_2_index(test_set, index_2_word, index_2_tag)
# num of tag
k = len(index_2_tag)

# test
output = sys.argv[7]
prediction(test_set_with_indices, k, A, B, C, output, index_2_word, index_2_tag)