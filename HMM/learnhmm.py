#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri April 13 14:56:37 2018

@author: ziyicui
"""
################################################################################
# Program:    learnhmm.py
# Author:     Ziyi (Echo) Cui
# AndrewID:   ziyic
# Data:       4/13/2018
# Usage:      python learnhmm.py <train input> <index to word> <index to tag>
#             <hmmprior> <hmmemit> <hmmtrans>
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
###############################################################################
###read_as_list()
#
def read_as_list(file):
    list = []
    with open (file, 'r') as f:
        for line in f.readlines():
            if line != '\n':
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
###learn_initialization()
def learn_initialization(train_set,num_tag,hmmprior_file):
    pi = [1]*num_tag
    num_train_set = len(train_set)
    for set in train_set:
        # set[0] is the first word_tag, set[0][1] is the first tag
        pi[set[0][1]] += 1
    total = sum(pi)
    pi = [float(i)/total for i in pi]
    hmmprior = open(hmmprior_file, 'w')
    hmmprior.writelines('%s\n' % item for item in pi)
    hmmprior.close()
    return
################################################################################
###learn_transition()
#
def learn_transition(train_set,num_tag, hmmtrans_file):
    transition_matrix = [[1]*num_tag for i in range(num_tag)]
    hmmtrans = open(hmmtrans_file,'w')

    for sequence in train_set:
        for i in range(len(sequence)-1):
            current_tag = sequence[i][1]
            next_tag = sequence[i+1][1]
            transition_matrix[current_tag][next_tag] += 1


    for i, row in enumerate(transition_matrix):
        count = sum(row)
        for j, element in enumerate(row):
            transition_matrix[i][j] = float(element)/float(count)
        hmmtrans.writelines('%s ' % item for item in row)
        hmmtrans.write('\n')
    hmmtrans.close()

################################################################################
###learn_emission()
#
def learn_emission(tran_set, num_word, num_tag, hmmemit_file):
    emission_matrix = [[1]*num_word for i in range(num_tag)]
    hmmemit = open(hmmemit_file, 'w')

    for sequence in tran_set:
        for word_tag in sequence:
            word = word_tag[0]
            tag = word_tag[1]
            emission_matrix[tag][word] += 1

    for i, row in enumerate(emission_matrix):
        count = sum(row)
        for j, element in enumerate(row):
            emission_matrix[i][j] = float(element)/float(count)
        hmmemit.writelines('%s ' % item for item in row)
        hmmemit.write('\n')
    hmmemit.close()
################################################################################
###main program:
#
test_open(sys.argv[1])


### read files
#<train input> <index to word> <index to tag> <hmmprior> <hmmemit> <hmmtrans>
# state: tag
# observation: word
train_set = read_as_matrix(sys.argv[1])
index_2_word = read_as_list(sys.argv[2])
index_2_tag = read_as_list(sys.argv[3])


# each: [observation, state]
train_set_with_indices = convert_2_index(train_set, index_2_word, index_2_tag)
num_tag = len(index_2_tag)
num_word = len(index_2_word)

### learn parameters
hmmprior_file = sys.argv[4]
hmmtrans_file = sys.argv[5]
hmmemit_file = sys.argv[6]
learn_initialization(train_set_with_indices, num_tag,hmmprior_file)
learn_transition(train_set_with_indices,num_tag, hmmtrans_file)
learn_emission(train_set_with_indices, num_word, num_tag, hmmemit_file)
