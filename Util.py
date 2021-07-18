#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
import numpy as np
from torchvision import datasets, transforms
from numpy.random import choice


def Sample_node(k, node_idx, list_of_candidates, probability_distribution):
    
    fintuned_node_sets = []
    set_length = k

    while(len(fintuned_node_sets) < set_length):
        draw = choice(list_of_candidates, 1,
                  p=probability_distribution)
        if draw not in fintuned_node_sets:
            fintuned_node_sets.append(int(draw))

    return fintuned_node_sets


def FedAvg(w,alpha):
    w_avg = w[0]
    n_clients = len(w)
    
    alpha = alpha/np.sum(alpha)
    #print(np.sum(alpha))
    #print(alpha)
    #alpha = np.random.uniform(0,1,n_clients)
    
    for l in w_avg.keys():
        w_avg[l] = w_avg[l] - w_avg[l]

    for l, layer in enumerate(w_avg.keys()): #for each layer
        w_kl = []
        for k in range(0,n_clients): #for each client
            w_avg[layer] += alpha[k]*w[k][layer]
    return w_avg


def Preparedata(dataset_train, dataset_test, ntrain, ntest, class_number, percent):
    ntrain_major = int((ntrain*percent)/len(class_number))
    ntest_major = int((ntest*percent)/len(class_number))
    ntrain_minor = ntrain-len(class_number)*ntrain_major
    ntest_minor = ntest-len(class_number)*ntest_major

    train_set = []
    test_set = []

    iter_train = 0
    iter_test = 0
    iter_train_major = [0]*len(class_number)
    iter_test_major = [0]*len(class_number)
    iter_train_minor = 0
    iter_test_minor = 0



    for (images, labels)  in dataset_train:
        if (iter_train < ntrain):
            for i in range(len(class_number)):
                if (labels == class_number[i] and iter_train_major[i] < ntrain_major):
                    train_set.append((images, labels))
                    iter_train_major[i] = iter_train_major[i] + 1
                    iter_train = iter_train + 1
            if (labels not in class_number and iter_train_minor < ntrain_minor ):
                train_set.append((images, labels))
                iter_train_minor = iter_train_minor + 1
                iter_train = iter_train + 1
        else:
            break

    for (images, labels)  in dataset_test:
        if (iter_test < ntest):
            for i in range(len(class_number)):
                if (labels == class_number[i] and iter_test_major[i] < ntest_major):
                    test_set.append((images, labels))
                    iter_test_major[i] = iter_test_major[i] + 1
                    iter_test = iter_test + 1
            if (labels not in class_number and iter_test_minor < ntest_minor):
                test_set.append((images, labels))
                iter_test_minor = iter_test_minor + 1
                iter_test = iter_test + 1
        else:
            break

    return train_set, test_set

