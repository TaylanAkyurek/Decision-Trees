#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def safelog2(x):
    if x == 0:
        return(0)
    else:
        return(np.log2(x))
    
data_set = np.genfromtxt("/Users/alitaylanakyurek/Downloads/hw04_data_set.csv", delimiter = ",",skip_header=1)



N = data_set.shape[0]
D = data_set.shape[1]


X = data_set[0:,0]
Y = data_set[0:,1].astype(int)

K = np.max(Y)

train_indices = np.arange(0, 150)
test_indices = np.setdiff1d(range(N), train_indices)

x_train = X[train_indices,]
y_train = Y[train_indices,]
x_test = X[test_indices,]
y_test = Y[test_indices,]

N_train = len(y_train)
N_test = len(y_test)

node_indices = {}
is_terminal = {}
need_split = {}

node_features = {}
node_splits = {}
node_frequencies = {}

node_indices[1] = np.array(range(N_train))
is_terminal[1] = False
need_split[1] = True

def decisionTree(P):
    
    # learning algorithm
    while True:
        # find nodes that need splitting
        split_nodes = [key for key, value in need_split.items()
                       if value == True]
        # check whether we reach all terminal nodes
        if len(split_nodes) == 0:
            break
        # find best split positions for all nodes
        for split_node in  split_nodes:
            data_indices = node_indices[split_node]
            need_split[split_node] = False
            node_frequencies[split_node] = [np.sum(y_train[data_indices] == c + 1)
                                            for c in range(K)]
            if data_indices.shape[0] <= P:
                is_terminal[split_node] = True
            else:
                is_terminal[split_node] = False
    
                best_scores = 0.0
                best_splits = 0.0
                unique_values = np.sort(np.unique(x_train[data_indices]))
                split_positions = (unique_values[1:] + unique_values[:-1]) / 2
                split_scores = np.repeat(0.0, len(split_positions))

                for s in range(len(split_positions)):
                    
                    left_indices = data_indices[x_train[data_indices] > split_positions[s]]
                    right_indices = data_indices[x_train[data_indices] <= split_positions[s]]
                    
                    split_scores[s] = -len(left_indices) / len(data_indices) * \
                                        (np.mean(y_train[left_indices]) * \
                                                safelog2(np.mean(y_train[left_indices]))) - \
                                        len(right_indices) / len(data_indices) * \
                                            (np.mean(y_train[right_indices]) * \
                                             safelog2(np.mean(y_train[right_indices])))
                    

                 

                best_scores = np.min(split_scores)
                best_splits = split_positions[np.argmin(split_scores)]
                # decide where to split on which feature
                node_splits[split_node] = best_splits
                
                left_indices = data_indices[x_train[data_indices] <= best_splits]
                
                node_indices[2 * split_node] = left_indices
                is_terminal[2 * split_node] = False
                need_split[2 * split_node] = True
      
                # create right node using the selected split
                right_indices = data_indices[x_train[data_indices] > best_splits]
                node_indices[2 * split_node + 1] = right_indices
                is_terminal[2 * split_node + 1] = False
                need_split[2 * split_node + 1] = True
                y_predicted = np.repeat(0.0, N_test)
    
    
    
    
    
P=25
decisionTree(P)

minimum_value = np.min(x_train)
maximum_value = np.max(x_train)
sorted_splits = np.sort(np.array(list(node_splits.items()))[0:,1])

left_borders = np.append(minimum_value, sorted_splits)
right_borders = np.append(sorted_splits, maximum_value)

g = np.zeros(len(left_borders))


g = np.asarray([np.sum(((left_borders[b] < x_train) & (x_train <= right_borders[b])) * y_train) / np.sum((left_borders[b] < x_train) & (x_train <= right_borders[b])) for b in range(len(left_borders))])


plt.figure(figsize = (10, 8))
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.plot(x_train, y_train,"b.", markersize = 10, label="training")
plt.plot(x_test, y_test,"r.", markersize = 10, label="test")

for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [g[b], g[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [g[b], g[b + 1]], "k-")    
plt.show()


top_sum = 0
rmse = 0


def rmse(x_data, y_data, N_data):
    
    top_sum = 0
    
    for i in range(N_data):
        for b in range(len(left_borders)):
            if (left_borders[b] < x_data[i]) and (x_data[i] <= right_borders[b]):
                top_sum += (y_data[i] - g[b])**2
                
    return math.sqrt(top_sum / N_data) 


print("RMSE on training set is", rmse(x_train, y_train, N_train), "when P is", P)

print("RMSE on test set is", rmse(x_test, y_test, N_test), "when P is", P)



P_array = np.arange(start = 5,
                          stop = 55,
                          step = 5)
rmse_train = []
rmse_test = []


for i in P_array:
    
    
    node_indices = {}
    is_terminal = {}
    need_split = {}
    
    node_features = {}
    node_splits = {}
    node_frequencies = {}
    
    node_indices[1] = np.array(range(N_train))
    is_terminal[1] = False
    need_split[1] = True
    
    decisionTree(i)

    minimum_value = np.min(x_train)
    maximum_value = np.max(x_train)
    sorted_splits = np.sort(np.array(list(node_splits.items()))[0:,1])
    
    left_borders = np.append(minimum_value, sorted_splits)
    right_borders = np.append(sorted_splits, maximum_value)
    
    g = np.zeros(len(left_borders))
    
    
    g = np.asarray([np.sum(((left_borders[b] < x_train) & (x_train <= right_borders[b])) * y_train) / np.sum((left_borders[b] < x_train) & (x_train <= right_borders[b])) for b in range(len(left_borders))])

    
    rmse_train.append(rmse(x_train, y_train, N_train))
    
    rmse_test.append(rmse(x_test, y_test, N_test))


plt.figure(figsize = (10, 10))
plt.plot(P_array, rmse_train, marker = ".", markersize = 10, color = "b",label='train')
plt.plot(P_array, rmse_test, marker = ".", markersize = 10, color = "r",label='test')
plt.xlabel("Pre-pruning size (P)")
plt.ylabel("RMSE")
plt.legend(['training', 'test'])
plt.show()


    
    
