#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:24:27 2017

@author: fubao
"""

# for extra credit;
#three hidden layer


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier

from NN_one_layer import read_image_data
import kaggle
import time

#use stra
def stratifyDataTrainTest3layerNN():
    data = read_image_data()
    train_x = data[0]
    train_y_integers = data[1]
    test_x = data[2]
    
    #normalize
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.fit_transform(test_x)
    
    print ("train_x. shape:" , train_x.shape)

    #split
    xsplitTrain, xsplitTest, ysplitTrain_integer, ysplitTest_integer = train_test_split(train_x, train_y_integers, test_size=0.2, random_state=0, stratify=train_y_integers)

    hidden_layer_sizes_lst = [(5,5,5), (10,10,10), (40,40,40), (70,70,70)]
    
    smalleAccuracy = 2**32
    best_hidden_layer_size = None
    for hidden_layer_sizes in hidden_layer_sizes_lst:
        beginTime = time.time()
        mlp = MLPClassifier(hidden_layer_sizes, activation='tanh', max_iter=500, momentum=0.9, epsilon=1e-8)
        mlp.fit(xsplitTrain,ysplitTrain_integer)
        #pred = mlp.predict(xsplitTest)              #predict validation set
        meanAccuracy = mlp.score(xsplitTest, ysplitTest_integer)
        if meanAccuracy < smalleAccuracy:
            smalleAccuracy = meanAccuracy
            best_hidden_layer_size = hidden_layer_sizes
        
        print ("SstratifyDataTrainTest3layerNN. smalleAccuracy:" , time.time()-beginTime, meanAccuracy)

    print ("SstratifyDataTrainTest3layerNN. smalleAccuracy:" , smalleAccuracy, best_hidden_layer_size)
    #train and test the whole data
    mlp = MLPClassifier(best_hidden_layer_size, activation='tanh', max_iter=500, momentum=0.9, epsilon=1e-8)
    mlp.fit(train_x, train_y_integers)
    predyTest = mlp.predict(test_x)

    #output to file
    fileTestOutput3LayerNN = "../Predictions/best_3HiddenNN.csv"
    if fileTestOutput3LayerNN != "":
        kaggle.kaggleize(predyTest, fileTestOutput3LayerNN)    
    
    
if __name__== "__main__": 
    stratifyDataTrainTest3layerNN()