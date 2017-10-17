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


from NN_one_layer.py import read_image_data


#use stra
def SstratifyDataTrainTest3layerNN():
    data = read_image_data()
    train_x = data[0]
    train_y_integers = data[1]
    test_x = data[2]
    
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)
    
    
    xsplitTrain, xsplitTest, ysplitTrain_integer, ysplitTest_integer = train_test_split(train_x, train_y_integers, test_size=0.2, random_state=0, stratify=train_y_integers)


    mlp = MLPClassifier(hidden_layer_sizes=(30,30,30), activation='tanh', max_iter=500, momentum=0.9, epsilon=1e-8)
    mlp.fit(xsplitTrain,ysplitTrain_integer)
    predyTest = mlp.predict(xsplitTest)

	meanZeroOneLoss = (np.mean(predyTest != ysplitTest_integer))
    print ("x. shape:" , meanZeroOneLoss)

