#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Data Preprossing
#Importing the libraries

import numpy as np #for mathematical functions
import matplotlib.pyplot as plt #for plotting graphs
import pandas as pd #importing and managing datasets

#Importing the dataset using pandas
dataset = pd.read_csv('Data.csv') 

#matrix of features
X = dataset.iloc[:, :-1].values  #[rows, all columns except last one]

#Dependant variable vector
Y = dataset.iloc[:, 3].values

#Taking care of missing data
#Replace all missing data with mean of columns
from sklearn.preprocessing import Imputer 
#imputer object
imputer = Imputer(missing_values = np.nan, strategy = "mean", axis = 0)
imputer = imputer.fit(X[:, 1:3])  #fitting columns required into imputer object
X[:, 1:3] = imputer.transform(X[:, 1:3])    #changing the values with mean

#Encoding categorical data
''' Categorical data: Values are only from a finite set of variables '''
#encode categorical data with numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder() #create object
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) #encode and assign to first col
#This is only for independant variables
onehotencoder = OneHotEncoder(categorical_features = [0]) #dummy encoding
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder() #create object
Y = labelencoder_Y.fit_transform(Y)

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)