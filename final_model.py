# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 13:14:22 2018

@author: Jorik
"""

import pandas as pd
import xgboost as xgb

#Load data
def load_data():
    X = pd.read_csv('train_X2.csv', header = 0, index_col = 0)
    y = pd.read_csv('train_y2.csv', header = 0, index_col = 0)
    test = pd.read_csv('test_extended2.csv', header = 0, index_col = 0)
    return X, y, test

#Get data
X, y, test = load_data()

#Setup a parameter grid to search
params = {
            'max_depth': [10],
            'learning_rate': [0.1],
            'n_estimators': [600],
            'min_child_weight': [5],
            'subsample': [0.8],
            'gamma': [2]
            }
#Create XGB model
xgbclf = xgb.XGBClassifier(max_depth= 5,learning_rate= 0.1,n_estimators= 600,min_child_weight= 10,gamma= 0.5,subsample= 0.8, objective = 'binary:logistic', silent = True, nthread = -2)
xgbclf.fit(X,y)
y_hat = xgbclf.predict_proba(test)

#Prepare results for submission
y_hat = pd.DataFrame(y_hat).drop(0, axis = 1) #Drop first column, as this predicts when flight is NOT delayed, we are only interested in flights that are delayed
y_hat['id']= test.index #Add the test id's
y_hat = y_hat.set_index('id') #Set index to be the id's
y_hat = y_hat.rename(columns = {1 : 'is_delayed'}) #Rename the only column to 'is_delayed'
y_hat.to_csv('results_jsn235_jor250.csv')