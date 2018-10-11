#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 12:30:25 2018

@author: york
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib as plt
import sklearn
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
from imblearn.over_sampling import SMOTE

#Load data
def load_data():
    X = pd.read_csv('train_X.csv', header = 0, index_col = 0)
    y = pd.read_csv('train_y.csv', header = 0, index_col = 0)
    test = pd.read_csv('test_extended.csv', header = 0, index_col = 0)
    return X, y, test

#Stratified K-fold cross validation (using ROC AUC)
def crossVal(model, X, y, n):
    skf = StratifiedKFold(n_splits = n )
    scores = cross_validate(model, X, y, scoring = 'roc_auc', cv = skf, return_train_score = True)
    return scores

#Stratified Kfold cross validation with SMOTE
def crossVal_smote(model, X, y, splits):
    skf = StratifiedKFold(n_splits=splits, shuffle = True)
    scores = []
    
    for train_i, test_i in skf.split(X,y):
        s = get_smote_training_scores(model, X.iloc[train_i], y.iloc[train_i], X.iloc[test_i], y.iloc[test_i])
        scores.append(s)
    return scores

#Train a model on synthetically augmented data, using SMOTE
def get_smote_training_scores(model, X, y, X_test, y_test):
    X_s, y_s = smote(X,y)
    m = model.fit(X_s,y_s)
    y_hat = m.predict(X_test)
    #score = matthews_corrcoef(y_test, y_hat) #MCC score
    score = roc_auc_score(y_test, y_hat) #ROC AUC score
    return score

#Oversample the minority class
def smote(X,y):
    sm = SMOTE(random_state = 42, ratio = 'all', k_neighbors = 2)
    X_s, y_s = sm.fit_sample(X, y) #Oversample to 1:1
    x_synth = pd.DataFrame(X_s, columns = X.columns) #Rename columns
    y_synth = pd.Series(y_s)
    return x_synth, y_synth

#Train on all the data and predict test set
def predict_prob(model, X, y, test):
    #Train XGBoost model
    xgb_model = XGBClassifier(silent = False)
    xgb_model.fit(X, y)
    
    #Predict test
    y_hat = xgb_model.predict_proba(test)
    return y_hat
    
#Get data
X, y, test = load_data()

#Setup a predictive model
xgb_model = XGBClassifier(silent = False)

#Cross validation method
#cross_val_scores = crossVal(xgb_model, X, y, 10)
cross_val_scores_smote = crossVal_smote(xgb_model, X, y, 10)

#Prepare results for submission
#y_hat = predict_prob(xgb_model,X,y,test)
#y_hat = pd.DataFrame(y_hat).drop(0, axis = 1) #Drop first column, as this predicts when flight is NOT delayed, we are only interested in flights that are delayed
#y_hat['id']= test.index #Add the test id's
#y_hat = y_hat.set_index('id') #Set index to be the id's
#y_hat = y_hat.rename(columns = {1 : 'is_delayed'}) #Rename the only column to 'is_delayed'
#y_hat.to_csv('results_jsn235_jor250.csv')