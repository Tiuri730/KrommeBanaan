import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib as plt
import sklearn
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate



#Loads Data from .csv file and puts it into DataFrame
def loadData(datapath):
    dataset = pd.read_csv(datapath)
    return dataset

#Stratified K-fold cross validation (using ROC AUC)
def crossVal(model, X, y, n):
    skf = StratifiedKFold(n_splits = n )
    scores = cross_validate(model, X, y, scoring = 'roc_auc', cv = skf, return_train_score = True)

    return scores

#Create one-hot encoding of a column
def one_hot(column, prefix):
    df = pd.get_dummies(column, prefix = prefix)
    return df

#Merge a one hot encoded column into dataframe df and drop the original column
def merge_and_drop(df, column):
    oh = one_hot(df[column], column)
    new_df = df.drop(column, axis = 1)
    new_df = pd.concat([new_df, oh], axis = 1)
    return new_df

#Load all the data
train = loadData('train.csv')
train = train.set_index('id')
weather = loadData('weather.csv')
airports = loadData('airports.csv')
test = loadData('test.csv')

#one-hot encoding of the categorical feature
train = merge_and_drop(train, 'origin')
train = merge_and_drop(train, 'dest')
train = merge_and_drop(train, 'carrier')


#sum_df = df.sum(axis = 0)
#sum_df.plot(kind = 'pie' )

##TODO: Encode the categorical values (all one-hot?)

##TODO: Combine train/weather/airports datasets into 1. train.departure -> airports.index
# then include the weather data for each airport at each time point.


y = train['is_delayed']
X = train.drop('is_delayed', axis = 1)

xgb_model = XGBClassifier(silent = False)

cross_val_scores = crossVal(xgb_model, X, y, 10)