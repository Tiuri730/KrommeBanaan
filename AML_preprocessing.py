import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib as plot
import sklearn
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

from util.VisualizeDataset2 import VisualizeDataset
from PrepareDatasetForLearning import PrepareDatasetForLearning
from LearningAlgorithms import ClassificationAlgorithms
from Evaluation import ClassificationEvaluation
from FeatureSelection import FeatureSelectionClassification
import copy
from util import util
import numpy as np
from sklearn.model_selection import train_test_split
import os

#Vizualize DataSet
DataViz = VisualizeDataset()

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


#convert times
train.loc[train['sched_dep_time'] <=100,'sched_dep_time'] = 0
train.loc[(train['sched_dep_time'] >100) & (train['sched_dep_time'] <=200),'sched_dep_time'] = 1
train.loc[(train['sched_dep_time'] >200) & (train['sched_dep_time'] <=300),'sched_dep_time'] = 2
train.loc[(train['sched_dep_time'] >300) & (train['sched_dep_time'] <=400),'sched_dep_time'] = 3
train.loc[(train['sched_dep_time'] >400) & (train['sched_dep_time'] <=500),'sched_dep_time'] = 4
train.loc[(train['sched_dep_time'] >500) & (train['sched_dep_time'] <=600),'sched_dep_time'] = 5
train.loc[(train['sched_dep_time'] >600) & (train['sched_dep_time'] <=700),'sched_dep_time'] = 6
train.loc[(train['sched_dep_time'] >700) & (train['sched_dep_time'] <=800),'sched_dep_time'] = 7
train.loc[(train['sched_dep_time'] >800) & (train['sched_dep_time'] <=900),'sched_dep_time'] = 8
train.loc[(train['sched_dep_time'] >900) & (train['sched_dep_time'] <=1000),'sched_dep_time'] = 9
train.loc[(train['sched_dep_time'] >1000) & (train['sched_dep_time'] <=1100),'sched_dep_time'] = 10
train.loc[(train['sched_dep_time'] >1100) & (train['sched_dep_time'] <=1200),'sched_dep_time'] = 11
train.loc[(train['sched_dep_time'] >1200) & (train['sched_dep_time'] <=1300),'sched_dep_time'] = 12
train.loc[(train['sched_dep_time'] >1300) & (train['sched_dep_time'] <=1400),'sched_dep_time'] = 13
train.loc[(train['sched_dep_time'] >1400) & (train['sched_dep_time'] <=1500),'sched_dep_time'] = 14
train.loc[(train['sched_dep_time'] >1500) & (train['sched_dep_time'] <=1600),'sched_dep_time'] = 15
train.loc[(train['sched_dep_time'] >1600) & (train['sched_dep_time'] <=1700),'sched_dep_time'] = 16
train.loc[(train['sched_dep_time'] >1700) & (train['sched_dep_time'] <=1800),'sched_dep_time'] = 17
train.loc[(train['sched_dep_time'] >1800) & (train['sched_dep_time'] <=1900),'sched_dep_time'] = 18
train.loc[(train['sched_dep_time'] >1900) & (train['sched_dep_time'] <=2000),'sched_dep_time'] = 19
train.loc[(train['sched_dep_time'] >2000) & (train['sched_dep_time'] <=2100),'sched_dep_time'] = 20
train.loc[(train['sched_dep_time'] >2100) & (train['sched_dep_time'] <=2200),'sched_dep_time'] = 21
train.loc[(train['sched_dep_time'] >2200) & (train['sched_dep_time'] <=2300),'sched_dep_time'] = 22
train.loc[(train['sched_dep_time'] >2300) & (train['sched_dep_time'] <=2400),'sched_dep_time'] = 23

#merge both dataframes
train = pd.merge(train, weather,  how='left', left_on=['origin','year','month','day','sched_dep_time'], right_on = ['origin','year','month','day','hour'])
train = train.drop('time_hour', axis=1)
train = pd.merge(train, airports, how='left', left_on=['dest'], right_on=['faa'])
train = train.drop('faa', axis=1)
train = train.drop('name', axis=1)
train = train.drop('dst', axis=1)
train = train.drop('tzone', axis=1)

#one-hot encoding of the categorical feature
train = merge_and_drop(train, 'origin')
train = merge_and_drop(train, 'dest')
train = merge_and_drop(train, 'carrier')

##prepare dataset for learning
#prepare = PrepareDatasetForLearning()
#train_X, test_X, train_y, test_y = prepare.split_single_dataset_classification(train, ['is_delayed'],"",0.7, filter=False, temporal=False)
#
#print('Training set length is: ', len(train_X.index))
#print('Test set length is: ', len(test_X.index))
#
## First, let us consider the performance over a selection of features:
#
#fs = FeatureSelectionClassification()
#
#features, ordered_features, ordered_scores = fs.forward_selection(50, train_X, train_y)
#print(ordered_scores)
#print(ordered_features)
#
#plot.plot(range(1, 51), ordered_scores)
#plot.xlabel('number of features')
#plot.ylabel('accuracy')
#plot.show()

#sum_df = df.sum(axis = 0)
#sum_df.plot(kind = 'pie' )

##TODO: Encode the categorical values (all one-hot?)

##TODO: Combine train/weather/airports datasets into 1. train.departure -> airports.index
# then include the weather data for each airport at each time point.


y = train['is_delayed']
X = train.drop('is_delayed', axis = 1)

xgb_model = XGBClassifier(silent = False)

cross_val_scores = crossVal(xgb_model, X, y, 10)