import pandas as pd

#Loads Data from .csv file and puts it into DataFrame
def loadData(datapath):
    dataset = pd.read_csv(datapath)
    return dataset

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
train = train.set_index('id') #Use ID as index
weather = loadData('weather.csv')
airports = loadData('airports.csv')
test = loadData('test.csv')

#one-hot encoding of the categorical feature
train = merge_and_drop(train, 'origin')
train = merge_and_drop(train, 'dest')
train = merge_and_drop(train, 'carrier')

##TODO: Combine train/weather/airports datasets into 1. train.departure -> airports.index
# then include the weather data for each airport at each time point.

#Create train set and target set
y = train['is_delayed']
X = train.drop('is_delayed', axis = 1)

#Prepare test set(create/align columns to that of train set)
test = test.set_index('id')
test = merge_and_drop(test, 'origin')
test = merge_and_drop(test, 'dest')
test = merge_and_drop(test, 'carrier')
test['dest_LGA'] = 0 #Not found in test columns, add colums and set to zero
test['dest_LEX'] = 0
test = test[X.columns]

X.to_csv('train_X.csv')
y.to_csv('train_y.csv', header = True)
test.to_csv('test_extended.csv')