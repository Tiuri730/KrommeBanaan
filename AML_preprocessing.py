import pandas as pd
import numpy as np
from sklearn import preprocessing

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

#Merges the weather database with the df
def merge_weather(df, weather):
    df = df.copy()
    weather = weather.copy()
    weather = weather.drop('time_hour', axis = 1)
    
    #Some preprocessing:
    weather['wind_gust'] = weather.groupby(['origin'])['wind_gust'].apply(lambda x: x.interpolate(method = 'spline', order = 2, limit = 10, limit_direction = 'both')) #interpolate windgust values
    weather['wind_gust'] = weather['wind_gust'].apply(lambda x : 0 if x < 0 else x) #remove any gustvalues that turned out to be negative, this might be better of by taking the absolute value (shift in wind direction)
    weather['wind_gust'] = weather['wind_gust'].fillna(-1)
    interpolate_list = ['pressure', 'wind_dir','wind_speed', 'dewp', 'humid','temp','precip','visib']
    for c in interpolate_list:
        weather[c] = weather.groupby(['origin'])[c].apply(lambda x : x.interpolate(limit = 100, limit_direction = 'both')) #Linear interpolation (to get rid of NaNs)
        if c == 'visib':
            avg = weather[c].mean()
            lower = avg -5 * weather[c].std()
            upper= avg + 5 * weather[c].std()
        else:
            avg = weather[c].mean()
            lower = avg -3 * weather[c].std()
            upper= avg + 3 * weather[c].std()
        weather[c] = weather[c].clip(lower, upper, axis = c)
        
    #Normalization:
    norm_cols = ['wind_gust','pressure', 'wind_dir','wind_speed', 'dewp', 'humid','temp','precip','visib']
    min_max_scalar = preprocessing.MinMaxScaler()
    weather[norm_cols] = pd.DataFrame(min_max_scalar.fit_transform(weather[norm_cols].values)) #Replace the norm_cols with the normalized values found by min_max_scalar
    
    df = df.merge(weather, left_on = ['origin','year', 'month', 'day','sched_dep_time'], right_on = ['origin','year', 'month', 'day','hour'], how = 'left')
    new_names = [(i,'o_'+i) for i in df.iloc[:,-(len(weather.columns) - 5):].columns.values]
    df.rename(columns = dict(new_names), inplace=True) #Prefix the columns with 'o_' for origin
    return df

def encode(df, column):
    le = preprocessing.LabelEncoder()
    df[column] = le.fit_transform(df[column])
    return df

#Load all the data
train = loadData('train.csv')
train = train.set_index('id') #Use ID as index
train['sched_dep_time'] = (train['sched_dep_time'] / 100).astype(int) #Prepare hours to align with that of the weather dataset
train['sched_arr_time'] = (train['sched_arr_time'] / 100).astype(int)
weather = loadData('weather.csv')
airports = loadData('airports.csv')
test = loadData('test.csv')
test['sched_dep_time'] = (test['sched_dep_time'] / 100).astype(int) #Prepare hours to align with that of the weather dataset
test['sched_arr_time'] = (test['sched_arr_time'] / 100).astype(int)

airports = airports.set_index('faa')
airports = airports.drop(['name','tzone', 'dst', 'tz'], axis = 1)

#Fix longitude and latitude to not exceed the ranges (data seems to be missing the floating point, increasing it by 1000)
airports['lat'] = airports['lat'].apply(lambda x: float(x)/1000.0 if np.abs(x) > 90 else x)
airports['lon'] = airports['lon'].apply(lambda x: float(x)/1000.0 if np.abs(x) > 180 else x)

#airports = airports['lat'].mask(airports['lat'] > 3*airports.lat.std())
#airports = airports[np.abs(airports.lat - airports.lat.mean() <= 3*airports.lat.std())]
#airports = airports[np.abs(airports.lon - airports.lon.mean() <= 3*airports.lon.std())]
#airports = merge_and_drop(airports, 'dst')

#Merge for origin airport
train = train.merge(airports, left_on = 'origin', right_index = True, how = 'left')
test = test.merge(airports, left_on = 'origin', right_index = True, how = 'left')
new_names = [(i,'o_'+i) for i in train.iloc[:,-len(airports.columns):-1].columns.values]
train.rename(columns = dict(new_names), inplace=True) #Prefix the columns with 'o_' for origin
test.rename(columns = dict(new_names), inplace=True) #Prefix the columns with 'o_' for origin
#Merge for destination airport
train = train.merge(airports, left_on ='dest', right_index = True, how = 'left')
test = test.merge(airports, left_on ='dest', right_index = True, how = 'left')
new_names = [(i,'d_'+i) for i in train.iloc[:,-len(airports.columns):-1].columns.values]
train.rename(columns = dict(new_names), inplace=True) #Prefix the columns with 'd_' for destination
test.rename(columns = dict(new_names), inplace=True) #Prefix the columns with 'd_' for destination

#Merge weather dataset into sets
train = merge_weather(train,weather)
test = merge_weather(test,weather) 

##one-hot encoding of the categorical feature
#train = merge_and_drop(train, 'origin')
#train = merge_and_drop(train, 'dest')
#train = merge_and_drop(train, 'carrier')

cat_cols = ['origin', 'dest', 'carrier']
##Cat encoding of data
for c in cat_cols:
    train = encode(train, c)
    test = encode(test, c)

#Create train set and target set
y = train['is_delayed']
X = train.drop('is_delayed', axis = 1)


#Prepare test set(create/align columns to that of train set)
test = test.set_index('id')
#test = merge_and_drop(test, 'origin')
#test = merge_and_drop(test, 'dest')
#test = merge_and_drop(test, 'carrier')
#test['dest_LGA'] = 0 #Not found in test columns, add colums and set to zero
#test['dest_LEX'] = 0
test = test[X.columns] #Align columns according to train (X)set

X.to_csv('train_X2.csv')
y.to_csv('train_y2.csv', header = True)
test.to_csv('test_extended2.csv')