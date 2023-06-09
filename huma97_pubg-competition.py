import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt  

from timeit import default_timer as timer
from sklearn import preprocessing

from sklearn.neural_network import MLPRegressor


import gc, sys
gc.enable()
def state(message,start = True, time = 0):
    if(start):
        print('Working on {}... '.format(message))
    else :
        print('Working on {} took ({}) Sec \n'.format(message,round(time , 3)))
def feature_engineering(is_train=True):
    # When this function is used for the training data, load train_V2.csv :
    if is_train: 
        print("processing train_V2.csv")
        df = pd.read_csv('../input/train_V2.csv')
        
        # Only take the samples with matches that have more than 1 player 
        # there are matches with no players or just one player ( those samples could affect our model badly) 
        df = df[df['maxPlace'] > 1]
    
    # When this function is used for the test data, load test_V2.csv :
    else:
        print("processing test_V2.csv")
        df = pd.read_csv('../input/test_V2.csv')
        
    # Make a new feature indecating the total distance a player cut :
    state('totalDistance')
    s = timer()
    df['totalDistance'] = df['rideDistance'] + df["walkDistance"] + df["swimDistance"]
    e = timer()
    state('totalDistance', False, e - s)
          

    state('rankPoints')
    s = timer()
    # Process the 'rankPoints' feature by replacing any value of (-1) to be (0) :
    df['rankPoints'] = np.where(df['rankPoints'] <= 0 ,0 , df['rankPoints'])
    e = timer()                                  
    state('rankPoints', False, e-s)
    

    target = 'winPlacePerc'
    # Get a list of the features to be used
    features = list(df.columns)
    
    # Remove some features from the features list :
    features.remove("Id")
    features.remove("matchId")
    features.remove("groupId")
    features.remove("matchDuration")
    features.remove("matchType")
    
    y = None
    
    # If we are processing the training data, process the target
    # (group the data by the match and the group then take the mean of the target) 
    if is_train: 
        y = np.array(df.groupby(['matchId','groupId'])[target].agg('mean'), dtype=np.float64)
        # Remove the target from the features list :
        features.remove(target)
    
    # Make new features indicating the mean of the features ( grouped by match and group ) :
    print("get group mean feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('mean')
    # Put the new features into a rank form ( max value will have the highest rank)
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    
    
    # If we are processing the training data let df_out = the grouped  'matchId' and 'groupId'
    if is_train: df_out = agg.reset_index()[['matchId','groupId']]
    # If we are processing the test data let df_out = 'matchId' and 'groupId' without grouping 
    else: df_out = df[['matchId','groupId']]
    
    # Merge agg and agg_rank (that we got before) with df_out :
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_mean", "_mean_rank"], how='left', on=['matchId', 'groupId'])
    
    # Make new features indicating the max value of the features for each group ( grouped by match )
    print("get group max feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('max')
    # Put the new features into a rank form ( max value will have the highest rank)
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    
    # Merge the new (agg and agg_rank) with df_out :
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_max", "_max_rank"], how='left', on=['matchId', 'groupId'])
    
    # Make new features indicating the minimum value of the features for each group ( grouped by match )
    print("get group min feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('min')
    # Put the new features into a rank form ( max value will have the highest rank)
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    
    # Merge the new (agg and agg_rank) with df_out :
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_min", "_min_rank"], how='left', on=['matchId', 'groupId'])
    
    # Make new features indicating the number of players in each group ( grouped by match )
    print("get group size feature")
    agg = df.groupby(['matchId','groupId']).size().reset_index(name='group_size')
     
    # Merge the group_size feature with df_out :
    df_out = df_out.merge(agg, how='left', on=['matchId', 'groupId'])
    
    # Make new features indicating the mean value of each features for each match :
    print("get match mean feature")
    agg = df.groupby(['matchId'])[features].agg('mean').reset_index()
    
    # Merge the new agg with df_out :
    df_out = df_out.merge(agg, suffixes=["", "_match_mean"], how='left', on=['matchId'])
    
    # Make new features indicating the number of groups in each match :
    print("get match size feature")
    agg = df.groupby(['matchId']).size().reset_index(name='match_size')
    
    # Merge the match_size feature with df_out :
    df_out = df_out.merge(agg, how='left', on=['matchId'])
    
    # Drop matchId and groupId
    df_out.drop(["matchId", "groupId"], axis=1, inplace=True)
    
    # X is the output dataset (without the target) and y is the target :
    X = np.array(df_out, dtype=np.float64)
    
    
    del df, df_out, agg, agg_rank
    gc.collect()

    return X, y
df_test = pd.read_csv('../input/test_V2.csv')
# Process the training data :
x_train, y = feature_engineering(True)
# Scale the data to be in the range (-1 , 1)
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False).fit(x_train)
print("x_train", x_train.shape, x_train.max(), x_train.min())
scaler.transform(x_train)
print("x_train", x_train.shape, x_train.max(), x_train.min())
y = y * 2 - 1
print("y", y.shape, y.max(), y.min())
epoch_train = 30
mlp = MLPRegressor(hidden_layer_sizes=(60, 60, 28), alpha=0.001, learning_rate='adaptive', max_iter=epoch_train)
mlp.fit(x_train,y)

del x_train, y
gc.collect()
x_test, _ = feature_engineering(False)
scaler.transform(x_test)
print("x_test", x_test.shape, x_test.max(), x_test.min())
np.clip(x_test, out=x_test, a_min=-1, a_max=1)
print("x_test", x_test.shape, x_test.max(), x_test.min())
pred = mlp.predict(x_test)
del x_test
gc.collect()
pred = pred.reshape(-1)
pred = (pred + 1) / 2
print("fix winPlacePerc")
for i in range(len(df_test)):
    winPlacePerc = pred[i]
    maxPlace = int(df_test.iloc[i]['maxPlace'])
    if maxPlace == 0:
        winPlacePerc = 0.0
    elif maxPlace == 1:
        winPlacePerc = 1.0
    else:
        gap = 1.0 / (maxPlace - 1)
        winPlacePerc = round(winPlacePerc / gap) * gap
    
    if winPlacePerc < 0: winPlacePerc = 0.0
    if winPlacePerc > 1: winPlacePerc = 1.0    
    pred[i] = winPlacePerc
df_test['winPlacePerc'] = pred
submission = df_test[['Id', 'winPlacePerc']]
submission.to_csv("pubg-submission.csv", index=False)
