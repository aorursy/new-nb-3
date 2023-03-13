import gc

import os

import logging

import datetime

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import lightgbm as lgb

from tqdm import tqdm_notebook

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import StratifiedKFold

from scipy import signal

from sklearn.decomposition import FastICA, PCA

warnings.filterwarnings('ignore')
IS_LOCAL = False

if(IS_LOCAL):

    PATH="../input/dont-overfit-2/"

else:

    PATH="../input/"

os.listdir(PATH)

train_df = pd.read_csv(os.path.join(PATH,"train.csv"))

test_df = pd.read_csv(os.path.join(PATH,"test.csv"))
train_df.shape, test_df.shape
train_df.head()
test_df.head()
def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return(np.transpose(tt))

missing_data(train_df)

missing_data(test_df)

train_df.describe()

test_df.describe()
sns.countplot(train_df['target'])
print("There are {}% target values with 1".format(100 * train_df["target"].value_counts()[1]/train_df.shape[0]))
t0 = train_df.loc[train_df['target'] == 0, train_df.columns.values[2:302]]

t1 = train_df.loc[train_df['target'] == 1, train_df.columns.values[2:302]]

plt.subplots(1,1,figsize=(20, 7.2))

plt.title("Values in training set for target = 0")

sns.heatmap(t0, cmap="Spectral")

plt.show()
plt.subplots(1,1,figsize=(20, 12.8))

plt.title("Values in training set for target = 1")

sns.heatmap(t1, cmap='Spectral')

plt.show()
def plot_feature_distribution(df1, df2, label1, label2, features):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(10,10,figsize=(18,22))



    for feature in features:

        i += 1

        plt.subplot(10,10,i)

        sns.kdeplot(df1[feature], bw=0.5,label=label1)

        sns.kdeplot(df2[feature], bw=0.5,label=label2)

        plt.xlabel(feature, fontsize=9)

        locs, labels = plt.xticks()

        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)

        plt.tick_params(axis='y', which='major', labelsize=6)

    plt.show();
t0 = train_df.loc[train_df['target'] == 0]

t1 = train_df.loc[train_df['target'] == 1]

features = train_df.columns.values[2:102]

plot_feature_distribution(t0, t1, '0', '1', features)
features = train_df.columns.values[102:202]

plot_feature_distribution(t0, t1, '0', '1', features)
features = train_df.columns.values[202:302]

plot_feature_distribution(t0, t1, '0', '1', features)
features = train_df.columns.values[2:102]

plot_feature_distribution(train_df, test_df, 'train', 'test', features)
features = train_df.columns.values[102:202]

plot_feature_distribution(train_df, test_df, 'train', 'test', features)
features = train_df.columns.values[202:302]

plot_feature_distribution(train_df, test_df, 'train', 'test', features)
plt.figure(figsize=(16,6))

features = train_df.columns.values[2:302]

plt.title("Distribution of mean values per row in the train and test set")

sns.distplot(train_df[features].mean(axis=1),color="green", kde=True,bins=120, label='train')

sns.distplot(test_df[features].mean(axis=1),color="blue", kde=True,bins=120, label='test')

plt.legend()

plt.show()
plt.figure(figsize=(16,6))

plt.title("Distribution of mean values per column in the train and test set")

sns.distplot(train_df[features].mean(axis=0),color="magenta",kde=True,bins=120, label='train')

sns.distplot(test_df[features].mean(axis=0),color="darkblue", kde=True,bins=120, label='test')

plt.legend()

plt.show()
plt.figure(figsize=(16,6))

plt.title("Distribution of std values per row in the train and test set")

sns.distplot(train_df[features].std(axis=1),color="black", kde=True,bins=120, label='train')

sns.distplot(test_df[features].std(axis=1),color="red", kde=True,bins=120, label='test')

plt.legend();plt.show()
plt.figure(figsize=(16,6))

plt.title("Distribution of std values per column in the train and test set")

sns.distplot(train_df[features].std(axis=0),color="blue",kde=True,bins=120, label='train')

sns.distplot(test_df[features].std(axis=0),color="green", kde=True,bins=120, label='test')

plt.legend(); plt.show()
t0 = train_df.loc[train_df['target'] == 0]

t1 = train_df.loc[train_df['target'] == 1]

plt.figure(figsize=(16,6))

plt.title("Distribution of mean values per row in the train set")

sns.distplot(t0[features].mean(axis=1),color="red", kde=True,bins=120, label='target = 0')

sns.distplot(t1[features].mean(axis=1),color="blue", kde=True,bins=120, label='target = 1')

plt.legend(); plt.show()
plt.figure(figsize=(16,6))

plt.title("Distribution of mean values per column in the train set")

sns.distplot(t0[features].mean(axis=0),color="green", kde=True,bins=120, label='target = 0')

sns.distplot(t1[features].mean(axis=0),color="darkblue", kde=True,bins=120, label='target = 1')

plt.legend(); plt.show()
plt.figure(figsize=(16,6))

features = train_df.columns.values[2:302]

plt.title("Distribution of min values per row in the train and test set")

sns.distplot(train_df[features].min(axis=1),color="red", kde=True,bins=120, label='train')

sns.distplot(test_df[features].min(axis=1),color="orange", kde=True,bins=120, label='test')

plt.legend()

plt.show()
plt.figure(figsize=(16,6))

features = train_df.columns.values[2:302]

plt.title("Distribution of min values per column in the train and test set")

sns.distplot(train_df[features].min(axis=0),color="magenta", kde=True,bins=120, label='train')

sns.distplot(test_df[features].min(axis=0),color="darkblue", kde=True,bins=120, label='test')

plt.legend()

plt.show()
plt.figure(figsize=(16,6))

features = train_df.columns.values[2:302]

plt.title("Distribution of max values per row in the train and test set")

sns.distplot(train_df[features].max(axis=1),color="brown", kde=True,bins=120, label='train')

sns.distplot(test_df[features].max(axis=1),color="lightblue", kde=True,bins=120, label='test')

plt.legend()

plt.show()
plt.figure(figsize=(16,6))

features = train_df.columns.values[2:302]

plt.title("Distribution of max values per column in the train and test set")

sns.distplot(train_df[features].max(axis=0),color="blue", kde=True,bins=120, label='train')

sns.distplot(test_df[features].max(axis=0),color="red", kde=True,bins=120, label='test')

plt.legend()

plt.show()
t0 = train_df.loc[train_df['target'] == 0]

t1 = train_df.loc[train_df['target'] == 1]

plt.figure(figsize=(16,6))

plt.title("Distribution of min values per row in the train set")

sns.distplot(t0[features].min(axis=1),color="red", kde=True,bins=120, label='target = 0')

sns.distplot(t1[features].min(axis=1),color="blue", kde=True,bins=120, label='target = 1')

plt.legend(); plt.show()
plt.figure(figsize=(16,6))

plt.title("Distribution of min values per column in the train set")

sns.distplot(t0[features].min(axis=0),color="red", kde=True,bins=120, label='target = 0')

sns.distplot(t1[features].min(axis=0),color="blue", kde=True,bins=120, label='target = 1')

plt.legend(); plt.show()
plt.figure(figsize=(16,6))

plt.title("Distribution of max values per row in the train set")

sns.distplot(t0[features].max(axis=1),color="red", kde=True,bins=120, label='target = 0')

sns.distplot(t1[features].max(axis=1),color="blue", kde=True,bins=120, label='target = 1')

plt.legend(); plt.show()
plt.figure(figsize=(16,6))

plt.title("Distribution of max values per column in the train set")

sns.distplot(t0[features].max(axis=0),color="red", kde=True,bins=120, label='target = 0')

sns.distplot(t1[features].max(axis=0),color="blue", kde=True,bins=120, label='target = 1')

plt.legend(); plt.show()
plt.figure(figsize=(16,6))

features = train_df.columns.values[2:302]

plt.title("Distribution of skew per row in the train and test set")

sns.distplot(train_df[features].skew(axis=1),color="red", kde=True,bins=120, label='train')

sns.distplot(test_df[features].skew(axis=1),color="orange", kde=True,bins=120, label='test')

plt.legend()

plt.show()
plt.figure(figsize=(16,6))

features = train_df.columns.values[2:302]

plt.title("Distribution of skew per column in the train and test set")

sns.distplot(train_df[features].skew(axis=0),color="magenta", kde=True,bins=120, label='train')

sns.distplot(test_df[features].skew(axis=0),color="darkblue", kde=True,bins=120, label='test')

plt.legend()

plt.show()
plt.figure(figsize=(16,6))

features = train_df.columns.values[2:302]

plt.title("Distribution of kurtosis per row in the train and test set")

sns.distplot(train_df[features].kurtosis(axis=1),color="darkblue", kde=True,bins=120, label='train')

sns.distplot(test_df[features].kurtosis(axis=1),color="yellow", kde=True,bins=120, label='test')

plt.legend()

plt.show()
plt.figure(figsize=(16,6))

features = train_df.columns.values[2:302]

plt.title("Distribution of kurtosis per column in the train and test set")

sns.distplot(train_df[features].kurtosis(axis=0),color="magenta", kde=True,bins=120, label='train')

sns.distplot(test_df[features].kurtosis(axis=0),color="green", kde=True,bins=120, label='test')

plt.legend()

plt.show()
t0 = train_df.loc[train_df['target'] == 0]

t1 = train_df.loc[train_df['target'] == 1]

plt.figure(figsize=(16,6))

plt.title("Distribution of skew values per row in the train set")

sns.distplot(t0[features].skew(axis=1),color="red", kde=True,bins=120, label='target = 0')

sns.distplot(t1[features].skew(axis=1),color="blue", kde=True,bins=120, label='target = 1')

plt.legend(); plt.show()
t0 = train_df.loc[train_df['target'] == 0]

t1 = train_df.loc[train_df['target'] == 1]

plt.figure(figsize=(16,6))

plt.title("Distribution of skew values per column in the train set")

sns.distplot(t0[features].skew(axis=0),color="red", kde=True,bins=120, label='target = 0')

sns.distplot(t1[features].skew(axis=0),color="blue", kde=True,bins=120, label='target = 1')

plt.legend(); plt.show()
t0 = train_df.loc[train_df['target'] == 0]

t1 = train_df.loc[train_df['target'] == 1]

plt.figure(figsize=(16,6))

plt.title("Distribution of kurtosis values per row in the train set")

sns.distplot(t0[features].kurtosis(axis=1),color="red", kde=True,bins=120, label='target = 0')

sns.distplot(t1[features].kurtosis(axis=1),color="blue", kde=True,bins=120, label='target = 1')

plt.legend(); plt.show()
t0 = train_df.loc[train_df['target'] == 0]

t1 = train_df.loc[train_df['target'] == 1]

plt.figure(figsize=(16,6))

plt.title("Distribution of kurtosis values per column in the train set")

sns.distplot(t0[features].kurtosis(axis=0),color="red", kde=True,bins=120, label='target = 0')

sns.distplot(t1[features].kurtosis(axis=0),color="blue", kde=True,bins=120, label='target = 1')

plt.legend(); plt.show()

correlations = train_df[features].corr().abs().unstack().sort_values(kind="quicksort").reset_index()

correlations = correlations[correlations['level_0'] != correlations['level_1']]

correlations.head(10)
correlations.tail(10)
correlations.head(10)

features = [c for c in train_df.columns if c not in ['id', 'target']]

for df in [test_df, train_df]:

    df['sum'] = df[features].sum(axis=1)  

    df['min'] = df[features].min(axis=1)

    df['max'] = df[features].max(axis=1)

    df['mean'] = df[features].mean(axis=1)

    df['std'] = df[features].std(axis=1)

    df['skew'] = df[features].skew(axis=1)

    df['kurt'] = df[features].kurtosis(axis=1)

    df['med'] = df[features].median(axis=1)
train_df[train_df.columns[302:]].head()
test_df[test_df.columns[301:]].head()
def plot_new_feature_distribution(df1, df2, label1, label2, features):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(2,4,figsize=(18,8))



    for feature in features:

        i += 1

        plt.subplot(2,4,i)

        sns.kdeplot(df1[feature], bw=0.5,label=label1)

        sns.kdeplot(df2[feature], bw=0.5,label=label2)

        plt.xlabel(feature, fontsize=11)

        locs, labels = plt.xticks()

        plt.tick_params(axis='x', which='major', labelsize=8)

        plt.tick_params(axis='y', which='major', labelsize=8)

    plt.show();
t0 = train_df.loc[train_df['target'] == 0]

t1 = train_df.loc[train_df['target'] == 1]

features = train_df.columns.values[302:]

plot_new_feature_distribution(t0, t1, 'target: 0', 'target: 1', features)
features = train_df.columns.values[302:]

plot_new_feature_distribution(train_df, test_df, 'train', 'test', features)
print('Train and test columns: {} {}'.format(len(train_df.columns), len(test_df.columns)))
def apply_noise(data, noise_level):

    idxt = data[['id', 'target']]

    features = data.columns.values[2:]

    appended_data = []

    for feature in features:

        signal = data[feature]

        noise_factor = (np.abs(signal)).mean() * noise_level

        noise =  np.random.normal(0, noise_level, signal.shape)

        jittered = signal + noise

        appended_data.append(pd.DataFrame(jittered))

    appended_data = pd.concat(appended_data, axis=1)

    data_jittered = pd.concat([idxt, pd.DataFrame(appended_data)], axis=1)

    return data_jittered
noise_train_df = []

for i in tqdm_notebook(range(0,2)):

    t = apply_noise(train_df, noise_level = i * 0.025)

    noise_train_df.append(t)

noise_train_df = pd.concat(noise_train_df, axis = 0)
print("Shape train with additional rows with noise added:",noise_train_df.shape)

train_df = noise_train_df
features = [c for c in train_df.columns if c not in ['id', 'target']]

#using https://www.kaggle.com/alexandregeorges/glmnet-train-rfe-boruta-cv ?

#features = ['33', '65', '217', '91', '199', '69', '82', '117', '73', '295',

#           '130', '108', '258', '18', '189', '194', '43', '145', '80','24', 

#           '56', '214', '268', 'max', 'min', 'std', 'mean', 'skew','kurt' ]

target = train_df['target']
len(features)
param = {

    'bagging_freq': 5,

    'bagging_fraction': 0.83,

    'boost_from_average':'false',

    'boost': 'gbdt',

    'feature_fraction': 0.81,

    'learning_rate': 0.005,

    'max_depth': -1,  

    'metric':'auc',

    'min_data_in_leaf': 90,

    'min_sum_hessian_in_leaf': 10.0,

    'num_leaves': 11,

    'num_threads': 8,

    'tree_learner': 'serial',

    'objective': 'binary', 

    'verbosity': 1

}
folds = StratifiedKFold(n_splits=9, shuffle=True, random_state=4422)

oof = np.zeros(len(train_df))

predictions = np.zeros(len(test_df))

feature_importance_df = pd.DataFrame()



for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):

    print("Fold {}".format(fold_))

    trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])

    val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])



    num_round = 100000

    clf = lgb.train(param, trn_data, 

                    num_round, 

                    valid_sets = [trn_data, val_data], 

                    verbose_eval=400, 

                    early_stopping_rounds = 400)

    oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = features

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits



print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
def plot_feature_importance():

    cols = (feature_importance_df[["Feature", "importance"]]

            .groupby("Feature")

            .mean()

            .sort_values(by="importance", ascending=False)[:50].index)

    best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

    plt.figure(figsize=(12,10))

    sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))

    plt.title('Features importance (averaged/folds)')

    plt.tight_layout()

    plt.savefig('FI.png')

plot_feature_importance()
sub_df = pd.DataFrame({"id":test_df["id"].values})

sub_df["target"] = predictions

sub_df.to_csv("submission.csv", index=False)