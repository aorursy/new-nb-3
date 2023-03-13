# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Load Packages

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import log_loss, accuracy_score, roc_auc_score

from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint as sp_randint

import matplotlib.pyplot as plt

from sklearn.model_selection import validation_curve
# Read Data

train = pd.read_csv('../input/train.csv')
# Subsampling

train = train.sample(5000)
# define tfidf vectorizer 

tfidf = TfidfVectorizer(analyzer = 'word',

                        stop_words = 'english',

                        lowercase = True,

                        max_features = 300,

                        norm = 'l1')
BagOfWords = pd.concat([train.question1, train.question2], axis = 0)
tfidf.fit(BagOfWords)
train_q1_tfidf = tfidf.transform(train.question1)

train_q2_tfidf = tfidf.transform(train.question2)
X = abs(train_q1_tfidf - train_q2_tfidf)

y = train['is_duplicate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LogisticRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict_proba(X_test)[:,1]

logloss_lr = log_loss(y_test, pred_lr)
logloss_lr
rf = RandomForestClassifier(n_estimators = 200,

                            min_samples_leaf = 10,

                            n_jobs = -1)
rf.fit(X_train, y_train)
pred_rf = rf.predict_proba(X_test)[:,1]

logloss_rf = log_loss(y_test, pred_rf)
logloss_rf
param_dist = {"max_depth": [3, None],

              "max_features": sp_randint(1, 50),

              "min_samples_split": sp_randint(2, 10),

              "min_samples_leaf": sp_randint(1, 10),

              "criterion": ["gini", "entropy"]}



# run randomized search

n_iter_search = 5

random_search = RandomizedSearchCV(rf, param_distributions=param_dist,

                                   n_iter=n_iter_search, scoring='neg_log_loss')
random_search.fit(X, y)
random_search.best_score_
random_search.best_params_