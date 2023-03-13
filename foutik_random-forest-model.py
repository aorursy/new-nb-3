# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

from xgboost.sklearn import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/train_users_2.csv")

df_test = pd.read_csv("../input/test_users.csv")
labels = df_train['country_destination'].values # Get the values of the country destination for each row

df_train = df_train.drop(['country_destination'], axis=1) # It's the output variable for the decision tree

id_test = df_test['id']

piv_train = df_train.shape[0]
df_all = pd.concat((df_train, df_test), axis = 0, ignore_index = True)
df_all = df_all.drop(['id','date_first_booking'], axis=1)
df_all.gender.replace('-unknown-', np.nan, inplace=True) # -unknown- is not considered as a missing value so we replace it by nan

print(df_all.isnull().sum())

df_all = df_all.fillna(-1)
dac = np.vstack(df_all.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)

print(dac)

df_all['dac_year'] = dac[:,0]

df_all['dac_mounth'] = dac[:,1]

df_all['dac_day'] = dac[:,2]

df_all = df_all.drop(['date_account_created'], axis = 1)
tfa = np.vstack(df_all.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)

print(tfa)

df_all['tfa_year'] = tfa[:,0]

df_all['tfa_month'] = tfa[:,1]

df_all['tfa_day'] = tfa[:,2]

df_all = df_all.drop(['timestamp_first_active'], axis=1)
print(df_all.age.describe()) # We can see that the age has some inconsistancy variables

av = df_all.age.values

df_all['age'] = np.where(np.logical_or(av<14, av>100), -1, av)
features = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
for f in features:

    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)

    df_all = df_all.drop([f], axis=1)

    df_all = pd.concat((df_all, df_all_dummy), axis=1)
vals = df_all.values

X = vals[:piv_train]

le = LabelEncoder()

y = le.fit_transform(labels)   

X_test = vals[piv_train:]
model = RandomForestClassifier()

model.fit(X,y)

y_pred = model.predict_proba(X_test)
ids = []  #list of ids

cts = []  #list of countries

for i in range(len(id_test)):

    idx = id_test[i]

    ids += [idx] * 5

    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()