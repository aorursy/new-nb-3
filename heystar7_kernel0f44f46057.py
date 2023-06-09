# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns 

pd.options.display.max_columns  = None # 모든 column 확인

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
train.shape, test.shape
train.describe()
train.info()
train.info(verbose = 1, null_counts = True)
test.info(verbose = 1, null_counts = True)
train.select_dtypes(np.int64).nunique().value_counts().sort_index().plot.bar(color = 'blue', 

                                                                             figsize = (8, 6),

                                                                            edgecolor = 'k', linewidth = 2);

plt.xlabel('Number of Unique Values'); plt.ylabel('Count');

plt.title('Count of Unique Values in Integer Columns');
from collections import OrderedDict



plt.figure(figsize = (20, 16))

plt.style.use('fivethirtyeight')



# Color mapping

colors = OrderedDict({1: 'red', 2: 'orange', 3: 'blue', 4: 'green'})

poverty_mapping = OrderedDict({1: 'extreme', 2: 'moderate', 3: 'vulnerable', 4: 'non vulnerable'})



# Iterate through the float columns

for i, col in enumerate(train.select_dtypes('float')):

    ax = plt.subplot(4, 2, i + 1)

    # Iterate through the poverty levels

    for poverty_level, color in colors.items():

        # Plot each poverty level as a separate line

        sns.kdeplot(train.loc[train['Target'] == poverty_level, col].dropna(), 

                    ax = ax, color = color, label = poverty_mapping[poverty_level])

        

    plt.title(f'{col.capitalize()} Distribution'); plt.xlabel(f'{col}'); plt.ylabel('Density')



plt.subplots_adjust(top = 2)
train.select_dtypes('object').head()
mapping = {"yes": 1, "no": 0}



for df in [train, test]:

    # Fill in the values with the correct mapping

    df['dependency'] = df['dependency'].replace(mapping).astype(np.float64)

    df['edjefa'] = df['edjefa'].replace(mapping).astype(np.float64)

    df['edjefe'] = df['edjefe'].replace(mapping).astype(np.float64)



train[['dependency', 'edjefa', 'edjefe']].describe()
plt.figure(figsize = (16, 12))



# Iterate through the float columns

for i, col in enumerate(['dependency', 'edjefa', 'edjefe']):

    ax = plt.subplot(3, 1, i + 1)

    # Iterate through the poverty levels

    for poverty_level, color in colors.items():

        # Plot each poverty level as a separate line

        sns.kdeplot(train.loc[train['Target'] == poverty_level, col].dropna(), 

                    ax = ax, color = color, label = poverty_mapping[poverty_level])

        

    plt.title(f'{col.capitalize()} Distribution'); plt.xlabel(f'{col}'); plt.ylabel('Density')



plt.subplots_adjust(top = 2)
# Add null Target column to test

test['Target'] = np.nan

data = train.append(test, ignore_index = True)
# Heads of household

heads = data.loc[data['parentesco1'] == 1].copy()



# Labels for training

train_labels = data.loc[(data['Target'].notnull()) & (data['parentesco1'] == 1), ['Target', 'idhogar']]



# Value counts of target

label_counts = train_labels['Target'].value_counts().sort_index()



# Bar plot of occurrences of each label

label_counts.plot.bar(figsize = (8, 6), 

                      color = colors.values(),

                      edgecolor = 'k', linewidth = 2)

# Formatting

plt.xlabel('Poverty Level'); plt.ylabel('Count'); 

plt.xticks([x - 1 for x in poverty_mapping.keys()], 

           list(poverty_mapping.values()), rotation = 60)

plt.title('Poverty Level Breakdown');



label_counts
# Groupby the household and figure out the number of unique values

all_equal = train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)



not_equal = all_equal[all_equal != True]

print('There are {} households where the family members do not all have the same target.'.format(len(not_equal)))
train[train['idhogar'] == not_equal.index[0]][['idhogar', 'parentesco1', 'Target']]
households_leader = train.groupby('idhogar')['parentesco1'].sum()



households_no_head = train.loc[train['idhogar'].isin(households_leader[households_leader == 0].index), :]



print('There are {} households without a head.'.format(households_no_head['idhogar'].nunique()))


households_no_head_equal = households_no_head.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)

print('{} Households with no head have different labels.'.format(sum(households_no_head_equal == False)))


for household in not_equal.index:

    

    true_target = int(train[(train['idhogar'] == household) & (train['parentesco1'] == 1.0)]['Target'])

    

    

    train.loc[train['idhogar'] == household, 'Target'] = true_target

    

    



all_equal = train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)





not_equal = all_equal[all_equal != True]

print('There are {} households where the family members do not all have the same target.'.format(len(not_equal)))


missing = pd.DataFrame(data.isnull().sum()).rename(columns = {0: 'total'})





missing['percent'] = missing['total'] / len(data)



missing.sort_values('percent', ascending = False).head(10).drop('Target')
def plot_value_counts(df, col, heads_only = False):

    """Plot value counts of a column, optionally with only the heads of a household"""

    # Select heads of household

    if heads_only:

        df = df.loc[df['parentesco1'] == 1].copy()

        

    plt.figure(figsize = (8, 6))

    df[col].value_counts().sort_index().plot.bar(color = 'blue',

                                                 edgecolor = 'k',

                                                 linewidth = 2)

    plt.xlabel(f'{col}') 

    plt.title(f'{col} Value Counts')

    plt.ylabel('Count')

    plt.show()
plot_value_counts(heads, 'v18q1')
heads.groupby('v18q')['v18q1'].apply(lambda x: x.isnull().sum())
data['v18q1'] = data['v18q1'].fillna(0)
own_variables = [x for x in data if x.startswith('tipo')]



data.loc[data['v2a1'].isnull(), own_variables].sum().plot.bar(figsize = (10, 8),

                                                                        color = 'green',

                                                              edgecolor = 'k', linewidth = 2);

plt.xticks([0, 1, 2, 3, 4],

           ['Owns and Paid Off', 'Owns and Paying', 'Rented', 'Precarious', 'Other'],

          rotation = 60)

plt.title('Home Ownership Status for Households Missing Rent Payments', size = 18)
data.loc[(data['tipovivi1'] == 1), 'v2a1'] = 0





data['v2a1-missing'] = data['v2a1'].isnull()



data['v2a1-missing'].value_counts()
data.loc[data['rez_esc'].notnull()]['age'].describe()
data.loc[data['rez_esc'].isnull()]['age'].describe()
# If individual is over 19 or younger than 7 and missing years behind, set it to 0

data.loc[((data['age'] > 19) | (data['age'] < 7)) & (data['rez_esc'].isnull()), 'rez_esc'] = 0



# Add a flag for those between 7 and 19 with a missing value

data['rez_esc-missing'] = data['rez_esc'].isnull()
data.loc[data['rez_esc'] > 5, 'rez_esc'] = 5
plot_value_counts(data[(data['rez_esc-missing'] == 1)], 

                  'Target')
plot_value_counts(data[(data['v2a1-missing'] == 1)], 

                  'Target')
id_ = ['Id', 'idhogar', 'Target']
ind_bool = ['v18q', 'dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3', 

            'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 

            'parentesco1', 'parentesco2',  'parentesco3', 'parentesco4', 'parentesco5', 

            'parentesco6', 'parentesco7', 'parentesco8',  'parentesco9', 'parentesco10', 

            'parentesco11', 'parentesco12', 'instlevel1', 'instlevel2', 'instlevel3', 

            'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 

            'instlevel9', 'mobilephone', 'rez_esc-missing']



ind_ordered = ['rez_esc', 'escolari', 'age']
hh_bool = ['hacdor', 'hacapo', 'v14a', 'refrig', 'paredblolad', 'paredzocalo', 

           'paredpreb','pisocemento', 'pareddes', 'paredmad',

           'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisoother', 

           'pisonatur', 'pisonotiene', 'pisomadera',

           'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 

           'abastaguadentro', 'abastaguafuera', 'abastaguano',

            'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 

           'sanitario2', 'sanitario3', 'sanitario5',   'sanitario6',

           'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 

           'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 

           'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3',

           'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 

           'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 

           'computer', 'television', 'lugar1', 'lugar2', 'lugar3',

           'lugar4', 'lugar5', 'lugar6', 'area1', 'area2', 'v2a1-missing']



hh_ordered = [ 'rooms', 'r4h1', 'r4h2', 'r4h3', 'r4m1','r4m2','r4m3', 'r4t1',  'r4t2', 

              'r4t3', 'v18q1', 'tamhog','tamviv','hhsize','hogar_nin',

              'hogar_adul','hogar_mayor','hogar_total',  'bedrooms', 'qmobilephone']



hh_cont = ['v2a1', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'overcrowding']
sqr_ = ['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 

        'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq']
x = ind_bool + ind_ordered + id_ + hh_bool + hh_ordered + hh_cont + sqr_



from collections import Counter



print('There are no repeats: ', np.all(np.array(list(Counter(x).values())) == 1))

print('We covered every variable: ', len(x) == data.shape[1])
sns.lmplot('age', 'SQBage', data = data, fit_reg=False); 

plt.title('Squared Age versus Age'); 
data = data.drop(columns = sqr_)

data.shape
heads = data.loc[data['parentesco1'] == 1, :]

heads = heads[id_ + hh_bool + hh_cont + hh_ordered]

print(heads.shape)

heads.head()
# Create correlation matrix

corr_matrix = heads.corr()



# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# Find index of feature columns with correlation greater than 0.95

to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]



to_drop
corr_matrix.loc[corr_matrix['tamhog'].abs() > 0.9, corr_matrix['tamhog'].abs() > 0.9]
sns.heatmap(corr_matrix.loc[corr_matrix['tamhog'].abs() > 0.9, corr_matrix['tamhog'].abs() > 0.9],

            annot=True, cmap = plt.cm.autumn_r, fmt='.3f');
heads = heads.drop(columns = ['tamhog', 'hogar_total', 'r4t3'])
sns.lmplot('tamviv', 'hhsize', data, fit_reg=False, size = 8);

plt.title('Household size vs number of persons living in the household');
heads['phones-per-capita'] = heads['qmobilephone'] / heads['tamviv']

heads['tablets-per-capita'] = heads['v18q1'] / heads['tamviv']

heads['rooms-per-capita'] = heads['rooms'] / heads['tamviv']

heads['rent-per-capita'] = heads['v2a1'] / heads['tamviv']
train_heads = heads.loc[heads['Target'].notnull(), :].copy()



pcorrs = pd.DataFrame(train_heads.corr()['Target'].sort_values()).rename(columns = {'Target': 'pcorr'}).reset_index()

pcorrs = pcorrs.rename(columns = {'index': 'feature'})



print('Most negatively correlated variables:')

print(pcorrs.head())



print('\nMost positively correlated variables:')

print(pcorrs.dropna().tail())
ind = data[id_ + ind_bool + ind_ordered]

ind.shape
# Create correlation matrix

corr_matrix = ind.corr()



# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# Find index of feature columns with correlation greater than 0.95

to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.90)]



to_drop
ind = ind.drop(columns = 'male')
ind[[c for c in ind if c.startswith('instl')]].head()
ind.shape
# Define custom function

range_ = lambda x: x.max() - x.min()

range_.__name__ = 'range_'



# Group and aggregate

ind_agg = ind.drop(columns = 'Target').groupby('idhogar').agg(['min', 'max', 'sum', 'count', 'std', range_])

ind_agg.head()
new_col = []

for c in ind_agg.columns.levels[0]:

    for stat in ind_agg.columns.levels[1]:

        new_col.append(f'{c}-{stat}')

        

ind_agg.columns = new_col

ind_agg.head()
# Create correlation matrix

corr_matrix = ind_agg.corr()



# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# Find index of feature columns with correlation greater than 0.95

to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]



print(f'There are {len(to_drop)} correlated columns to remove.')
ind_agg = ind_agg.drop(columns = to_drop)

ind_feats = list(ind_agg.columns)



# Merge on the household id

final = heads.merge(ind_agg, on = 'idhogar', how = 'left')



print('Final features shape: ', final.shape)
final.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score, make_scorer

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline



# Custom scorer for cross validation

scorer = make_scorer(f1_score, greater_is_better=True, average = 'macro')
# Labels for training

train_labels = np.array(list(final[final['Target'].notnull()]['Target'].astype(np.uint8)))



# Extract the training data

train_set = final[final['Target'].notnull()].drop(columns = ['Id', 'idhogar', 'Target'])

test_set = final[final['Target'].isnull()].drop(columns = ['Id', 'idhogar', 'Target'])



# Submission base which is used for making submissions to the competition

submission_base = test[['Id', 'idhogar']].copy()
features = list(train_set.columns)



pipeline = Pipeline([('imputer', Imputer(strategy = 'median')), 

                      ('scaler', MinMaxScaler())])



# Fit and transform training data

train_set = pipeline.fit_transform(train_set)

test_set = pipeline.transform(test_set)
model = RandomForestClassifier(n_estimators=100, random_state=10, 

                               n_jobs = -1)

# 10 fold cross validation

cv_score = cross_val_score(model, train_set, train_labels, cv = 10, scoring = scorer)



print(f'10 Fold Cross Validation F1 Score = {round(cv_score.mean(), 4)} with std = {round(cv_score.std(), 4)}')
model.fit(train_set, train_labels)



# Feature importances into a dataframe

feature_importances = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})

feature_importances.head()
def plot_feature_importances(df, n = 10, threshold = None):

    """Plots n most important features. Also plots the cumulative importance if

    threshold is specified and prints the number of features needed to reach threshold cumulative importance.

    Intended for use with any tree-based feature importances. 

    

    Args:

        df (dataframe): Dataframe of feature importances. Columns must be "feature" and "importance".

    

        n (int): Number of most important features to plot. Default is 15.

    

        threshold (float): Threshold for cumulative importance plot. If not provided, no plot is made. Default is None.

        

    Returns:

        df (dataframe): Dataframe ordered by feature importances with a normalized column (sums to 1) 

                        and a cumulative importance column

    

    Note:

    

        * Normalization in this case means sums to 1. 

        * Cumulative importance is calculated by summing features from most to least important

        * A threshold of 0.9 will show the most important features needed to reach 90% of cumulative importance

    

    """

    plt.style.use('fivethirtyeight')

    

    # Sort features with most important at the head

    df = df.sort_values('importance', ascending = False).reset_index(drop = True)

    

    # Normalize the feature importances to add up to one and calculate cumulative importance

    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])

    

    plt.rcParams['font.size'] = 12

    

    # Bar plot of n most important features

    df.loc[:n, :].plot.barh(y = 'importance_normalized', 

                            x = 'feature', color = 'darkgreen', 

                            edgecolor = 'k', figsize = (12, 8),

                            legend = False, linewidth = 2)



    plt.xlabel('Normalized Importance', size = 18); plt.ylabel(''); 

    plt.title(f'{n} Most Important Features', size = 18)

    plt.gca().invert_yaxis()

    

    

    if threshold:

        # Cumulative importance plot

        plt.figure(figsize = (8, 6))

        plt.plot(list(range(len(df))), df['cumulative_importance'], 'b-')

        plt.xlabel('Number of Features', size = 16); plt.ylabel('Cumulative Importance', size = 16); 

        plt.title('Cumulative Feature Importance', size = 18);

        

        # Number of features needed for threshold cumulative importance

        # This is the index (will need to add 1 for the actual number)

        importance_index = np.min(np.where(df['cumulative_importance'] > threshold))

        

        # Add vertical line to plot

        plt.vlines(importance_index + 1, ymin = 0, ymax = 1.05, linestyles = '--', colors = 'red')

        plt.show();

        

        print('{} features required for {:.0f}% of cumulative importance.'.format(importance_index + 1, 

                                                                                  100 * threshold))

    

    return df
norm_fi = plot_feature_importances(feature_importances, threshold=0.95)
from sklearn.svm import LinearSVC

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier



import warnings 

from sklearn.exceptions import ConvergenceWarning



# Filter out warnings from models

warnings.filterwarnings('ignore', category = ConvergenceWarning)

warnings.filterwarnings('ignore', category = DeprecationWarning)

warnings.filterwarnings('ignore', category = UserWarning)



# Dataframe to hold results

model_results = pd.DataFrame(columns = ['model', 'cv_mean', 'cv_std'])



def cv_model(train, train_labels, model, name, model_results=None):

    """Perform 10 fold cross validation of a model"""

    

    cv_scores = cross_val_score(model, train, train_labels, cv = 10, scoring=scorer, n_jobs = -1)

    print(f'10 Fold CV Score: {round(cv_scores.mean(), 5)} with std: {round(cv_scores.std(), 5)}')

    

    if model_results is not None:

        model_results = model_results.append(pd.DataFrame({'model': name, 

                                                           'cv_mean': cv_scores.mean(), 

                                                            'cv_std': cv_scores.std()},

                                                           index = [0]),

                                             ignore_index = True)



        return model_results
model_results = cv_model(train_set, train_labels, LinearSVC(), 

                         'LSVC', model_results)
model_results = cv_model(train_set, train_labels, 

                         GaussianNB(), 'GNB', model_results)
model_results = cv_model(train_set, train_labels, 

                         MLPClassifier(hidden_layer_sizes=(32, 64, 128, 64, 32)),

                         'MLP', model_results)
model_results = cv_model(train_set, train_labels, 

                          LinearDiscriminantAnalysis(), 

                          'LDA', model_results)
model_results = cv_model(train_set, train_labels, 

                         RidgeClassifierCV(), 'RIDGE', model_results)
for n in [5, 10, 20]:

    print(f'\nKNN with {n} neighbors\n')

    model_results = cv_model(train_set, train_labels, 

                             KNeighborsClassifier(n_neighbors = n),

                             f'knn-{n}', model_results)
from sklearn.ensemble import ExtraTreesClassifier



model_results = cv_model(train_set, train_labels, 

                         ExtraTreesClassifier(n_estimators = 100, random_state = 10),

                         'EXT', model_results)
model_results = cv_model(train_set, train_labels,

                          RandomForestClassifier(100, random_state=10),

                              'RF', model_results)
model_results.set_index('model', inplace = True)

model_results['cv_mean'].plot.bar(color = 'blue', figsize = (8, 6),

                                  yerr = list(model_results['cv_std']),

                                  edgecolor = 'k', linewidth = 2)

plt.title('Model F1 Score Results');

plt.ylabel('Mean F1 Score (with error bar)');

model_results.reset_index(inplace = True)
def submit(model):

    model.fit(train_set, train_labels)

    predictions = model.predict(test_set)

    predictions = pd.DataFrame({'idhogar': test_ids,

                               'Target': predictions})



     # Make a submission dataframe

    submission = submission_base.merge(predictions, 

                                       on = 'idhogar',

                                       how = 'left').drop(columns = ['idhogar'])



    # Fill in households missing a head

    submission['Target'] = submission['Target'].fillna(4).astype(np.int8)

    

    return submission 
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(train_set, train_labels, test_size = 0.2, 

                                                    random_state =42)
print('train shape', X_train.shape, y_train.shape)

print('test shape', X_test.shape, y_test.shape)
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import xgboost 

from xgboost import XGBClassifier

from sklearn.metrics import f1_score
xgb_model  = XGBClassifier(objective = 'multi:softmax', random_state = 40)

xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

f1_score(y_test, y_pred, average = 'macro')
test_ids = list(final.loc[final['Target'].isnull(), 'idhogar'])
xgb_submission = submit(xgb_model)
import lightgbm

from lightgbm import LGBMClassifier
lgb_model  = LGBMClassifier(objective = 'multiclass', random_state = 42)

lgb_model.fit(X_train, y_train)

y_pred = lgb_model.predict(X_test)

print(y_pred)

f1_score(y_test, y_pred, average = 'macro')
lgb_submission = submit(lgb_model)
lgb_submission = lgb_submission.set_index('Id')
lgb_submission.head()
lgb_submission.to_csv('submission.csv')