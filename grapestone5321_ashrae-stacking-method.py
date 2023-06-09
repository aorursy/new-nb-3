# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print(os.listdir('/kaggle/input/ashrae-ensembling-1'))



# Any results you write to the current directory are saved as output.
import os

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from scipy.stats.mstats import gmean

import seaborn as sns


from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
sub_path = "../input/ashrae-ensembling-1"

all_files = os.listdir(sub_path)

all_files
import warnings

warnings.filterwarnings("ignore")

outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files]

concat_sub = pd.concat(outs, axis=1)

cols = list(map(lambda x: "mol" + str(x), range(len(concat_sub.columns))))

concat_sub.columns = cols

concat_sub.reset_index(inplace=True)

concat_sub.head()

ncol = concat_sub.shape[1]
concat_sub.iloc[:,1:].corr()
corr = concat_sub.iloc[:,1:].corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap='prism', vmin=0.96, center=0, square=True, linewidths=1, annot=True, fmt='.4f')
concat_sub['m_max'] = concat_sub.iloc[:, 1:].max(axis=1)

concat_sub['m_min'] = concat_sub.iloc[:, 1:].min(axis=1)

concat_sub['m_median'] = concat_sub.iloc[:, 1:].median(axis=1)
concat_sub.describe()
cutoff_lo = 0.8

cutoff_hi = 0.2
rank = np.tril(concat_sub.iloc[:,1:ncol].corr().values,-1)

m_gmean = 0

n = 8

while rank.max()>0:

    mx = np.unravel_index(rank.argmax(), rank.shape)

    m_gmean += n*(np.log(concat_sub.iloc[:, mx[0]+1]) + np.log(concat_sub.iloc[:, mx[1]+1]))/2

    rank[mx] = 0

    n += 1
concat_sub['m_mean'] = np.exp(m_gmean/(n-1)**2)
concat_sub['meter_reading'] = concat_sub['m_mean']

concat_sub[['row_id', 'meter_reading']].to_csv('stack_mean.csv', 

                                        index=False, float_format='%.6f')
concat_sub['meter_reading'] = concat_sub['m_median']

concat_sub[['row_id', 'meter_reading']].to_csv('stack_median.csv', 

                                        index=False, float_format='%.6f')
concat_sub['meter_reading'] = np.where(np.all(concat_sub.iloc[:,1:7] > cutoff_lo, axis=1), 1, 

                                    np.where(np.all(concat_sub.iloc[:,1:7] < cutoff_hi, axis=1),

                                             0, concat_sub['m_median']))

concat_sub[['row_id', 'meter_reading']].to_csv('stack_pushout_median.csv', 

                                        index=False, float_format='%.6f')
concat_sub['meter_reading'] = np.where(np.all(concat_sub.iloc[:,1:7] > cutoff_lo, axis=1), 

                                    concat_sub['m_max'], 

                                    np.where(np.all(concat_sub.iloc[:,1:7] < cutoff_hi, axis=1),

                                             concat_sub['m_min'], 

                                             concat_sub['m_mean']))

concat_sub[['row_id', 'meter_reading']].to_csv('stack_minmax_mean.csv', 

                                        index=False, float_format='%.6f')
concat_sub['meter_reading'] = np.where(np.all(concat_sub.iloc[:,1:7] > cutoff_lo, axis=1), 

                                    concat_sub['m_max'], 

                                    np.where(np.all(concat_sub.iloc[:,1:7] < cutoff_hi, axis=1),

                                             concat_sub['m_min'], 

                                             concat_sub['m_median']))

concat_sub[['row_id', 'meter_reading']].to_csv('stack_minmax_median.csv', 

                                        index=False, float_format='%.6f')
concat_sub['meter_reading'] = concat_sub['mol0'].rank(method ='min') + concat_sub['mol1'].rank(method ='min') + concat_sub['mol2'].rank(method ='min') 

concat_sub['meter_reading'] = (concat_sub['meter_reading']-concat_sub['meter_reading'].min())/(concat_sub['meter_reading'].max() - concat_sub['meter_reading'].min())

concat_sub.describe()

concat_sub[['row_id', 'meter_reading']].to_csv('stack_rank.csv', index=False, float_format='%.8f')