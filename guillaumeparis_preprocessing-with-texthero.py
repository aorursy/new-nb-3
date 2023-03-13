# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import texthero as hero
pd.set_option('display.max_colwidth', None)
train = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')



train.drop(['severe_toxic','obscene','threat','insult','identity_hate'],axis=1,inplace=True)





#train

display(train.head())
train['comment_text_clean'] = (

    train['comment_text']

    .pipe(hero.remove_digits)

    .pipe(hero.remove_diacritics)

    .pipe(hero.remove_whitespace)

    .pipe(hero.remove_urls)

    .pipe(hero.remove_html_tags)

    .pipe(hero.remove_brackets))
train.head()