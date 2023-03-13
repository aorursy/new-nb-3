import numpy as np
import pandas as pd 
import string
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import ktrain
from ktrain import text
from sklearn.model_selection import train_test_split
import csv
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')
def remove_punctuation(text):
    return ''.join([char for char in text if char not in string.punctuation])
    
def tokenize(text):
    return word_tokenize(text)
    
def to_lowercase(text):
    return ''.join([char.lower() for char in text])
    
def stemming(text):
    snowball_stemmer = SnowballStemmer('english')
    return ' '.join([snowball_stemmer.stem(word) for word in tokenize(text)])

def preprocess_text(dataframe_column):
    dataframe_column = dataframe_column.map(lambda comment : remove_punctuation(comment))
    dataframe_column = dataframe_column.map(lambda comment : to_lowercase(comment))
    dataframe_column = dataframe_column.map(lambda comment : stemming(comment))
    return dataframe_column
data['comment_text'] = preprocess_text(data['comment_text'])
data['toxic'].value_counts(dropna=False, normalize=True)
data.isnull().sum()
data[data['comment_text'].duplicated()]['toxic']
wordcloud_text = ''.join([comment for comment in data['comment_text'][1:1000] if pd.notna(comment)])
wordcloud = WordCloud().generate(wordcloud_text)
plt.imshow(wordcloud)
plt.show()
x_train = data["comment_text"].head(2000).values.tolist()
y_train=  data["toxic"].head(2000).values.tolist()
x_test = data["comment_text"].tail(660).values.tolist()
y_test = data["toxic"].tail(660).values.tolist()
len(x_train), len(y_train), len(x_test), len(y_test)
(x_train,  y_train), (x_test, y_test), preproc = text.texts_from_array(x_train=x_train, y_train=y_train,
                                                                       x_test=x_test, y_test=y_test,
                                                                       class_names=['0','1'],
                                                                       preprocess_mode='bert',
                                                                       maxlen=338)
model = text.text_classifier('bert', train_data=(x_train, y_train), preproc=preproc)
learner = ktrain.get_learner(model, train_data=(x_train, y_train), batch_size=3)
result = learner.fit_onecycle(2e-5, 2)
validate = learner.validate(val_data=(x_test, y_test), class_names=['0', '1'])
predictor = ktrain.get_predictor(learner.model, preproc)
predictor.get_classes()
prediction = predictor.predict(data['comment_text'].tail(660).tolist())
submission_index = list(data.index[:-661:-1])
submission_index.reverse()
submission_index[0], submission_index[-1]
len(submission_index), len(prediction)
submission_dataframe = pd.DataFrame({'id': submission_index, 'toxic': prediction})

submission_dataframe.to_csv('submission.csv', index=False)
with open('submission.csv', mode='a+') as submission_csv:
    submission_writer = csv.writer(submission_csv, delimiter=',')
    submission_writer.writerow(['id', 'toxic'])
    for index, row in submission_dataframe.iterrows():
        submission_writer.writerow([row['id'], row['toxic']])

