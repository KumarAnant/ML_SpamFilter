import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

import os
os.chdir('E:/OneDrive/Data Science/Udemy/Python for Data Science and Machine Learning/Resources/20-Natural-Language-Processing/smsspamcollection')

messages = [line.rstrip() for line in open('SMSSpamCollection')]

messages = pd.read_csv('SMSSpamCollection', sep = '\t', names = ['label', 'message'])
messages.head()
messages['length'] = messages['message'].apply(len)

g = sns.FacetGrid(messages, col = 'label')
g = g.map(plt.hist, 'length', bins = 50)
plt.show()
punctuation = string.punctuation
stopwords = nltk.corpus.stopwords.words('english')

def text_analyzer(message_string):
    noPunc = ''.join([char for char in message_string if char not in punctuation])
    string = [word for word in noPunc.split() if word.lower() not in stopwords]
    return ' '.join(string)

text_analyzer('Hello how are you *&^ skdjh aweuo akfh')

