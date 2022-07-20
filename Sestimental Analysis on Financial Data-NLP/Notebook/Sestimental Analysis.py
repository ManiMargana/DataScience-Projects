#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install nltk
#Standard Libraries #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Other Libraries

## Classification Algorithms ##
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

## For building models ##
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

## For measuring performance ##
from sklearn import metrics
from sklearn.model_selection import cross_val_score

## To visualize decision tree ##
# from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
# import pydotplus

## Ignore warnings ##
import warnings
warnings.filterwarnings('ignore')


# In[3]:


### Load the dataset & Analyse


# In[4]:


headlines = pd.read_csv("C:\\Users\\HP-DK0272TX\\OneDrive\\Desktop\\file\\Codingrad\\FinancialData.csv",
                      names = ['lables','messages'],encoding='ISO-8859-1')
headlines.head()


# In[5]:


headlines.info()


# In[6]:


headlines.isnull().sum()


# # Introducing new column: 'target'

# In[7]:


News_copy=headlines.copy()
News_copy.shape


# In[8]:


def func(df_new):
    if df_new == 'neutral':
        return 0
    elif df_new == 'negative':
        return 1
    else:
        return 2


# In[9]:


News_copy['target'] = News_copy.lables.apply(func)
News_copy.head()


# # Visualization:

# In[10]:


headlines.lables.value_counts()


# In[11]:


plt.figure(figsize=(7,7))

headlines.lables.value_counts().plot(kind='pie', autopct='%1.0f%%')
plt.title("Percentages of different lables to the headlines")


# In[12]:


# Count plot for labels feature

plt.figure(figsize=(7,6))
headlines.lables.value_counts().plot(kind='bar',color=['blue','green','red'])
plt.xlabel("News Type")
plt.ylabel("count")
plt.title("Count Plot for labels");


# # Data PreProcessing:

# In[13]:


import nltk #NLP library
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords #removes unnecessary(repetitive) words
import re #regex
import string
# nltk.download_shell()


# In[14]:


import string
from nltk.corpus import stopwords
nltk.download('stopwords')
def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
headlines['messages'].head(5).apply(text_process)


# # Vectorization:

# In[15]:


from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_process).fit(headlines['messages'])

# Print total number of vocab words
print(len(bow_transformer.vocabulary_))


# In[16]:


message4 = headlines['messages'][3]
print(message4)


# In[17]:


bow4 = bow_transformer.transform([message4])
print(bow4)
print(bow4.shape)


# In[18]:


print(bow_transformer.get_feature_names()[6987])
print(bow_transformer.get_feature_names()[12206])


# In[19]:


messages_bow = bow_transformer.transform(headlines['messages'])


# In[20]:


print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)


# In[21]:


sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format(round(sparsity)))


# In[22]:


from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)


# In[23]:


messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)


# # Training a model:

# In[24]:


from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf, headlines['lables'])


# In[25]:


print('predicted:', spam_detect_model.predict(tfidf4)[0])
print('expected:', headlines.lables[1])


# # Test the data:

# In[26]:


all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)


# In[27]:


from sklearn.metrics import classification_report
print (classification_report(headlines['lables'], all_predictions))


# In[28]:


from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(headlines['messages'], headlines['lables'], test_size=0.2)

print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))


# In[29]:


from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


# In[30]:


pipeline.fit(msg_train,label_train)


# In[31]:


predictions = pipeline.predict(msg_test)
print(classification_report(predictions,label_test))


# # Saving the model to pickle file:

# In[32]:


import pickle
pickle.dump(headlines,open('Sentiment_model.pkl','wb'))


# In[ ]:




