#!/usr/bin/env python
# coding: utf-8

# In[1]:


##### Standard Libraries #####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#data gathering
data=pd.read_csv('C:\\Users\\HP-DK0272TX\\OneDrive\\Desktop\\file\\Codingrad\\lymphography.csv')
data.head(5)


# In[3]:


#List columns names based on the description

col_names = ['class', 'lymphatics', 'block of affere', 'bl. of lymph. c', 'bl. of lymph. s', 'by pass', 
 'extravasates', 'regeneration of', 'early uptake in', 'lym.nodes dimin', 'lym.nodes enlar', 
'changes in lym.', 'defect in node', 'changes in node', 'changes in stru', 'special forms', 
'dislocation of', 'exclusion of no', 'no. of nodes in']


# Here, it can be seen that the data file does not have column headers. To add headers, let's list the column names then specify this list when loading the data.

# In[4]:


#Load the data
data = pd.read_csv('C:\\Users\\HP-DK0272TX\\OneDrive\\Desktop\\file\\Codingrad\\lymphography.csv', names=col_names)
print(data.shape)
data.head(5)


# In[5]:


data.info()


# In[6]:


data.isnull().sum()


# In[7]:


data.describe()


# In[8]:


data['class'].value_counts()


# In[9]:


sns.set_style('whitegrid')
plt.figure(figsize = (5,6))
sns.countplot(data['class'])


# In[10]:


#Heatmap to show the correlation between various variables of the dataset
plt.figure(figsize=(18, 16))
cor = data.corr()
ax = sns.heatmap(cor,annot=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()


# In[11]:


sns.set_style("darkgrid")
plt.figure(figsize = (8,6))
sns.countplot(data['class'], hue=data['lymphatics'])


# In[12]:


sns.distplot(data['class'],kde=False,color='darkblue',bins=30)


# # Imblearn SMOTH algorithm and test the model:

# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[14]:


#selecting features and target
X = data.drop(columns=['class'])
y = data['class']


# In[15]:


X


# In[16]:


y


# In[17]:


#train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,y,random_state=101,test_size=0.3,stratify=y)


# In[18]:


model = LogisticRegression()
model.fit(x_train,y_train)
pred = model.predict(x_test)
print('Accuracy ',accuracy_score(y_test,pred))
print(classification_report(y_test,pred))

sns.heatmap(confusion_matrix(y_test,pred),annot=True,fmt='.2g')


# In[19]:


np.bincount(y_train)


# In[20]:


#!pip install imblearn
from imblearn.over_sampling import SMOTE 
# transform the dataset
oversample = SMOTE(k_neighbors = 1)
X_train_res,y_train_res  = oversample.fit_resample(X,y)


# In[21]:


lr = LogisticRegression() 
lr.fit(X_train_res, y_train_res.ravel()) 
predictions = lr.predict(x_test) 
  
print('Accuracy ',accuracy_score(y_test,predictions))
print(classification_report(y_test, predictions)) 
sns.heatmap(confusion_matrix(y_test,predictions),annot=True,fmt='.2g')


# In[22]:


np.bincount(y_train_res)


# # Tensorflow keras neural networks:

# In[23]:


#!pip install tensorflow
#!pip install keras
from tensorflow.keras.layers import Dense #NN
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


# In[24]:


features = data.drop(columns=['class'])
target = data['class']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, target)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)


# In[25]:


model = Sequential()

model.add(Dense(100, input_shape=(features.shape[1],)))
model.add(Dense(30, activation="relu"))
# model.add(Dense(45, activation="relu"))
# model.add(Dense(32, activation="relu"))
# model.add(Dense(23, activation="relu"))
model.add(Dense(3, activation="softmax"))


# In[26]:


model.summary()


# In[27]:


import tensorflow

model.compile(optimizer="sgd", 
              loss=tensorflow.keras.losses.CategoricalCrossentropy(), 
              metrics=['accuracy'])


# In[28]:


X_train.shape, y_train.shape


# In[29]:


y_pred = model.predict(X_test)
y_pred


# In[30]:


#Tune the Model using GridSearchCv

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Create first pipeline for base without reducing features.
pipe = Pipeline([('classifier' , RandomForestClassifier())])

# Create param grid.
param_grid = [
    {'classifier' : [LogisticRegression()],
     'classifier__penalty' : ['l1', 'l2'],
    'classifier__C' : np.logspace(-4, 4, 20),
    'classifier__solver' : ['liblinear']},
    {'classifier' : [RandomForestClassifier()],
    'classifier__n_estimators' : list(range(10,101,10)),
    'classifier__max_features' : list(range(6,32,5))}
]

# Create grid search object
clf = GridSearchCV(pipe, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)

# Fit on data
best_clf = clf.fit(X_train, y_train)


# In[31]:


print(clf.best_params_)
print(clf.best_score_)


# # Plot the Accuracy and Loss graphs:

# In[32]:


X = data.drop(columns=['class'])
y = data['class']
model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])


history = model.fit(X,y,steps_per_epoch=3, epochs=3,validation_steps=5)


# In[33]:


history.history.keys()


# In[34]:


plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()


# In[35]:


# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()


# In[36]:


#Saving the model
import pickle
with open('Lymphography_model.pkl', 'wb') as f:
    pickle.dump(model, f)


# In[ ]:




