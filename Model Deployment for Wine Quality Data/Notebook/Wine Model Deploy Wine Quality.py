#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[3]:


wine_data = pd.read_csv('C:\\Users\\HP-DK0272TX\\OneDrive\\Desktop\\file\\Codingrad\\winequality-red.csv')
wine_data.head()


# In[4]:


wine_data.shape


# In[5]:


wine_data.info()


# In[6]:


wine_data.isnull().sum()


# In[7]:


wine_data.describe()


# # DATA-VISUALIZATION:

# In[8]:


wine_data.quality.value_counts()


# In[9]:


#Number of values for each quality
sns.catplot(x='quality',data = wine_data, kind = 'count')


# In[10]:


#volatile acidity vs quality
plt.figure(figsize=(6,5))
sns.barplot(x='quality',y='volatile acidity',data=wine_data)


# In[11]:


#constructing a heatmap to understand the correlation between columns
correlation = wine_data.corr()
plt.figure(figsize = (12,11))
sns.heatmap(correlation, cbar = True, square=True, fmt='.2f', annot=True, annot_kws={'size':8}, cmap = 'Blues')


# # DATA-PREPROCESSING:

# In[12]:


#SEPERATE THE DATA AS FEATURES & LABELS

X = wine_data.drop(['quality'],axis=1)


# In[13]:


print(X.shape)
X.head()


# In[14]:


#LABEL BINARIZATION

Y = wine_data['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)


# In[15]:


Y


# # TRAIN & TEST SPLIT :

# In[16]:


from sklearn.model_selection import train_test_split 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(Y.shape, Y_train.shape, Y_test.shape)


# # Model Training:

# Random Forest Classifier: In simple terms we can say it as a multiple decision tree model

# In[17]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()


# In[18]:


model.fit(X_train, Y_train)


# # Model Evaluation :

# # Accuracy score:

# In[19]:


# Accuracy on test data
from sklearn.metrics import accuracy_score
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy:', test_data_accuracy) 


# # Building a predictive system:

# In[20]:


input_data = (7.3,0.65,0.0,1.2,0.065,15.0,21.0,0.9946,3.39,0.47,10.0)

# Changing the input data in to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the data as we are predicting the label for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if prediction[0]==1: 
    print('Good Quality Wine')
else:
    print('Bad Quality Wine')


# # Save the model in pickle file:

# In[21]:


import pickle

#save file
save = pickle.dump(model,open('wine_model.pkl','wb'))


# In[22]:


model = pickle.load(open('wine_model.pkl','rb'))
pred=model.predict([[7.3,0.65,0.0,1.2,0.065,15.0,21.0,0.9946,3.39,0.47,10.0]])
print(pred)
if pred[0]==1:
    print('Best quality wine')
else:
    print('Poor quality wine')


# In[ ]:




