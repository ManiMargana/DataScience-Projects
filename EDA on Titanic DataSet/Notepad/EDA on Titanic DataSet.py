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


# In[2]:


#Reading data
dataset = pd.read_csv('C:\\Users\\HP-DK0272TX\\OneDrive\\Desktop\\file\\Codingrad\\Titanic-Train-Data.csv')
dataset


# # Missing data

# In[3]:


dataset.isnull()


# In[4]:


sns.heatmap(dataset.isnull(), yticklabels = False, cbar=False, cmap = 'viridis')


# In[5]:


sns.set_style("whitegrid")
sns.countplot(x = "Survived", data = dataset)


# In[6]:


sns.set_style("whitegrid")
sns.countplot(x ='Survived',hue ='Sex', data = dataset, palette = "muted")


# In[7]:


sns.set_style('whitegrid')
sns.countplot(x = 'Survived', hue = 'Pclass', data = dataset, palette = 'dark')


# In[8]:


dataset['Age'].hist(bins=30, color='darkred', alpha=0.3)


# In[9]:


sns.countplot(x='SibSp', data = dataset)


# In[10]:


dataset['Fare'].hist(bins=30, color ='green', figsize=(7,4))


# # Data cleaning

# we want to fill in missing age data instead of dropping it.
# 
# one way to do this is by filling the mean age of all the passengers (imputation).
# 
# however we can be smarter about this and check the average age by passenger class.

# In[11]:


plt.figure(figsize=(12,7))
sns.boxplot(x = 'Pclass', y = 'Age', data = dataset, palette = 'summer')


# In[12]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        if Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


# In[13]:


dataset['Age'] = dataset[['Age', 'Pclass']].apply(impute_age, axis=1)


# In[14]:


sns.heatmap(dataset.isnull(), yticklabels = False, cbar=False, cmap = 'viridis')


# In[15]:


dataset.drop('Cabin',axis=1,inplace=True)


# In[16]:


dataset.head()


# In[17]:


dataset.dropna(inplace = True)


# In[18]:


dataset.info()


# In[19]:


pd.get_dummies(dataset['Embarked'], drop_first=True).head()


# In[20]:


sex = pd.get_dummies(dataset['Sex'], drop_first=True)
embark = pd.get_dummies(dataset['Embarked'], drop_first=True)


# In[21]:


dataset.head()


# In[22]:


dataset.drop(['Embarked','Sex','Name','Ticket'], axis=1, inplace=True)


# In[23]:


dataset.head()


# In[24]:


dataset = pd.concat([dataset, sex, embark], axis=1)


# In[25]:


dataset.head()


# Now, data is ready for our model!

# # Building a model using Logistic Regression

# # Train Test Split

# In[26]:


dataset.drop('Survived', axis=1).head()


# In[27]:


dataset['Survived'].head()


# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(dataset.drop('Survived', axis=1),
                                               dataset['Survived'], test_size=0.3, random_state=101)


# # Training & Predicting

# In[30]:


from sklearn.linear_model import LogisticRegression


# In[31]:


logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)


# In[32]:


predictions = logmodel.predict(X_test)


# In[33]:


from sklearn.metrics import confusion_matrix


# In[34]:


accuracy = confusion_matrix(y_test, predictions)


# In[35]:


accuracy


# In[36]:


from sklearn.metrics import accuracy_score
accuracy1 = accuracy_score(y_test, predictions)
accuracy1


# # SAVE MODEL:

# In[37]:


import pickle

with open('model_titanic.pkl', 'wb') as f:
    pickle.dump(logmodel, f)


# In[ ]:




