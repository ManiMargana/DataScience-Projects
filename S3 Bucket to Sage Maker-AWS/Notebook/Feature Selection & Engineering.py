#!/usr/bin/env python
# coding: utf-8

# # Feature Selection / Engineering

# # Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('df_for_feature_engineering.csv')

train = pd.read_csv("C:/Users/HP-DK0272TX/OneDrive/Desktop/file/Codingrad/S3 train-1.csv")
test = pd.read_csv("C:/Users/HP-DK0272TX/OneDrive/Desktop/file/Codingrad/S3 test-2.csv")
df


# # Drop feature

# In[3]:


df = df.drop(['YrSold',
 'LowQualFinSF',
 'MiscVal',
 'BsmtHalfBath',
 'BsmtFinSF2',
 '3SsnPorch',
 'MoSold'],axis=1)


# In[4]:


quan = list(df.loc[:,df.dtypes != 'object'].columns.values)
quan


# In[5]:


skewd_feat = ['1stFlrSF',
 '2ndFlrSF',
 'BedroomAbvGr',
 'BsmtFinSF1',
 'BsmtFullBath',
 'BsmtUnfSF',
 'EnclosedPorch',
 'Fireplaces',
 'FullBath',
 'GarageArea',
 'GarageCars',
 'GrLivArea',
 'HalfBath',
 'KitchenAbvGr',
 'LotArea',
 'LotFrontage',
 'MasVnrArea',
 'OpenPorchSF',
 'PoolArea',
 'ScreenPorch',
 'TotRmsAbvGrd',
 'TotalBsmtSF',
 'WoodDeckSF']
#  '3SsnPorch',  'BsmtFinSF2',  'BsmtHalfBath',  'LowQualFinSF', 'MiscVal'


# In[6]:


# Decrease the skewness of the data
for i in skewd_feat:
    df[i] = np.log(df[i] + 1)
    
SalePrice = np.log(train['SalePrice'] + 1)


# # decrease the skewnwnes of the data

# for i in skewed_features: df[i] = np.log(df[i] + 1)

# In[7]:


df


# In[8]:


obj_feat = list(df.loc[:, df.dtypes == 'object'].columns.values)
print(len(obj_feat))

obj_feat


# In[9]:


# dummy varaibale
dummy_drop = []
for i in obj_feat:
    dummy_drop += [i + '_' + str(df[i].unique()[-1])]

df = pd.get_dummies(df, columns = obj_feat)
df = df.drop(dummy_drop, axis = 1)


# In[10]:


df.shape


# In[11]:


# scaling dataset with robust scaler
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler.fit(df)
df = scaler.transform(df)


# # Model Bulding

# In[12]:


train_len = len(train)
X_train = df[:train_len]
X_test = df[train_len:]
y_train = SalePrice

print("Shape of X_train: ", len(X_train))
print("Shape of X_test: ", len(X_test))
print("Shape of y_train: ", len(y_train))


# # Cross Validation

# In[13]:


from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, r2_score

def test_model(model, X_train=X_train, y_train=y_train):
    cv = KFold(n_splits = 3, shuffle=True, random_state = 45)
    r2 = make_scorer(r2_score)
    r2_val_score = cross_val_score(model, X_train, y_train, cv=cv, scoring = r2)
    score = [r2_val_score.mean()]
    return score


# first cross validation with df with log second without log
# 

# # Linear Model

# In[14]:


import sklearn.linear_model as linear_model
LR = linear_model.LinearRegression()
test_model(LR)


# In[15]:


rdg = linear_model.Ridge()
test_model(rdg)


# In[16]:


lasso = linear_model.Lasso(alpha=1e-4)
test_model(lasso)


# # Support vector machine

# In[17]:


from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf')
test_model(svr_reg)


# # svm hyper parameter tuning

# In[18]:


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
params = {'kernel': ['rbf'],
         'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
         'C': [0.1, 1, 10, 100, 1000],
         'epsilon': [1, 0.2, 0.1, 0.01, 0.001, 0.0001]}
rand_search = RandomizedSearchCV(svr_reg, param_distributions=params, n_jobs=-1, cv=11)
rand_search.fit(X_train, y_train)
rand_search.best_score_


# In[19]:


rand_search.best_estimator_


# In[20]:


svr_reg1=SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.001,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
test_model(svr_reg1)


# In[21]:


svr_reg= SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.01, gamma=0.0001,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
test_model(svr_reg)


# In[22]:


get_ipython().system('pip install xgboost')


# # XGBoost

# In[23]:


import xgboost
#xgb_reg=xgboost.XGBRegressor()
xgb_reg = xgboost.XGBRegressor(bbooster='gbtree', random_state=51)
test_model(xgb_reg)


# In[24]:


xgb2_reg=xgboost.XGBRegressor(n_estimators= 899,
 mon_child_weight= 2,
 max_depth= 4,
 learning_rate= 0.05,
 booster= 'gbtree')

test_model(xgb2_reg)


# # Solution

# In[25]:


xgb2_reg.fit(X_train,y_train)
y_pred = np.exp(xgb2_reg.predict(X_test)).round(2)
submit_test = pd.concat([test['Id'],pd.DataFrame(y_pred)], axis=1)
submit_test.columns=['Id', 'SalePrice']
submit_test.to_csv('sample_submission.csv', index=False)
submit_test

"""
Rank: 1444
Red AI Productionnovice 
tier
0.12278
5
now
Your Best Entry 
Your submission scored 0.13481, which is not an improvement of your best score. Keep trying!"""


# In[26]:


svr_reg.fit(X_train,y_train)
y_pred = np.exp(svr_reg.predict(X_test)).round(2)
submit_test = pd.concat([test['Id'],pd.DataFrame(y_pred)], axis=1)
submit_test.columns=['Id', 'SalePrice']
submit_test.to_csv('sample_submission.csv', index=False)
submit_test

"""
file: sample_submission-v1-fs
rank: 1444
Red AI Productionnovice tier
0.12278
4
3m
Your Best Entry 
You advanced 140 places on the leaderboard!

Your submission scored 0.12278, which is an improvement of your previous score of 0.12484. Great job!"""


# # Model Save

# In[28]:


import pickle

save = pickle.dump(svr_reg, open('house_price_pred_AWS.pkl', 'wb')) #save file in pkl


# In[29]:


model_house_price_prediction = pickle.load(open('house_price_pred_AWS.pkl', 'rb'))
model_house_price_prediction.predict(X_test)


# In[30]:


test_model(model_house_price_prediction)


# # SVM Accuracy = 90%

# In[ ]:




