#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np #linear algebra
import pandas as pd #data processing,csv file I/o
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#data gathering
data=pd.read_csv('C:\\Users\\HP-DK0272TX\\OneDrive\\Desktop\\file\\Codingrad\\Car-details-v3.csv')
data


# In[3]:


#pre process(cleaning)
data.info()


# In[4]:


data.isnull().sum()#check null values


# In[5]:


final_data = data[['year', 'selling_price', 'km_driven', 'fuel', 'seller_type',
       'transmission', 'owner', 'mileage', 'engine', 'max_power',
       'seats']]


# In[6]:


final_data.head()


# In[7]:


#s=data.drop(['name','mileage','engine','max_power','torque','seats','owner'],axis='columns',inplace=True)#remove or del column
#now we are going to remove kmpl, cc, bhp

#removed kmpl
final_data['mileage'] = final_data['mileage'].apply(lambda x: x.split()[0] if type(x)==str else np.nan)

#removed cc
final_data['engine'] = final_data['engine'].apply(lambda x: x.split()[0] if type(x)==str else np.nan)

#removed bhp
final_data['max_power'] = final_data['max_power'].apply(lambda x: x.split()[0] if type(x)==str else np.nan)


# In[8]:


final_data.head(3)


# In[9]:


final_data['current_year'] = 2020


# In[10]:


final_data['no_years'] = final_data['current_year']-final_data['year']
final_data.head()


# In[11]:


final_data.drop(['year'], axis=1, inplace=True)


# In[12]:


final_data.drop(['current_year'], axis=1, inplace=True)


# In[13]:


final_data.head()


# In[14]:


final_data.isnull().sum()


# In[15]:


# check missing values percentage in each column
final_data.isnull().mean().round(4).mul(100).sort_values(ascending=False).head()


# In[16]:


final_data.dropna(inplace = True)


# In[17]:


final_data.isnull().sum()


# In[18]:


final_data.info()


# In[19]:


final_data[final_data['max_power'].str.contains('bhp')]


# In[20]:


# we have one row containing only bhp with no value in it, lets drop it

final_data = final_data[final_data['max_power'].str.contains('bhp') == False]


# In[21]:


# convert from object type to their respective types

final_data['mileage'] = final_data['mileage'].astype('float')
final_data['engine'] = final_data['engine'].astype('int64')
final_data['max_power'] = final_data['max_power'].astype('float')


# In[22]:


final_data.info()


# In[23]:


final_data.head()


# In[24]:


final_data = pd.get_dummies(final_data, drop_first= True) #drop_first because of dummie variable trap


# In[25]:


final_data.head()


# In[26]:


# corr() - it tells us, how one feature is related to the other feature
final_data.corr()


# In[27]:


import seaborn as sns


# In[28]:


sns.pairplot(final_data)


# In[29]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[30]:


corrmat = final_data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(10,10))
g = sns.heatmap(final_data[top_corr_features].corr(), annot=True, cmap="RdYlGn")


# In[31]:


#independent and dependent features

X = final_data.iloc[:, 1:]
y = final_data.iloc[:, 0]


# In[32]:


X.head()


# In[33]:


y.head()


# In[34]:


#feature importance
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X,y)


# In[35]:


print(model.feature_importances_)


# In[36]:


#plot graph of feature importance for better visualization
feat_importances = pd.Series(model.feature_importances_, index = X.columns)
feat_importances.nlargest(10).plot(kind = 'barh')
plt.show()


# In[37]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)


# In[38]:


from sklearn.ensemble import RandomForestRegressor
rf_random = RandomForestRegressor()


# In[39]:


#hyperparameters
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)


# In[40]:


#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]

# max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[41]:


#we are using randomized searchcv as it is pretty much fast
from sklearn.model_selection import RandomizedSearchCV


# In[42]:


# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[43]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()


# In[44]:


#applying my randomized searchCV
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[45]:


rf_random.fit(X_train, y_train)


# In[46]:


predictions = rf_random.predict(X_test)


# In[47]:


predictions


# In[48]:


sns.displot(y_test-predictions)


# In[49]:


plt.scatter(y_test, predictions)


# In[50]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[51]:


from sklearn.metrics import r2_score
print('r2score: ', r2_score(y_test, predictions))


# In[52]:


import pickle 
#open a file, where you want to store the data
file = open('random_forest_regression_model.pkl', 'wb')

#dump information to the file
pickle.dump(rf_random, file)


# In[ ]:




