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


train = pd.read_csv("C:\\Users\\HP-DK0272TX\\OneDrive\\Desktop\\file\\Codingrad\\train.csv")
test = pd.read_csv("C:\\Users\\HP-DK0272TX\\OneDrive\\Desktop\\file\\Codingrad\\test.csv")


# In[3]:


train.head(3)


# In[4]:


test.head(3)


# In[5]:


train.shape, test.shape


# In[6]:


train.head()


# In[7]:


train.isnull().sum()


# In[8]:


train.describe()


# In[9]:


train.info()


# In[10]:


train.corr()


# In[11]:


list1 = ["Age", "Annual_Premium", "Region_Code", "Vintage", "Response"]
plt.figure(figsize = (10,9))
sns.heatmap(train.corr(), annot = True, fmt = ".2f")
plt.show()


# In[12]:


train.Driving_License.value_counts()


# In[13]:


sns.set_style('whitegrid')
plt.figure(figsize = (4,4))
sns.countplot(x ='Driving_License', data = train)


# In[14]:


train.Previously_Insured.value_counts()


# In[15]:


sns.set_style('whitegrid')
plt.figure(figsize = (5,5))
sns.countplot(x ='Previously_Insured', data = train)


# In[16]:


train.Vehicle_Age.value_counts()


# In[17]:


sns.set_style('whitegrid')
plt.figure(figsize = (5,5))
sns.countplot(x ='Vehicle_Age', data = train)


# In[18]:


sns.set_style('whitegrid')
plt.figure(figsize = (14,14))
sns.countplot(x ='Age', data = train)


# In[19]:


train.Vehicle_Damage.value_counts()


# In[20]:


sns.set_style('whitegrid')
plt.figure(figsize = (5,5))
sns.countplot(x ='Vehicle_Damage', data = train)


# In[21]:


sns.set_style('whitegrid')
plt.figure(figsize = (5,5))
sns.countplot(x ='Gender', data = train)


# In[22]:


sns.boxplot(y = 'Annual_Premium', data = train,palette='Accent')


# In[23]:


sns.distplot(train.Vintage)


# In[24]:


s=sns.displot(data=train, x="Age", hue="Gender")


# In[25]:


fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
fig.suptitle('Visuallization of categorical columns')

# Fuel_Type
sns.barplot(x = 'Gender', y = 'Response', data = train, ax = axes[0],palette="dark")

# Seller_Type
sns.barplot(x = 'Age', y = 'Response', data = train, ax = axes[1],palette="Blues_d")

# Transmission
sns.barplot(x = 'Driving_License', y = 'Response', data = train, ax = axes[2],palette="Blues_d")


# # Respose by gender

# In[26]:


data_genderwise = train.groupby(['Gender','Response'])['id'].count().to_frame().rename(columns={'id':'count'}).reset_index()
data_genderwise


# In[27]:


g = sns.catplot(x="Gender", y="count",col="Response",
                data=data_genderwise, kind="bar",
                height=4, aspect=.7);


# # DrivingLicence By Gender

# In[28]:


data_DrivingLicenceBy = train.groupby(['Gender'])['Driving_License'].count().to_frame().reset_index()
data_DrivingLicenceBy


# In[29]:


sns.catplot(x="Gender", y="Driving_License", data=data_DrivingLicenceBy, kind="bar")


# In[30]:


data_DrivingLicence=train.groupby(['Driving_License','Response'])['id'].count().to_frame().rename(columns={'id':'count'}).reset_index()
data_DrivingLicence


# # Response By Vehicle_Age

# In[31]:


data_Vehicleage=train.groupby(['Vehicle_Age','Response'])['id'].count().to_frame().rename(columns={'id':'count'}).reset_index()
data_Vehicleage


# In[32]:


g2 = sns.catplot(x="Vehicle_Age", y="count",col="Response",
                data=data_Vehicleage, kind="bar",
                height=4, aspect=.7);


# # Response By Damaged_Vehicle

# In[33]:


data_Vehicle_Damage=train.groupby(['Vehicle_Damage','Response'])['id'].count().to_frame().rename(columns={'id':'count'}).reset_index()
data_Vehicle_Damage


# In[34]:


g3 = sns.catplot(x="Vehicle_Damage", y="count",col="Response",
                data=data_Vehicle_Damage, kind="bar",
                height=4, aspect=.7);


# In[35]:


Gender=pd.get_dummies(train['Gender'],drop_first=True)
Gender


# In[36]:


train.drop(labels=['Gender'],axis=1,inplace=True)


# In[37]:


train.shape


# In[38]:


train['Gender']=Gender


# In[39]:


train.Vehicle_Age.unique()


# In[40]:


VAge=pd.get_dummies(train['Vehicle_Age'],)
VAge


# In[41]:


VAge=VAge.rename(columns={"< 1 Year": "Veh_Age_lessthan_1_Yr", 
                          "> 2 Years": "Veh_Age_greaterthan_2_Yrs",
                          "1-2 Year": "Veh_Age_blw(1-2)_Yrs"})
VAge


# In[42]:


traindata=pd.concat([train,VAge],axis=1)


# In[43]:


traindata.head()


# In[44]:


traindata.drop(labels=['Vehicle_Age'],axis=1,inplace=True)


# In[45]:


traindata['Vehicle_Damage'] = traindata['Vehicle_Damage'].map( {'Yes': 1, 'No': 0} ).astype(int)


# In[46]:


traindata=traindata.rename(columns={"Vehicle_Damage": "Vehicle_Damage_Yes"})


# In[47]:


traindata.shape


# In[48]:


traindata.drop(labels=['id'],axis=1,inplace=True)


# In[49]:


traindata.shape


# In[50]:


traindata.head()


# # Saving the cleaned train dataset to pickle file

# In[51]:


import pickle
with open('EDA_Vehicledata.pkl', 'wb') as f:
    pickle.dump(traindata, f)


# In[52]:


# Numerical and categorical columns
num_df= ['Age','Vintage']
cat_df = ['Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Age_lt_1_Year',
            'Vehicle_Age_gt_2_Years','Vehicle_Damage','Region_Code','Policy_Sales_Channel']


# In[53]:


from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer



cat_pipeline = Pipeline([
        # we will fill the NaNs with the mode
        ('imputer', SimpleImputer(strategy="most_frequent")),
        # the features has order meaning
        ('encoder', OrdinalEncoder()),
    ])
X=traindata.drop(['Response'], axis = 1)

# prepare the df form the ML models by calling the full_pipeline
X_prepared = cat_pipeline.fit_transform(X)
# inspect the number of rows & columns of the prepared df
X_prepared.shape


# Now let's build a pipeline for preprocessing all the attributes:

# In[54]:


traindata.columns


# In[55]:


traindata


# In[56]:


traindata.to_csv('traindata.csv', index=False)


# In[ ]:




