#!/usr/bin/env python
# coding: utf-8

# # Goal of the Project

# Predict the price of a house by its features. If you are a buyer or seller of the house but you don't know the exact price of the house, so supervised machine learning regression algorithms can help you to predict the price of the house just providing features of the target house.

# # Import essential libraries

# In[1]:


# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Load Data Set

# In[2]:


train = pd.read_csv("C:/Users/HP-DK0272TX/OneDrive/Desktop/file/Codingrad/S3 train-1.csv")
test = pd.read_csv("C:/Users/HP-DK0272TX/OneDrive/Desktop/file/Codingrad/S3 test-2.csv")

print("Shape of train: ", train.shape)
print("Shape of test: ", test.shape)


# In[3]:


train.head(10)


# In[4]:


test.head(10)


# In[5]:


# concat train and test
df = pd.concat((train, test))
temp_df = df
print("Shape of df: ", df.shape)


# In[6]:


df.head(5)


# In[7]:


df.tail(5)


# # Exploratory Data Analysis (EDA)

# In[8]:


# To show the all columns
pd.set_option("display.max_columns", 2000)
pd.set_option("display.max_rows", 85)


# In[9]:


df.head(5)


# In[10]:


df.tail(5)


# In[11]:


df.info()


# In[12]:


df.describe()


# In[13]:


df.select_dtypes(include=['int64', 'float64']).columns


# In[14]:


df.select_dtypes(include=['object']).columns


# In[15]:


# Set index as Id column
df = df.set_index("Id")


# In[16]:


df.head(5)


# In[17]:


# Show the null values using heatmap
plt.figure(figsize=(16,9))
sns.heatmap(df.isnull())


# In[18]:


# Get the percentages of null value
null_percent = df.isnull().sum()/df.shape[0]*100
null_percent


# In[19]:


col_for_drop = null_percent[null_percent > 20].keys() # if the null value % 20 or > 20 so need to drop it


# In[20]:


# drop columns
df = df.drop(col_for_drop, "columns")
df.shape


# In[21]:


# find the unique value count
for i in df.columns:
    print(i + "\t" + str(len(df[i].unique())))


# In[22]:


# find unique values of each column
for i in df.columns:
    print("Unique value of:>>> {} ({})\n{}\n".format(i, len(df[i].unique()), df[i].unique()))


# In[23]:


# Describe the target 
train["SalePrice"].describe()


# In[24]:


# Plot the distplot of target
plt.figure(figsize=(10,8))
bar = sns.distplot(train["SalePrice"])
bar.legend(["Skewness: {:.2f}".format(train['SalePrice'].skew())])


# In[25]:


# correlation heatmap
plt.figure(figsize=(25,25))
ax = sns.heatmap(train.corr(), cmap = "coolwarm", annot=True, linewidth=2)

# to fix the bug "first and last row cut in half of heatmap plot"
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)


# In[26]:


# correlation heatmap of higly correlated features with SalePrice
hig_corr = train.corr()
hig_corr_features = hig_corr.index[abs(hig_corr["SalePrice"]) >= 0.5]
hig_corr_features


# In[27]:


plt.figure(figsize=(10,8))
ax = sns.heatmap(train[hig_corr_features].corr(), cmap = "coolwarm", annot=True, linewidth=3)
# to fix the bug "first and last row cut in half of heatmap plot"
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)


# In[28]:


# Plot regplot to get the nature of highly correlated data
plt.figure(figsize=(16,9))
for i in range(len(hig_corr_features)):
    if i <= 9:
        plt.subplot(3,4,i+1)
        plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
        sns.regplot(data=train, x = hig_corr_features[i], y = 'SalePrice')


# # Handling Missing Value

# In[29]:


missing_col = df.columns[df.isnull().any()]
missing_col


# # Handling missing value of Bsmt feature

# In[30]:


bsmt_col = ['BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1',
       'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual', 'BsmtUnfSF', 'TotalBsmtSF']
bsmt_feat = df[bsmt_col]
bsmt_feat


# In[31]:


bsmt_feat.info()


# In[32]:


bsmt_feat.isnull().sum()


# In[33]:


bsmt_feat = bsmt_feat[bsmt_feat.isnull().any(axis=1)]
bsmt_feat


# In[34]:


bsmt_feat_all_nan = bsmt_feat[(bsmt_feat.isnull() | bsmt_feat.isin([0])).all(1)]
bsmt_feat_all_nan


# In[35]:


bsmt_feat_all_nan.shape


# In[36]:


qual = list(df.loc[:, df.dtypes == 'object'].columns.values)
qual


# In[37]:


# Fillinf the mising value in bsmt features
for i in bsmt_col:
    if i in qual:
        bsmt_feat_all_nan[i] = bsmt_feat_all_nan[i].replace(np.nan, 'NA') # replace the NAN value by 'NA'
    else:
        bsmt_feat_all_nan[i] = bsmt_feat_all_nan[i].replace(np.nan, 0) # replace the NAN value inplace of 0

bsmt_feat.update(bsmt_feat_all_nan) # update bsmt_feat df by bsmt_feat_all_nan
df.update(bsmt_feat_all_nan) # update df by bsmt_feat_all_nan

"""
>>> df = pd.DataFrame({'A': [1, 2, 3],
...                    'B': [400, 500, 600]})
>>> new_df = pd.DataFrame({'B': [4, 5, 6],
...                        'C': [7, 8, 9]})
>>> df.update(new_df)
>>> df
   A  B
0  1  4
1  2  5
2  3  6
"""


# In[38]:


bsmt_feat = bsmt_feat[bsmt_feat.isin([np.nan]).any(axis=1)]
bsmt_feat


# In[39]:


bsmt_feat.shape


# In[40]:


print(df['BsmtFinSF2'].max())
print(df['BsmtFinSF2'].min())


# In[41]:


pd.cut(range(0,1526),5) # create a bucket


# In[42]:


df_slice = df[(df['BsmtFinSF2'] >= 305) & (df['BsmtFinSF2'] <= 610)]
df_slice


# In[43]:


bsmt_feat.at[333,'BsmtFinType2'] = df_slice['BsmtFinType2'].mode()[0] # replace NAN value of BsmtFinType2 by mode of buet ((305.0, 610.0)


# In[44]:


bsmt_feat


# In[45]:


bsmt_feat['BsmtExposure'] = bsmt_feat['BsmtExposure'].replace(np.nan, df[df['BsmtQual'] =='Gd']['BsmtExposure'].mode()[0])


# In[46]:


bsmt_feat['BsmtCond'] = bsmt_feat['BsmtCond'].replace(np.nan, df['BsmtCond'].mode()[0])
bsmt_feat['BsmtQual'] = bsmt_feat['BsmtQual'].replace(np.nan, df['BsmtQual'].mode()[0])


# In[47]:


df.update(bsmt_feat)


# In[48]:


bsmt_feat.isnull().sum()


# # Handling missing value of Garage feature

# In[49]:


df.columns[df.isnull().any()]


# In[50]:


garage_col = ['GarageArea', 'GarageCars', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'GarageYrBlt',]
garage_feat = df[garage_col]
garage_feat = garage_feat[garage_feat.isnull().any(axis=1)]
garage_feat


# In[51]:


garage_feat.shape


# In[52]:


garage_feat_all_nan = garage_feat[(garage_feat.isnull() | garage_feat.isin([0])).all(1)]
garage_feat_all_nan.shape


# In[53]:


for i in garage_feat:
    if i in qual:
        garage_feat_all_nan[i] = garage_feat_all_nan[i].replace(np.nan, 'NA')
    else:
        garage_feat_all_nan[i] = garage_feat_all_nan[i].replace(np.nan, 0)
        
garage_feat.update(garage_feat_all_nan)
df.update(garage_feat_all_nan)


# In[54]:


garage_feat = garage_feat[garage_feat.isnull().any(axis=1)]
garage_feat


# In[55]:


for i in garage_col:
    garage_feat[i] = garage_feat[i].replace(np.nan, df[df['GarageType'] == 'Detchd'][i].mode()[0])


# In[56]:


garage_feat.isnull().any()


# In[57]:


df.update(garage_feat)


# # Handling missing value of remain feature

# In[58]:


df.columns[df.isnull().any()]


# In[59]:


df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])
df['Functional'] = df['Functional'].fillna(df['Functional'].mode()[0])
df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])
df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])
df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])
df['Utilities'] = df['Utilities'].fillna(df['Utilities'].mode()[0])
df['MasVnrType'] = df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])


# In[60]:


df.columns[df.isnull().any()]


# In[61]:


df[df['MasVnrArea'].isnull() == True]['MasVnrType'].unique()


# In[62]:


df.loc[(df['MasVnrType'] == 'None') & (df['MasVnrArea'].isnull() == True), 'MasVnrArea'] = 0


# In[63]:


df.isnull().sum()/df.shape[0] * 100


# # Handling missing value of LotFrontage feature

# In[64]:


lotconfig = ['Corner', 'Inside', 'CulDSac', 'FR2', 'FR3']
for i in lotconfig:
    df['LotFrontage'] = pd.np.where((df['LotFrontage'].isnull() == True) & (df['LotConfig'] == i) , df[df['LotConfig'] == i] ['LotFrontage'].mean(), df['LotFrontage'])


# In[65]:


df.isnull().sum()


# # Feature Transformation

# In[66]:


df.columns


# In[67]:


# converting columns in str which have categorical nature but in int64
feat_dtype_convert = ['MSSubClass', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']
for i in feat_dtype_convert:
    df[i] = df[i].astype(str)


# In[68]:


df['MoSold'].unique() # MoSold = Month of sold


# In[69]:


# conver in month abbrevation
import calendar
df['MoSold'] = df['MoSold'].apply(lambda x : calendar.month_abbr[x])


# In[70]:


df['MoSold'].unique()


# In[71]:


quan = list(df.loc[:, df.dtypes != 'object'].columns.values)


# In[72]:


quan


# In[73]:


len(quan)


# In[74]:


obj_feat = list(df.loc[:, df.dtypes == 'object'].columns.values)
obj_feat


# # Convert categorical code into order

# In[75]:


from pandas.api.types import CategoricalDtype
df['BsmtCond'] = df['BsmtCond'].astype(CategoricalDtype(categories=['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes


# In[76]:


df['BsmtCond'].unique()


# In[77]:


df['BsmtExposure'] = df['BsmtExposure'].astype(CategoricalDtype(categories=['NA', 'Mn', 'Av', 'Gd'], ordered = True)).cat.codes


# In[78]:


df['BsmtExposure'].unique()


# In[79]:


df['BsmtFinType1'] = df['BsmtFinType1'].astype(CategoricalDtype(categories=['NA', 'Unf', 'LwQ', 'Rec', 'BLQ','ALQ', 'GLQ'], ordered = True)).cat.codes
df['BsmtFinType2'] = df['BsmtFinType2'].astype(CategoricalDtype(categories=['NA', 'Unf', 'LwQ', 'Rec', 'BLQ','ALQ', 'GLQ'], ordered = True)).cat.codes
df['BsmtQual'] = df['BsmtQual'].astype(CategoricalDtype(categories=['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df['ExterQual'] = df['ExterQual'].astype(CategoricalDtype(categories=['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df['ExterCond'] = df['ExterCond'].astype(CategoricalDtype(categories=['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df['Functional'] = df['Functional'].astype(CategoricalDtype(categories=['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod','Min2','Min1', 'Typ'], ordered = True)).cat.codes
df['GarageCond'] = df['GarageCond'].astype(CategoricalDtype(categories=['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df['GarageQual'] = df['GarageQual'].astype(CategoricalDtype(categories=['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df['GarageFinish'] = df['GarageFinish'].astype(CategoricalDtype(categories=['NA', 'Unf', 'RFn', 'Fin'], ordered = True)).cat.codes
df['HeatingQC'] = df['HeatingQC'].astype(CategoricalDtype(categories=['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df['KitchenQual'] = df['KitchenQual'].astype(CategoricalDtype(categories=['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df['PavedDrive'] = df['PavedDrive'].astype(CategoricalDtype(categories=['N', 'P', 'Y'], ordered = True)).cat.codes
df['Utilities'] = df['Utilities'].astype(CategoricalDtype(categories=['ELO', 'NASeWa', 'NASeWr', 'AllPub'], ordered = True)).cat.codes


# In[80]:


df['Utilities'].unique()


# # Show skewness of feature with distplot

# In[81]:


skewed_features = ['1stFlrSF',
 '2ndFlrSF',
 '3SsnPorch',
 'BedroomAbvGr',
 'BsmtFinSF1',
 'BsmtFinSF2',
 'BsmtFullBath',
 'BsmtHalfBath',
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
 'LowQualFinSF',
 'MasVnrArea',
 'MiscVal',
 'OpenPorchSF',
 'PoolArea',
 'ScreenPorch',
 'TotRmsAbvGrd',
 'TotalBsmtSF',
 'WoodDeckSF']


# In[82]:


quan == skewed_features


# In[83]:


plt.figure(figsize=(25,20))
for i in range(len(skewed_features)):
    if i <= 28:
        plt.subplot(7,4,i+1)
        plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
        ax = sns.distplot(df[skewed_features[i]])
        ax.legend(["Skewness: {:.2f}".format(df[skewed_features[i]].skew())], fontsize = 'xx-large')


# In[84]:


df_back = df


# In[85]:


# decrease the skewnwnes of the data
for i in skewed_features:
    df[i] = np.log(df[i] + 1)


# In[86]:


plt.figure(figsize=(25,20))
for i in range(len(skewed_features)):
    if i <= 28:
        plt.subplot(7,4,i+1)
        plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
        ax = sns.distplot(df[skewed_features[i]])
        ax.legend(["Skewness: {:.2f}".format(df[skewed_features[i]].skew())], fontsize = 'xx-large')


# In[87]:


SalePrice = np.log(train['SalePrice'] + 1)


# In[88]:


# get object feature to conver in numeric using dummy variable
obj_feat = list(df.loc[:,df.dtypes == 'object'].columns.values)
len(obj_feat)


# In[89]:


#dummy varaibale
dummy_drop = []
clean_df = df
for i in obj_feat:
    dummy_drop += [i + '_' + str(df[i].unique()[-1])]

df = pd.get_dummies(df, columns = obj_feat)
df = df.drop(dummy_drop, axis = 1)


# In[90]:


df.shape


# In[91]:


# scaling dataset with robust scaler
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler.fit(df)
df = scaler.transform(df)


# # Machine Learning Model Building

# In[92]:


train_len = len(train)


# In[93]:


X_train = df[:train_len]
X_test = df[train_len:]
y_train = SalePrice

print(X_train.shape)
print(X_test.shape)
print(len(y_train))


# # Cross Validation

# In[94]:


from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, r2_score

def test_model(model, X_train=X_train, y_train=y_train):
    cv = KFold(n_splits = 3, shuffle=True, random_state = 45)
    r2 = make_scorer(r2_score)
    r2_val_score = cross_val_score(model, X_train, y_train, cv=cv, scoring = r2)
    score = [r2_val_score.mean()]
    return score


# # Linear Regression

# In[95]:


import sklearn.linear_model as linear_model
LR = linear_model.LinearRegression()
test_model(LR)


# In[96]:


# Cross validation
cross_validation = cross_val_score(estimator = LR, X = X_train, y = y_train, cv = 10)
print("Cross validation accuracy of LR model = ", cross_validation)
print("\nCross validation mean accuracy of LR model = ", cross_validation.mean())


# In[97]:


rdg = linear_model.Ridge()
test_model(rdg)


# In[98]:


lasso = linear_model.Lasso(alpha=1e-4)
test_model(lasso)


# # Fitting Polynomial Regression to the dataset

# from sklearn.preprocessing import PolynomialFeatures
# 
# poly_reg = PolynomialFeatures(degree = 2) X_poly = 
# 
# poly_reg.fit_transform(X_train) poly_reg.fit(X_poly, y_train) lin_reg_2 = 
# 
# LinearRegression()

# # lin_reg_2.fit(X_poly, y_train)

# # test_model(lin_reg_2,X_poly)

# import sklearn.linear_model as linear_model lin_reg_2 = linear_model.LinearRegression()

# # lin_reg_2.fit(X_poly, y_train)

# test_model(lin_reg_2,X_poly)

# # Support Vector Machine

# In[99]:


from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf')
test_model(svr_reg)


# # Decision Tree Regressor

# In[100]:


from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(random_state=21)
test_model(dt_reg)


# # Random Forest Regressor

# In[101]:


from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 1000, random_state=51)
test_model(rf_reg)


# # Bagging & boosting

# In[102]:


from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
br_reg = BaggingRegressor(n_estimators=1000, random_state=51)
gbr_reg = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, loss='ls', random_state=51)


# In[103]:


test_model(br_reg)


# In[104]:


test_model(gbr_reg)


# In[105]:


get_ipython().system('pip install xgboost')


# # XGBoost

# In[106]:


import xgboost
#xgb_reg=xgboost.XGBRegressor()
xgb_reg = xgboost.XGBRegressor(bbooster='gbtree', random_state=51)
test_model(xgb_reg)


# # SVM Model Bulding

# In[107]:


svr_reg.fit(X_train,y_train)
y_pred = np.exp(svr_reg.predict(X_test)).round(2)


# In[108]:


y_pred


# In[109]:


submit_test1 = pd.concat([test['Id'],pd.DataFrame(y_pred)], axis=1)
submit_test1.columns=['Id', 'SalePrice']


# In[110]:


submit_test1


# In[111]:


submit_test1.to_csv('sample_submission.csv', index=False )


# # SVM Model Bulding Hyperparameter Tuning

# # Hyperparameter Tuning

# from sklearn.model_selection import RandomizedSearchCV,
# 
# GridSearchCV params = {'kernel': ['linear', 'rbf', 'sigmoid'], 
# 
# 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'C': [0.1, 1, 10, 100, 1000], 
# 
# 'epsilon': [1, 0.2, 0.1, 0.01, 0.001, 0.0001]}

# rand_search = RandomizedSearchCV(svr_reg, 
# 
# param_distributions=params, n_jobs=-1, cv=11) 
# 
# rand_search.fit(X_train, y_train) rand_search.bestparams

# In[112]:


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
params = {'kernel': ['rbf'],
         'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
         'C': [0.1, 1, 10, 100, 1000],
         'epsilon': [1, 0.2, 0.1, 0.01, 0.001, 0.0001]}
rand_search = RandomizedSearchCV(svr_reg, param_distributions=params, n_jobs=-1, cv=11)
rand_search.fit(X_train, y_train)
rand_search.best_score_


# In[113]:


svr_reg= SVR(C=100, cache_size=200, coef0=0.0, degree=3, epsilon=0.01, gamma=0.0001,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
test_model(svr_reg)


# In[114]:


svr_reg.fit(X_train,y_train)
y_pred = np.exp(svr_reg.predict(X_test)).round(2)


# In[115]:


y_pred


# In[116]:


submit_test3 = pd.concat([test['Id'],pd.DataFrame(y_pred)], axis=1)
submit_test3.columns=['Id', 'SalePrice']


# In[117]:


submit_test3.to_csv('sample_submission.csv', index=False)
submit_test3


# Name Submitted Wait time Execution time Score 
# 
# sample_submission.csv 3 days ago 0 seconds 0 seconds 0.12612

# # XGBoost parameter tuning

# xgb2_reg = xgboost.XGBRegressor() params_xgb = { 
# 
# 'max_depth': range(2, 20, 2), 'n_estimators': range(99, 2001,
# 
# 80), 'learning_rate': [0.2, 0.1, 0.01, 0.05], 'booster': ['gbtree'],
# 
# 'mon_child_weight': range(1, 8, 1) } rand_search_xgb =
# 
# RandomizedSearchCV(estimator = xgb2_reg, 
# 
# param_distributions=params_xgb, n_iter=100, n_jobs=-1,
# 
# cv=11, verbose=11, random_state=51, return_train_score 
# 
# =True, scoring='neg_mean_absolute_error') 
# 
# rand_search_xgb.fit(X_train,y_train)
# 
# rand_search_xgb.bestscore
# 
# rand_search_xgb.bestparams

# In[118]:


xgb2_reg=xgboost.XGBRegressor(n_estimators= 899,
 mon_child_weight= 2,
 max_depth= 4,
 learning_rate= 0.05,
 booster= 'gbtree')

test_model(xgb2_reg)


# In[119]:


xgb2_reg.fit(X_train,y_train)
y_pred_xgb_rs=xgb2_reg.predict(X_test)


# In[120]:


np.exp(y_pred_xgb_rs).round(2)


# In[121]:


y_pred_xgb_rs = np.exp(xgb2_reg.predict(X_test)).round(2)
xgb_rs_solution = pd.concat([test['Id'], pd.DataFrame(y_pred_xgb_rs)], axis=1)
xgb_rs_solution.columns=['Id', 'SalePrice']
xgb_rs_solution.to_csv('sample_submission.csv', index=False)


# In[122]:


xgb_rs_solution


# 1603 0.12484 2 1d Your Best Entry Your submission scored 0.12484,
# 
# which is an improvement of your previous score of 0.12612.
# 
# Great job! Tweet this!

# # Feature Engineering / Selection to improve accuracy

# In[123]:


# correlation Barplot
plt.figure(figsize=(9,16))
corr_feat_series = pd.Series.sort_values(train.corrwith(train.SalePrice))
sns.barplot(x=corr_feat_series, y=corr_feat_series.index, orient='h')


# In[124]:


df_back1 = df_back


# In[125]:


df_back1.to_csv('df_for_feature_engineering.csv', index=False)


# In[126]:


list(corr_feat_series.index)


# In[ ]:




