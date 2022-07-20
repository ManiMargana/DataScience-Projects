#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#data gathering
data=pd.read_csv('C:\\Users\\HP-DK0272TX\\OneDrive\\Desktop\\file\\Codingrad\\fetal_health.csv')
data.head()


# In[3]:


data.info()


# In[4]:


data.describe()


# In[5]:


data.isnull().sum()


# In[6]:


data.corr()


# In[7]:


corr = data.corr()
plt.figure(figsize=(15,14))
sns.heatmap(corr, annot = True)


# # Visualization:

# In[8]:


data.fetal_health.value_counts()


# In[9]:


sns.set_style(style='whitegrid')
sns.countplot(data=data, x='fetal_health')
plt.title("Number of samples of each class")


# In[10]:


grouped = data.groupby(by='fetal_health').mean()
grouped


# In[11]:


for index,i in enumerate(grouped.columns,start=1):
    plt.figure(figsize=(5,4))
    sns.barplot(data=grouped, x=grouped.index, y=grouped[i])
    plt.show() 


# In[12]:


data.columns


# In[13]:


sns.displot(x="baseline value", data=data, hue="fetal_health")


# In[14]:


sns.barplot(x="fetal_health",y="fetal_movement",data=data)


# In[15]:


list = data[['light_decelerations', 'severe_decelerations','prolongued_decelerations']]
plt.figure(figsize = (6,5))   
sns.histplot(data=list)


# In[16]:


plt.figure(figsize=(10,9))
sns.scatterplot(data=list)


# In[17]:


plt.figure(figsize=(10, 10))

plt.pie(
    data['fetal_health'].value_counts(),
    autopct='%.2f%%',
    labels=["NORMAL", "SUSPECT", "PATHOLOGICAL"],
    colors=sns.color_palette('Greens')
)

plt.title("Fetal Health Distribution")
plt.show()


# In[18]:


cor=data.select_dtypes(exclude="object").corr()
Num_feature = cor["fetal_health"].sort_values(ascending=False).head(20).to_frame()

cm = sns.light_palette("#5F9EA0", as_cmap=True)

style = Num_feature.style.background_gradient(cmap=cm)
style


# We can observe here that the three features ("prolongued_decelerations", "abnormal_short_term_variability", "percentage_of_time_with_abnormal_long_term_variability") have high correlation with the target culumn (fetal_health).

# # Taking features and target columns:

# In[19]:


features=data.iloc[:,:-1]
target=data.iloc[:,-1]


# In[20]:


features


# In[21]:


target


# # MODEL:

# In[22]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score


# In[23]:


## Train Test split

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=101)


# In[24]:


X_train.shape , X_test.shape , y_train.shape , y_test.shape


# In[25]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)

X_train = pd.DataFrame(scaler.transform(X_train), 
                       index = X_train.index, columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), 
                      index=X_test.index, columns=X_test.columns)


# Comparing the accuracies of different models:

# In[26]:


models = {
    "                   Logistic Regression": LogisticRegression(),
    "                   K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
    "                         Decision Tree": DecisionTreeClassifier(),
    "Support Vector Machine (Linear Kernel)": LinearSVC(),
    "   Support Vector Machine (RBF Kernel)": SVC(),
    "                        Neural Network": MLPClassifier(),
    "                         Random Forest": RandomForestClassifier(),
    "                     Gradient Boosting": GradientBoostingClassifier()
                                   
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(name + " : {:.2f}%".format(accuracy_score(y_pred,y_test)*100))


# In[27]:


from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error
cv_method = StratifiedKFold(n_splits=3,random_state=42,shuffle=True)


# In[28]:


knn_scores=[]
for k in range(1,20):
    knn=KNeighborsClassifier(n_neighbors=k)
    scores=cross_val_score(knn,X_train,y_train,cv=5)
    knn_scores.append(scores.mean())

x_ticks = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
x_labels = x_ticks

plt.plot([k for k in range(1,20)],knn_scores)
# plt.xticks(ticks=x_ticks, labels=x_labels)
plt.grid()


# In[29]:


knn_scores


# In[30]:


knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
from sklearn.metrics import confusion_matrix
confusion_knn=confusion_matrix(y_test,knn.predict(X_test))
sns.heatmap(confusion_knn,annot=True)

pd.DataFrame(confusion_matrix(y_test,knn.predict(X_test)),columns=['Predicted_Normal','Predicted_Suspect','Predicted_Pathological'],
             index=['Actual_Normal','Actual_Suspect','Actual_Pathological'])


# # Tuning the Model using GridSearchCV:

# In[31]:


#Parameter tuning with GridSearchCV 

estimator_KNN = KNeighborsClassifier(algorithm='auto')
parameters_KNN = {
    'n_neighbors': (1,10, 1),
    'leaf_size': (20,40,1),
    'p': (1,2),
    'weights': ('uniform', 'distance'),
    'metric': ('minkowski', 'chebyshev')}
                   
# with GridSearch
grid_search_KNN = GridSearchCV(estimator=estimator_KNN,param_grid=parameters_KNN,
                               scoring = 'accuracy',n_jobs = -1,cv = 5)


# In[32]:


KNN = grid_search_KNN.fit(X_train, y_train)


# In[33]:


ypred_KNN =KNN.predict(X_test)
ypred_KNN


# In[34]:


# Get the best estimator values.

best_estimator_knn = grid_search_KNN.best_estimator_
print("Best estimator values for KNN model:", {best_estimator_knn})

#Parameter setting that gave the best results on the hold out data.
print("\nBest parameter values: ", grid_search_KNN.best_params_ ) 

#Mean cross-validated score of the best_estimator
print('\nBest Score - KNN:', grid_search_KNN.best_score_ )


# In[35]:


y_pred = model.predict(X_test)
ax= plt.subplot()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, ax = ax, cmap = "Blues");

# labels, title and ticks
ax.set_xlabel("Predicted labels");
ax.set_ylabel("Actual labels"); 
ax.set_title("Confusion Matrix"); 
ax.xaxis.set_ticklabels(["Normal", "Suspect", "Pathological"]);


# In[36]:


#Saving the Tuned KNN model to the pickle file
pickle.dump(grid_search_KNN,open('GridSearchCV_knnModel_fetal-health.pkl','wb'))


# In[ ]:




