#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


from matplotlib import pyplot as plt


# In[14]:


import seaborn as sns


# In[18]:


# Loading dataset
df = pd.read_csv("Titanic-Dataset.csv")
df.head(5)


# In[19]:


df.shape


# In[20]:


df.info()


# In[21]:


# To find out the missing percentage of values in all rows 
# Since cabin column having 77% of missing data, we can remove that column

df.isnull().sum()/len(df)


# In[22]:


# To find out the Qualitative(numerical) analysis

df.describe()


# In[23]:


# Exploring and Analysing Data


# In[27]:


sns.countplot(x = 'Survived', data = df)


# In[ ]:


# 0 : Non-Survived and 1 : Survived
# From this Graph we could understand that the survived people are less(near to 300) that the unsurvived(greater than 500) 


# In[26]:


sns.countplot(x ='Survived',hue ='Sex',data = df)


# In[28]:


sns.countplot(x ='Survived',hue ='Pclass',data = df)


# In[ ]:


# Analysis - 0 : Non-survived, 1 : Survived
# Females have more ssurvival chance of nearly 50% than Male
# People travelling in 1St class have more survival chance, where 3rd class passengers have the least chance to survive.


# In[25]:


df['Age'].plot.hist()


# In[ ]:


# Most of the passengers Travelling in titanic are young aged people(20-40)
# Old aged people between age 60-80 are very less


# In[30]:


df['Fare'].plot.hist()


# In[ ]:


# Most of  the tickets sold are below 150, only few tickets were sold in range 200-350


# In[31]:


sns.countplot(x ='Parch',data = df)


# In[32]:


sns.countplot(x ='SibSp',data = df)


# In[ ]:


# The number of siblings and parents who aboard the ship are less


# In[ ]:


# Data Wrangling


# In[33]:


sns.heatmap(df.isnull(), cmap = 'spring')


# In[34]:


sns.boxplot(x = "Pclass", y = "Age", data = df)


# In[ ]:


# Older aged people are travelling in class 1 than in class 2 and class 3


# In[35]:


df.drop('Cabin', axis = 1, inplace = True)


# In[36]:


df.dropna(inplace = True)


# In[37]:


sns.heatmap(df.isnull())


# In[ ]:


MODEL BUILDING


# In[ ]:


# Before building a model, we are converting other categorical values to numerical data as ML Algorithms works with numerical values only.
# We use One-Hot encoding technique to convert these to 0's and 1's


# In[39]:


pd.get_dummies(df['Sex']).head(10)


# In[40]:


sex = pd.get_dummies(df['Sex'],drop_first = True)
sex.head(5)


# In[ ]:


# We removed the first column, since we need only one parameter (either 1 as male or 0 as Female)


# In[41]:


Embrk = pd.get_dummies(df['Embarked'])
Embrk.head()


# In[42]:


Pcl = pd.get_dummies(df['Pclass'])
Pcl.head(5)


# In[73]:


df = pd.concat([df,sex,Embrk,Pcl],axis = 1)
df


# In[85]:


df.info()


# In[58]:


import warnings
warnings.filterwarnings('ignore')


# In[79]:


X = df.drop('Survived',axis = 1)
y = df['Survived']


# In[96]:


X= X.rename(str,axis="columns") 


# In[97]:


from sklearn.model_selection import train_test_split


# In[98]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state = 42)


# In[115]:


from sklearn.metrics import accuracy_score,f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# In[100]:


rf = RandomForestClassifier()


# In[101]:


rf.fit(X_train, y_train)


# In[102]:


y_pred = rf.predict(X_test)


# In[108]:


print(classification_report(y_test, y_pred))


# In[106]:


# By using Random Forest algorithm we achieved 75% accuracy score of predicting the survival of people in titanic


# In[112]:


lm = LogisticRegression()


# In[109]:


lm.fit(X_train, y_train)


# In[110]:


prediction = lm.predict(X_test)


# In[111]:


classification_report(y_test, prediction)


# In[113]:


from sklearn.metrics import accuracy_score


# In[116]:


accuracy_score(y_test, prediction)


# In[ ]:


We could observe  here that accuracy score is 81% which makes our model a good model to predict the values accurately , here in Titanic data set our model accurately predicts as to who will survive and who will not survive.

Through Visualization we found out that females have more chances of survival than males, class 1 have more changes of survival, youth age group 20-35 yrs male from class 3 have not survived.

Further, other Machine Learning Algorithms can be applied on the same data set, Ensemble algorithms to boost the performance of the model and get good predictions

