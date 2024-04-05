#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn import metrics
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import iplot
from warnings import filterwarnings

filterwarnings("ignore")


# In[3]:


df = pd.read_csv("HR Data.CSV")
df.head()


# In[ ]:


##Classification Meaning of Few columns 

#Education: 1 'Below College' 2 'College' 3 'Bachelor' 4 'Master' 5 'Doctor'

#EnvironmentSatisfaction: 1 'Low' 2 'Medium' 3 'High' 4 'Very High'

#JobInvolvement 1 'Low' 2 'Medium' 3 'High' 4 'Very High'

#JobSatisfaction: 1 'Low' 2 'Medium' 3 'High' 4 'Very High'

#PerformanceRating: 1 'Low' 2 'Good' 3 'Excellent' 4 'Outstanding'

#RelationshipSatisfaction: 1 'Low' 2 'Medium' 3 'High' 4 'Very High'

#WorkLifeBalance: 1 'Bad' 2 'Good' 3 'Better' 4 'Best'


# In[4]:


df.info()


# In[5]:


df1 = df.select_dtypes("object")
df1


# In[6]:


df2 = df.select_dtypes("int64")
df2


# Observation: 
# 
# There are 35 columns and 1470 rows. This means we have 34 features, 1 target to investigate, and 1470 different person's information.
# 
# The target is binary.
# 
# Out of the 34 available features, 8 of them are in the form of objects and the rest are in the form of numbers.

# In[10]:


df1.describe().T


# In[11]:


df.describe(include='int64').T


# In[12]:


df.isnull().sum()


# In[13]:


have_duplicate_rows = df.duplicated().any()
have_duplicate_rows


# There is no missing value and duplicate rows in dataset.
# 
# 🧹Some of columns can be removed, because their values do not affect the analysis results.
# 
# Over18: All values are Y
# EmployeeCount: all values are 1.0
# StandardHours: all values are 80.0
# EmployeeNumber: is the id of the employee that their values do not affect the analysis results.

# In[14]:


# remove 4 unimpactfull columns from data
df = df.drop(['Over18', 'EmployeeNumber','EmployeeCount','StandardHours'],axis=1)


# In[15]:


# Seperating Numerical and Categorial Data 

cat = df.select_dtypes(['object']).columns
num = df.select_dtypes(['number']).columns
print(cat)
print(num)


# In[16]:


for i in cat:
    print('Unique values of ', i, set(df[i]))


# In[17]:


# univariate analysis of categorical data:
sns.set(rc={"axes.facecolor":"white","figure.facecolor":"#9ed9cd"})
sns.set_palette("pastel")
for i, col in enumerate(cat):

    fig, axes = plt.subplots(1,2,figsize=(10,5))

    # count of col (countplot)
    
    ax=sns.countplot(data=df, x=col, ax=axes[0])
    activities = [var for var in df[col].value_counts().sort_index().index]
    ax.set_xticklabels(activities,rotation=90)
    for container in axes[0].containers:
        axes[0].bar_label(container)
        
    #count of col (pie chart)
    
    index = df[col].value_counts().index
    size = df[col].value_counts().values
    explode = (0.05, 0.05)

    axes[1].pie(size, labels=index,autopct='%1.1f%%', pctdistance=0.85)

    # Inner circle
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.suptitle(col,backgroundcolor='black',color='white',fontsize=15)

    plt.show()


# In[18]:


for column in cat : 
    plt.figure(figsize=(10,5))
    
    ax=sns.countplot(x=df[column], data=df,hue="Attrition")
    for container in ax.containers:
        ax.bar_label(container)
    plt.title(column,backgroundcolor='black',color='white',fontsize=20)
    plt.xticks(rotation=90)
    plt.xlabel(column,fontsize=20)
    plt.grid()
    plt.show()


# In[19]:


plt.figure(figsize = (15,25))
for idx, i in enumerate(num):
    plt.subplot(12, 2, idx + 1)
    sns.boxplot(x = i, data = df)
    plt.title(i,backgroundcolor='black',color='white',fontsize=15)
    plt.xlabel(i, size = 12)
plt.tight_layout()                     
plt.show()


# In[24]:


plt.figure(figsize=(10,5))

ax = sns.countplot(x=df['JobLevel'], data=df,hue="Attrition")
#for container in ax.containers:
        #ax.bar_label(container)
plt.title('JobLevel',backgroundcolor='black',color='white',fontsize=20)
plt.xlabel('JobLevel',fontsize=20)
plt.grid()
plt.show()


# In[25]:


plt.figure(figsize=(5,10))
sns.relplot(data=df, y="MonthlyIncome", x="Age", hue="Attrition",col='Gender')
plt.show()


# In[26]:


plt.figure(figsize=(5,10))
sns.relplot(data=df, y="MonthlyIncome", x="MaritalStatus", hue="Attrition",col='Gender')
plt.show()


# In[27]:


plt.figure(figsize=(5,10))
sns.relplot(data=df, y="MonthlyIncome", x="Department", hue="Attrition",col='Gender')
plt.show()


# In[28]:


plt.figure(figsize=(5,10))
sns.relplot(data=df, y="MonthlyIncome", x="Education", hue="Attrition",col='Gender')
plt.show()


# In[29]:


plt.figure(figsize=(5,10))
ax=sns.relplot(data=df, y="MonthlyIncome", x="JobRole", hue="Attrition",col='Gender')
rotation = 90 
for i, ax in enumerate(ax.fig.axes):   ## getting all axes of the fig object
     ax.set_xticklabels(ax.get_xticklabels(), rotation = rotation)
#plt.xticks(rotation=90)
plt.show()


# In[30]:


plt.figure(figsize=(5,10))
sns.relplot(data=df, y="MonthlyIncome", x="BusinessTravel", hue="Attrition",col='Gender')
plt.xticks(rotation=90)
plt.show()


#  Analysis of graphs
#  
# Attrition is the highest for both men and women from 18 to 35 years of age and gradually decreases.
# As income increases, attrition decreases.
# Attrition is much, much less in divorced women.
# Attrition is higher for employees who usually travel than others, and this rate is higher for women than for men.
# Attrition is the highest for those in level 1 jobs.
# Women with the job position of manager, research director and technician laboratory have almost no attrition.
# Men with the position of sales expert have a lot of attrition.

# In[31]:


df_copy = df.copy()


# In[32]:


# Converting categorial values having 2 attributes to numerical using LabelEncoder

from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
df_copy['Attrition']=label_encoder.fit_transform(df['Attrition'])
df_copy['OverTime']=label_encoder.fit_transform(df['OverTime'])
df_copy['Gender']=label_encoder.fit_transform(df['Gender'])


# In[33]:


#convert category attributes with more than 2 distinct values to numeric using one-hot vector
df_copy=pd.get_dummies(df_copy, columns=['BusinessTravel', 'Department', 'EducationField', 
                               'JobRole', 'MaritalStatus'])


# In[34]:


# To find the correlation between attributes
plt.figure(figsize=(20,10))
correlations=df_copy.corr()
correlations['Attrition'].sort_values(ascending = False).plot(kind='bar');


# In[35]:


df2_copy = df2.copy()
df2_copy = df2_copy.drop(['EmployeeNumber','EmployeeCount','StandardHours'],axis=1)

corr = df2_copy.corr(method = "spearman")
sns.set(style="white")

mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=(12, 10), dpi = 100)
sns.heatmap(corr, mask = mask, cmap= "winter", vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .8}, annot = True, fmt = ".2f")
plt.show()


# There are high correlation between some features:
# 
# MonthlyIncome & joblevel
# year in currentrol , year at company, year with current manager & year in current role

# In[36]:


DF = df.copy()


# In[37]:


DF['BusinessTravel'] = DF['BusinessTravel'].replace('Travel_Rarely',2)
DF['BusinessTravel'] = DF['BusinessTravel'].replace('Travel_Frequently',3)
DF['BusinessTravel'] = DF['BusinessTravel'].replace('Non-Travel',4)

DF['Attrition'] = DF['Attrition'].replace('Yes',2)
DF['Attrition'] = DF['Attrition'].replace('No',3)

DF['OverTime'] = DF['OverTime'].replace('Yes',2)
DF['OverTime'] = DF['OverTime'].replace('No',3)

DF['Gender'] = DF['Gender'].replace('Male',2)
DF['Gender'] = DF['Gender'].replace('Female',3)

DF['MaritalStatus'] = DF['MaritalStatus'].replace('Single',2)
DF['MaritalStatus'] = DF['MaritalStatus'].replace('Married',3)
DF['MaritalStatus'] = DF['MaritalStatus'].replace('Divorced',4)

DF['Department'] = DF['Department'].replace('Sales',2)
DF['Department'] = DF['Department'].replace('Human Resources',3)
DF['Department'] = DF['Department'].replace('Research & Development',4)

DF['EducationField'] = DF['EducationField'].replace('Life Sciences',2)
DF['EducationField'] = DF['EducationField'].replace('Medical',3)
DF['EducationField'] = DF['EducationField'].replace('Marketing',4)
DF['EducationField'] = DF['EducationField'].replace('Technical Degree',2)
DF['EducationField'] = DF['EducationField'].replace('Human Resources',3)
DF['EducationField'] = DF['EducationField'].replace('Other',4)

DF['JobRole'] = DF['JobRole'].replace('Sales Executive',2)
DF['JobRole'] = DF['JobRole'].replace('Manufacturing Director',3)
DF['JobRole'] = DF['JobRole'].replace('Healthcare Representative',4)
DF['JobRole'] = DF['JobRole'].replace('Manager',2)
DF['JobRole'] = DF['JobRole'].replace('Research Director',3)
DF['JobRole'] = DF['JobRole'].replace('Laboratory Technician',4)
DF['JobRole'] = DF['JobRole'].replace('Sales Representative',2)
DF['JobRole'] = DF['JobRole'].replace('Research Scientist',3)
DF['JobRole'] = DF['JobRole'].replace('Human Resources',4)


# In[38]:


DF = DF.drop(['MonthlyIncome' ,'YearsInCurrentRole' , 'YearsAtCompany', 'YearsWithCurrManager'],axis=1)


# In[39]:


#Normalizeing using MinMax scaler 
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler(feature_range = (0,1))
DF1 = DF.drop(columns=['Attrition'])
norm = scaler.fit_transform(DF)
norm_df = pd.DataFrame(norm,columns=DF.columns)


# In[40]:


X = pd.DataFrame(norm_df.drop(columns='Attrition'))
Y = pd.DataFrame(norm_df.Attrition).values.reshape(-1, 1)


# In[41]:


x_train  , x_test , y_train, y_test = train_test_split (X ,Y ,test_size = 0.2 , random_state = 0)


# In[43]:


from sklearn.ensemble import RandomForestClassifier


# In[47]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

rfc = RandomForestClassifier()
rfc = rfc.fit(X_train , Y_train)
Y_pred = rfc.predict(X_test)

print ('accuracy',metrics.accuracy_score(Y_test, Y_pred))


# In[49]:


from sklearn.metrics import accuracy_score,f1_score, confusion_matrix, classification_report, roc_auc_score

fig, ax = plt.subplots(figsize=(10,5))
cm = metrics.confusion_matrix(Y_test,Y_pred)
sns.heatmap(metrics.confusion_matrix(Y_test,Y_pred),annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.xlabel('y prediction')
plt.ylabel('y actual')
plt.show()

print(classification_report(Y_test, Y_pred))


# In[ ]:




