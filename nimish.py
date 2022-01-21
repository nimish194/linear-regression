#!/usr/bin/env python
# coding: utf-8

# In[3]:


# set pandas options
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[4]:


# 1. Import data
titanic = pd.read_csv('titanic.csv')
    
titanic_soham = pd.DataFrame(titanic)


# In[154]:


titanic_soham.head(3)


# In[155]:



print(titanic_soham.shape)
titanic_soham.dtypes


# In[156]:


columnList = []
for column in titanic_soham.columns:
    columnList.append(column)


# In[157]:


#Missing values
for column in columnList:
    print("The number of missing values in "+column+" is ", titanic_soham[column].isnull().sum())


# In[158]:


titanic_soham.head(10)


# In[159]:


# print unique values of Sex and Pclass column
unique_values = ['Sex', 'Pclass']
for column in unique_values:
    print("The unique values in "+column+" are: ", titanic_soham[column].unique())


# In[2]:


#a.	A bar chart showing the # of survived versus the passenger class
pd.crosstab(titanic_soham.Pclass, titanic_soham.Survived).plot(kind='bar')
plt.title('"sohams" Survived vs Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survived')



# In[161]:


#b.	A bar chart showing the # of survived versus gender
pd.crosstab(titanic_soham.Sex, titanic_soham.Survived).plot(kind="bar")
plt.title("sohams Survived vs Gender")
plt.xlabel('Gender')
plt.ylabel('Survived')


# In[162]:


# 2.scatter matrix to plot the relationships between the number of survived 
pd.DataFrame(np.random.randn(1000, 5), columns =['Sex', 'Pclass', 'SibSp', 'Fare', 'Parch'])
pd.plotting.scatter_matrix(titanic_soham, alpha=0.2)


# In[163]:


#Dropping the columns with not enough data
titanic_soham = titanic_soham.drop(['Name', 'Ticket', 'Cabin'], axis=1)


# In[164]:


# Dummy variables

titanic_soham = pd.get_dummies(titanic_soham, columns=['Sex'], prefix='Sex')
titanic_soham


# In[165]:



titanic_soham = pd.get_dummies(titanic_soham, columns=['Embarked'], prefix='Embarked')
titanic_soham


# In[166]:


#Replace the missing values in the Age with the mean of the age

titanic_soham['Age'] = titanic_soham['Age'].fillna(titanic_soham['Age'].mean())
titanic_soham


# In[167]:


for col in titanic_soham.columns:
    titanic_soham[col] = pd.Series(titanic_soham[col],dtype=np.dtype("float"))


# In[168]:


titanic_soham.info()


# In[169]:


def normalizes(df):
    for column in df.columns:
        df[column] = (df[column] - df[column].min()) /             (df[column].max() - df[column].min())
    return df


# In[187]:


titanic_soham = normalizes(titanic_soham)
print(titanic_soham)


# In[171]:


hist = titanic_soham.hist(figsize=(9,10))


# In[172]:


titanic_soham.columns


# In[173]:


cols = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']


# In[174]:


X = titanic_soham[cols]
y = titanic_soham['Survived']


# In[175]:


from sklearn.model_selection import train_test_split
X_train_soham,X_test_soham,y_train_soham,y_test_soham=train_test_split(X,y,test_size=0.3,random_state=2)


# In[176]:


# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
soham_model = LogisticRegression(solver='lbfgs', multi_class='multinomial')  

# fit the model with data
soham_model.fit(X_train_soham,y_train_soham)


# In[177]:


probs = soham_model.predict_proba(X_test_soham)
probs


# In[178]:


predicted = soham_model.predict(X_test_soham)
predicted


# In[179]:


accuracy_score = metrics.accuracy_score(y_test_soham, predicted)
accuracy_score*100


# In[180]:


metrics.confusion_matrix(y_test_soham, predicted) 


# In[181]:


from sklearn.model_selection import cross_val_score, cross_validate
modelCV = LogisticRegression()
scoring = {'accuracy': 'accuracy', 'log_loss': 'neg_log_loss', 'auc': 'roc_auc'}
results = cross_validate(modelCV, X_train_soham, y_train_soham, cv=10, scoring=list(scoring.values()), 
                         return_train_score=False)


# In[182]:


results


# In[183]:


results['test_accuracy'].mean()


# In[184]:


#classification report
from sklearn.metrics import classification_report

print(classification_report(y_test_soham, predicted, target_names=['Died', 'Survived']))


# In[185]:


prob = probs[:, 1]
prob_df = pd.DataFrame(prob)
prob_df['predict'] = np.where(prob_df[0] >= 0.05, 1, 0)
y_test_soham = (y_test_soham == 'yes').astype(int)
Y_A = y_test_soham.values
Y_P = np.array(prob_df['predict'])


# In[186]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_A, Y_P)
print(confusion_matrix)

