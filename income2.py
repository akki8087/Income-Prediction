# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 00:06:52 2018

@author: akshay
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Loading the data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#data type
train.dtypes

# data exploration
## numeical variables
train.describe()
'''
## categorical variables
categorical_variables = train.dtypes.loc[train.dtypes == 'object'].index 
print(categorical_variables)
train[categorical_variables].apply(lambda x: len(x.unique()))

#univariate analysis
train["Workclass"].value_counts()
train["Sex"].value_counts()/train.shape[0]
train["Race"].value_counts()
train["Marital.Status"].value_counts()/train.shape[0]

#multivariate analysis
#cat-cat
ct = pd.crosstab(train['Sex'],train['Income.Group'],margins = True)
ct.iloc[:-1,:-1].plot(kind = "bar",stacked = True,color = ['red','blue'],grid = False)
def percConvert(ser):
    return ser/float(ser[-1])
ct2 = ct.apply(percConvert,axis=1)
ct2.iloc[:-1,:-1].plot(kind = "bar",stacked = True,color = ['red','blue'],grid = False)

#cont-cont
train.plot('Age','Hours.Per.Week',kind ='scatter')

#cont-cat
train.boxplot(column = 'Hours.Per.Week',by = 'Sex')
'''
#missing values
train.apply(lambda x: sum(x.isnull()))
test.apply(lambda x: sum(x.isnull()))

#from scipy.stats import mode
#mode(train['Workclass'],nan_policy ='omit')

train['Occupation'].fillna('Prof-specialty',inplace=True)
train['Workclass'].fillna('Private',inplace=True)
train['Native.Country'].fillna('United-States',inplace=True)
train.apply(lambda x: sum(x.isnull()))

test['Occupation'].fillna('Prof-specialty',inplace=True)
test['Workclass'].fillna('Private',inplace=True)
test['Native.Country'].fillna('United-States',inplace=True)
test.apply(lambda x: sum(x.isnull()))

#variable transformation
train["Workclass"].value_counts()/train.shape[0]

var_combi = ['State-gov','Self-emp-inc','Federal-gov','Without-pay','Never-worked']

for var in var_combi:
    train['Workclass'].replace({var:'Others'},inplace = True)
    test['Workclass'].replace({var:'Others'},inplace = True)
test["Workclass"].value_counts()/train.shape[0]    
    
categorical_variables = train.dtypes.loc[train.dtypes == 'object'].index 
print(categorical_variables)
train[categorical_variables].apply(lambda x: len(x.unique()))    

categorical_variables = categorical_variables[1:]

for cat in categorical_variables:
    frq = train[cat].value_counts()/train.shape[0]    
    cat_to_combine = frq.loc[frq.values<0.05].index
    for var in cat_to_combine:
        train[cat].replace({var:'Others'},inplace = True)
        test[cat].replace({var:'Others'},inplace = True)

from sklearn.preprocessing import LabelEncoder
categorical_variables = train.dtypes.loc[train.dtypes == 'object'].index 
print(categorical_variables)

le = LabelEncoder()
for var in categorical_variables[:-1]:
    train[var] = le.fit_transform(train[var])
    test[var] = le.fit_transform(test[var])
    
train['Income.Group'] = le.fit_transform(train['Income.Group'])     
train.dtypes    


depe_var = train.iloc[:,-1]
independent_variable = train.iloc[:,1:-1]
test_i = test.iloc[:,1:]
'''
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth = 10,min_samples_leaf = 100,max_features = 'sqrt',random_state = 0)
model.fit(independent_variable,depe_var)
'''


# Random forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 300,criterion = 'entropy',random_state = 0)
cl = classifier.fit(independent_variable,depe_var)
#y_pred = classifier.predict(independent_variable)

train_predict = classifier.predict(independent_variable)

test_predict = classifier.predict(test_i)

from sklearn.metrics import accuracy_score

aac_train = accuracy_score(depe_var,train_predict)
print(aac_train)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = independent_variable, y = depe_var, cv = 10)
print(accuracies.mean())
print(accuracies.std())
fi = cl.feature_importances_
print(fi)

result = pd.DataFrame()
result['ID'] = test['ID']

result['Income.Group'] = test_predict
result['Income.Group'].replace({1:'>50K',0:'<=50K'},inplace = True)

result.to_csv('income2.csv',index = False)
