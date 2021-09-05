#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import requests
import json
import networkx as nx
import pickle
import time
from networkx.readwrite import json_graph
import time
from networkx.algorithms.distance_measures import diameter
import csv
import operator
import argparse
import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support


# In[3]:


import inspect

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]


# In[4]:


# Train Test Split
users = pd.read_csv('company.csv')

print(len(list(users.columns)))


# In[5]:


users[users['company'] != None].head()


# In[6]:


users_with_company = users[users['is_good'] == 1]
users_without_company = users[users['is_good'] == 0]

print(users_without_company.shape)
print(users_with_company.shape)


# In[18]:


from sklearn.model_selection import train_test_split
if 'is_good' in list(users.columns):
    is_good = users['is_good']
    del users['is_good']

if 'company' in list(users.columns):
    del users['company']


# In[20]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing

users.fillna('NONE')

# create the Labelencoder object
le = preprocessing.LabelEncoder()

#convert the categorical columns into numeric
# users['company'] = le.fit_transform(users['company'].astype(str))

X_train, X_test, Y_train, Y_test = train_test_split(
    users, 
    is_good, 
    train_size=0.7, 
    test_size=0.3, 
    random_state=42
)

for v in [X_train, X_test, Y_train, Y_test]:
    print("Size of {}: {}".format(retrieve_name(v), v.shape))


# In[21]:


model = RandomForestClassifier()
model.fit(X_train, Y_train)
target = model.predict(X_test)
score = accuracy_score(Y_test,target)
print('Model Name: RandomForestClassifier, Accuracy: {}'.format(score))
print(precision_recall_fscore_support(Y_test, target, average='macro'))

model = LogisticRegression()
model.fit(X_train, Y_train)
target = model.predict(X_test)
score = accuracy_score(Y_test,target)
print('Model Name: LogisticRegression, Accuracy: {}'.format(score))
print(precision_recall_fscore_support(Y_test, target, average='macro'))

model = MultinomialNB()
model.fit(X_train, Y_train)
target = model.predict(X_test)
score = accuracy_score(Y_test,target)
print('Model Name: MultinomialNB, Accuracy: {}'.format(score))
print(precision_recall_fscore_support(Y_test, target, average='macro'))

model = KNeighborsClassifier()
model.fit(X_train, Y_train)
target = model.predict(X_test)
score = accuracy_score(Y_test,target)
print('Model Name: KNeighborsClassifier, Accuracy: {}'.format(score))
print(precision_recall_fscore_support(Y_test, target, average='macro'))


# In[22]:


import xgboost as xgb
model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1,objective= 'multi:softprob',n_jobs=-1, num_class=2)
model.fit(X_train, Y_train)
target = model.predict(X_test)
score = accuracy_score(Y_test,target)
print('Model Name: XGBoostClassifier, Accuracy: {}'.format(score))
print(precision_recall_fscore_support(Y_test, target, average='macro'))


# In[23]:


import matplotlib.pylab as plt
from matplotlib import pyplot
from xgboost import plot_importance
plot_importance(model, max_num_features=10) # top 10 most important features
plt.show()


# In[ ]:




