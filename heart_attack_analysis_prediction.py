
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 21:24:49 2021

@author: khemrajregmi
"""

#importing libraries
import pandas as pnd                       

#importing the dataset
datum = pnd.read_csv('heart.csv')     #getting dataset

#independent variable matrix
X = datum.iloc[:,0:8].values         #splitting into independed variable matrix

#dependent variable
y = datum.iloc[:,8].values           #splitting into depended variable matrix

#train test split
from sklearn.model_selection import train_test_split                                    #splitting train and             
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2 ,random_state = 0) #test data

#scaling the dataset to take dataset to a common scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler();
X_train = sc_X.fit_transform(X_train);
X_test = sc_X.fit_transform(X_test);

#performing logistic regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train,y_train)

#prediction
y_predict = log_reg.predict(X_test)

#confusion matrix for the accuracy of system
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predict)
accuracy = (cm[0,0]+cm[1,1])/len(X_test) * 100
print("accuracy " + str(accuracy) + "%" )
# print('this is y test')
print(y_test.reshape(-1,1))
print(y_predict.reshape(-1,1))
# y_test
# y_predict
