#!/usr/bin/env python
# coding: utf-8

import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.svm import SVC

#reading the data
df = pd.read_excel(r"C:\Users\Desktop\classification\studentMarks.xlsx", index=False)

#separating to X, y
X=df.iloc[:,2:5]
#print(X)
y=df.iloc[:,5:]
#print(y)

#splitting the data in test and train
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)

#creating model
model = SVC(kernel='linear',C=1,gamma=1)

#training the model, vlaue.ravel for converting 1d array
model.fit(X_train, y_train.values.ravel())

#finding accuracy
predicted_classes = model.predict(X_test)
accuracy = accuracy_score((y_test.values.flatten()),predicted_classes)
print(accuracy)

#saving the trained model
joblib.dump(model,'marksmodel_joblib')

