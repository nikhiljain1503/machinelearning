import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('G:/50_Startups.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
print("##",X[0])
labelEncoder = LabelEncoder()
X[:,3] = labelEncoder.fit_transform(X[:,3])

print("@@",X[0])

onehotencoder = OneHotEncoder(categorical_features = [3])
X=onehotencoder.fit_transform(X).toarray()


X=X[:,1:]

from sklearn.model_selection import  train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=1,test_size=0.2)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train,Y_train)

y_pred  = regressor.predict(X_test)

print("^^^",regressor.coef_)
