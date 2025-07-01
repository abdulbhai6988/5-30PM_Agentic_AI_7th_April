import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv(r"C:\Users\rahee\Downloads\Investment.csv")
data
x=data.iloc[:,:-1]
y=data.iloc[:,4]

x=pd.get_dummies(x,dtype=int)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train, y_train)

y_pred=regressor.predict(x_test)

intercept=regressor.intercept_
print(intercept)

slope=regressor.coef_
print(slope)

bias=regressor.score(x_train,y_train)
print(bias)

variance=regressor.score(x_test,y_test)
print(variance)

x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)

import statsmodels.api as sm
x_opt=x[:,[0,1,2,3,4,5]]
#Ordinary Least Squares
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
x_opt=x[:,[0,1,2,3,5]]
#Ordinary Least Squares
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
x_opt=x[:,[0,1,2,3]]
#Ordinary Least Squares
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
x_opt=x[:,[0,1,3]]
#Ordinary Least Squares
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
x_opt=x[:,[0,1]]
#Ordinary Least Squares
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()



