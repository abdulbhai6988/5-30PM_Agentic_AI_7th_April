import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

exp=pd.read_csv(r"C:\Users\rahee\Downloads\Salary_Data.csv")

x=exp.iloc[:,:-1]
y=exp.iloc[:,-1]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary Vs Experience (Test Set)')
plt.xlabel('Years of Experiemce')
plt.ylabel('Salary')
plt.show()

m_slope=regressor.coef_
print(m_slope)

c_intercept=regressor.intercept_
print(c_intercept)

y_12=m_slope*12+c_intercept
print(y_12)

#Comparision
comparision=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(comparision)

bias=regressor.score(x_train,y_train)
print(bias)

variance=regressor.score(x_test,y_test)
print(variance)
#stats concept need to  add on the code
exp.mean()
exp['Salary'].mean()
exp['YearsExperience'].mean()

exp.median()
exp['YearsExperience'].median()
exp['Salary'].median()

exp.mode()

exp.var()
exp['Salary'].var()

exp.std()
exp['Salary'].std()

#Coefficient of variation
from scipy.stats import variation
variation(exp.values)

variation(exp['Salary'])

#Correlation

exp.corr()

exp['Salary'].corr(exp['YearsExperience'])

#Skewness

exp.skew()

exp['Salary'].skew()

#Standard Erroe
exp.sem()

exp['Salary'].sem()

#Z-Score
#for calculating Z-score we have to import a library first
 
import scipy.stats as stats

exp.apply(stats.zscore)

stats.zscore(exp['Salary'])

#degree of freedom

a=exp.shape[0]
b=exp.shape[1]

degree_of_freedom=a-b
print(degree_of_freedom)

#Sum of square regression
y_mean=np.mean(y)
SSR=np.sum((y_pred-y_mean)**2)
print(SSR)

#SSE
y=y[0:6]
SSE=np.sum((y-y_pred)**2)
print(SSE)
#SST
mean_total=np.mean(exp.values)
SST=np.sum((exp.values-mean_total)**2)
print(SST)

#R2
r_square=1-(SSR/SST)
r_square

#Save the trained model to disk
import pickle
filename='linear_regression_model.pkl'
with open(filename,'wb')as file:
    pickle.dump(regressor,file)
print("Model has been saved")

import os
print(os.getcwd())