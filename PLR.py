import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

data=pd.read_csv(r"C:\Users\rahee\Downloads\emp_sal.csv")

X=data.iloc[:,1:2].values
y=data.iloc[:,2].values


lin_reg=LinearRegression()
lin_reg.fit(X,y)

#Plot the graph of linear regression
plt.scatter(X, y, color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('Linear Regression model(Linear Regression')
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

lin_model_pred=lin_reg.predict([[6.5]])
lin_model_pred

#Non linear model
poly_reg=PolynomialFeatures(degree=6)
X_poly=poly_reg.fit_transform(X)

poly_reg.fit(X_poly,y)
#Comparisoion
lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,y)

#1st model lin_reg_2 (linear model)
#2nd model poly_reg(polynomial model)

plt.scatter(X, y, color='red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('Polynomial Regression model(Polynomail Regression')
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

poly_model_pred=lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred