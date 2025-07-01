import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (LinearRegression, Ridge, Lasso,ElasticNet,SGDRegressor,HuberRegressor)
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
import xgboost as xgb
import pickle

#load the dataset
data=pd.read_csv(r"C:\Users\rahee\OneDrive\Desktop\FSDS_AI\VS Code\17.Capstone\Housing_Project\USA_Housing.csv")

#Preprocess the data
X=data.drop(['Price','Address'],axis=1)
y=data['Price']

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Define the models
models = {
    'Linear Regression': LinearRegression(),
    'Robust Regression': HuberRegressor(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'ElasticNet Regression': ElasticNet(),
    'SGD Regressor': SGDRegressor(),
    'Random Forest Regressor': RandomForestRegressor(),
    'Support Vector Regressor': SVR(),
    'K-Neighbors Regressor': KNeighborsRegressor(),
    'Polynomial Regression': Pipeline([
        ('poly', PolynomialFeatures(degree=4)),
        ('linear', LinearRegression())
    ]),
    'Artificial Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000),
    'LightGBM Regressor': lgb.LGBMRegressor(),
    'XGBoost Regressor': xgb.XGBRegressor() 
}   

#Train and evaluate the models
results = []
 
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred=model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae= mean_absolute_error(y_test, y_pred)

    results.append({
        'Model': name,
        'MSE': mse,
        'R2 Score': r2,
        'MAE': mae
    })

    with open(f'{name}_model.pkl', 'wb') as file:
        pickle.dump(model, file)    


#Convert results to DataFrame
results_df = pd.DataFrame(results)      
#Save the results to a CSV file
results_df.to_csv('model_results.csv', index=False)

#Print the results
print("Model Evaluation Results:")
print(results_df)

models