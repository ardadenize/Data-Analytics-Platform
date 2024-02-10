import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dateutil import parser
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge

from catboost import CatBoostRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

def regresyon_model_tekSet(X_train, X_val, Y_train, Y_val):
   # Linear Regression
  regressorLR = LinearRegression()
  regressorLR.fit(X_train, Y_train)
 
  test_LinearRegression = regressorLR.predict(X_val)
   
  rmse_LinearRegression = np.sqrt(mean_squared_error(Y_val, test_LinearRegression))
  st.write("**Linear Regression**")
  st.write("RMSE Değeri: ", rmse_LinearRegression)
  

  # Gradient Boosting Regressor
  regressorGB = GradientBoostingRegressor()
  regressorGB.fit(X_train, Y_train)
  test_GradientBoosting = regressorGB.predict(X_val)
  rmse_GradientBoosting = np.sqrt(mean_squared_error(Y_val, test_GradientBoosting))
  st.write("**Gradient Boosting Regressor**")
  st.write("RMSE Değeri: ", rmse_GradientBoosting)

  # Elastic Net Regression
  regressorENet = ElasticNet()
  regressorENet.fit(X_train, Y_train)
  test_ElasticNet = regressorENet.predict(X_val)
  rmse_ElasticNet = np.sqrt(mean_squared_error(Y_val, test_ElasticNet))
  st.write("**Elastic Net Regression**")
  st.write("RMSE Değeri: ", rmse_ElasticNet)

    # Bayesian Ridge Regression
  regressorBayesianRidge = BayesianRidge()
  regressorBayesianRidge.fit(X_train, Y_train)
  test_BayesianRidge = regressorBayesianRidge.predict(X_val)
  rmse_BayesianRidge = np.sqrt(mean_squared_error(Y_val, test_BayesianRidge))
  st.write("**Bayesian Ridge Regression**")
  st.write("RMSE Değeri: ", rmse_BayesianRidge)

    # CatBoost Regression
  regressorCatBoost = CatBoostRegressor()
  regressorCatBoost.fit(X_train, Y_train)
  test_CatBoost = regressorCatBoost.predict(X_val)
  rmse_CatBoost = np.sqrt(mean_squared_error(Y_val, test_CatBoost))
  st.write("**CatBoost Regression**")
  st.write("RMSE Değeri: ", rmse_CatBoost)


    # XGB Regression
  regressorXGB = XGBRegressor()
  regressorXGB.fit(X_train, Y_train)
  test_XGB = regressorXGB.predict(X_val)
  rmse_XGB = np.sqrt(mean_squared_error(Y_val, test_XGB))
  st.write("**XGB Regression**")
  st.write("RMSE Değeri: ", rmse_XGB)

  
    # LGBM Regression
  regressorLGBM = LGBMRegressor()
  regressorLGBM.fit(X_train, Y_train)
  test_LGBM = regressorENet.predict(X_val)
  rmse_LGBM = np.sqrt(mean_squared_error(Y_val, test_LGBM))
  st.write("**LGBM Regression**")
  st.write("RMSE Değeri: ", rmse_LGBM)


  # Decision Tree Regression
  regressorDT = DecisionTreeRegressor()
  regressorDT.fit(X_train, Y_train)
  test_DT = regressorDT.predict(X_val)
  rmse_DT = np.sqrt(mean_squared_error(Y_val, test_DT))
  st.write("**Decision Tree Regression**")
  st.write("RMSE Değeri: ", rmse_DT)

  # Random Forest Regression
  regressorRF = RandomForestRegressor()
  regressorRF.fit(X_train, Y_train)
  test_RF = regressorRF.predict(X_val)
  rmse_RF = np.sqrt(mean_squared_error(Y_val, test_RF))
  st.write("**Random Forest Regression**")
  st.write("RMSE Değeri: ", rmse_RF)


  # AdaBoost Regression
  regressorAB = AdaBoostRegressor()
  regressorAB.fit(X_train, Y_train)
  test_AB = regressorAB.predict(X_val)
  rmse_AB = np.sqrt(mean_squared_error(Y_val, test_AB))
  st.write("**AdaBoost Regression**")
  st.write("RMSE Değeri: ", rmse_AB)

  # karşılaştırma yapmak için rmse değerlerini bir sözlükte topla
  rmse_dict = {'Decision Tree Regression': rmse_DT,
             'Random Forest Regression': rmse_RF,
             'Elastic Net Regression': rmse_ElasticNet,
             'Bayesian Ridge Regression': rmse_BayesianRidge,
             'CatBoost Regression': rmse_CatBoost,
             'XGB Regression': rmse_XGB,
             'LGBM Regression': rmse_LGBM,
             'Linear Regression': rmse_LinearRegression,
             'Gradient Boosting Regressor': rmse_GradientBoosting,
             'AdaBoost Regression': rmse_AB}

  st.table(rmse_dict)
  # en küçük rmse değerini ve hangi modelin bu değere sahip olduğunu bulmak için min() fonksiyonunu kullanın
  min_rmse = min(rmse_dict.values())
  best_model = [model for model, rmse in rmse_dict.items() if rmse == min_rmse][0]

  # Sonucu ekrana yazdırın
  st.write("En Düşük RMSE Değeri:", min_rmse)
  st.write("En İyi Model:", best_model)



  
  # RMSE değerleri ve modellere ait isimler listesi 
  rmse_degerleri = [rmse_LinearRegression, rmse_GradientBoosting, rmse_ElasticNet, rmse_BayesianRidge, rmse_CatBoost, rmse_XGB, rmse_LGBM, rmse_DT, rmse_RF, rmse_AB]
  model_isimleri = ["Linear Regression", "Gradient Boosting Regressor", "Elastic Net Regression", "Bayesian Ridge Regression", "CatBoost Regression", "XGB Regression", "LGBM Regression", "Decision Tree Regression", "Random Forest Regression", "AdaBoost Regression"]

  # verileri bir DataFrame'e aktarın
  df = pd.DataFrame({'Model İsimleri': model_isimleri, 'RMSE Değerleri': rmse_degerleri})

  # bar chart'ı oluşturun
  fig = px.bar(df, x='Model İsimleri', y='RMSE Değerleri', text='RMSE Değerleri', color='RMSE Değerleri', color_continuous_scale='blues')

  # plot'u gösterin
  st.write("RMSE DEĞERLERİ")
  st.plotly_chart(fig)



def regresyon_model(X_train, X_val, Y_train, Y_val, test_set, test_target):
   # Linear Regression
  regressorLR = LinearRegression()
  regressorLR.fit(X_train, Y_train)
 
  val_LinearRegression = regressorLR.predict(X_val)
  test_LinearRegression = regressorLR.predict(test_set)
   
  rmse_LinearRegression = np.sqrt(mean_squared_error(test_target, test_LinearRegression))
  st.write("**Linear Regression**")
  st.write("RMSE Değeri: ", rmse_LinearRegression)
  

  # Gradient Boosting Regressor
  regressorGB = GradientBoostingRegressor()
  regressorGB.fit(X_train, Y_train)
  val_LinearRegression = regressorGB.predict(X_val)
  test_GradientBoosting = regressorGB.predict(test_set)
  rmse_GradientBoosting = np.sqrt(mean_squared_error(test_target, test_GradientBoosting))
  st.write("**Gradient Boosting Regressor**")
  st.write("RMSE Değeri: ", rmse_GradientBoosting)

  # Elastic Net Regression
  regressorENet = ElasticNet()
  regressorENet.fit(X_train, Y_train)
  val_LinearRegression = regressorENet.predict(X_val)
  test_ElasticNet = regressorENet.predict(test_set)
  rmse_ElasticNet = np.sqrt(mean_squared_error(test_target, test_ElasticNet))
  st.write("**Elastic Net Regression**")
  st.write("RMSE Değeri: ", rmse_ElasticNet)

    # Bayesian Ridge Regression
  regressorBayesianRidge = BayesianRidge()
  regressorBayesianRidge.fit(X_train, Y_train)
  val_LinearRegression = regressorBayesianRidge.predict(X_val)
  test_BayesianRidge = regressorBayesianRidge.predict(test_set)
  rmse_BayesianRidge = np.sqrt(mean_squared_error(test_target, test_BayesianRidge))
  st.write("**Bayesian Ridge Regression**")
  st.write("RMSE Değeri: ", rmse_BayesianRidge)

    # CatBoost Regression
  regressorCatBoost = CatBoostRegressor()
  regressorCatBoost.fit(X_train, Y_train)
  val_CatBoost = regressorCatBoost.predict(X_val)
  test_CatBoost = regressorCatBoost.predict(test_set)
  rmse_CatBoost = np.sqrt(mean_squared_error(test_target, test_CatBoost))
  st.write("**CatBoost Regression**")
  st.write("RMSE Değeri: ", rmse_CatBoost)


    # XGB Regression
  regressorXGB = XGBRegressor()
  regressorXGB.fit(X_train, Y_train)
  val_XGB = regressorXGB.predict(X_val)
  test_XGB = regressorXGB.predict(test_set)
  rmse_XGB = np.sqrt(mean_squared_error(test_target, test_XGB))
  st.write("**XGB Regression**")
  st.write("RMSE Değeri: ", rmse_XGB)

  
    # LGBM Regression
  regressorLGBM = LGBMRegressor()
  regressorLGBM.fit(X_train, Y_train)
  val_LGBM = regressorENet.predict(X_val)
  test_LGBM = regressorENet.predict(test_set)
  rmse_LGBM = np.sqrt(mean_squared_error(test_target, test_LGBM))
  st.write("**LGBM Regression**")
  st.write("RMSE Değeri: ", rmse_LGBM)


  # Decision Tree Regression
  regressorDT = DecisionTreeRegressor()
  regressorDT.fit(X_train, Y_train)
  val_DT = regressorDT.predict(X_val)
  test_DT = regressorDT.predict(test_set)
  rmse_DT = np.sqrt(mean_squared_error(test_target, test_DT))
  st.write("**Decision Tree Regression**")
  st.write("RMSE Değeri: ", rmse_DT)

  # Random Forest Regression
  regressorRF = RandomForestRegressor()
  regressorRF.fit(X_train, Y_train)
  val_RF = regressorRF.predict(X_val)
  test_RF = regressorRF.predict(test_set)
  rmse_RF = np.sqrt(mean_squared_error(test_target, test_RF))
  st.write("**Random Forest Regression**")
  st.write("RMSE Değeri: ", rmse_RF)


  # AdaBoost Regression
  regressorAB = AdaBoostRegressor()
  regressorAB.fit(X_train, Y_train)
  val_AdaBoost = regressorAB.predict(X_val)
  test_AB = regressorAB.predict(test_set)
  rmse_AB = np.sqrt(mean_squared_error(test_target, test_AB))
  st.write("**AdaBoost Regression**")
  st.write("RMSE Değeri: ", rmse_AB)

  # karşılaştırma yapmak için rmse değerlerini bir sözlükte topla
  rmse_dict = {'Decision Tree Regression': rmse_DT,
             'Random Forest Regression': rmse_RF,
             'Elastic Net Regression': rmse_ElasticNet,
             'Bayesian Ridge Regression': rmse_BayesianRidge,
             'CatBoost Regression': rmse_CatBoost,
             'XGB Regression': rmse_XGB,
             'LGBM Regression': rmse_LGBM,
             'Linear Regression': rmse_LinearRegression,
             'Gradient Boosting Regressor': rmse_GradientBoosting,
             'AdaBoost Regression': rmse_AB}
  st.table(rmse_dict)
  # en küçük rmse değerini ve hangi modelin bu değere sahip olduğunu bulmak için min() fonksiyonunu kullanın
  min_rmse = min(rmse_dict.values())
  best_model = [model for model, rmse in rmse_dict.items() if rmse == min_rmse][0]

  # Sonucu ekrana yazdırın
  st.write("En Düşük RMSE Değeri:", min_rmse)
  st.write("En İyi Model:", best_model)



  
  # RMSE değerleri ve modellere ait isimler listesi 
  rmse_degerleri = [rmse_LinearRegression, rmse_GradientBoosting, rmse_ElasticNet, rmse_BayesianRidge, rmse_CatBoost, rmse_XGB, rmse_LGBM, rmse_DT, rmse_RF, rmse_AB]
  model_isimleri = ["Linear Regression", "Gradient Boosting Regressor", "Elastic Net Regression", "Bayesian Ridge Regression", "CatBoost Regression", "XGB Regression", "LGBM Regression", "Decision Tree Regression", "Random Forest Regression", "AdaBoost Regression"]

  # verileri bir DataFrame'e aktarın
  df = pd.DataFrame({'Model İsimleri': model_isimleri, 'RMSE Değerleri': rmse_degerleri})

  # bar chart'ı oluşturun
  fig = px.bar(df, x='Model İsimleri', y='RMSE Değerleri', text='RMSE Değerleri', color='RMSE Değerleri', color_continuous_scale='blues')

  # plot'u gösterin
  st.write("RMSE DEĞERLERİ")
  st.plotly_chart(fig)
