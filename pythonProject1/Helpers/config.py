import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import warnings
warnings.simplefilter(action='ignore', category=Warning)


# Dataset path
dataset_path = "C:/Users/Emirhan Denizyol/Diabetes_ML/pythonProject1/Helpers/diabetes.csv"


# Automated Hyperparameter Optimization

knn_params = {'n_neighbors': range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               'min_samples_split': range(2, 30)}

rf_params = {'max_depth': [8, 15, None],
             'max_features': [5, 7, 'auto'],
             'min_samples_split': [10, 20, 30, 40],
             'n_estimators': [200, 300, 400]}

xgboost_params = {'learning_rate': [0.1, 0.01, 0.001],
                  'max_depth': [5, 10, 15],
                  'n_estimators': [100, 200, 350],
                  'colsample_bytree': [0.5, 1, 3]}

lightgbm_params = {'learning_rate': [0.1, 0.01, 0.001],
                   'n_estimators': [100, 200, 350],
                   'colsample_bytree': [0.5, 1, 3]}

catboost_params = {'iterations': [200, 500, 750],
                   'learning_rate': [0.1, 0.01, 0.001],
                   'depth': [3, 6, 9, 12]}

classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ('CART', DecisionTreeClassifier(), cart_params),
               ('RF', RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(), xgboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params),
               ('CATBoost', CatBoostClassifier(), cartboost_params)
               ]

