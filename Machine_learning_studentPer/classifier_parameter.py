#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
import lightgbm
from lightgbm.sklearn import LGBMRegressor
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPRegressor
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
import time

# Load labels data
train_labels = pd.read_csv("train_labels.csv")

# Split session id and question id
train_labels['session'] = train_labels['session_id'].apply(lambda x:int(x.split('_')[0]))
train_labels['q'] = train_labels['session_id'].apply(lambda x:int(x.split('_')[-1][1:]))
train_labels = train_labels.drop('session_id', axis=1)
train_labels = train_labels.rename({'session':'session_id'}, axis=1)
train_labels = train_labels.set_index('session_id')


# Load features data
df = pd.read_csv("feature_data.csv", index_col='session_id')

# Expand data according to level_group
df = df.reset_index()
df = df.replace({'0-4':4,'13-22':6,'5-12':8})
df = df.loc[df.index.repeat(df.level_group)]
df = df.set_index('session_id')

# X, y
train_x = df.drop('level_group', axis=1)
train_y = train_labels.drop('q', axis=1)


# Tune random forest model
# Set hyperparameter ranges
param_grid_rf = {
    "n_estimators": [100, 300, 500, 1000],
    "max_depth": [3, 4, 5, 6],    
}

# Fit model to select best params
print('RandomForest')
start = time.time() 
rf =  RandomForestClassifier() 

random_cv = RandomizedSearchCV(rf, param_grid_rf, n_iter=10, cv=3, n_jobs = -1, verbose=3, random_state=42)

_ = random_cv.fit(train_x, train_y['correct'])
print(time.time()-start)
print('Best params:\n')
print(random_cv.best_params_)


# Tune MLP model
# Set hyperparameter ranges
param_grid_mlp = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'solver': ['sgd', 'adam'],
    'alpha': [0.001, 0.05],
}

# Fit model to select best params
print('MLP')
start = time.time() 
mlp =  MLPRegressor() 

random_cv = RandomizedSearchCV(mlp, param_grid_mlp, n_iter=10, cv=3, n_jobs = -1, verbose=3, random_state=42)

_ = random_cv.fit(train_x, train_y['correct'])
print(time.time()-start)
print('Best params:\n')
print(random_cv.best_params_)


# Tune XGBoost model
# Set hyperparameter ranges
param_grid_xgb = {
    'n_estimators': [100, 500, 1000, 2000],
    'max_depth': [ 3, 4, 5, 6],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
}

# Fit model to select best params
print('XGBoost')
start = time.time() 
xgb =  XGBClassifier()

random_cv = RandomizedSearchCV(xgb, param_grid_xgb, n_iter=10, cv=3, n_jobs = -1, verbose=3, random_state=42)

_ = random_cv.fit(train_x, train_y['correct'])
print(time.time()-start)
print('Best params:\n')
print(random_cv.best_params_)


# Tune SVM model
# Set hyperparameter ranges
param_grid_svm = {
    'kernel': ['poly','sigmoid','rbf'],
}

# Fit model to select best params
print('SVM')
start = time.time() 

random_cv = RandomizedSearchCV(svm.SVR(), param_grid_svm, n_iter=10, cv=3, n_jobs = -1, verbose=3, random_state=42)

_ = random_cv.fit(train_x, train_y['correct'])
print(time.time()-start)
print('Best params:\n')
print(random_cv.best_params_)


# Tune LightGBM model
# Set hyperparameter ranges
param_grid_lgbm = {
    'n_estimators': [100, 500, 1000, 2000],
    'max_depth': [ 3, 4, 5, 6],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
}

# Fit model to select best params
print('LightGBM')
start = time.time() 
lgbm =  LGBMRegressor()

random_cv = RandomizedSearchCV(lgbm, param_grid_lgbm, n_iter=10, cv=3, n_jobs = -1, verbose=3, random_state=42)

_ = random_cv.fit(train_x, train_y['correct'])
print(time.time()-start)
print('Best params:\n')
print(random_cv.best_params_)


# Tune AdaBoost model
# Set hyperparameter ranges
param_grid_adab = {
    'n_estimators': [100, 500, 1000],
}

# Fit model to select best params
print('AdaBoost')
start = time.time() 
adab =  AdaBoostClassifier()

random_cv = RandomizedSearchCV(adab, param_grid_adab, n_iter=10, cv=3, n_jobs = -1, verbose=3, random_state=42)

_ = random_cv.fit(train_x, train_y['correct'])
print(time.time()-start)
print('Best params:\n')
print(random_cv.best_params_)

