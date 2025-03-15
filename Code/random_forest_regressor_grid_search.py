# Computing best hyperparameters for random forest regressor

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

df = pd.read_excel("../Cleaned Data/merged_city_year_panel milestone updated.xlsx")

# this list contains all possible columns we might use as features or outcomes
# we need to clean this data and remove NAN values 
to_clean = [
    "Liability Ratio(%)",
    "Debt Ratio(%)",
    "GDP(CNY,B)",
    "Growth Rate of GDP(%)",
    "Comprehensive Financial Resources(CNY,B)",
    "Fiscal Self-sufficiency(%)",
    "Budget Revenue(CNY,B)",
    "Revenue of Government-Managed Funds(CNY,B)",
    "State-owned Land Transfer Income/Budget Revenue(%)",
    "Real Estate Investment(CNY,B)",
    "LGFV Interest-bearing Debt(CNY,B)",
    "Balance of Urban Investment Bond(CNY,B)"
]

# remove all -- cells
for col in to_clean:
    df = df[df[col] != "--"]

# convert data type to numbers
df[to_clean] = df[to_clean].apply(pd.to_numeric, errors='raise')

# creating new columns by dividing
df["LGFV Interest-bearing Debt(CNY,B) / GDP(CNY,B)"] = df["LGFV Interest-bearing Debt(CNY,B)"]/df["GDP(CNY,B)"] # outcome 1
df["Balance of Urban Investment Bond(CNY,B) / GDP(CNY,B)"] = df["Balance of Urban Investment Bond(CNY,B)"]/df["GDP(CNY,B)"] # outcome 2
df["Real Estate Investment(CNY,B) / GDP(CNY,B)"] = df["Real Estate Investment(CNY,B)"]/df["GDP(CNY,B)"] # potential feature

# list of all possible features
all_features = [
    "Liability Ratio(%)",
    "Debt Ratio(%)",
    "Growth Rate of GDP(%)",
    "Comprehensive Financial Resources(CNY,B)",
    "Fiscal Self-sufficiency(%)",
    "Budget Revenue(CNY,B)",
    "Revenue of Government-Managed Funds(CNY,B)",
    "State-owned Land Transfer Income/Budget Revenue(%)",
    "Real Estate Investment(CNY,B)"
]

# possible outcomes
outcome1 = "LGFV Interest-bearing Debt(CNY,B) / GDP(CNY,B)" 
outcome2 = "Balance of Urban Investment Bond(CNY,B) / GDP(CNY,B)"

# optimal features for outcome 1
features1 = [
    "Liability Ratio(%)", 
    "Debt Ratio(%)", 
    "Comprehensive Financial Resources(CNY,B)",
    "Fiscal Self-sufficiency(%)",
    "Revenue of Government-Managed Funds(CNY,B)", 
    "Real Estate Investment(CNY,B)"
]

# optimal features for outcome 2
features2 = [
    "Comprehensive Financial Resources(CNY,B)", 
    "Fiscal Self-sufficiency(%)",
    "Revenue of Government-Managed Funds(CNY,B)", 
    "Real Estate Investment(CNY,B)" 
]

outcomes = [outcome1, outcome2] # list of possible outcomes 

def forest_grid_search(outcome, features, num):

    # note: we are not grouping by year when splitting x_train and y_train when running the grid search
    # because it allows us to automate the cross-validation and we don't believe it will have a significant
    # impact on the optimal hyperparameters
    x_train = df[features]
    y_train = df[outcome]

    pipeline = make_pipeline(
        StandardScaler(),
        RandomForestRegressor()
    )

    param_grid = {
            "randomforestregressor__n_estimators": [10, 100, 1000],
            "randomforestregressor__criterion": ["squared_error", "friedman_mse", "poisson"],
            "randomforestregressor__min_samples_split": [1.0, 2, 3],
            "randomforestregressor__min_samples_leaf": [1, 5]
    }

    # test different hyperparameters with grid_cv
    grid_cv = GridSearchCV(
        pipeline,
        param_grid,
        scoring="neg_mean_squared_error",
        cv=5
    )

    grid_cv.fit(x_train, y_train)

    print(grid_cv.best_params_)

# run forest_regression
forest_grid_search(outcome1, features1, 1)
forest_grid_search(outcome2, features2, 2)

# result:
# {'randomforestregressor__criterion': 'poisson', 'randomforestregressor__min_samples_leaf': 5, 'randomforestregressor__min_samples_split': 2, 'randomforestregressor__n_estimators': 1000}
# {'randomforestregressor__criterion': 'squared_error', 'randomforestregressor__min_samples_leaf': 5, 'randomforestregressor__min_samples_split': 2, 'randomforestregressor__n_estimators': 100}