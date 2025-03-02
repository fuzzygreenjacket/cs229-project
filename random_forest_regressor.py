# Calculating random forest regression for best feature combinations
# for respective outcomes and plotting them
# Uses best hyperparameters from random_forest_regressor_grid_search.py

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


df = pd.read_excel("merged_city_year_panel milestone updated.xlsx")

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

# optimal hyperparameters for each outcome
hyperparams1 = ["poisson", 5, 2, 1000]
hyperparams2 = ["squared_error", 5, 2, 100]


# arguments are: outcome to predict, features to use, hyperparameters to use, 
# and num is just to keep track of what number outcome this is (useful for printing)
def random_forest_regressor(outcome, features, hyperparams, num):

    # create train/test split
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    for year, group in df.groupby('year'):
        train, test = train_test_split(group, test_size=.2)
        train_data = pd.concat([train_data, train], ignore_index=True)
        test_data = pd.concat([test_data, test], ignore_index=True)                                                 

    x_train = train_data[features]
    x_test = test_data[features]
    y_train = train_data[outcome]
    y_test = test_data[outcome]

    # run random forest regressor
    # with scaled data
    pipeline = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(
            criterion=hyperparams[0],
            min_samples_leaf=hyperparams[1],
            min_samples_split=hyperparams[2],
            n_estimators=hyperparams[3]
        )
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    # compute MSE
    mse = mean_squared_error(y_test, predictions)

    # compute r2 score
    r2 = r2_score(y_test, predictions)

    print(f'MSE: {mse}')
    print(f'R2: {r2}')

    # select 100 random points to plot (otherwise too crowded)
    sample = np.random.choice(len(y_test), 50, replace=False)
    predictions_sample = predictions[sample]
    test_sample = y_test[sample]
    
    # scatterplot
    plt.figure(figsize=(7, 5))
    plt.scatter(range(len(test_sample)), test_sample, label="Actual Values", alpha=0.5, color="blue")
    plt.scatter(range(len(predictions_sample)), predictions_sample, label="Predicted Values", alpha=0.5, color="orange")

    plt.xlabel("Sample")
    plt.ylabel(outcome)
    plt.title("True vs. Predicted Values")
    plt.legend()
    plt.savefig("forest_plot_" + str(num) + ".png", dpi=300) 
    plt.show()


    # code borrowed from scikit learn: https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
    # this plots the feature importances from the random forest split based on mean decrease in impurity
    feature_names = [f"{features[i].split(' ', 1)[0]}" for i in range(x_train.shape[1])]
    importances = pipeline.named_steps["randomforestregressor"].feature_importances_
    std = np.std([tree.feature_importances_ for tree in pipeline.named_steps["randomforestregressor"].estimators_], axis=0)
    fig, ax = plt.subplots()
    forest_importances = pd.Series(importances, index=feature_names)
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    fig.savefig("feature_importances_" + str(num) + ".png")

# run forest_regression
random_forest_regressor(outcome1, features1, hyperparams1, 1)
random_forest_regressor(outcome2, features2, hyperparams2, 2)

