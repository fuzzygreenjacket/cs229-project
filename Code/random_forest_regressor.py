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
import seaborn as sns


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
    "Growth Rate of GDP(%)",
    "Fiscal Self-sufficiency(%)",
    "Budget Revenue(CNY,B)",
    "State-owned Land Transfer Income/Budget Revenue(%)"
]

# optimal features for outcome 2
features2 = [
    "Liability Ratio(%)", 
    "Debt Ratio(%)", 
    "Growth Rate of GDP(%)",
    "Fiscal Self-sufficiency(%)",
    "Budget Revenue(CNY,B)",
    "State-owned Land Transfer Income/Budget Revenue(%)"
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

    log_error = -np.log(mse + 1e-10)

    print(f'MSE: {mse}')
    print(f'R2: {r2}')
    print(f'log error: {log_error}')

    # select 100 random points to plot (otherwise too crowded)
    sample = np.random.choice(len(y_test), 50, replace=False)
    predictions_sample = predictions[sample]
    test_sample = y_test[sample]
    
    # scatterplot
    plt.figure(figsize=(7, 5))
    plt.scatter(range(len(test_sample)), test_sample, label="Actual Values", alpha=0.6, color="green")
    plt.scatter(range(len(predictions_sample)), predictions_sample, label="Predicted Values", alpha=0.6, color="red")
    plt.xlabel("Sample")
    if outcome == outcome1:
        plt.ylabel("LGFV Debt")
    elif outcome == outcome2:
        plt.ylabel("Urban Investment Bond")
    else:
        plt.ylabel(outcome)
    plt.title("True vs. Predicted Values")
    plt.legend()
    plt.savefig("forest_plot_" + str(num) + ".png", dpi=300) 
    plt.show()

    # histogram
    plt.clf()
    deltas = abs(predictions - y_test)
    if outcome == outcome1:
        sns.histplot(deltas, bins=np.arange(0, 1.25, 0.05), kde=True, legend=False)
        plt.xlim(0, 1.25)
        plt.ylim(0, 135)
        plt.title("Random Forest Deltas: LGFV Debt")
    elif outcome == outcome2:
        sns.histplot(deltas, bins=np.arange(0, 0.5, 0.025), kde=True, legend=False)
        plt.xlim(0, .5)
        plt.ylim(0, 185)
        plt.title("Random Forest Deltas: Urban Investment Bond")
    else:
        sns.histplot(deltas, kde=True)
    plt.xlabel("Difference Between Prediction and True Value")
    plt.ylabel("Count")
    plt.savefig("../Visualizations/random_forest_histogram" + str(num) + ".png", dpi=300) 
    plt.show()
    return mse, r2, log_error

# run forest_regression
random_forest_regressor(outcome1, features1, hyperparams1, 1)
random_forest_regressor(outcome2, features2, hyperparams2, 2)

# this code was used to run random forest regression 100 times and average the results
'''mse1_average = 0
mse2_average = 0
r21_average = 0
r22_average = 0
log_error1_average = 0
log_error2_average = 0

for i in range(1000):
    mse1, r21, log_error1 = random_forest_regressor(outcome1, features1, hyperparams1, 1)
    mse2, r22, log_error2 =  random_forest_regressor(outcome2, features2, hyperparams2, 2)
    mse1_average += mse1
    mse2_average += mse2
    r21_average += r21
    r22_average += r22
    log_error1_average += log_error1
    log_error2_average += log_error2


print(mse1_average/1000)
print(mse2_average/1000)
print(r21_average/1000)
print(r22_average/1000)
print(log_error1_average/1000)
print(log_error2_average/1000)'''
