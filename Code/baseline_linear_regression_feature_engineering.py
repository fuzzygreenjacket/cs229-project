# This calculates the baseline linear regression for EVERY possible combination of features
# for both respective outcomes. It does this five times and uses the mean as the final answer.
# The program then selects the top five feature combinations for each outcome
# and prints them out. This is done twice, once using MSE and once using R2. Thus there
# are four total top-five lists. 

# code structure inspired by https://www.geeksforgeeks.org/linear-regression-using-pytorch/ 

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


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

feature_combinations = []

# loop over every possible combination of features
for i in range(1, len(all_features) + 1):
    for features in combinations(all_features, i):
        feature_combinations.append(list(features))

outcomes = [outcome1, outcome2] # list of possible outcomes 

# dictionaries to track the MSE and R2 performances for each feature combination and outcome
# keys are tuples (outcome, feature) and entries are lists of values (because we are running
# linear regression multiple times)
feature_combo_performance_mse = {}
feature_combo_performance_r2 = {}

def feature_engineering():

    # create train/test split
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    for year, group in df.groupby('year'):
        train, test = train_test_split(group, test_size=.2)
        train_data = pd.concat([train_data, train], ignore_index=True)
        test_data = pd.concat([test_data, test], ignore_index=True)                                                 

    class LinearRegression(nn.Module):
        def __init__(self, num_features):
            super(LinearRegression, self).__init__()
            self.linear = nn.Linear(num_features, 1)

        def forward(self, x):
            return self.linear(x)

    for outcome in outcomes:
        for features in feature_combinations:
            # scale data
            feature_scaler = StandardScaler()
            feature_scaler.fit(train_data[all_features])
            train_data[all_features] = feature_scaler.transform(train_data[all_features])
            test_data[all_features] = feature_scaler.transform(test_data[all_features])
            outcome_scaler = StandardScaler()
            outcome_scaler.fit(train_data[[outcome]])
            train_data[outcome] = outcome_scaler.transform(train_data[[outcome]])
            test_data[outcome] = outcome_scaler.transform(test_data[[outcome]])

            x_train = torch.from_numpy(train_data[features].to_numpy()).type(torch.float32)
            x_test = torch.from_numpy(test_data[features].to_numpy()).type(torch.float32)
            y_train = torch.from_numpy(train_data[outcome].to_numpy()).type(torch.float32)
            y_test = torch.from_numpy(test_data[outcome].to_numpy()).type(torch.float32)

            model = LinearRegression(len(features))
            criterion = nn.MSELoss()
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

            num_epochs = 1000
            for i in range(num_epochs):
                optimizer.zero_grad()
                y_predict = model(x_train)
                loss = criterion(y_predict.squeeze(), y_train)
                loss.backward()
                optimizer.step()

            predictions = model(x_test).squeeze()

            # transform the data back
            predictions_original = outcome_scaler.inverse_transform(predictions.detach().numpy().reshape(-1, 1))
            y_test_original = outcome_scaler.inverse_transform(y_test.numpy().reshape(-1, 1))

            # convert to tensor
            predictions_tensor = torch.from_numpy(predictions_original).type(torch.float32)
            y_test_tensor = torch.from_numpy(y_test_original).type(torch.float32)
            
            # compute MSE
            mse = criterion(predictions_tensor.squeeze(), y_test_tensor.squeeze())

            # compute r2 score
            predictions_np = predictions_original
            y_test_np = y_test_original
            r2 = r2_score(y_test_np, predictions_np)

            # update dictionary of scores
            if (outcome, tuple(features)) not in feature_combo_performance_mse:
                feature_combo_performance_mse[(outcome, tuple(features))] = [mse.item()]
                feature_combo_performance_r2[(outcome, tuple(features))] = [r2]
            else:
                feature_combo_performance_mse[(outcome, tuple(features))].append(mse.item())
                feature_combo_performance_r2[(outcome, tuple(features))].append(r2)

                
# run five iterations of linear regression
feature_engineering()
feature_engineering()
feature_engineering()
feature_engineering()
feature_engineering()

# split our two dictionaries along the outcomes (makes it simpler to sort)
outcome_1_performance_mean_mse = {k: v for k, v in feature_combo_performance_mse.items() if k[0] == outcome1}
outcome_2_performance_mean_mse = {k: v for k, v in feature_combo_performance_mse.items() if k[0] == outcome2}
outcome_1_performance_mean_r2 = {k: v for k, v in feature_combo_performance_r2.items() if k[0] == outcome1}
outcome_2_performance_mean_r2 = {k: v for k, v in feature_combo_performance_r2.items() if k[0] == outcome2}

outcome_1_performances_mse = {}
outcome_2_performances_mse = {}
outcome_1_performances_r2 = {}
outcome_2_performances_r2 = {}

# calculate the mean MSE and R2 for each outcome
for key in outcome_1_performance_mean_mse:
    outcome_1_performances_mse[key] = np.mean(outcome_1_performance_mean_mse[key])
for key in outcome_2_performance_mean_mse:
    outcome_2_performances_mse[key] = np.mean(outcome_2_performance_mean_mse[key])
for key in outcome_1_performance_mean_r2:
    outcome_1_performances_r2[key] = np.mean(outcome_1_performance_mean_r2[key])
for key in outcome_2_performance_mean_r2:
    outcome_2_performances_r2[key] = np.mean(outcome_2_performance_mean_r2[key])

# select the top five feature combinations for each outcome and metric
top_five_outcome_1_mse = sorted(outcome_1_performances_mse.items(), key=lambda x: x[1])[:5]
top_five_outcome_2_mse = sorted(outcome_2_performances_mse.items(), key=lambda x: x[1])[:5]
top_five_outcome_1_r2 = sorted(outcome_1_performances_r2.items(), key=lambda x: x[1], reverse=True)[:5]
top_five_outcome_2_r2 = sorted(outcome_2_performances_r2.items(), key=lambda x: x[1], reverse=True)[:5]

print(top_five_outcome_1_mse)
print(top_five_outcome_2_mse)
print(top_five_outcome_1_r2)
print(top_five_outcome_2_r2)

