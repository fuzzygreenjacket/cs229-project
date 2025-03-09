# this is baseline linear regression for the four experiments that Cheryl suggested
# 1. All the listed features predicting LGFV interest bearing debt/GDP
# 2. All the listed features predicting balance of urban investment bond/GDP
# 3. All the listed features + real estate investment/GDP predicting LGFV interest bearing debt/GDP
# 4. All the listed features + real estate investment/GDP predicting urban investment bond/GDP
# feature engineering will happen in a different file

# code structure inspired by https://www.geeksforgeeks.org/linear-regression-using-pytorch/ 
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# random seeding
seed = 21
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

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

# create train/test split
train_data = pd.DataFrame()
test_data = pd.DataFrame()
for year, group in df.groupby('year'):
    train, test = train_test_split(group, test_size=.2, random_state=seed)
    train_data = pd.concat([train_data, train], ignore_index=True)
    test_data = pd.concat([test_data, test], ignore_index=True)

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
] # list of all possible features

features1 = [
    "Liability Ratio(%)",
    "Debt Ratio(%)",
    "Growth Rate of GDP(%)",
    "Comprehensive Financial Resources(CNY,B)",
    "Fiscal Self-sufficiency(%)",
    "Budget Revenue(CNY,B)",
    "Revenue of Government-Managed Funds(CNY,B)",
    "State-owned Land Transfer Income/Budget Revenue(%)"
] # without real estate investment


features2 = features1 + ["Real Estate Investment(CNY,B) / GDP(CNY,B)"] # with real estate investment

outcome1 = "LGFV Interest-bearing Debt(CNY,B) / GDP(CNY,B)" 
outcome2 = "Balance of Urban Investment Bond(CNY,B) / GDP(CNY,B)"

feature_combinations = [features1, features2] # list of lists of feature combinations
outcomes = [outcome1, outcome2] # list of possible outcomes
                                                                
class LinearRegression(nn.Module):
    def __init__(self, num_features):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.linear(x)

for features in feature_combinations:
    for outcome in outcomes:
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
            if i%100 == 0:
                print(f'epoch: {i}, loss: {loss}')

        predictions = model(x_test).squeeze()

        # transform the data back
        predictions_original = outcome_scaler.inverse_transform(predictions.detach().numpy().reshape(-1, 1))
        y_test_original = outcome_scaler.inverse_transform(y_test.numpy().reshape(-1, 1))

        # convert to tensor
        predictions_tensor = torch.from_numpy(predictions_original).type(torch.float32)
        y_test_tensor = torch.from_numpy(y_test_original).type(torch.float32)
        
        # compute MSE
        mse = criterion(predictions_tensor.squeeze(), y_test_tensor.squeeze())

        print(f'Features: {features}')
        print(f'Outcome: {outcome}')
        print(f'Test Set MSE: {mse}')

        plt.clf()
        plt.scatter(x_test[:, 0].numpy(), predictions.detach().numpy(), label='Predictions', alpha=0.4, color='blue') # pick one item on x_axis
        plt.scatter(x_test[:, 0].numpy(), y_test.detach().numpy(), label='True Values', alpha = 0.4, color='orange') # pick one item on x_axis
        plt.legend()
        plt.savefig("linear_regression_scatterplot.png")
