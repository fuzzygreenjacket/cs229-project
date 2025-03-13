# Calculating baseline linear regression for best feature combinations
# for respective outcomes and plotting them

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
from sklearn.metrics import mean_squared_log_error


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
features1 = ["Debt Ratio(%)"]

# optimal features for outcome 2
features2 = [
    "Comprehensive Financial Resources(CNY,B)", 
    "Fiscal Self-sufficiency(%)",
    "Revenue of Government-Managed Funds(CNY,B)", 
    "Real Estate Investment(CNY,B)" 
]

outcomes = [outcome1, outcome2] # list of possible outcomes 

def lin_reg(outcome, features, num):

    # create train/test split
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    for year, group in df.groupby('year'):
        train, test = train_test_split(group, test_size=.2)
        train_data = pd.concat([train_data, train], ignore_index=True)
        test_data = pd.concat([test_data, test], ignore_index=True)                                                 

    # scale data
    feature_scaler = StandardScaler()
    feature_scaler.fit(train_data[all_features])
    train_data[all_features] = feature_scaler.transform(train_data[all_features])
    test_data[all_features] = feature_scaler.transform(test_data[all_features])
    outcome_scaler = StandardScaler()
    outcome_scaler.fit(train_data[[outcome]])
    train_data[outcome] = outcome_scaler.transform(train_data[[outcome]])
    test_data[outcome] = outcome_scaler.transform(test_data[[outcome]])

    class LinearRegression(nn.Module):
        def __init__(self, num_features):
            super(LinearRegression, self).__init__()
            self.linear = nn.Linear(num_features, 1)

        def forward(self, x):
            return self.linear(x)

    x_train = torch.from_numpy(train_data[features].to_numpy()).type(torch.float32)
    x_test = torch.from_numpy(test_data[features].to_numpy()).type(torch.float32)
    y_train = torch.from_numpy(train_data[outcome].to_numpy()).type(torch.float32)
    y_test = torch.from_numpy(test_data[outcome].to_numpy()).type(torch.float32)

    model = LinearRegression(len(features))
    criterion = nn.MSELoss()
    # criterion = nn.HuberLoss()
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
    log_error = -np.log(mse + 1e-10)

    print(f'MSE: {mse}')
    # print(f'Huber: {huber}')
    print(f'R2: {r2}')
    print(f'log error: {log_error}')

    # select 100 random points to plot (otherwise too crowded)
    sample = np.random.choice(len(y_test_np), 50, replace=False)
    predictions_sample = predictions.detach().numpy()[sample]
    test_sample = y_test_np[sample]

    # scatterplot
    plt.figure(figsize=(7, 5))
    plt.scatter(range(len(test_sample)), test_sample, label="Actual Values", alpha=0.5, color="blue")
    plt.scatter(range(len(predictions_sample)), predictions_sample, label="Predicted Values", alpha=0.5, color="orange")

    plt.xlabel("Sample")
    plt.ylabel(outcome)
    plt.title("True vs. Predicted Values")
    plt.legend()
    plt.savefig("residual_plot_" + str(num) + ".png", dpi=300) 
    plt.show()

    # histogram
    plt.clf()
    deltas = abs(predictions_original - y_test_original)
    # manually set bin size based on outcome
    if outcome == outcome1:
        plt.hist(deltas, bins=np.arange(0, 1.25, 0.05))
    elif outcome == outcome2:
        plt.hist(deltas, bins=np.arange(0, 0.5, 0.025))
    else:
        plt.hist(deltas)
    plt.savefig("lin_reg_histogram" + str(num) + ".png", dpi=300) 
    plt.show()

    return mse, r2, log_error


# run linear regression
lin_reg(outcome1, features1, 1)
lin_reg(outcome2, features2, 2)


# this code was used to run linear regression 100 times and average the results
'''outcome_1_average = 0
outcome_2_average = 0

for i in range(100):
    outcome_1_average += lin_reg(outcome1, features1, 1)
    outcome_2_average += lin_reg(outcome2, features2, 2)

print(outcome_1_average/100)
print(outcome_2_average/100)'''

