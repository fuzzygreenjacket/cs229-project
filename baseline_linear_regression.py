# code structure inspired by https://www.geeksforgeeks.org/linear-regression-using-pytorch/ 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_excel("merged_city_year_panel milestone.xlsx")
features = [
    "GDP(CNY,B)",
    "Growth Rate of GDP(%)",
    "Comprehensive Financial Resources(CNY,B)",
    "Fiscal Self-sufficiency(%)",
    "Budget Revenue(CNY,B)",
    "Revenue of Government-Managed Funds(CNY,B)"
]

outcome = "Non-Standard Financing Balance /LGFV Interest-bearing Debt(%)"

for feature in features:
    df = df[df[feature] != "--"]

df = df[df[outcome] != "--"]

train_data = pd.DataFrame()
test_data = pd.DataFrame()

for year, group in df.groupby('year'):
    train, test = train_test_split(group, test_size=.2)
    train_data = pd.concat([train_data, train], ignore_index=True)
    test_data = pd.concat([test_data, test], ignore_index=True)

train_data[features] = train_data[features].apply(pd.to_numeric, errors='raise')
test_data[features] = test_data[features].apply(pd.to_numeric, errors='raise')
train_data[outcome] = train_data[outcome].apply(pd.to_numeric, errors='raise')
test_data[outcome] = test_data[outcome].apply(pd.to_numeric, errors='raise')

scaler = StandardScaler()
scaler.fit(train_data[features])
train_data[features] = scaler.transform(train_data[features])
scaler.fit(test_data[features])
test_data[features] = scaler.transform(test_data[features])
scaler.fit(train_data[outcome].values.reshape(-1, 1))
train_data[outcome] = scaler.transform(train_data[outcome].values.reshape(-1, 1))
scaler.fit(test_data[outcome].values.reshape(-1, 1))
test_data[outcome] = scaler.transform(test_data[outcome].values.reshape(-1, 1))

x_train = torch.from_numpy(train_data[features].to_numpy()).type(torch.float32)
x_test = torch.from_numpy(test_data[features].to_numpy()).type(torch.float32)
y_train = torch.from_numpy(train_data[outcome].to_numpy()).type(torch.float32)
y_test = torch.from_numpy(test_data[outcome].to_numpy()).type(torch.float32)

class LinearRegression(nn.Module):
    def __init__(self, num_features):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.linear(x)
    
model = LinearRegression(6)
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

print(f'Test Set MSE: {criterion(predictions.squeeze(), y_test)}')

plt.clf()
plt.scatter(x_test[:, 0].numpy(), predictions.detach().numpy(), label='Predictions', alpha=0.4, color='blue') # pick one item on x_axis
plt.scatter(x_test[:, 0].numpy(), y_test.detach().numpy(), label='True Values', alpha = 0.4, color='orange') # pick one item on x_axis
plt.legend()
plt.savefig("linear_regression_scatterplot.png")