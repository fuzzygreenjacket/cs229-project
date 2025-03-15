import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline

seed = 5
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

df = pd.read_excel("../Cleaned Data/merged_city_year_panel milestone updated.xlsx")
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
for col in to_clean:
    df = df[df[col] != "--"]
df[to_clean] = df[to_clean].apply(pd.to_numeric, errors='raise')

df["LGFV Interest-bearing Debt(CNY,B) / GDP(CNY,B)"] = df["LGFV Interest-bearing Debt(CNY,B)"] / df["GDP(CNY,B)"]
df["Balance of Urban Investment Bond(CNY,B) / GDP(CNY,B)"] = df["Balance of Urban Investment Bond(CNY,B)"] / df["GDP(CNY,B)"]

full_features = [
    "Liability Ratio(%)",
    "Debt Ratio(%)",
    "Growth Rate of GDP(%)",
    "Comprehensive Financial Resources(CNY,B)",
    "Fiscal Self-sufficiency(%)",
    "Budget Revenue(CNY,B)",
    "Revenue of Government-Managed Funds(CNY,B)",
    "State-owned Land Transfer Income/Budget Revenue(%)"
]
selected_features = [
    "Liability Ratio(%)",
    "Debt Ratio(%)",
    "Growth Rate of GDP(%)",
    "Fiscal Self-sufficiency(%)",
    "Budget Revenue(CNY,B)",
    "State-owned Land Transfer Income/Budget Revenue(%)"
]

outcome1 = "LGFV Interest-bearing Debt(CNY,B) / GDP(CNY,B)"
outcome2 = "Balance of Urban Investment Bond(CNY,B) / GDP(CNY,B)"

hyperparams1 = ["poisson", 5, 2, 1000]
hyperparams2 = ["squared_error", 5, 2, 100]

class RNNRegression(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNRegression, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

hidden_size = 10
num_epochs = 50000
lr = 0.005
momentum = 0.3

def evaluate_ensemble(outcome, features, rf_hyperparams):
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    for _, group in df.groupby('year'):
        train, test = train_test_split(group, test_size=0.2, random_state=seed)
        train_data = pd.concat([train_data, train], ignore_index=True)
        test_data = pd.concat([test_data, test], ignore_index=True)
    
    train_rnn = train_data.copy()
    test_rnn = test_data.copy()
    
    scaler_feat = StandardScaler()
    scaler_out = StandardScaler()
    scaler_feat.fit(train_rnn[features])
    train_rnn[features] = scaler_feat.transform(train_rnn[features])
    test_rnn[features] = scaler_feat.transform(test_rnn[features])
    
    scaler_out.fit(train_rnn[[outcome]])
    train_rnn[outcome] = scaler_out.transform(train_rnn[[outcome]])
    test_rnn[outcome] = scaler_out.transform(test_rnn[[outcome]])
    
    train_split_rnn, val_split_rnn = train_test_split(train_rnn, test_size=0.2, random_state=seed)
    
    x_train = torch.from_numpy(train_split_rnn[features].to_numpy()).float().unsqueeze(1)
    y_train = torch.from_numpy(train_split_rnn[outcome].to_numpy()).float()
    x_val = torch.from_numpy(val_split_rnn[features].to_numpy()).float().unsqueeze(1)
    y_val = torch.from_numpy(val_split_rnn[outcome].to_numpy()).float()
    
    model = RNNRegression(input_size=len(features), hidden_size=hidden_size)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=100)
    
    best_val_loss = float('inf')
    patience = 500
    patience_counter = 0
    
    for _ in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train).squeeze()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val).squeeze()
            val_loss = criterion(val_outputs, y_val)
        scheduler.step(val_loss)
        
        if val_loss.item() < best_val_loss - 1e-4:
            best_val_loss = val_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break
    
    model.eval()
    with torch.no_grad():
        rnn_val_preds = model(x_val).squeeze()
    rnn_val_preds_orig = scaler_out.inverse_transform(rnn_val_preds.numpy().reshape(-1,1)).squeeze()
    y_val_orig = scaler_out.inverse_transform(y_val.numpy().reshape(-1,1)).squeeze()
    mse_rnn_val = mean_squared_error(y_val_orig, rnn_val_preds_orig)
    
    x_test = torch.from_numpy(test_rnn[features].to_numpy()).float().unsqueeze(1)
    with torch.no_grad():
        rnn_test_preds = model(x_test).squeeze()
    rnn_test_preds_orig = scaler_out.inverse_transform(rnn_test_preds.numpy().reshape(-1,1)).squeeze()
    train_rf, val_rf = train_test_split(train_data, test_size=0.2, random_state=seed)
    
    rf_pipeline = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(
            criterion=rf_hyperparams[0],
            min_samples_leaf=rf_hyperparams[1],
            min_samples_split=rf_hyperparams[2],
            n_estimators=rf_hyperparams[3],
            random_state=seed
        )
    )
    rf_pipeline.fit(train_rf[features], train_rf[outcome])
    rf_val_preds = rf_pipeline.predict(val_rf[features])
    mse_rf_val = mean_squared_error(val_rf[outcome], rf_val_preds)
    
    rf_test_preds = rf_pipeline.predict(test_data[features])
    
    weight_rnn = (1/mse_rnn_val) / ((1/mse_rnn_val) + (1/mse_rf_val))
    weight_rf = 1 - weight_rnn
    ensemble_preds = weight_rnn * rnn_test_preds_orig + weight_rf * rf_test_preds
    
    test_true = test_data[outcome].values
    n = len(test_true)
    performance_scores = []
    for _ in range(1000):
        indices = np.random.choice(n, size=n, replace=True)
        mse_sample = np.mean((test_true[indices] - ensemble_preds[indices])**2)
        performance_scores.append(-np.log(mse_sample + 1e-10))
    performance_metric = np.mean(performance_scores)
    
    test_mse = mean_squared_error(test_true, ensemble_preds)
    test_r2 = r2_score(test_true, ensemble_preds)
    
    print(f"Outcome: {outcome}")
    print(f"Computed Weights - RNN: {weight_rnn:.4f}, RF: {weight_rf:.4f}")
    print(f"Test R^2: {test_r2:.4f}")
    print(f"Bootstrap: {performance_metric:.4f}")
    print("")
    return test_mse, test_r2, performance_metric, weight_rnn, weight_rf

print("Full Features")
evaluate_ensemble(outcome1, full_features, hyperparams1)
evaluate_ensemble(outcome2, full_features, hyperparams2)

print("Selected Features")
evaluate_ensemble(outcome1, selected_features, hyperparams1)
evaluate_ensemble(outcome2, selected_features, hyperparams2)