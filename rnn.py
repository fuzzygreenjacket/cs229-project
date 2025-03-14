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

seed = 5
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

df = pd.read_excel("merged_city_year_panel milestone updated.xlsx")

to_clean = [
    "Liability Ratio(%)", "Debt Ratio(%)", "GDP(CNY,B)", "Growth Rate of GDP(%)",
    "Comprehensive Financial Resources(CNY,B)", "Fiscal Self-sufficiency(%)",
    "Budget Revenue(CNY,B)", "Revenue of Government-Managed Funds(CNY,B)",
    "State-owned Land Transfer Income/Budget Revenue(%)", "LGFV Interest-bearing Debt(CNY,B)",
    "Balance of Urban Investment Bond(CNY,B)"
]
for col in to_clean:
    df = df[df[col] != "--"]
df[to_clean] = df[to_clean].apply(pd.to_numeric, errors='raise')

df["LGFV Interest-bearing Debt(CNY,B) / GDP(CNY,B)"] = df["LGFV Interest-bearing Debt(CNY,B)"] / df["GDP(CNY,B)"]
df["Balance of Urban Investment Bond(CNY,B) / GDP(CNY,B)"] = df["Balance of Urban Investment Bond(CNY,B)"] / df["GDP(CNY,B)"]

train_data = pd.DataFrame()
test_data = pd.DataFrame()
for year, group in df.groupby('year'):
    train, test = train_test_split(group, test_size=0.2, random_state=seed)
    train_data = pd.concat([train_data, train], ignore_index=True)
    test_data = pd.concat([test_data, test], ignore_index=True)


# Full features (8 predictors)
full_features = [
    "Liability Ratio(%)", "Debt Ratio(%)", "Growth Rate of GDP(%)",
    "Comprehensive Financial Resources(CNY,B)", "Fiscal Self-sufficiency(%)",
    "Budget Revenue(CNY,B)", "Revenue of Government-Managed Funds(CNY,B)",
    "State-owned Land Transfer Income/Budget Revenue(%)"
]
# Selected features (6 predictors)
selected_features = [
    "Liability Ratio(%)", "Debt Ratio(%)", "Growth Rate of GDP(%)",
    "Fiscal Self-sufficiency(%)", "Budget Revenue(CNY,B)",
    "State-owned Land Transfer Income/Budget Revenue(%)"
]

outcome1 = "LGFV Interest-bearing Debt(CNY,B) / GDP(CNY,B)"
outcome2 = "Balance of Urban Investment Bond(CNY,B) / GDP(CNY,B)"
outcomes = [outcome1, outcome2]

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

def evaluate_model(features_used, outcome_col):
    scaler_feat = StandardScaler()
    scaler_out = StandardScaler()
    
    scaler_feat.fit(train_data[full_features])
    train_scaled = train_data.copy()
    test_scaled = test_data.copy()
    train_scaled[full_features] = scaler_feat.transform(train_data[full_features])
    test_scaled[full_features] = scaler_feat.transform(test_data[full_features])
    
    scaler_out.fit(train_scaled[[outcome_col]])
    train_scaled[outcome_col] = scaler_out.transform(train_scaled[[outcome_col]])
    test_scaled[outcome_col] = scaler_out.transform(test_scaled[[outcome_col]])
    
    train_split, val_split = train_test_split(train_scaled, test_size=0.2, random_state=seed)
    
    x_train = torch.from_numpy(train_split[features_used].to_numpy()).float().unsqueeze(1)
    y_train = torch.from_numpy(train_split[outcome_col].to_numpy()).float()
    x_val = torch.from_numpy(val_split[features_used].to_numpy()).float().unsqueeze(1)
    y_val = torch.from_numpy(val_split[outcome_col].to_numpy()).float()
    x_test = torch.from_numpy(test_scaled[features_used].to_numpy()).float().unsqueeze(1)
    y_test = torch.from_numpy(test_scaled[outcome_col].to_numpy()).float()
    
    model = RNNRegression(input_size=len(features_used), hidden_size=hidden_size)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=100)
    
    best_val_loss = float('inf')
    patience = 500
    patience_counter = 0
    
    for epoch in range(num_epochs):
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
        
        if epoch % 100 == 0:
            print(f'Epoch: {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} with best Val Loss: {best_val_loss:.4f}")
            break
    
    model.eval()
    with torch.no_grad():
        preds = model(x_test).squeeze()
    preds_orig = scaler_out.inverse_transform(preds.numpy().reshape(-1,1)).squeeze()
    y_test_orig = scaler_out.inverse_transform(y_test.numpy().reshape(-1,1)).squeeze()
    
    n = len(y_test_orig)
    perf_scores = []
    for _ in range(1000):
        indices = np.random.choice(n, size=n, replace=True)
        mse_sample = np.mean((y_test_orig[indices] - preds_orig[indices])**2)
        perf_scores.append(-np.log(mse_sample))
    performance_metric = np.mean(perf_scores)
    mse_value = np.mean((y_test_orig - preds_orig)**2)
    r2 = r2_score(y_test_orig, preds_orig)
    
    return performance_metric, mse_value, r2


results = []
for feature_set, label in [(full_features, "Full Features"), (selected_features, "Selected Features")]:
    for outcome in outcomes:
        perf, mse_val, r2_val = evaluate_model(feature_set, outcome)
        results.append({
            "Feature_Set": label,
            "Outcome": outcome,
            "Performance": perf,
            "MSE": mse_val,
            "R^2": r2_val
        })
        print(f"Experiment with {label} for {outcome}: Performance = {perf:.4f}, R^2 = {r2_val:.4f}")
