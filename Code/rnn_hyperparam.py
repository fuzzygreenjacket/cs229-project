import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_excel("../Cleaned Data/merged_city_year_panel milestone updated.xlsx")
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

full_features = [
    "Liability Ratio(%)", "Debt Ratio(%)", "Growth Rate of GDP(%)",
    "Comprehensive Financial Resources(CNY,B)", "Fiscal Self-sufficiency(%)",
    "Budget Revenue(CNY,B)", "Revenue of Government-Managed Funds(CNY,B)",
    "State-owned Land Transfer Income/Budget Revenue(%)"
]

outcome = "LGFV Interest-bearing Debt(CNY,B) / GDP(CNY,B)"

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
num_epochs = 2500
lr_values = [0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.1]
momentum_values = [0.0, 0.3, 0.5, 0.9]
patience = 500

seeds_list = [5, 10, 15, 20, 25, 30]
all_results = []

for s in seeds_list:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    
    train_data_seed = pd.DataFrame()
    test_data_seed = pd.DataFrame()
    for year, group in df.groupby('year'):
        train, test = train_test_split(group, test_size=0.2, random_state=s)
        train_data_seed = pd.concat([train_data_seed, train], ignore_index=True)
        test_data_seed = pd.concat([test_data_seed, test], ignore_index=True)
    
    scaler_feature = StandardScaler()
    scaler_output = StandardScaler()
    scaler_feature.fit(train_data_seed[full_features])
    train_data_scaled = train_data_seed.copy()
    test_data_scaled = test_data_seed.copy()
    train_data_scaled[full_features] = scaler_feature.transform(train_data_seed[full_features])
    test_data_scaled[full_features] = scaler_feature.transform(test_data_seed[full_features])
    scaler_output.fit(train_data_scaled[[outcome]])
    train_data_scaled[outcome] = scaler_output.transform(train_data_scaled[[outcome]])
    test_data_scaled[outcome] = scaler_output.transform(test_data_scaled[[outcome]])
    
    for m in momentum_values:
        for l in lr_values:
            train_split, val_split = train_test_split(train_data_scaled, test_size=0.2, random_state=s)
            x_train = torch.from_numpy(train_split[full_features].to_numpy()).float().unsqueeze(1)
            y_train = torch.from_numpy(train_split[outcome].to_numpy()).float()
            x_val = torch.from_numpy(val_split[full_features].to_numpy()).float().unsqueeze(1)
            y_val = torch.from_numpy(val_split[outcome].to_numpy()).float()
            x_test = torch.from_numpy(test_data_scaled[full_features].to_numpy()).float().unsqueeze(1)
            y_test = torch.from_numpy(test_data_scaled[outcome].to_numpy()).float()
            
            model = RNNRegression(input_size=len(full_features), hidden_size=hidden_size)
            criterion = nn.MSELoss()
            optimizer = optim.SGD(model.parameters(), lr=l, momentum=m)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=100, verbose=False)
            
            best_val_loss = float('inf')
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
                
                if patience_counter >= patience:
                    break
            
            model.eval()
            with torch.no_grad():
                predictions = model(x_test).squeeze()
            prediction_original = scaler_output.inverse_transform(predictions.numpy().reshape(-1, 1)).squeeze()
            y_test_original = scaler_output.inverse_transform(y_test.numpy().reshape(-1, 1)).squeeze()
            mse = np.mean((y_test_original - prediction_original)**2)
            
            n = len(y_test_original)
            perf_scores = []
            for _ in range(1000):
                indices = np.random.choice(n, size=n, replace=True)
                mse_sample = np.mean((y_test_original[indices] - prediction_original[indices])**2)
                perf_scores.append(-np.log(mse_sample))
            performance = np.mean(perf_scores)
            
            all_results.append({
                "Seed": s,
                "Learning_Rate": l,
                "Momentum": m,
                "Performance": performance,
                "MSE": mse
            })
            print(f"Tuning: Seed={s}, LR={l}, Momentum={m}, Performance={performance}, MSE={mse}")

results_df = pd.DataFrame(all_results)
grouped = results_df.groupby(["Learning_Rate", "Momentum"]).mean().reset_index()

plt.figure(figsize=(8,6))
for m in momentum_values:
    subset = grouped[grouped["Momentum"] == m]
    plt.plot(subset["Learning_Rate"], subset["Performance"], marker='o', label=f'Momentum={m}')

plt.xticks(lr_values, rotation=45, ha="right")
plt.xlabel("Learning Rate")
plt.ylabel("Performance (Avg. -log(MSE))")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("hyperparam_tuning_performance.png")
plt.show()