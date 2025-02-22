# XGBoost to predict the two outcomes
# random split


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import shap
import jinja2


df = pd.read_excel("merged_city_year_panel milestone updated.xlsx")

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

# Replace '--' with NaN
df.replace("--", np.nan, inplace=True)

# convert data type to numbers
df[to_clean] = df[to_clean].apply(pd.to_numeric, errors='raise')

df.info()
latex_summary = df.describe().to_latex()
print(latex_summary)

# creating new variables
df["LGFV Interest-bearing Debt(CNY,B) / GDP(CNY,B)"] = df["LGFV Interest-bearing Debt(CNY,B)"]/df["GDP(CNY,B)"] # outcome 1
df["Balance of Urban Investment Bond(CNY,B) / GDP(CNY,B)"] = df["Balance of Urban Investment Bond(CNY,B)"]/df["GDP(CNY,B)"] # outcome 2
df["Real Estate GDP"] = df["Real Estate Investment(CNY,B)"]/df["GDP(CNY,B)"] # potential feature

# list of all possible features
all_features = [
    "Liability Ratio(%)",
    "Debt Ratio(%)",
    "Growth Rate of GDP(%)",
    "Comprehensive Financial Resources(CNY,B)",
    "Fiscal Self-sufficiency(%)",
    "Budget Revenue(CNY,B)",
    "Revenue of Government-Managed Funds(CNY,B)",
    "State-owned Land Transfer Income/Budget Revenue(%)"
]

best_features = [
    "Liability Ratio(%)",
    "Debt Ratio(%)",
    "Growth Rate of GDP(%)",
    "Comprehensive Financial Resources(CNY,B)",
    "Fiscal Self-sufficiency(%)",
    "Budget Revenue(CNY,B)",
    "State-owned Land Transfer Income/Budget Revenue(%)"
]

# possible outcomes
outcome1 = "LGFV Interest-bearing Debt(CNY,B) / GDP(CNY,B)"
outcome2 = "Balance of Urban Investment Bond(CNY,B) / GDP(CNY,B)"

df = df.dropna(subset=all_features)


# Define X (features) and y (targets)
X = df[all_features]
y1 = df[outcome1]  # Target variable for model1
y2 = df[outcome2]  # Target variable for model2

# Perform a random split (80% train, 20% test)
X_train, X_test, y_train1, y_test1 = train_test_split(X, y1, test_size=0.2, random_state=123)
X_train, X_test, y_train2, y_test2 = train_test_split(X, y2, test_size=0.2, random_state=123)

# Convert data to DMatrix format (for XGBoost)
dtrain1 = xgb.DMatrix(X_train, label=y_train1)
dtest1 = xgb.DMatrix(X_test, label=y_test1)

dtrain2 = xgb.DMatrix(X_train, label=y_train2)
dtest2 = xgb.DMatrix(X_test, label=y_test2)

# Define XGBoost parameters
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.05,
    "max_depth": 5,  # Reduce max_depth to prevent deep trees#
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 5,  # L1 regularization (adds sparsity, removes irrelevant features)
    "reg_lambda": 10  # L2 regularization (reduces model complexity)
}


model1 = xgb.train(params, dtrain1, num_boost_round=100)
model2 = xgb.train(params, dtrain2, num_boost_round=100)


# Predictions
y_pred1 = model1.predict(dtest1)
y_pred2 = model2.predict(dtest2)

# Evaluate using MSE
mse1 = mean_squared_error(y_test1, y_pred1)
mse2 = mean_squared_error(y_test2, y_pred2)

# Training MSE
y_train_pred1 = model1.predict(dtrain1)
mse_train1 = mean_squared_error(y_train1, y_train_pred1)

y_train_pred2 = model2.predict(dtrain2)
mse_train2 = mean_squared_error(y_train2, y_train_pred2)

# Test MSE (already calculated)
print(f"Training MSE for {outcome1}: {mse_train1}")
print(f"Test MSE for {outcome1}: {mse1}")

print(f"Training MSE for {outcome2}: {mse_train2}")
print(f"Test MSE for {outcome2}: {mse2}")

r2_train1 = r2_score(y_train1, y_train_pred1)
r2_test1 = r2_score(y_test1, y_pred1)

r2_train2 = r2_score(y_train2, y_train_pred2)
r2_test2 = r2_score(y_test2, y_pred2)

print(f"Training R² for {outcome1}: {r2_train1}")
print(f"Test R² for {outcome1}: {r2_test1}")

print(f"Training R² for {outcome2}: {r2_train2}")
print(f"Test R² for {outcome2}: {r2_test2}")



# Feature importance plots
xgb.plot_importance(model1)
plt.title(f"Feature Importance for {outcome1}")
plt.show()

xgb.plot_importance(model2)
plt.title(f"Feature Importance for {outcome2}")
plt.show()

# Initialize SHAP explainer for model 1 (LGFV Interest-bearing Debt)
explainer1 = shap.Explainer(model1, X_train)
shap_values1 = explainer1(X_test)

# Initialize SHAP explainer for model 2 (Balance of Urban Investment Bond)
explainer2 = shap.Explainer(model2, X_train)
shap_values2 = explainer2(X_test)

feature_names = list(X_test.columns)

shap.summary_plot(shap_values1, X_test, feature_names=feature_names)
shap.summary_plot(shap_values2, X_test, feature_names=feature_names)
