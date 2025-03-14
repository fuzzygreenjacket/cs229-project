# XGBoost to predict the two outcomes
# random split


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import shap
from bayes_opt import BayesianOptimization



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

# explain why I chose these
best_features = [
    "Liability Ratio(%)",
    "Debt Ratio(%)",
    "Growth Rate of GDP(%)",
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

# Initialize the StandardScaler
scaler = StandardScaler()

# Create empty DataFrames for train and test sets
train_data = pd.DataFrame()
test_data = pd.DataFrame()

# Perform 80/20 split for each year separately
for year, group in df.groupby('Year'):
    train, test = train_test_split(group, test_size=0.2, random_state=123)
    train_data = pd.concat([train_data, train], ignore_index=True)
    test_data = pd.concat([test_data, test], ignore_index=True)

# Extract features and target variables after splitting
X_train = train_data[all_features]
X_test = test_data[all_features]

y_train1 = train_data[outcome1]
y_test1 = test_data[outcome1]

y_train2 = train_data[outcome2]
y_test2 = test_data[outcome2]

# Apply StandardScaler on training features and transform both train & test sets
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Convert data to DMatrix format (for XGBoost)
dtrain1 = xgb.DMatrix(X_train, label=y_train1)
dtest1 = xgb.DMatrix(X_test, label=y_test1)

dtrain2 = xgb.DMatrix(X_train, label=y_train2)
dtest2 = xgb.DMatrix(X_test, label=y_test2)

def xgb_evaluate(learning_rate, max_depth, subsample, colsample_bytree):
    params = {
        'objective': 'reg:pseudohubererror',
        'eval_metric': 'rmse',
        'learning_rate': learning_rate,
        'max_depth': int(max_depth),
        'subsample': subsample,
        'colsample_bytree': colsample_bytree
    }
    model = xgb.train(params, dtrain1, num_boost_round=100, evals=[(dtest1, "eval")], verbose_eval=False)
    preds = model.predict(dtest1)
    return -mean_squared_error(y_test1, preds)  # Minimize MSE

optimizer = BayesianOptimization(
    f=xgb_evaluate,
    pbounds={'learning_rate': (0.01, 0.2), 'max_depth': (3, 10), 'subsample': (0.7, 1), 'colsample_bytree': (0.7, 1)},
    random_state=123
)
optimizer.maximize(init_points=5, n_iter=25)

print("Best Parameters:", optimizer.max)

# Define XGBoost parameters
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.14,
    "max_depth": 10,  # Reduce max_depth to prevent deep trees#
    "subsample": 0.8,
    "colsample_bytree": 0.77,
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

# Bootstrap resampling
n = len(y_test1)
bootstrap_iterations = 1000

def bootstrap_evaluation(y_test, y_pred):
    n = len(y_test)
    perf_scores = []
    for _ in range(bootstrap_iterations):
        indices = np.random.choice(n, size=n, replace=True)
        mse_sample = np.mean((y_test.iloc[indices] - y_pred[indices])**2)
        perf_scores.append(-np.log(mse_sample + 1e-10))
    performance_metric = np.mean(perf_scores)
    mse_value = np.mean((y_test - y_pred)**2)
    r2 = r2_score(y_test, y_pred)
    return performance_metric, mse_value, r2

perf1, mse1, r2_1 = bootstrap_evaluation(y_test1, y_pred1)
perf2, mse2, r2_2 = bootstrap_evaluation(y_test2, y_pred2)

print(f"Bootstrap Performance for {outcome1}: Mean={perf1:.4f}, MSE={mse1:.4f}, R^2={r2_1:.4f}")
print(f"Bootstrap Performance for {outcome2}: Mean={perf2:.4f}, MSE={mse2:.4f}, R^2={r2_2:.4f}")


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

# SHAP Feature interaction analysis
shap_interaction_values = shap.TreeExplainer(model1).shap_interaction_values(X_test)
shap.summary_plot(shap_interaction_values, X_test)

shap_interaction_values = shap.TreeExplainer(model2).shap_interaction_values(X_test)
shap.summary_plot(shap_interaction_values, X_test)


