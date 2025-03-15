# XGBoost to predict the two outcomes
# random split, groupby year


import numpy as np
import pandas as pd
import shap
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization
import xgboost as xgb
from sklearn.metrics import mean_squared_error

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

# replace '--' with NaN
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


# two outcomes
outcome1 = "LGFV Interest-bearing Debt(CNY,B) / GDP(CNY,B)"
outcome2 = "Balance of Urban Investment Bond(CNY,B) / GDP(CNY,B)"

df = df.dropna(subset=all_features)


# define X (features) and y (targets)
X = df[all_features]
y1 = df[outcome1]  # Target variable for model1
y2 = df[outcome2]  # Target variable for model2

# initialize the StandardScaler
scaler = StandardScaler()

# create empty DataFrames for train and test sets
train_data = pd.DataFrame()
test_data = pd.DataFrame()

# perform 80/20 split for each year separately
for year, group in df.groupby('Year'):
    train, test = train_test_split(group, test_size=0.2, random_state=123)
    train_data = pd.concat([train_data, train], ignore_index=True)
    test_data = pd.concat([test_data, test], ignore_index=True)

# extract features and target variables after splitting
X_train = train_data[all_features]
X_test = test_data[all_features]

y_train1 = train_data[outcome1]
y_test1 = test_data[outcome1]

y_train2 = train_data[outcome2]
y_test2 = test_data[outcome2]

# apply StandardScaler on training features and transform both train & test sets
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# convert data to DMatrix format for XGBoost
dtrain1 = xgb.DMatrix(X_train, label=y_train1)
dtest1 = xgb.DMatrix(X_test, label=y_test1)

dtrain2 = xgb.DMatrix(X_train, label=y_train2)
dtest2 = xgb.DMatrix(X_test, label=y_test2)

# Function to evaluate XGBoost for the first outcome variable
def xgb_evaluate_1(learning_rate, max_depth, subsample, colsample_bytree):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': learning_rate,
        'max_depth': int(max_depth),
        'subsample': subsample,
        'colsample_bytree': colsample_bytree
    }
    model = xgb.train(params, dtrain1, num_boost_round=100, evals=[(dtest1, "eval")], verbose_eval=False)
    preds = model.predict(dtest1)
    return -mean_squared_error(y_test1, preds)  # Minimize MSE

# for the second outcome variable
def xgb_evaluate_2(learning_rate, max_depth, subsample, colsample_bytree):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': learning_rate,
        'max_depth': int(max_depth),
        'subsample': subsample,
        'colsample_bytree': colsample_bytree
    }
    model = xgb.train(params, dtrain2, num_boost_round=100, evals=[(dtest2, "eval")], verbose_eval=False)
    preds = model.predict(dtest2)
    return -mean_squared_error(y_test2, preds)  # Minimize MSE

# Bayesian Optimization for first outcome
optimizer_1 = BayesianOptimization(
    f=xgb_evaluate_1,
    pbounds={'learning_rate': (0.01, 0.2), 'max_depth': (3, 10), 'subsample': (0.7, 1), 'colsample_bytree': (0.7, 1)},
    random_state=123
)
optimizer_1.maximize(init_points=5, n_iter=25)
print("Best Parameters for Outcome 1:", optimizer_1.max)

# Bayesian Optimization for second outcome
optimizer_2 = BayesianOptimization(
    f=xgb_evaluate_2,
    pbounds={'learning_rate': (0.01, 0.2), 'max_depth': (3, 10), 'subsample': (0.7, 1), 'colsample_bytree': (0.7, 1)},
    random_state=123
)
optimizer_2.maximize(init_points=5, n_iter=25)
print("Best Parameters for Outcome 2:", optimizer_2.max)

# Define XGBoost parameters based on Bayesian Optimization
params1 = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.14,
    "max_depth": 10,
    "subsample": 0.99,
    "colsample_bytree": 0.97,
    "reg_alpha": 5,
    "reg_lambda": 10
}

params2 = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.08,
    "max_depth": 10,
    "subsample": 0.77,
    "colsample_bytree": 0.97,
    "reg_alpha": 5,
    "reg_lambda": 10
}


model1 = xgb.train(params1, dtrain1, num_boost_round=100)
model2 = xgb.train(params2, dtrain2, num_boost_round=100)


# predictions
y_pred1 = model1.predict(dtest1)
y_pred2 = model2.predict(dtest2)

# evaluate using MSE
mse1 = mean_squared_error(y_test1, y_pred1)
mse2 = mean_squared_error(y_test2, y_pred2)

# training MSE
y_train_pred1 = model1.predict(dtrain1)
mse_train1 = mean_squared_error(y_train1, y_train_pred1)

y_train_pred2 = model2.predict(dtrain2)
mse_train2 = mean_squared_error(y_train2, y_train_pred2)

# print test MSE
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


# Initialize SHAP explainer for model 1
explainer1 = shap.Explainer(model1, X_train)
shap_values1 = explainer1(X_test)

# Initialize SHAP explainer for model 2
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



