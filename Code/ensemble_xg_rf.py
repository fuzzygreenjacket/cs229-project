import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
import xgboost as xgb

seed = 8
random.seed(seed)
np.random.seed(seed)

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

df = df.dropna(subset=full_features)

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.14,
    "max_depth": 10,
    "subsample": 0.99,
    "colsample_bytree": 0.97,
    "reg_alpha": 5,
    "reg_lambda": 10
}

hyperparams1 = ["poisson", 5, 2, 1000]
hyperparams2 = ["squared_error", 5, 2, 100]

def evaluate_ensemble(outcome, features, rf_hyperparams):
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    for _, group in df.groupby('year'):
        train, test = train_test_split(group, test_size=0.2, random_state=seed)
        train_data = pd.concat([train_data, train], ignore_index=True)
        test_data = pd.concat([test_data, test], ignore_index=True)
    
    train_split_xgb, val_split_xgb = train_test_split(train_data, test_size=0.2, random_state=seed)
    dtrain_xgb = xgb.DMatrix(train_split_xgb[features], label=train_split_xgb[outcome])
    dval_xgb = xgb.DMatrix(val_split_xgb[features], label=val_split_xgb[outcome])
    
    model_xgb = xgb.train(xgb_params, dtrain_xgb, num_boost_round=100, evals=[(dval_xgb, "eval")], verbose_eval=False)
    preds_xgb_val = model_xgb.predict(dval_xgb)
    mse_xgb_val = mean_squared_error(val_split_xgb[outcome], preds_xgb_val)
    
    dtest_xgb = xgb.DMatrix(test_data[features])
    xgb_test_preds = model_xgb.predict(dtest_xgb)
    
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

    weight_xgb = (1/mse_xgb_val) / ((1/mse_xgb_val) + (1/mse_rf_val))
    weight_rf = 1 - weight_xgb
    ensemble_preds = weight_xgb * xgb_test_preds + weight_rf * rf_test_preds
    
    test_true = test_data[outcome].values
    n = len(test_true)
    bootstrap_iterations = 1000
    performance_scores = []
    for _ in range(bootstrap_iterations):
        indices = np.random.choice(n, size=n, replace=True)
        mse_sample = np.mean((test_true[indices] - ensemble_preds[indices])**2)
        performance_scores.append(-np.log(mse_sample + 1e-10))
    performance_metric = np.mean(performance_scores)
    
    test_mse = mean_squared_error(test_true, ensemble_preds)
    test_r2 = r2_score(test_true, ensemble_preds)
    
    print(f"Outcome: {outcome}")
    print(f"Validation MSE - XGBoost: {mse_xgb_val:.4f}, RF: {mse_rf_val:.4f}")
    print(f"Computed Weights - XGBoost: {weight_xgb:.4f}, RF: {weight_rf:.4f}")
    print(f"Performance Metric (bootstrap): {performance_metric:.4f}")
    print(f"Test R²: {test_r2:.4f}\n")

    
    return test_mse, test_r2, performance_metric, weight_xgb, weight_rf

print("Full Features")
evaluate_ensemble(outcome1, full_features, hyperparams1)
evaluate_ensemble(outcome2, full_features, hyperparams2)

print("Selected Features")
evaluate_ensemble(outcome1, selected_features, hyperparams1)
evaluate_ensemble(outcome2, selected_features, hyperparams2)