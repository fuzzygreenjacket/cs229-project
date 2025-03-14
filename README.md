# Assessing Debt Risks in China's Local Government Financing Vehicles (LGFVs)

## Overview
This repository contains code and analysis related to evaluating the debt sustainability and risk associated with China's Local Government Financing Vehicles (LGFVs). Using machine learning techniques, we examine the relationships between macroeconomic conditions, local financial indicators, and LGFV debt metrics, specifically focusing on two outcomes:
- LGFV Interest-Bearing Debt / GDP
- Balance of Urban Investment Bond / GDP
Our goal is to provide predictive insights that aid in identifying municipalities at higher risk of financial difficulties.

## Dataset

We analyze LGFV debt data alongside macroeconomic indicators across all prefecture-level cities in China from 2018 to 2023. Data sources include:
- Wind Database
- Chinese Statistical Yearbooks

## File Description
Our repository contains the following key files:
- `Summary of statistics.Rmd` — R Markdown document summarizing data statistics.
- `lgfv_data_cleaning.py` — Cleans and preprocesses the raw dataset.
- `baseline_linear_regression.py` — Baseline linear regression implementation with all features.
- `baseline_linear_regression_best_features.py` — Linear regression with selected optimal features.
- `baseline_linear_regression_feature_engineering.py` — Linear regression with engineered features.
- `XGBoost random split.py` — XGBoost model implementation.
- `random_forest_regressor.py` — Random forest model for prediction.
- `random_forest_regressor_grid_search.py` — Hyperparameter tuning via grid search for Random Forest.
- `rnn.py` — Recurrent Neural Network implementation.
- `rnn_hyperparam.py` — Hyperparameter tuning for the RNN.
- `ensemble.py` — Weighted ensemble model (RNN and Random Forest) Implementation.


## Methods
- **Linear Regression** (Baseline)
- **XGBoost & Random Forest** (Tree-based models to capture non-linear relationships)
- **Recurrent Neural Network (RNN)** (Capturing sequential dependencies in financial data)
- **Ensemble Model** (Integration of RNN and Random Forest)

## Evaluation
Models are evaluated using Mean Squared Error (MSE) with bootstrap resampling and $R^2$ scores. To be precise, our performance metric is defined as follows: $\frac{1}{1000}\sum_{j=1}^{1000}\left[-\log\left(\frac{1}{n}\sum_{i=1}^{n}\left(y_{ij}-\hat{y}_{ij}\right)^2\right)\right]$. We then use SHAP analysis to identify key predictors influencing model decisions.

## Usage
To replicate the analyses:
1. Run data cleaning: python lgfv_data_cleaning.py.
2. Train and evaluate models sequentially (linear regression → tree-based models → RNN).
3. Conduct hyperparameter tuning where relevant (Random Forest, XGBoost, RNN).

## Team Members
- Jong Beom (JB) Lim, Stanford Computer Science
- Xinru Pan, Stanford Political Science
- Matt Hsu, Stanford Mathematics

## Acknowledgments
This project utilizes data from the Wind Database and Chinese Statistical Yearbooks, and references prior works in the field, including methods adapted from [Zhang et al., 2022] and others.




