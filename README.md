# Assessing Debt Risks in China's Local Government Financing Vehicles (LGFVs)

## Overview
This repository contains code and analysis related to evaluating the debt sustainability and risk associated with China's Local Government Financing Vehicles (LGFVs). Using machine learning techniques, we examine the relationships between macroeconomic conditions, local financial indicators, and LGFV debt metrics, specifically focusing on two outcomes:
\begin{itemize}
	\item LGFV Interest-Bearing Debt / GDP
	\item Balance of Urban Investment Bond / GDP
\end{itemize}

Our goal is to provide predictive insights that aid in identifying municipalities at higher risk of financial difficulties.

##Dataset

We analyze LGFV debt data alongside macroeconomic indicators across all prefecture-level cities in China from 2018 to 2023. Data sources include:
\begin{itemize}
	\item Wind Database
	\item Chinese Statistical Yearbooks
\end{itemize}

## File Description
Our repository contains the following key files:

\begin{itemize}
    \item \texttt{Summary of statistics.Rmd} -- R Markdown document summarizing data statistics
    \item \texttt{lgfv\_data\_cleaning.py} -- Cleans and preprocesses the raw dataset.
    \item \texttt{baseline\_linear\_regression.py} -- Baseline linear regression implementation with all features
    \item \texttt{baseline\_linear\_regression\_best\_features.py} -- Linear regression with selected optimal features
    \item \texttt{baseline\_linear\_regression\_feature\_engineering.py} -- Linear regression with engineered features
    \item \texttt{XGBoost random split.py} -- XGBoost model implementation
    \item \texttt{random\_forest\_regressor.py} -- Random forest model for prediction
    \item \texttt{random\_forest\_regressor\_grid\_search.py} -- Hyperparameter tuning via grid search for Random Forest
    \item \texttt{rnn.py} -- Recurrent Neural Network implementation
    \item \texttt{rnn\_hyperparam.py} -- Hyperparameter tuning for the RNN..
\end{itemize}

#Methods
\begin{itemize}
	\item Linear Regression (Baseline)
	\item XGBoost \& Random Forest (Tree-based models to capture non-linear relationships)
	\item Recurrent Neural Network (RNN) (Capturing sequential dependencies in financial data)
	\item Ensemble Model (Integration of RNN and Random Forest)
\end{itemize}

#Evaluation
Models are evaluated using Mean Squared Error (MSE) with bootstrap resampling and $R^2$ scores. SHAP analysis to identify key predictors influencing model decisions

#Usage
To replicate the analyses:
\begin{itemize}
	\item Run data cleaning: python lgfv_data_cleaning.py.
	\item Train and evaluate models sequentially (linear regression → tree-based models → RNN).
	\item Conduct hyperparameter tuning where relevant (Random Forest, XGBoost, RNN).
\end{itemize}

## Team Members
- Jong Beom (JB) Lim, Stanford Computer Science
- Xinru Pan, Stanford Political Science
- Matt Hsu, Stanford Mathematics

#Acknowledgments
This project utilizes data from the Wind Database and Chinese Statistical Yearbooks, and references prior works in the field, including methods adapted from [Zhang et al., 2022] and others.




