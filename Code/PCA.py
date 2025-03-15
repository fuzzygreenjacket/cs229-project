# PCA Analysis for features

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df = pd.read_excel("../Cleaned Data/merged_city_year_panel milestone updated.xlsx")

to_clean = [
    "Liability Ratio(%)", "Debt Ratio(%)", "GDP(CNY,B)", "Growth Rate of GDP(%)",
    "Comprehensive Financial Resources(CNY,B)", "Fiscal Self-sufficiency(%)",
    "Budget Revenue(CNY,B)", "Revenue of Government-Managed Funds(CNY,B)",
    "State-owned Land Transfer Income/Budget Revenue(%)", "Real Estate Investment(CNY,B)",
    "LGFV Interest-bearing Debt(CNY,B)", "Balance of Urban Investment Bond(CNY,B)"
]

df.replace("--", np.nan, inplace=True)
df[to_clean] = df[to_clean].apply(pd.to_numeric, errors='raise')

df["LGFV Interest-bearing Debt(CNY,B) / GDP(CNY,B)"] = df["LGFV Interest-bearing Debt(CNY,B)"] / df["GDP(CNY,B)"]
df["Balance of Urban Investment Bond(CNY,B) / GDP(CNY,B)"] = df["Balance of Urban Investment Bond(CNY,B)"] / df["GDP(CNY,B)"]

df = df.dropna(subset=to_clean)

# Define features and targets
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
outcome1 = "LGFV Interest-bearing Debt(CNY,B) / GDP(CNY,B)"
outcome2 = "Balance of Urban Investment Bond(CNY,B) / GDP(CNY,B)"

# rename for my plots
feature_names = [
    "Liability Ratio",
    "Debt Ratio",
    "GDP Growth",
    "Financial Resources",
    "Fiscal Self-sufficiency",
    "Budget Revenue",
    "Government-Managed Funds",
    "State-owned Land Transfer"
]

train_data = pd.DataFrame()
test_data = pd.DataFrame()

# Train-test split
for year, group in df.groupby('Year'):
    train, test = train_test_split(group, test_size=0.2, random_state=123)
    train_data = pd.concat([train_data, train], ignore_index=True)
    test_data = pd.concat([test_data, test], ignore_index=True)

X_train = train_data[all_features]
X_test = test_data[all_features]

y_train1 = train_data[outcome1]
y_test1 = test_data[outcome1]

y_train2 = train_data[outcome2]
y_test2 = test_data[outcome2]

# standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA with automatic component selection with 95% variance
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# convert transformed data to DataFrame
n_components = X_train_pca.shape[1]
X_train_pca_df = pd.DataFrame(X_train_pca, columns=[f'PC{i+1}' for i in range(n_components)])
X_test_pca_df = pd.DataFrame(X_test_pca, columns=[f'PC{i+1}' for i in range(n_components)])

# perform K-Means Clustering on PCA-transformed data
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(X_train_pca_df)

# scatter plot of PCA clusters
plt.figure(figsize=(8,6))
plt.scatter(X_train_pca_df.iloc[:,0], X_train_pca_df.iloc[:,1], c=clusters, cmap='viridis', alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Clustering (K-Means, 3 Clusters)")
plt.colorbar(label="Cluster Label")
plt.show()


# Plot explained variance
plt.figure(figsize=(8,5))
plt.plot(range(1, n_components+1), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Components')
plt.show()

# get the principal component loadings
pc_loadings = pd.DataFrame(pca.components_, columns=all_features, index=[f"PC{i+1}" for i in range(n_components)])
print(pc_loadings)

pc_loadings.columns = feature_names

# Plot heatmap with updated labels
plt.figure(figsize=(12, 6))  # Adjust figure size
sns.heatmap(pc_loadings, annot=True, cmap="coolwarm", center=0)

plt.xticks(rotation=45, ha='right')  # Rotate feature names for readability
plt.yticks(rotation=0)  # Keep principal component names horizontal

plt.title("Principal Component Loadings")
plt.xlabel("Original Features")
plt.ylabel("Principal Components")

plt.show()