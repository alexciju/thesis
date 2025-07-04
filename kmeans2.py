# -*- coding: utf-8 -*-
"""kmeans2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1rNt6zlg1McziuHTdhu7fF0P5-5axg22X
"""

pip install linearmodels

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.cluster import KMeans

# Load and clean data
df = pd.read_csv('/content/Melbourne_housing_FULL.csv')
df = df[['Price', 'Rooms', 'Bathroom', 'Landsize', 'Suburb', 'Date',
         'BuildingArea', 'Bedroom2', 'Car', 'YearBuilt', 'Method', 'SellerG', 'Type']].dropna()
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Month'] = df['Date'].dt.to_period('M').astype(int)
df['log_price'] = np.log(df['Price'])

# Features
features_to_scale = ['BuildingArea', 'Bedroom2', 'Car', 'YearBuilt', 'Rooms', 'Bathroom', 'Landsize', 'Month']
categorical_vars = ['Suburb', 'Month', 'Method', 'SellerG', 'Type']

# Cluster range
cluster_range = range(2, 128, 1)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results_list = []

for k in cluster_range:
    # Scale features
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[features_to_scale])

    # Add cluster labels and distances
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(scaled)
    centroids = kmeans.cluster_centers_
    df['DistanceToCentroid'] = np.linalg.norm(scaled - centroids[df['Cluster']], axis=1)

    # One-hot encoding
    df_encoded = pd.get_dummies(df[categorical_vars + features_to_scale + ['Cluster', 'DistanceToCentroid']])
    X_base = df_encoded.drop(columns=['Cluster', 'DistanceToCentroid'])  # base model without cluster features
    X_ext = df_encoded.copy()  # extended model
    y = df['log_price']

    r2_base_list, r2_ext_list = [], []
    rmse_base_list, rmse_ext_list = [], []
    mape_base_list, mape_ext_list = [], []

    for train_index, test_index in kf.split(X_ext):
        Xb_train, Xb_test = X_base.iloc[train_index], X_base.iloc[test_index]
        Xe_train, Xe_test = X_ext.iloc[train_index], X_ext.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Fit models
        base_model = LinearRegression().fit(Xb_train, y_train)
        ext_model = LinearRegression().fit(Xe_train, y_train)

        # Predict
        yb_pred = base_model.predict(Xb_test)
        ye_pred = ext_model.predict(Xe_test)

        # Evaluate
        rmse_base_list.append(np.sqrt(mean_squared_error(y_test, yb_pred)))
        rmse_ext_list.append(np.sqrt(mean_squared_error(y_test, ye_pred)))
        r2_base_list.append(r2_score(y_test, yb_pred))
        r2_ext_list.append(r2_score(y_test, ye_pred))
        mape_base_list.append(mean_absolute_percentage_error(y_test, yb_pred))
        mape_ext_list.append(mean_absolute_percentage_error(y_test, ye_pred))

    # Append results
    results_list.append({
        "Clusters": k,
        "OLS Base R² (CV Mean)": np.mean(r2_base_list),
        "OLS Ext R² (CV Mean)": np.mean(r2_ext_list),
        "Diff in R²": np.mean(r2_ext_list) - np.mean(r2_base_list),
        "OLS Base RMSE (CV Mean)": np.mean(rmse_base_list),
        "OLS Base MAPE (CV Mean)": np.mean(mape_base_list),
        "OLS Ext RMSE (CV Mean)": np.mean(rmse_ext_list),
        "OLS Ext MAPE (CV Mean)": np.mean(mape_ext_list),
    })

# Save and display results
results_cluster_df = pd.DataFrame(results_list)
print(results_cluster_df.round(3))
results_cluster_df.to_csv('/content/results_cluster_sklearn.csv', index=False)
