import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load and clean data
df = pd.read_csv('data/Melbourne_housing_FULL.csv')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Month'] = df['Date'].dt.to_period('M').astype(str)
df['Log_Price'] = np.log(df['Price'])

df.drop(columns=['Address', 'Postcode', 'Regionname', 'CouncilArea', 'Longtitude', 'Lattitude'], inplace=True)
df.dropna(subset=['Log_Price', 'YearBuilt', 'BuildingArea', 'Landsize', 'Car'], inplace=True)

# Filter valid test data
def filter_valid_test(df_test, df_train, cat_cols):
    valid_mask = pd.Series(True, index=df_test.index)
    for col in cat_cols:
        if col in df_train.columns:
            valid_values = df_train[col].dropna().unique()
            valid_mask &= df_test[col].isin(valid_values)
    return df_test[valid_mask].copy()

# Identify feature columns (exclude target and date)
base_features = [col for col in df.columns if col not in ['Price', 'Log_Price', 'Date']]
num_cols = df[base_features].select_dtypes('number').columns
cat_cols = df[base_features].select_dtypes('object').columns

# Define OLS model variants
ols_variants = {
    "OLS_base": base_features,
    "OLS_no_month": [col for col in base_features if col != 'Month'],
    "OLS_no_suburb": [col for col in base_features if col != 'Suburb'],
    "OLS_no_both": [col for col in base_features if col not in ['Month', 'Suburb']]
}

# Storage for results
results = []

# Loop through each variant
for name, features in ols_variants.items():
    X = df[features].copy()
    y = df['Log_Price'].copy()

    num_cols_variant = X.select_dtypes('number').columns
    cat_cols_variant = X.select_dtypes('object').columns

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test = filter_valid_test(X_test, X_train, cat_cols_variant)
    y_test = y.loc[X_test.index]

    print(f"{name}: train {X_train.shape}, test {X_test.shape}")
    assert 'Log_Price' not in X.columns

    cat_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='Missing', add_indicator=True)),
        ('encode', OneHotEncoder(drop='first', handle_unknown='ignore', min_frequency=30))
    ])
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols_variant),
        ('cat', cat_pipeline, cat_cols_variant)
    ])

    pipe = Pipeline([
        ('prep', preprocessor),
        ('model', LinearRegression())
    ])

    pipe.fit(X_train, y_train)

    y_pred_train_log = pipe.predict(X_train)
    y_pred_train = np.exp(y_pred_train_log)
    y_true_train = np.exp(y_train)
    r2_train = r2_score(y_train, y_pred_train_log)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train_log))
    mape_train = np.mean(np.abs((y_true_train - y_pred_train) / y_true_train)) * 100

    y_pred_log = pipe.predict(X_test)
    y_pred = np.exp(y_pred_log)
    y_true = np.exp(y_test)
    r2 = r2_score(y_test, y_pred_log)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_log))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"\n{name} - Training Set:")
    print(f"R²: {r2_train:.3f} | RMSE: {rmse_train:.3f} | MAPE: {mape_train:.3f}%")
    print(f"{name} - Test Set:")
    print(f"R²: {r2:.3f} | RMSE: {rmse:.3f} | MAPE: {mape:.3f}%")

    results.append({"Model": name, "R2_train": r2_train, "RMSE_train": rmse_train, "MAPE_train": mape_train,
                    "R2_test": r2, "RMSE_test": rmse, "MAPE_test": mape})

# Show summary
df_ols_results = pd.DataFrame(results)
print("\nSummary of OLS Variants:")
print(df_ols_results.round(3))