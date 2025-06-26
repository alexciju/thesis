import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


 # Load the cleaned dataset
df = pd.read_csv('my_cleaned_data.csv')
df = df.dropna(subset=['Price'])
df['log_price'] = np.log(df['Price'])

# 3) Convert Date to datetime, extract year and month, compute building age
# Convert Date to datetime, infer the format, coerce bad parses to NaT
df['Date'] = pd.to_datetime(
    df['Date'],
    dayfirst=True,
    infer_datetime_format=True,
    errors='coerce'
)

# # Drop any rows where Date couldn’t be parsed
df = df.dropna(subset=['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
# df['Age'] = df['Year'] - df['YearBuilt']
# Combine Year and Month into a single period for fixed effects
df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)

# # 2) Define your features and target
# #    — numeric columns:
# numeric_feats = ['BuildingArea', 'Bedroom2', 'Car', 'YearBuilt',
#                  'Rooms', 'Bathroom', 'Landsize']
# #    — categorical columns:
# categorical_feats = ['Suburb', 'Month']
# # 4) Drop rows with any missing values in the selected features

# Old numeric features (no longer in this dataset):
# numeric_feats = ['BuildingArea', 'Bedroom2', 'Car', 'YearBuilt',
#                  'Rooms', 'Bathroom', 'Landsize']
# New numeric features from MELBOURNE_HOUSE_PRICES_LESS.csv:
numeric_feats = ['Rooms', 'Distance', 'Propertycount', 'Postcode']
# Old categorical features (month-of-year):
# categorical_feats = ['Suburb', 'Type', 'Method', 'SellerG', 'Regionname', 'CouncilArea', 'Month']
# Use YearMonth period dummies instead of plain Month
categorical_feats = ['Suburb', 'Type', 'Method', 'SellerG', 'Regionname', 'CouncilArea', 'YearMonth']

df = df.dropna(subset=numeric_feats + categorical_feats + ['log_price'])

X = df[numeric_feats + categorical_feats]
y = df['log_price']

 # 3) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4) Build a preprocessing + Lasso pipeline
preprocessor = ColumnTransformer([
    # scale numeric features to mean 0 / sd 1
    ('num', StandardScaler(), numeric_feats),
    # one‐hot encode the categorical ones
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_feats)
])

pipe = Pipeline([
    ('prep', preprocessor),
    # LassoCV will automatically select alpha via 5‐fold CV by default
    ('lasso', LassoCV(cv=5, n_alphas=50, max_iter=5000, random_state=42))
])

 # 5) Fit it
pipe.fit(X_train, y_train)

# 6) Inspect the chosen penalty and the nonzero coefficients
lasso: LassoCV = pipe.named_steps['lasso']
print("Optimal alpha:", lasso.alpha_)

# Retrieve transformed feature names and display nonzero coefficients
feature_names = pipe.named_steps['prep'].get_feature_names_out()
coefs = pd.Series(lasso.coef_, index=feature_names)
print("Nonzero coefficients:\n", coefs[coefs.abs()>1e-6])

# 7) Evaluate out-of‐sample
from sklearn.metrics import mean_squared_error, r2_score

y_pred = pipe.predict(X_test)
print("Lasso Test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("Lasso Test R²:", r2_score(y_test, y_pred))

# Plot true vs predicted for Lasso
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('True log_price')
plt.ylabel('Predicted log_price')
plt.title('Lasso: True vs. Predicted')
plt.show()

# 8) Fixed‑effects regression with Suburb and Month fixed effects
#    Use pandas get_dummies so we can safely predict on test data that may
#    contain suburbs or months not present in the training set.
# Reconstruct training DataFrame for fixed-effects from X_train and y_train
# train_fe = X_train.copy()
# train_fe['log_price'] = y_train
# Dummy‐encode all categorical features for FE model
# train_fe = pd.get_dummies(train_fe, columns=categorical_feats + ['Month'], drop_first=True)

# X_fe_train = train_fe.drop(columns=['log_price'])
# y_fe_train = train_fe['log_price']
# Ensure endog is numeric
# y_fe_train = y_fe_train.astype(float)

# Add a constant term for the intercept
# X_fe_train_const = sm.add_constant(X_fe_train, has_constant='add')
# Ensure exog is numeric
# X_fe_train_const = X_fe_train_const.astype(float)
# fe_model = sm.OLS(y_fe_train, X_fe_train_const).fit()

# print("\nFixed‑Effects Regression Results (suburb & month dummies shown):")
# print(fe_model.summary().tables[1])

# Prepare the test design matrix with the same dummy columns
# test_fe = X_test.copy()
# Dummy‐encode all categorical features for FE test data
# test_fe = pd.get_dummies(test_fe, columns=categorical_feats + ['Month'], drop_first=True)
# Align columns to match the training design; fill any missing cols with 0
# test_fe = test_fe.reindex(columns=X_fe_train.columns, fill_value=0)
# X_fe_test_const = sm.add_constant(test_fe, has_constant='add')

# fe_pred = fe_model.predict(X_fe_test_const)
# print("FE Test RMSE:", np.sqrt(mean_squared_error(y_test, fe_pred)))
# print("FE Test R²:", r2_score(y_test, fe_pred))

# 9) Random‑Forest regressor (with the same preprocessing pipeline)
rf_pipe = Pipeline([
    ('prep', preprocessor),
    ('rf', RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
        max_features= 0.5,
        max_depth= 50
    ))
])
rf_pipe.fit(X_train, y_train)
rf_pred = rf_pipe.predict(X_test)
print("\nRandom Forest Test RMSE:", np.sqrt(mean_squared_error(y_test, rf_pred)))
print("Random Forest Test R²:", r2_score(y_test, rf_pred))

# Plot true vs predicted for Random Forest
plt.figure()
plt.scatter(y_test, rf_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('True log_price')
plt.ylabel('Predicted log_price')
plt.title('Random Forest: True vs. Predicted')
plt.show()

# 11) Ridge regression with cross-validated alpha
ridge_pipe = Pipeline([
    ('prep', preprocessor),
    ('ridge', RidgeCV(alphas=np.logspace(-3, 3, 13), cv=5))
])
ridge_pipe.fit(X_train, y_train)
ridge_pred = ridge_pipe.predict(X_test)
print("\nRidge Regression Test RMSE:", np.sqrt(mean_squared_error(y_test, ridge_pred)))
print("Ridge Regression Test R²:", r2_score(y_test, ridge_pred))

# Plot true vs predicted for Ridge
plt.figure()
plt.scatter(y_test, ridge_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('True log_price')
plt.ylabel('Predicted log_price')
plt.title('Ridge: True vs. Predicted')
plt.show()

# 10) Gradient‑Boosting regressor (scikit‑learn)
gb_pipe = Pipeline([
    ('prep', preprocessor),
    ('gb', GradientBoostingRegressor(random_state=42))
])
gb_pipe.fit(X_train, y_train)
gb_pred = gb_pipe.predict(X_test)
print("\nGradient Boosting Test RMSE:", np.sqrt(mean_squared_error(y_test, gb_pred)))
print("Gradient Boosting Test R²:", r2_score(y_test, gb_pred))

# Plot true vs predicted for Gradient Boosting
plt.figure()
plt.scatter(y_test, gb_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('True log_price')
plt.ylabel('Predicted log_price')
plt.title('Gradient Boosting: True vs. Predicted')
plt.show()

# 13) Prepare data for neural network
# Transform training and test sets with the same preprocessor
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

# Define a simple feed-forward neural network
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_proc.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])
nn_model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mean_absolute_error']
)

# Early stopping to prevent overfitting
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
nn_history = nn_model.fit(
    X_train_proc, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[es],
    verbose=1
)

# Evaluate on test set
nn_pred = nn_model.predict(X_test_proc).flatten()
from sklearn.metrics import mean_squared_error, r2_score
print("\nNeural Network Test RMSE:", np.sqrt(mean_squared_error(y_test, nn_pred)))
print("Neural Network Test R²:", r2_score(y_test, nn_pred))

# Plot true vs predicted for Neural Network
plt.figure()
plt.scatter(y_test, nn_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('True log_price')
plt.ylabel('Predicted log_price')
plt.title('Neural Network: True vs. Predicted')
plt.show()

# 12) 5-Fold Cross-Validation Summary for each model
scoring = {
    'rmse': 'neg_root_mean_squared_error',
    'mae': 'neg_mean_absolute_error',
    'r2': 'r2'
}
models = {
    'Lasso': pipe,
    'Random Forest': rf_pipe,
    'Ridge': ridge_pipe,
    'Gradient Boosting': gb_pipe
}

print("\n5-Fold Cross-Validation Results (means ± std):")
for name, model in models.items():
    cv_results = cross_validate(
        model, X, y,
        cv=5,
        scoring=scoring,
        return_train_score=False
    )
    rmse_scores = -cv_results['test_rmse']
    mae_scores = -cv_results['test_mae']
    r2_scores = cv_results['test_r2']

    print(f"\n{name}:")
    print(f"  RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")
    print(f"   MAE: {mae_scores.mean():.4f} ± {mae_scores.std():.4f}")
    print(f"    R2: {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")