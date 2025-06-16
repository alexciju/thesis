import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from scikeras.wrappers import KerasRegressor
from keras import Input, Model
from keras.layers import Dense
from keras.callbacks import EarlyStopping

# TODO: Import function from Max 
def filter_valid_test(df_test, df_train, cat_cols):
    valid_mask = pd.Series(True, index=df_test.index)
    for col in cat_cols:
        valid_values = df_train[col].unique()
        valid_mask &= df_test[col].isin(valid_values)
    return df_test[valid_mask].copy()

# Load and clean data
df = pd.read_csv('data/Melbourne_housing_FULL.csv')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Month'] = df['Date'].dt.to_period('M').astype(str)
df['Log_Price'] = np.log(df['Price'])

df.drop(columns=['Address', 'Postcode', 'Regionname', 'CouncilArea', 'Longtitude', 'Lattitude'], inplace=True)
df.dropna(subset=['Log_Price', 'YearBuilt', 'BuildingArea', 'Landsize', 'Car'], inplace=True)

X = df.drop(columns=['Price', 'Log_Price', 'Date'])
y = df['Log_Price']

num_cols = X.select_dtypes('number').columns
cat_cols = X.select_dtypes('object').columns

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test = filter_valid_test(X_test, X_train, cat_cols)
y_test = y.loc[X_test.index]

# Preprocessing pipeline
cat_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='constant', fill_value='Missing', add_indicator=True)),
    ('encode', OneHotEncoder(drop='first', handle_unknown='ignore', min_frequency=30))
])
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', cat_pipeline, cat_cols)
])
preprocessor.fit(X_train)
input_shape = (preprocessor.transform(X_train[:1]).shape[1],)

# Define Keras model
def build_keras_model(input_shape, **kwargs):
    inputs = Input(shape=input_shape)
    x = Dense(32, activation='relu')(inputs)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['RootMeanSquaredError'])
    return model

early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

# Define model pipelines
pipe_ols = Pipeline([
    ('prep', preprocessor),
    ('model', LinearRegression())
])
pipe_xgb = Pipeline([
    ('prep', preprocessor),
    ('model', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=42))
])
pipe_nn = Pipeline([
    ('prep', preprocessor),
    ('model', KerasRegressor(
        model=build_keras_model,
        model__input_shape=input_shape,
        epochs=150,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0,
        random_state=42
    ))
])

# Final test evaluation
print("\n=== Test Set Evaluation ===")
for name, model in [('OLS', pipe_ols), ('XGBoost', pipe_xgb), ('KerasNN', pipe_nn)]:
    model.fit(X_train, y_train)
    y_pred_log = model.predict(X_test)
    y_pred = np.exp(y_pred_log)
    y_true = np.exp(y_test)

    r2 = r2_score(y_test, y_pred_log)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_log))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 

    print(f"{name} | R²: {r2:.3f} | RMSE: {rmse:.3f} | MAPE: {mape:.3f}%")

# Cross-validation (only for OLS and XGBoost)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model_names, r2_list, rmse_list, mape_list = [], [], [], []

print("\n=== Cross-Validation (5-Fold) ===")
for name, pipe in [('OLS', pipe_ols), ('XGBoost', pipe_xgb)]:
    print(f"\n{name} Cross-Validation Results:")
    fold = 1
    for train_index, test_index in kf.split(X):
        X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
        y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]

        X_test_cv = filter_valid_test(X_test_cv, X_train_cv, cat_cols)
        y_test_cv = y.loc[X_test_cv.index]

        pipe.fit(X_train_cv, y_train_cv)
        y_pred_log = pipe.predict(X_test_cv)
        y_pred = np.exp(y_pred_log)
        y_true = np.exp(y_test_cv)

        r2 = r2_score(y_test_cv, y_pred_log)
        rmse = np.sqrt(mean_squared_error(y_test_cv, y_pred_log))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 

        print(f"Fold {fold} | R²: {r2:.3f} | RMSE: {rmse:.3f} | MAPE: {mape:.3f}%")

        model_names.append(name)
        r2_list.append(r2)
        rmse_list.append(rmse)
        mape_list.append(mape)
        fold += 1

# Combine into DataFrame
df_cv_results = pd.DataFrame({
    "Model": model_names,
    "R²": r2_list,
    "RMSE": rmse_list,
    "MAPE": mape_list
})

print("\nSummary of Cross-Validation:")
print(df_cv_results.groupby("Model").mean().round(3))
