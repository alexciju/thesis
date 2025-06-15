from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import RidgeCV, LassoCV, LinearRegression
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import make_scorer

# ----------------- Custom metrics -----------------
"""
This section defines custom metrics for evaluating regression models.
These metrics include Absolute Percentage Error (APE), Mean Absolute Percentage Error (MAPE),
Median Absolute Percentage Error (MdAPE), and Root Mean Squared Percentage Error (RMSPE).
These metrics are used to assess the accuracy of predictions made by the models.
"""
def ape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.abs((y_true - y_pred) / y_true)

def mape(y_true, y_pred):
    return ape(y_true, y_pred).mean()

def mdape(y_true, y_pred):
    return np.median(ape(y_true, y_pred))

def rmspe(y_true, y_pred):
    return np.sqrt((ape(y_true, y_pred) ** 2).mean())

def filter_valid_test(df_test, df_train, cat_cols):
    valid_rows = pd.Series(True, index=df_test.index)
    for col in cat_cols:
        valid_values = df_train[col].unique()
        valid_rows &= df_test[col].isin(valid_values)
    return df_test[valid_rows].copy()

# ---------------- Load & clean data ----------------
"""
This section loads the Melbourne housing dataset, processes the date column,
and creates new features such as 'Month' and 'Log_Price'.
It also drops unnecessary columns and handles missing values in key features.
"""
df = pd.read_csv('data/Melbourne_housing_FULL.csv')

df['Date']  = pd.to_datetime(df['Date'], dayfirst=True)
df['Month'] = df['Date'].dt.to_period('M').astype(str)
df['Log_Price'] = np.log(df['Price'])

df = df.drop(columns=['Address', 'Postcode', 'Regionname', 'CouncilArea', 'Longtitude', 'Lattitude'])
df = df.dropna(subset=['Log_Price', 'YearBuilt', 'BuildingArea', 'Landsize', 'Car'])

X = df.drop(columns=['Price','Log_Price','Date'])
y = df['Log_Price']

num_cols = X.select_dtypes('number').columns
cat_cols = X.select_dtypes('object').columns

print(df.info)

print(cat_cols)

# ----------------- Preprocessing and modeling -----------------
"""
This code sets up a machine learning pipeline to predict house 
prices in Melbourne using various regression techniques. It 
includes data preprocessing steps such as handling missing values, 
encoding categorical variables, and scaling numerical features. 
The code also performs model training and evaluation using Ridge, 
Lasso, and OLS regression models, with cross-validation to assess 
performance.
"""
cat_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='constant', fill_value='Missing', add_indicator=True)),
    ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', min_frequency=30))])

pre = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', cat_pipe, cat_cols)
])

pipe_ridge = Pipeline([('prep', pre),
                       ('model', RidgeCV(alphas=np.logspace(-3,3,50), cv=5))])

pipe_lasso = Pipeline([('prep', pre),
                       ('model', LassoCV(cv=5, n_alphas=50, max_iter=10000,
                                         random_state=42))])

pipe_ols   = Pipeline([('prep', pre),
                       ('model', LinearRegression())])

# ------------ First train/test split and evaluation -------------
"""
This section splits the dataset into training and testing sets,
filters the test set to ensure it only contains valid categories
from the training set, and evaluates the performance of three
different regression models (OLS, Ridge, and Lasso) using R² and RMSE metrics.
"""
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   test_size=0.2, random_state=42)

X_test = filter_valid_test(X_test, X_train, cat_cols)
y_test = y.loc[X_test.index]

for name, pipe in [('OLS',pipe_ols), ('Ridge',pipe_ridge), ('Lasso',pipe_lasso)]:
    pipe.fit(X_train, y_train)
    y_pred_log = pipe.predict(X_test)
    y_pred_price = np.exp(y_pred_log)
    y_true_price = np.exp(y_test)
    ape_vals = ape(y_true_price, y_pred_price)
    print(
        f"{name} R2: {r2_score(y_test, y_pred_log):.3f} "
        f"RMSE_log: {np.sqrt(mean_squared_error(y_test, y_pred_log)):.3f} "
        f"MAPE: {mape(y_true_price, y_pred_price):.3%} "
        f"MdAPE: {mdape(y_true_price, y_pred_price):.3%} "
        f"RMSPE: {rmspe(y_true_price, y_pred_price):.3%}"
    )

# ------------- Make scorers  --------------
mape_scorer = make_scorer(mape, greater_is_better=False)
mdape_scorer = make_scorer(mdape, greater_is_better=False)
rmspe_scorer = make_scorer(rmspe, greater_is_better=False)
scoring_metrics = {'neg_root_mean_squared_error':'neg_root_mean_squared_error',
                   'r2':'r2',
                   'mape': mape_scorer,
                   'mdape': mdape_scorer,
                   'rmspe': rmspe_scorer}

# ------------------  cross-validation -------------------
"""
This section performs cross-validation on the three regression models
(OLS, Ridge, and Lasso) to evaluate their performance across multiple folds.
It calculates the R² and RMSE scores for each fold and stores the results in a dictionary.
"""

results = {}
for name, pipe in [('OLS',pipe_ols), ('Ridge',pipe_ridge), ('Lasso',pipe_lasso)]:
    results[name] = {}
    scores = cross_validate(pipe, X, y, cv=5, scoring=scoring_metrics)
    for score_name in scores:
        if score_name.startswith('test_'):
            results[name][score_name] = scores[score_name]


# ------------------  Prepare DataFrames for each model -------------------
"""
This section prepares DataFrames for each model's cross-validation results,
including R², RMSE, MAPE, MdAPE, and RMSPE scores.
"""
model_dfs = {}
for name, score_dict in results.items():
    data = {
        'Fold': np.arange(1, len(score_dict["test_r2"]) + 1),
        'R²': score_dict["test_r2"],
        'RMSE': [-x for x in score_dict["test_neg_root_mean_squared_error"]]
    }
    if "test_mape" in score_dict:
        data['MAPE'] = [-x for x in score_dict["test_mape"]]
    if "test_mdape" in score_dict:
        data['MdAPE'] = [-x for x in score_dict["test_mdape"]]
    if "test_rmspe" in score_dict:
        data['RMSPE'] = [-x for x in score_dict["test_rmspe"]]
    model_dfs[name] = pd.DataFrame(data).set_index('Fold')
    model_dfs[name].index.name = f"{name} Fold"

# Print each model's DataFrame
for model, df_model in model_dfs.items():
    print(f"\n=== {model} ===")
    print(df_model.round(3))