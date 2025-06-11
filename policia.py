from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import RidgeCV, LassoCV, LinearRegression
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

def filter_valid_test(df_test, df_train, cat_cols):
    valid_rows = pd.Series(True, index=df_test.index)
    for col in cat_cols:
        valid_values = df_train[col].unique()
        valid_rows &= df_test[col].isin(valid_values)
    return df_test[valid_rows].copy()

# ---------------- Load & basic cleaning ----------------
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

# ---------------- Feature engineering ----------------
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

# ------------ Train/test split for final evaluation -------------
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   test_size=0.2, random_state=42)

X_test = filter_valid_test(X_test, X_train, cat_cols)
y_test = y.loc[X_test.index]

for name, pipe in [('OLS',pipe_ols), ('Ridge',pipe_ridge), ('Lasso',pipe_lasso)]:
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(
        f"{name} R2: {r2_score(y_test, y_pred)}"
        f"{name} RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}"
          )

# ------------- Honest cross-validation (optional) --------------
results = {}
for name, pipe in [('OLS',pipe_ols), ('Ridge',pipe_ridge), ('Lasso',pipe_lasso)]:
    results[name] = {}
    scores = cross_validate(pipe, X, y, cv=5, scoring=('neg_root_mean_squared_error', 'r2'))
    for score_name in scores:
        if 'test_' in score_name:
            results[name][score_name] = scores[score_name]

# Create individual DataFrames for each model
model_dfs = {}
for name, score_dict in results.items():
    model_dfs[name] = pd.DataFrame({
        'Fold': np.arange(1, len(score_dict["test_r2"]) + 1),
        'R²': score_dict["test_r2"],
        'RMSE': [-x for x in score_dict["test_neg_root_mean_squared_error"]]
    }).set_index('Fold')
    model_dfs[name].index.name = f"{name} Fold"

# Print each model's DataFrame
for model, df_model in model_dfs.items():
    print(f"\n=== {model} ===")
    print(df_model.round(3))