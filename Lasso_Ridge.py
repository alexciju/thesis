from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso, LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
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

def mape_price(y_true_log, y_pred_log):
    """
    Compute MAPE on original price by exponentiating log-price inputs.
    """
    y_true = np.exp(y_true_log)
    y_pred = np.exp(y_pred_log)
    return np.mean(np.abs((y_true - y_pred) / y_true))

def filter_valid_test(df_test, df_train, cat_cols):
    valid_rows = pd.Series(True, index=df_test.index)
    for col in cat_cols:
        valid_values = df_train[col].unique()
        valid_rows &= df_test[col].isin(valid_values)
    return df_test[valid_rows].copy()

# ------------- Make scorers  --------------
mape_scorer = make_scorer(mape_price, greater_is_better=False)
mdape_scorer = make_scorer(mdape, greater_is_better=False)
rmspe_scorer = make_scorer(rmspe, greater_is_better=False)
scoring_metrics = {'neg_root_mean_squared_error':'neg_root_mean_squared_error',
                   'r2':'r2',
                   'mape': mape_scorer,
                   'mdape': mdape_scorer,
                   'rmspe': rmspe_scorer}

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

# ----------------- Correlation Table for Regressors -----------------
# Compute correlation matrix for numeric regressors
corr_matrix = X[num_cols].corr()
print("\nCorrelation matrix for numeric regressors:")
print(corr_matrix.round(3))
# Export the correlation matrix to CSV
corr_matrix.round(3).to_csv('data/correlation_matrix.csv', index=True)

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
                       ('model', RidgeCV(alphas=np.logspace(-3,3,50), cv=5,
                                         ))])

pipe_lasso = Pipeline([('prep', pre),
                       ('model', LassoCV(cv=5, alphas=np.logspace(-3,3,50), max_iter=10000,
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
        f"RMSPE: {rmspe(y_true_price, y_pred_price):.3%} "
        f"Num_Coef: {len(pipe.named_steps['model'].coef_)}"
    )



# ------------------  cross-validation -------------------
"""
This section performs cross-validation on the three regression models
(OLS, Ridge, and Lasso) to evaluate their performance across multiple folds.
It calculates the R² and RMSE scores for each fold and stores the results in a dictionary.
For Lasso, also store the number of nonzero coefficients in each fold.
"""

results = {}
for name, pipe in [('OLS',pipe_ols), ('Ridge',pipe_ridge)]:
    results[name] = {}
    scores = cross_validate(pipe, X, y, cv=5, scoring=scoring_metrics)
    for score_name in scores:
        if score_name.startswith('test_'):
            results[name][score_name] = scores[score_name]

# For Lasso, also track number of nonzero coefficients per fold
results['Lasso'] = {}
lasso_cv = cross_validate(
    pipe_lasso, X, y, cv=5, scoring=scoring_metrics, return_estimator=True
)
for score_name in lasso_cv:
    if score_name.startswith('test_'):
        results['Lasso'][score_name] = lasso_cv[score_name]

# Count nonzero coefficients per fold
lasso_nonzero_counts = []
for estimator in lasso_cv['estimator']:
    
    # estimator is a pipeline; get last step's coef_
    lasso_model = estimator.named_steps['model']
    lasso_nonzero_counts.append(np.sum(lasso_model.coef_ != 0))
results['Lasso']['nonzero_coef_count'] = np.array(lasso_nonzero_counts)


# ------------------  Prepare DataFrames for each model -------------------
"""
This section prepares DataFrames for each model's cross-validation results,
including R², RMSE, MAPE, MdAPE, and RMSPE scores.
For Lasso, also include the number of nonzero coefficients per fold.
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
    if name == "Lasso" and "nonzero_coef_count" in score_dict:
        data['NonZero_Coef'] = score_dict["nonzero_coef_count"]
    model_dfs[name] = pd.DataFrame(data).set_index('Fold')
    model_dfs[name].index.name = f"{name} Fold"

# Print each model's DataFrame
for model, df_model in model_dfs.items():
    print(f"\n=== {model} ===")
    print(df_model.round(3))

# ----------------- Export coefficients table -----------------
# Use already-fitted pipelines to extract coefficients
feature_names_num = num_cols
feature_names_cat = pipe_ols.named_steps['prep'].named_transformers_['cat'].named_steps['ohe'].get_feature_names_out(cat_cols)


coefs_ols    = pipe_ols.named_steps['model'].coef_
coefs_ridge  = pipe_ridge.named_steps['model'].coef_
coefs_lasso  = pipe_lasso.named_steps['model'].coef_

# ---------- Average absolute coefficient magnitude ----------
avg_coef_ols    = np.mean(np.abs(coefs_ols))
avg_coef_ridge  = np.mean(np.abs(coefs_ridge))
avg_coef_lasso  = np.mean(np.abs(coefs_lasso))

print("\nAverage |β| across all coefficients")
print(f"  OLS   : {avg_coef_ols:.6f}")
print(f"  Ridge : {avg_coef_ridge:.6f}")
print(f"  Lasso : {avg_coef_lasso:.6f}")

# Number of numeric features
n_num = len(feature_names_num)

# Export feature coefficients
df_feature_coefs = pd.DataFrame({
    'Feature': feature_names_num,
    'OLS': coefs_ols[:n_num],
    'Ridge': coefs_ridge[:n_num],
    'Lasso': coefs_lasso[:n_num]
}).round(4)
df_feature_coefs.to_csv('data/feature_coefficients.csv', index=False)
print("\nFeature coefficients saved to 'data/feature_coefficients.csv'")

# Export dummy (one-hot) coefficients
df_dummy_coefs = pd.DataFrame({
    'Dummy': feature_names_cat,
    'OLS': coefs_ols[n_num:],
    'Ridge': coefs_ridge[n_num:],
    'Lasso': coefs_lasso[n_num:]
}).round(4)
df_dummy_coefs.to_csv('data/dummy_coefficients.csv', index=False)
print("\nDummy coefficients saved to 'data/dummy_coefficients.csv'")


# ---------------- Export results to LateX ----------------
metrics = ["R²", "RMSE", "MAPE"]    

def mean_metric(model_name, metric):
    return model_dfs[model_name][metric].mean()

# ---------------------------------------------------------------
# 2.  Build the summary table  (OLS = Base)
# ---------------------------------------------------------------
summary = pd.DataFrame({
    "Metric" : metrics,
    "Base Fixed Effects" : [mean_metric("OLS", m)   for m in metrics],
    "Ridge"              : [mean_metric("Ridge", m) for m in metrics],
    "Lasso"              : [mean_metric("Lasso", m) for m in metrics],
})

# Add Δ-columns (model – base)
summary["Δ (Ridge – Base)"] = summary["Ridge"] - summary["Base Fixed Effects"]
summary["Δ (Lasso – Base)"] = summary["Lasso"] - summary["Base Fixed Effects"]

# Optional rounding
summary = summary.round(3)

print("\n=== Summary table ===")
print(summary)

# ---------- Append non‑zero coefficients row ----------
nonzero_row = pd.Series({
    "Metric": "Non‑zero coefficients",
    "Base Fixed Effects": len(coefs_ols),                  # OLS keeps all slopes
    "Ridge": len(coefs_ridge),                             # Ridge keeps all slopes
    "Lasso": results["Lasso"]["nonzero_coef_count"].mean(),# avg per fold
    "Δ (Ridge – Base)": len(coefs_ridge) - len(coefs_ols),
    "Δ (Lasso – Base)": results["Lasso"]["nonzero_coef_count"].mean() - len(coefs_ols)
})
summary = pd.concat([summary, pd.DataFrame([nonzero_row])], ignore_index=True)

# Optional: round Lasso average to one decimal
summary.loc[summary['Metric'] == 'Non‑zero coefficients',
            ["Base Fixed Effects","Ridge","Lasso",
             "Δ (Ridge – Base)","Δ (Lasso – Base)"]] = \
    summary.loc[summary['Metric'] == 'Non‑zero coefficients',
                ["Base Fixed Effects","Ridge","Lasso",
                 "Δ (Ridge – Base)","Δ (Lasso – Base)"]].round(1)

# ---------- Append average |β| row ----------
avg_row = pd.Series({
    "Metric": "Avg |β|",
    "Base Fixed Effects": avg_coef_ols,
    "Ridge": avg_coef_ridge,
    "Lasso": avg_coef_lasso,
    "Δ (Ridge – Base)": avg_coef_ridge - avg_coef_ols,
    "Δ (Lasso – Base)": avg_coef_lasso - avg_coef_ols
})
summary = pd.concat([summary, pd.DataFrame([avg_row])], ignore_index=True)

# Optional rounding for the new row
summary.loc[summary['Metric'] == 'Avg |β|', ["Base Fixed Effects","Ridge","Lasso",
                                             "Δ (Ridge – Base)","Δ (Lasso – Base)"]] = \
    summary.loc[summary['Metric'] == 'Avg |β|', ["Base Fixed Effects","Ridge","Lasso",
                                                 "Δ (Ridge – Base)","Δ (Lasso – Base)"]].round(4)

# --------------- Transpose for display ----------------
summary_T = summary.set_index("Metric").T
print("\n=== Summary table (transposed) ===")
print(summary_T)


summary_T.to_clipboard(index=True, excel=True)
print("\n[Table copied to clipboard - paste into Word (Keep Source Formatting)]")

# ------------------  λ‑path plots for Ridge and Lasso -------------------
"""
Plot cross‑validated R² as a function of the regularisation strength λ
to visualise how Ridge flattens while Lasso peaks and declines.
"""
alphas = np.logspace(-3, 3, 30)
ridge_r2 = []
lasso_r2 = []

for a in alphas:
    pipe_ridge_curve = Pipeline([('prep', pre),
                                 ('model', Ridge(alpha=a))])
    pipe_lasso_curve = Pipeline([('prep', pre),
                                 ('model', Lasso(alpha=a, max_iter=10000))])

    ridge_r2.append(cross_val_score(pipe_ridge_curve, X, y,
                                    cv=5, scoring='r2').mean())
    lasso_r2.append(cross_val_score(pipe_lasso_curve, X, y,
                                    cv=5, scoring='r2').mean())

# Baseline OLS R² (λ = 0)
ols_r2 = cross_val_score(pipe_ols, X, y, cv=5, scoring='r2').mean()

plt.figure(figsize=(7, 5))
plt.plot(alphas, ridge_r2, label='Ridge')
plt.plot(alphas, lasso_r2, label='Lasso')
plt.axhline(ols_r2, linestyle='--', label='OLS (λ=0)')

plt.xscale('log')
plt.xlabel('Regularisation strength λ (log scale)')
plt.ylabel('Mean 5‑fold CV $R^{2}$')
plt.title('$R^{2}$ vs. λ for Ridge and Lasso')
plt.legend()
plt.tight_layout()
plt.show()
