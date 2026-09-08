"""
House Price Prediction using Linear Regression
Dataset: Kaggle "House Prices - Advanced Regression Techniques"
(https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

Usage:
    1. Download train.csv from Kaggle and place it in the same folder as this script.
    2. Run: python house_price_prediction_linear.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
DATA_PATH = "train.csv"   # change path if needed
TARGET = "SalePrice"

df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print(df.head())

# Drop ID column if present
if "Id" in df.columns:
    df = df.drop(columns=["Id"])

# ---------------------------------------------------------
# 2. Split features / target
# ---------------------------------------------------------
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Optional: log-transform the target since house prices are usually
# right-skewed. This tends to help linear models a lot.
y_log = np.log1p(y)

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print(f"\nNumeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")

# ---------------------------------------------------------
# 3. Preprocessing pipelines
# ---------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ---------------------------------------------------------
# 4. Build full pipeline (preprocessing + Linear Regression)
# ---------------------------------------------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# ---------------------------------------------------------
# 5. Train / test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 6. Cross-validation (sanity check before final fit)
# ---------------------------------------------------------
cv_scores = cross_val_score(
    model, X_train, y_train, cv=5,
    scoring="neg_root_mean_squared_error"
)
print(f"\nCross-validated RMSE (log scale): {-cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ---------------------------------------------------------
# 7. Fit on full training set
# ---------------------------------------------------------
model.fit(X_train, y_train)

# ---------------------------------------------------------
# 8. Evaluate on test set
# ---------------------------------------------------------
y_pred_log = model.predict(X_test)

# Convert back from log scale to actual dollar values
y_pred = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_test)

rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
mae = mean_absolute_error(y_test_actual, y_pred)
r2 = r2_score(y_test_actual, y_pred)

print("\n--- Linear Regression Performance on Test Set ---")
print(f"RMSE: {rmse:,.2f}")
print(f"MAE:  {mae:,.2f}")
print(f"R^2:  {r2:.4f}")

# ---------------------------------------------------------
# 9. Inspect coefficients (top 15 by absolute value)
# ---------------------------------------------------------
try:
    ohe = model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    cat_names = ohe.get_feature_names_out(categorical_features)
    all_feature_names = np.concatenate([numeric_features, cat_names])

    coefs = model.named_steps["regressor"].coef_
    coef_series = pd.Series(coefs, index=all_feature_names)
    top_coefs = coef_series.reindex(coef_series.abs().sort_values(ascending=False).index)

    print("\nTop 15 features by coefficient magnitude:")
    print(top_coefs.head(15))
except Exception as e:
    print("Could not extract coefficients:", e)

# ---------------------------------------------------------
# 10. Save the trained model
# ---------------------------------------------------------
joblib.dump(model, "house_price_model_linear.pkl")
print("\nModel saved as house_price_model_linear.pkl")

# ---------------------------------------------------------
# 11. Example: predict on new data
# ---------------------------------------------------------
# new_house = X_test.iloc[[0]]
# predicted_price_log = model.predict(new_house)
# predicted_price = np.expm1(predicted_price_log)[0]
# print("Predicted price:", predicted_price)
