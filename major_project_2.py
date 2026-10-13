# -*- coding: utf-8 -*-
"""Major_Project_2.ipynb

Original file is located at
    https://colab.research.google.com/drive/1PCqq3kkbN9NnB9UuD_-F9bxE1DR7zqgX
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Loading the files
TRAIN_FILE = "Cap_Training_Data_2025.csv"
TEST_FILE = "Cap_Test_Data_2025.csv"

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

print("Training data:", train.shape)
print("Test data    :", test.shape)
train.head()

train.info()

train.describe()

train.drop(['ID'],axis=1,inplace=True) # Dropping the useless column

train.isnull().sum() # Null values present

train.duplicated().sum()

print(train.columns.tolist())
 #Distance has space at the end

# Removing the spaces
train.columns = train.columns.str.strip()
test.columns = test.columns.str.strip()
print(train.columns.tolist())

# Filling the missing numeric values with median
missing_numeric = ['seat_count','door_count','engine_power','manufacture_year','Distance']

for miss in missing_numeric:
  train[miss] = train[miss].fillna(train[miss].median())

print(train['body_type'].value_counts())

train['body_type'] = train['body_type'].fillna('Unknown')  #Replacing the missing values with unknown because we cant replace them with mode as there are only 9 vans
train['Location'] = train['Location'].fillna(train['Location'].mode()[0])

train.isnull().sum() # No null values present

train_clean = train[train["Price"] >= 10_000] # A car worth less than 10,000 rupess is almost certainly a typo

#Outliers: The numbers which are too low or too high of the range using IQR

outlier_columns = ["Distance", "Age of car", "engine_displacement", "engine_power"]

for col in outlier_columns:
      plt.figure(figsize=(8,4))
      sns.boxplot(x = train_clean[col])
      plt.title(f"Box plot of {col}")
      plt.show()

for col in ["Distance", "Age of car"]:
    Q1 = train_clean[col].quantile(0.25)
    Q3 = train_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    before = len(train_clean)
    train_clean = train_clean[(train_clean[col] >= lower) & (train_clean[col] <= upper)]
    print(f"Outliers Removed from {col}")

print("Rows remaining:", len(train_clean))

# Checking after the IQR
for col in ["Distance", "Age of car"]:
      plt.figure(figsize=(8,4))
      sns.boxplot(x = train_clean[col])
      plt.title(f"Box plot of {col}")
      plt.show()

train_clean["Maker_model"] = train_clean["Maker"] + "_" + train_clean["model"]

NUMERIC_FEATURES = ["Distance", "Age of car", "engine_displacement", "engine_power",
                    "Vroom Audit Rating", "door_count", "seat_count"]
CATEGORICAL_FEATURES = ["Maker_model", "Location", "Owner Type", "transmission", "fuel_type", "body_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

print("Maker_model" in train_clean.columns)

X = train_clean[FEATURES]
y = np.log1p(train_clean["Price"])

print(X.shape)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Relaxed clipping: use actual min/max so luxury car specs are NOT capped
# (Previously 1st/99th percentile was cutting off high-end engine_power etc.)
clip_low  = X_train[NUMERIC_FEATURES].quantile(0.0)   # actual min
clip_high = X_train[NUMERIC_FEATURES].quantile(1.0)   # actual max

def clip_numeric(df):
    df = df.copy()
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].clip(clip_low, clip_high, axis=1)
    return df

X_train_c = clip_numeric(X_train)
X_val_c   = clip_numeric(X_val)

pd.DataFrame({"low": clip_low, "high": clip_high})

from sklearn.preprocessing import OneHotEncoder

#One hot encoding
enc = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)

cat_train = enc.fit_transform(X_train_c[CATEGORICAL_FEATURES])
cat_val   = enc.transform(X_val_c[CATEGORICAL_FEATURES])

X_train_enc = np.hstack([X_train_c[NUMERIC_FEATURES].values, cat_train]) # joins both arrays side by side column by column
X_val_enc   = np.hstack([X_val_c[NUMERIC_FEATURES].values,   cat_val])

feature_names = NUMERIC_FEATURES + list(enc.get_feature_names_out())
print("Encoded train shape:", X_train_enc.shape)

# Evaluation Function
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate(name, model, X_v, y_v):
    predicted = np.expm1(model.predict(X_v))
    actual = np.expm1(y_v)
    result = {
        "Model": name,
        "RMSE": mean_squared_error(actual, predicted)** 0.5,
        "MAE": mean_absolute_error(actual, predicted),
        "R2": r2_score(actual, predicted),
    }
    return result

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train_enc, y_train)
lr_result = evaluate("Linear Regression", lr, X_val_enc, y_val)
lr_result

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_leaf=2,
                           n_jobs=-1, random_state=42)
rf.fit(X_train_enc, y_train)

rf_result = evaluate("Random Forest", rf, X_val_enc, y_val)
rf_result

# XGBoost 
import xgboost as xgb

xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,       # L1 regularisation 
    reg_lambda=1.0,    # L2
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
    eval_metric="rmse",
)

xgb_model.fit(
    X_train_enc, y_train,
    eval_set=[(X_val_enc, y_val)],
    verbose=100,
)

xgb_result = evaluate("XGBoost", xgb_model, X_val_enc, y_val)
xgb_result

results = pd.DataFrame([lr_result, rf_result, xgb_result]).set_index("Model")
results.style.format({"RMSE": "{:,.0f}", "MAE": "{:,.0f}", "R2": "{:.3f}"})

from sklearn.model_selection import cross_val_score

for name, model in [("Linear Regression", lr), ("Random Forest", rf)]:
    scores = cross_val_score(model, X_train_enc, y_train, cv=3, scoring="neg_root_mean_squared_error")
    print(f"{name:18s} CV RMSE (log scale): {-scores.mean():.4f}")

# Pick the best model by R2 score on validation set
all_results = {"Linear Regression": lr_result, "Random Forest": rf_result, "XGBoost": xgb_result}
best_name = max(all_results, key=lambda k: all_results[k]["R2"])
best_model = {"Linear Regression": lr, "Random Forest": rf, "XGBoost": xgb_model}[best_name]
print(f"\nBest model: {best_name} (R2 = {all_results[best_name]['R2']:.3f})")

test_clean = test.copy()

for col in missing_numeric:
    test_clean[col] = test_clean[col].fillna(train[col].median())
test_clean["body_type"] = test_clean["body_type"].fillna("Unknown")
test_clean["Location"] = test_clean["Location"].fillna(train["Location"].mode()[0])

test_clean["Maker_model"] = test_clean["Maker"] + "_" + test_clean["model"]

X_test = test_clean[FEATURES].copy()

X_test_c = clip_numeric(X_test)

# Same encoding as the training
cat_test = enc.transform(X_test_c[CATEGORICAL_FEATURES])
X_test_enc = np.hstack([X_test_c[NUMERIC_FEATURES].values, cat_test])

print("Missing values:", np.isnan(X_test_enc).sum())

from sklearn.base import clone

X_all_enc = np.vstack([X_train_enc, X_val_enc])
y_all = pd.concat([y_train, y_val])

if best_name == "XGBoost":
    # Exclude keys we set explicitly to avoid "multiple values" TypeError
    exclude = {"early_stopping_rounds", "random_state", "n_jobs"}
    params = {k: v for k, v in best_model.get_params().items() if k not in exclude}
    final_model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
    final_model.fit(X_all_enc, y_all, verbose=False)
else:
    final_model = clone(best_model)
    final_model.fit(X_all_enc, y_all)

test_pred = np.expm1(final_model.predict(X_test_enc))

submission = pd.DataFrame({"ID": test["ID"], "Price": test_pred.round(2)})
submission.to_csv("submission.csv", index=False)

print(submission.shape)
print(submission["Price"].describe().round(0))
submission.head()

import joblib

joblib.dump({
    "model": final_model,
    "encoder": enc,
    "clip_low": clip_low,
    "clip_high": clip_high,
    "features": FEATURES,
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "feature_names": NUMERIC_FEATURES + list(enc.get_feature_names_out()),
}, "car_price_model.joblib")

print("Model saved to car_price_model.joblib")