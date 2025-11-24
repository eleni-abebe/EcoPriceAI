import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

df=pd.read_csv('./data/co2.csv')
print(df.head(5))
print(df.describe())
print(df.info())
print(df.isnull().sum())
# ------------------------------
# PHASE 2: FEATURE SELECTION
# ------------------------------

# Step 6: Choose the target variable
target = "CO2 Emissions(g/km)"
y = df[target]

# Step 7: Select numerical features
num_features = [
    "Engine Size(L)",
    "Cylinders",
    "Fuel Consumption City (L/100 km)",
    "Fuel Consumption Hwy (L/100 km)",
    "Fuel Consumption Comb (L/100 km)",
    "Fuel Consumption Comb (mpg)"
]

# Step 8: Select categorical features
cat_features = [
    "Make",
    "Model",
    "Vehicle Class",
    "Transmission",
    "Fuel Type"
]

# Combine all features
all_features = num_features + cat_features

# Step 9: Build X matrix (input features)
X = df[all_features]

# Print shapes to verify
print("\nSelected Numerical Features:", num_features)
print("Selected Categorical Features:", cat_features)
print("\nX Shape:", X.shape)
print("y Shape:", y.shape)

# ------------------------------
# PHASE 3: PREPROCESSING + PIPELINE
# ------------------------------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Step 10: Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Step 11: Numerical pipeline
numeric_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

# Step 12: Categorical pipeline
categorical_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Step 13: Combine both pipelines
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, num_features),
    ("cat", categorical_pipeline, cat_features)
])

# Step 14: Full ML pipeline (preprocessing + model)
model = Pipeline([
    ("preprocess", preprocessor),
    ("regressor", LinearRegression())
])

print("\nPipeline created successfully!")


# ============================
# PHASE 4: Train/Test Split + Pipeline + Model Training
# ============================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Train the model
model.fit(X_train, y_train)

# 2. Predict
y_pred = model.predict(X_test)

# 5. Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Performance ---")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R²  :", r2)

# ============================
# PHASE 5: Visualization
# ============================

# --- Scatter Plot: Actual vs Predicted ---
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual CO2 Emissions (g/km)")
plt.ylabel("Predicted CO2 Emissions (g/km)")
plt.title("Actual vs Predicted CO2 Emissions")
plt.grid(True)
plt.show()

# --- Residual Plot ---
residuals = y_test - y_pred

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals)
plt.xlabel("Predicted CO2 Emissions (g/km)")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.axhline(0, linestyle='--')
plt.grid(True)
plt.show()


# ==============================
# PHASE 6: Export Model (Joblib)
# ==============================

from joblib import dump, load
import os

# --- Save the model ---
dump(model, "model/model.joblib")
print("\nModel saved successfully as model/model.joblib")

# --- Load back the model to verify ---
loaded_model = load("model/model.joblib")
print("Model loaded successfully!")

# --- Test prediction using loaded model ---

sample_df = X_test.iloc[[0]]  # double brackets -> keeps 2D

# Predict using the loaded pipeline
loaded_pred = loaded_model.predict(sample_df)[0]

print("\n--- Test Loaded Model ---")
print("Sample Input:", sample_df.to_dict())
print("Predicted CO2 Emission:", loaded_pred)
