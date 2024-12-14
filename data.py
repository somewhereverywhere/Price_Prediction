import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

# Step 1: Load the dataset
df = pd.read_csv("data.csv")

# Step 2: Preprocess the data

# Encode categorical variables
le_weather = LabelEncoder()
df["weather"] = le_weather.fit_transform(df["weather"])

le_day = LabelEncoder()
df["Day"] = le_day.fit_transform(df["Day"])

# Normalize numerical features
scaler = StandardScaler()
df[["Temperature", "Hour"]] = scaler.fit_transform(df[["Temperature", "Hour"]])
df = df[df["Adjusted price"] != "Closed"].dropna(subset=["Adjusted price"])

df["Adjusted price"] = pd.to_numeric(df["Adjusted price"], errors="coerce")

df.to_csv("final2_processed.csv", index=False)

# Step 3: Define features and target
X = df.drop(columns=["Item", "Adjusted price"])
y = df["Adjusted price"]

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],        # Number of trees
    'max_depth': [None, 10, 20, 30],      # Tree depth
    'min_samples_split': [2, 5, 10],      # Minimum samples for a split
    'min_samples_leaf': [1, 2, 4]         # Minimum samples per leaf
}

# Initialize Random Forest Regressor
rf = RandomForestRegressor(random_state=42)

# Step 6: Hyperparameter tuning with GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='r2',         # Optimize for R² score
    cv=5,                 # 5-fold cross-validation
    verbose=2,            # Display progress
    n_jobs=-1             # Use all processors
)

# Fit GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Get the best parameters and the best model
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

best_model = grid_search.best_estimator_

# Step 7: Evaluate the tuned model on the test set
y_pred = best_model.predict(X_test)
comparison_df = pd.DataFrame({
    'Actual Value': y_test,
    'Predicted Value': y_pred
})

print(comparison_df.head())

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")


# Step 8: Save the tuned model
with open("adjusted_price_model.pkl", "wb") as model_file:
    pickle.dump(best_model, model_file)

with open("adjusted_price_model.pkl", "wb") as model_file, \
        open("label_encoder_weather.pkl", "wb") as le_weather_file, \
        open("scaler.pkl", "wb") as scaler_file, \
        open("label_encoder_day.pkl", "wb") as le_day_file:
    # Save the trained model, LabelEncoders, and Scaler
    pickle.dump(best_model, model_file)
    pickle.dump(le_weather, le_weather_file)
    pickle.dump(scaler, scaler_file)
    pickle.dump(le_day, le_day_file)

print("Model, LabelEncoder for Weather, Scaler, and LabelEncoder for Day saved successfully.")
