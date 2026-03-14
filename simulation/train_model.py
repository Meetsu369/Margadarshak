"""
=========================================================
File: train_model.py
Purpose: Train a machine learning model with 3-step 
temporal memory to detect traffic shockwaves.
=========================================================
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils import resample

def train_traffic_model():

    # ---------------------------------------------------------
    # STEP 1: Load dataset
    # ---------------------------------------------------------
    try:
        data = pd.read_csv("traffic_data.csv")
        print("\n[1] Dataset Loaded Successfully")
    except FileNotFoundError:
        print("Error: traffic_data.csv not found.")
        return

    # ---------------------------------------------------------
    # STEP 2: Create prediction target
    # ---------------------------------------------------------
    data["congestion_next"] = data["congestion"].shift(-1)

    # ---------------------------------------------------------
    # STEP 3: Create Traffic Features (Current & Changes)
    # ---------------------------------------------------------
    data["speed_change"] = data["avg_speed"].diff().fillna(0)
    data["vehicle_change"] = data["vehicles"].diff().fillna(0)
    data["traffic_pressure"] = data["vehicles"] / (data["avg_speed"] + 0.1)

    # ---------------------------------------------------------
    # STEP 4: EXTENDED TEMPORAL MEMORY (3-Step Window)
    # Helps the model identify shockwave trends over time
    # ---------------------------------------------------------
    # Speed History
    data["prev_speed"] = data["avg_speed"].shift(1).fillna(data["avg_speed"])
    data["prev2_speed"] = data["avg_speed"].shift(2).fillna(data["avg_speed"])
    data["prev3_speed"] = data["avg_speed"].shift(3).fillna(data["avg_speed"])

    # Vehicle History
    data["prev_vehicles"] = data["vehicles"].shift(1).fillna(data["vehicles"])
    data["prev2_vehicles"] = data["vehicles"].shift(2).fillna(data["vehicles"])
    data["prev3_vehicles"] = data["vehicles"].shift(3).fillna(data["vehicles"])

    # ---------------------------------------------------------
    # STEP 5: Clean up and Format
    # ---------------------------------------------------------
    data = data.dropna().copy()
    data["congestion_next"] = data["congestion_next"].astype(int)

    # ---------------------------------------------------------
    # STEP 6: Define features and labels
    # ---------------------------------------------------------
    X = data[[
        "vehicles",
        "avg_speed",
        "speed_change",
        "vehicle_change",
        "prev_speed",
        "prev_vehicles",
        "prev2_speed",
        "prev3_speed",
        "prev2_vehicles",
        "prev3_vehicles",
        "traffic_pressure"
    ]]
    y = data["congestion_next"]

    # ---------------------------------------------------------
    # STEP 7: Split dataset (Stratified to maintain class ratios)
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------------------------------------------------------
    # STEP 8: Balance Training Data (Avoid Data Leakage)
    # ---------------------------------------------------------
    train_data = pd.concat([X_train, y_train], axis=1)
    majority = train_data[train_data["congestion_next"] == 0]
    minority = train_data[train_data["congestion_next"] == 1]

    minority_upsampled = resample(
        minority,
        replace=True,
        n_samples=len(majority),
        random_state=42
    )

    balanced_train = pd.concat([majority, minority_upsampled])
    X_train = balanced_train.drop("congestion_next", axis=1)
    y_train = balanced_train["congestion_next"]

    # ---------------------------------------------------------
    # STEP 9: Train Random Forest model
    # ---------------------------------------------------------
    print("\n[2] Training model with Windowed Temporal Memory...")
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=15, # Increased depth slightly for more complex features
        class_weight="balanced_subsample",
        random_state=42
    )
    model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # STEP 10: Evaluate
    # ---------------------------------------------------------
    predictions = model.predict(X_test)
    print("\n[3] Model Performance Report:")
    print(classification_report(y_test, predictions))

    # Feature Importance check
    importances = pd.Series(model.feature_importances_, index=X.columns)
    print("\n[4] Top 5 Most Predictive Features:")
    print(importances.sort_values(ascending=False).head(5))

if __name__ == "__main__":
    train_traffic_model()