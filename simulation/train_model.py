"""
=========================================================
File: train_model.py
Purpose: Train, evaluate, and save a Random Forest model
with feature importance reporting.
=========================================================
"""

import pandas as pd
import joblib
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
        print("Error: traffic_data.csv not found. Run your SUMO simulation first.")
        return

    # ---------------------------------------------------------
    # STEP 2: Feature Engineering & Target Creation
    # ---------------------------------------------------------
    # Predict congestion in the next step
    data["congestion_next"] = data["congestion"].shift(-1)

    # Calculate current dynamics
    data["speed_change"] = data["avg_speed"].diff().fillna(0)
    data["vehicle_change"] = data["vehicles"].diff().fillna(0)
    data["traffic_pressure"] = data["vehicles"] / (data["avg_speed"] + 0.1)

    # 3-Step Temporal Memory (Speed & Vehicles)
    data["prev_speed"] = data["avg_speed"].shift(1).fillna(data["avg_speed"])
    data["prev2_speed"] = data["avg_speed"].shift(2).fillna(data["avg_speed"])
    data["prev3_speed"] = data["avg_speed"].shift(3).fillna(data["avg_speed"])

    data["prev_vehicles"] = data["vehicles"].shift(1).fillna(data["vehicles"])
    data["prev2_vehicles"] = data["vehicles"].shift(2).fillna(data["vehicles"])
    data["prev3_vehicles"] = data["vehicles"].shift(3).fillna(data["vehicles"])

    # Final cleanup
    data = data.dropna().copy()
    data["congestion_next"] = data["congestion_next"].astype(int)

    # ---------------------------------------------------------
    # STEP 3: Dataset Preparation
    # ---------------------------------------------------------
    feature_cols = [
        "vehicles", "avg_speed", "speed_change", "vehicle_change", 
        "prev_speed", "prev_vehicles", "prev2_speed", "prev3_speed", 
        "prev2_vehicles", "prev3_vehicles", "traffic_pressure"
    ]
    X = data[feature_cols]
    y = data["congestion_next"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------------------------------------------------------
    # STEP 4: Training Set Balancing
    # ---------------------------------------------------------
    train_data = pd.concat([X_train, y_train], axis=1)
    maj = train_data[train_data["congestion_next"] == 0]
    min = train_data[train_data["congestion_next"] == 1]

    min_upsampled = resample(min, replace=True, n_samples=len(maj), random_state=42)
    balanced_train = pd.concat([maj, min_upsampled])

    X_train = balanced_train.drop("congestion_next", axis=1)
    y_train = balanced_train["congestion_next"]

    # ---------------------------------------------------------
    # STEP 5: Model Training
    # ---------------------------------------------------------
    print("\n[2] Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        class_weight="balanced_subsample",
        random_state=42
    )
    model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # STEP 6: Evaluation
    # ---------------------------------------------------------
    predictions = model.predict(X_test)
    print("\n[3] Model Performance Report:")
    print(classification_report(y_test, predictions))

    # ---------------------------------------------------------
    # STEP 7: Feature Importance & Save
    # ---------------------------------------------------------
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    print("\n[4] Top Important Features (The AI's logic):")
    print(importances.sort_values(ascending=False).head(5))

    joblib.dump(model, "traffic_model.pkl")
    print("\n[SUCCESS] AI brain saved as traffic_model.pkl")

if __name__ == "__main__":
    train_traffic_model()