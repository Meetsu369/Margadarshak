import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# load dataset
data = pd.read_csv("traffic_data.csv")

# create congestion label based on speed
data["congestion"] = data["avg_speed"].apply(lambda x: 1 if x < 5 else 0)

# create next-step congestion label (prediction target)
data["congestion_next"] = data["congestion"].shift(-1)

# remove last row with NaN
data = data.dropna()

# features and labels
X = data[["vehicles", "avg_speed"]]
y = data["congestion_next"]

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = RandomForestClassifier()

model.fit(X_train, y_train)

# predictions
predictions = model.predict(X_test)

print(classification_report(y_test, predictions))