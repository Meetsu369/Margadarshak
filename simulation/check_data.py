import pandas as pd

data = pd.read_csv("traffic_data.csv")
print(data["congestion"].value_counts())
print("Minimum speed:", data["avg_speed"].min())
print("Maximum speed:", data["avg_speed"].max())
print("Average speed:", data["avg_speed"].mean())