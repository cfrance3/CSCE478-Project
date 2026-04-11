import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Read data from CSV files
weather = pd.read_csv("weather.csv")
rides = pd.read_csv("cab_rides.csv")

# Drop non-desired attributes from each file
# weather = weather.drop()
rides = rides.drop(columns=["id", "name"])

# Convert timestamps to seconds
weather["time_stamp_s_weather"] = weather["time_stamp"] # Already in seconds
rides["time_stamp_s_ride"] = rides["time_stamp"] // 1000

# Fill in 0's for missing rain data
weather["rain"] = weather["rain"].fillna(0)

# Merge on source/location
merged = rides.merge(weather, left_on="source", right_on="location", how="inner")

# Filter merge by 1 hour
merged["time_diff"] = (merged["time_stamp_s_ride"] - merged["time_stamp_s_weather"]).abs()
merged = merged[merged["time_diff"] <= 3600]

# Keep only closest weather row for each ride
merged = merged.sort_values("time_diff").groupby("time_stamp_s_ride", as_index=False).first()

# Extract day/hour for model
merged["hour"] = pd.to_datetime(rides["time_stamp"], unit="ms").dt.hour
merged["day_of_week"] = pd.to_datetime(rides["time_stamp"], unit="ms").dt.dayofweek

merged = pd.get_dummies(merged, columns=["cab_type", "destination", "product_id"], drop_first=True)

# Prepare features
ride_features = ["distance", "surge_multiplier", "hour", "day_of_week"] + \
                [c for c in merged.columns if c.startswith("cab_type_") or c.startswith("destination_") or c.startswith("product_id_")]

ride_weather_features = ride_features + ["temp", "clouds", "pressure", "rain", "humidity", "wind"]

# Features with requried data
required_features = ride_weather_features + ["price"]

# Clean data
merged_num = len(merged)
merged = merged.dropna(subset=required_features)
clean_num = len(merged)
# print("Removed ", merged_num-clean_num, " rows with missing data")

X_ride = merged[ride_features]
X_ride_weather = merged[ride_weather_features]
y = merged[["price"]]

# 80% training, 20% testing
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_ride, y, train_size=0.8, random_state=123)
X_train_rw, X_test_rw, y_train_rw, y_test_rw = train_test_split(X_ride_weather, y, train_size=0.8, random_state=123)

rideModel = LinearRegression()
rideWeatherModel = LinearRegression()

rideModel.fit(X_train_r, y_train_r)
rideWeatherModel.fit(X_train_rw, y_train_rw)

ride_score = rideModel.score(X_test_r, y_test_r)
ride_weather_score = rideWeatherModel.score(X_test_rw, y_test_rw)

print(f"Ride Score: {ride_score:.4f}")
print(f"Ride Weather Score: {ride_weather_score:.4f}")

# Error statistics
y_test_r_flat = np.ravel(y_test_r)
y_test_rw_flat = np.ravel(y_test_rw)

y_pred_r = np.ravel(rideModel.predict(X_test_r))
y_pred_rw = np.ravel(rideWeatherModel.predict(X_test_rw))

error_ride = y_test_r_flat - y_pred_r
error_ride_weather = y_test_rw_flat - y_pred_rw

mean_ride = np.mean(error_ride)
var_ride = np.var(error_ride, ddof=1)

mean_ride_weather = np.mean(error_ride_weather)
var_ride_weather = np.var(error_ride_weather, ddof=1)

print("\nRide Model Error Statistics:")
print(f"Mean: {mean_ride:.4f}")
print(f"Var: {var_ride:.2f}")
print("\nRide Weather Model Error Statistics:")
print(f"Mean: {mean_ride_weather:.4f}")
print(f"Var: {var_ride_weather:.2f}")

