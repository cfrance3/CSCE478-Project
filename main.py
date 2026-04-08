import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Read data from CSV files
weather = pd.read_csv("weather.csv")
rides = pd.read_csv("cab_rides.csv")

# Desired attributes from each file
weather = weather[["temp", "clouds", "rain", "humidity", "wind", "pressure", "time_stamp", "location"]]
rides = rides[["distance", "price", "time_stamp", "destination", "source", "surge_multiplier", "cab_type"]]

# Converts ms to s to match weather
rides["time_stamp"] = rides["time_stamp"] // 1000

# Rename source column to location for merging
rides = rides.rename(columns={"source": "location"})

# Fill in 0s for missing rain data
weather["rain"] = weather["rain"].fillna(0)

# Merges ride and weather data by finding the nearest match on time stamp at the same location
merged_data = pd.merge_asof(
    rides.sort_values("time_stamp"),
    weather.sort_values("time_stamp"),
    on="time_stamp",
    by="location",
    tolerance=3600  # In seconds
    )

# Number of rows before cleaning
initial_num_rows = len(merged_data)

merged_data = merged_data.dropna(subset=["distance", "price", "temp", "clouds", "humidity", "wind", "pressure"])

# Number of rows after cleaning
clean_num_rows = len(merged_data)

print("Removed ", initial_num_rows - clean_num_rows, " rows with missing data.")

X = merged_data[["distance", "surge_multiplier", "temp", "clouds", "rain", "humidity", "wind", "pressure"]]
y = merged_data[["price"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=123)

linearModel = LinearRegression()

linearModel.fit(X_train, y_train)

score = linearModel.score(X_test, y_test)
print(score)
