import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
import joblib
from util import time_aware_ewma

# Load your data
df = pd.read_csv("../sensor_data.csv", parse_dates=["timestamp"])
df = df.set_index("timestamp")

# Calculate the time difference (Δt in minutes) between each point
df["delta_t"] = df.index.to_series().diff().dt.total_seconds() / 60.0  # time in minutes
df["delta_t"].iloc[0] = 0  # Set first value to 0

# Create time-aware lag features
df["lag1_temp"] = df["temperature"].shift(1)
df["lag2_temp"] = df["temperature"].shift(2)
df["lag1_pres"] = df["pressure"].shift(1)
df["lag2_pres"] = df["pressure"].shift(2)
df["lag1_hum"] = df["humidity"].shift(1)
df["lag2_hum"] = df["humidity"].shift(2)
df["lag1_iaq"] = df["iaq"].shift(1)
df["lag2_iaq"] = df["iaq"].shift(2)

# Compute the time differences (Δt) for each lag
df["lag1_time_diff"] = df["delta_t"].shift(1)
df["lag2_time_diff"] = df["delta_t"].shift(2)

# Apply time-aware EWMA for temperature, pressure, and humidity
df["ewma_temp"] = time_aware_ewma(df["temperature"], df["delta_t"])
df["ewma_pres"] = time_aware_ewma(df["pressure"], df["delta_t"])
df["ewma_hum"] = time_aware_ewma(df["humidity"], df["delta_t"])
df["ewma_iaq"] = time_aware_ewma(df["iaq"], df["delta_t"])

# Example: rolling mean over a 15-minute window (adjust to time intervals)
df["roll_mean_temp"]  = df["temperature"].rolling("15min").mean()
df["roll_mean_pres"]  = df["pressure"].rolling("15min").mean()
df["roll_mean_hum"]   = df["humidity"].rolling("15min").mean()
df["roll_mean_iaq"]   = df["iaq"].rolling("15min").mean()

df["temp_target"] = df["temperature"].shift(-1)
df["pres_target"] = df["pressure"].shift(-1)
df["hum_target"] = df["humidity"].shift(-1)
df["iaq_target"] = df["iaq"].shift(-1)

targets = ["temp_target", "pres_target", "hum_target", "iaq_target"]

# Select the features and drop rows with missing target values
feature_cols = [
    "lag1_temp", "lag2_temp", "lag1_pres", "lag2_pres",
    "lag1_hum", "lag2_hum", "lag1_iaq", "lag2_iaq",
    "lag1_time_diff", "lag2_time_diff",
    "ewma_temp", "ewma_pres", "ewma_hum", "ewma_iaq",
    "roll_mean_temp", "roll_mean_pres", "roll_mean_hum",
    "roll_mean_iaq", "delta_t"
]

df_model = df[feature_cols + targets].dropna()

X = df_model[feature_cols]
y = df_model[targets]

# Split the data (80% train, 20% test)
split = int(0.8 * len(df_model))
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Train a multi-output model
model = MultiOutputRegressor(Ridge(alpha=1.0))
model.fit(X_train, y_train)

# Predict the test data
y_pred = model.predict(X_test)

for i, var in enumerate(targets):
    mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
    print(f"{var} MAE: {mae:.3f}")

joblib.dump(model, "model.bin")
