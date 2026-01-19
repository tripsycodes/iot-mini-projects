from model.model import TrainedModel
from model.util import time_aware_ewma
import pandas as pd

sensor_data = pd.read_csv("./sensor_data.csv", parse_dates=["timestamp"]).set_index("timestamp")
# 3 readings must be skipped to get delta_t, lag1_time_diff, lag2_time_diff
# the feature vector is created here for sensor_data.iloc[3]
last_reading_index = 3
lag1_index = last_reading_index - 1
lag2_index = last_reading_index - 2
lag1_temp = sensor_data.iloc[lag1_index]["temperature"]
lag2_temp = sensor_data.iloc[lag2_index]["temperature"]

lag1_pres = sensor_data.iloc[lag1_index]["pressure"]
lag2_pres = sensor_data.iloc[lag2_index]["pressure"]

lag1_hum = sensor_data.iloc[lag1_index]["humidity"]
lag2_hum = sensor_data.iloc[lag2_index]["humidity"]

lag1_iaq = sensor_data.iloc[lag1_index]["iaq"]
lag2_iaq = sensor_data.iloc[lag2_index]["iaq"]

time_diffs = [0]
for i in range(last_reading_index):
    time_diffs.append((sensor_data.index.to_series().iloc[i+1] - sensor_data.index.to_series().iloc[i]).total_seconds() / 60)

lag1_time_diff = time_diffs[lag1_index]
lag2_time_diff = time_diffs[lag2_index]

ewma_temp = time_aware_ewma(sensor_data.iloc[:last_reading_index+1]["temperature"], time_diffs).iloc[-1]
ewma_pres = time_aware_ewma(sensor_data.iloc[:last_reading_index+1]["pressure"], time_diffs).iloc[-1]
ewma_hum = time_aware_ewma(sensor_data.iloc[:last_reading_index+1]["humidity"], time_diffs).iloc[-1]
ewma_iaq = time_aware_ewma(sensor_data.iloc[:last_reading_index+1]["iaq"], time_diffs).iloc[-1]

roll_mean_temp = sensor_data.iloc[:last_reading_index+1]["temperature"].rolling("15min").mean().iloc[-1]
roll_mean_pres = sensor_data.iloc[:last_reading_index+1]["pressure"].rolling("15min").mean().iloc[-1]
roll_mean_hum = sensor_data.iloc[:last_reading_index+1]["humidity"].rolling("15min").mean().iloc[-1]
roll_mean_iaq = sensor_data.iloc[:last_reading_index+1]["iaq"].rolling("15min").mean().iloc[-1]

delta_t = time_diffs[last_reading_index]

X = [lag1_temp, lag2_temp, lag1_pres, lag2_pres, lag1_hum, lag2_hum, lag1_iaq, lag2_iaq, lag1_time_diff, lag2_time_diff, ewma_temp, ewma_pres, ewma_hum, ewma_iaq, roll_mean_temp, roll_mean_pres, roll_mean_hum, roll_mean_iaq, delta_t]

model = TrainedModel("model/model.bin")
y = model.predict([X])
for pred in y:
    print("Predicted temperature:", pred[0])
    print("Predicted pressure:", pred[1])
    print("Predicted humidity:", pred[2])
    print("Predicted IAQ:", pred[3])
