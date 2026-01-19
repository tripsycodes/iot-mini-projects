# ML models

## LiteRT model

Dataset: (github.com/Hrushikraj/Data_Analysis_on_Weather_Dataset)[https://raw.githubusercontent.com/Hrushikraj/Data_Analysis_on_Weather_Dataset/refs/heads/main/1.%20Weather%20Data.csv]

1. Use `model.ipynb` to train and save the model in `model.tflite`.
2. Use `xxd -i model.tflite > model_data.cc` to generate `model_data.cc`.
3. Change `unsigned char` to `const unsigned char` in `model_data.cc` for better memory layout.
4. Integrate `model_test.cc` in the Arduino code.

## sklearn model (Multi-Output Ridge Regressor)

### Features
- lag1_temp
- lag2_temp
- lag1_pres
- lag2_pres
- lag1_hum
- lag2_hum
- lag1_iaq
- lag2_iaq
- lag1_time_diff
- lag2_time_diff
- ewma_temp
- ewma_pres
- ewma_hum
- ewma_iaq
- roll_mean_temp
- roll_mean_pres
- roll_mean_hum
- roll_mean_iaq
- delta_t

### Outputs
- temp_target
- pres_target
- hum_target
- iaq_target

1. Use `train.py` to train and save the model in `model.bin`.
2. Import `model.py` and use `model.predict(X)`.
