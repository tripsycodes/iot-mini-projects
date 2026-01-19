"""
Environmental Sensor Monitoring Dashboard
Minimalist, professional interface for IoT sensor data visualization
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os, time, sys, warnings
warnings.filterwarnings("ignore")

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Sensor Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== MODEL IMPORTS ====================
model_dir = os.path.join(os.path.dirname(__file__), "model")
if model_dir not in sys.path:
    sys.path.insert(0, model_dir)

TrainedModel = None
time_aware_ewma = None
import_error = None

try:
    import importlib.util

    model_spec = importlib.util.spec_from_file_location(
        "model_module", os.path.join(model_dir, "model.py")
    )
    model_module = importlib.util.module_from_spec(model_spec)
    model_spec.loader.exec_module(model_module)
    TrainedModel = model_module.TrainedModel

    util_spec = importlib.util.spec_from_file_location(
        "util_module", os.path.join(model_dir, "util.py")
    )
    util_module = importlib.util.module_from_spec(util_spec)
    util_spec.loader.exec_module(util_module)
    time_aware_ewma = util_module.time_aware_ewma

except Exception as e:
    import_error = str(e)

# ==================== FILE WATCH ====================
if "last_mtime" not in st.session_state:
    st.session_state.last_mtime = 0

def file_updated(path: str) -> bool:
    if not os.path.exists(path):
        return False
    mtime = os.path.getmtime(path)
    if mtime > st.session_state.last_mtime:
        st.session_state.last_mtime = mtime
        st.cache_data.clear()
        return True
    return False

# ==================== CSV LOADER ====================
@st.cache_data
def load_csv(path):
    df = pd.read_csv(path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    numeric_cols = [
        "device_id","sequence","uptime","temperature","humidity","pressure",
        "iaq","iaq_accuracy","static_iaq","co2_ppm","voc_ppm","gas_percent",
        "rssi","snr"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

# ==================== FEATURE CALC ====================
def calculate_features(df):
    """Calculate features from sensor data for model prediction"""
    if df.empty or len(df) < 3:
        return None, None
    
    try:
        df_sorted = df.sort_values('timestamp' if 'timestamp' in df.columns else df.index.name or 'index').reset_index(drop=True)
        
        if len(df_sorted) < 3:
            return None, None
        
        last_idx = len(df_sorted) - 1
        lag1_idx = last_idx - 1
        lag2_idx = last_idx - 2
        
        # Get lag values
        lag1_temp = df_sorted.iloc[lag1_idx].get('temperature', None)
        lag2_temp = df_sorted.iloc[lag2_idx].get('temperature', None)
        lag1_pres = df_sorted.iloc[lag1_idx].get('pressure', None)
        lag2_pres = df_sorted.iloc[lag2_idx].get('pressure', None)
        lag1_hum = df_sorted.iloc[lag1_idx].get('humidity', None)
        lag2_hum = df_sorted.iloc[lag2_idx].get('humidity', None)
        lag1_iaq = df_sorted.iloc[lag1_idx].get('iaq', None)
        lag2_iaq = df_sorted.iloc[lag2_idx].get('iaq', None)
        
        if any(pd.isna([lag1_temp, lag2_temp, lag1_pres, lag2_pres, lag1_hum, lag2_hum, lag1_iaq, lag2_iaq])):
            return None, None
        
        # Calculate time differences
        if 'timestamp' in df_sorted.columns:
            timestamps = pd.to_datetime(df_sorted['timestamp'], errors='coerce')
            time_diffs = [0]
            for i in range(min(last_idx, len(timestamps) - 1)):
                if pd.notna(timestamps.iloc[i]) and pd.notna(timestamps.iloc[i+1]):
                    diff_minutes = (timestamps.iloc[i+1] - timestamps.iloc[i]).total_seconds() / 60
                    time_diffs.append(diff_minutes)
                else:
                    time_diffs.append(0)
        else:
            time_diffs = [0] * (last_idx + 1)
        
        lag1_time_diff = time_diffs[lag1_idx] if lag1_idx < len(time_diffs) else 0
        lag2_time_diff = time_diffs[lag2_idx] if lag2_idx < len(time_diffs) else 0
        delta_t = time_diffs[last_idx] if last_idx < len(time_diffs) else 0
        
        # Calculate EWMA (simplified)
        temp_series = df_sorted.iloc[:last_idx+1]['temperature']
        pres_series = df_sorted.iloc[:last_idx+1]['pressure']
        hum_series = df_sorted.iloc[:last_idx+1]['humidity']
        iaq_series = df_sorted.iloc[:last_idx+1]['iaq']
        
        ewma_temp = temp_series.ewm(alpha=0.3).mean().iloc[-1]
        ewma_pres = pres_series.ewm(alpha=0.3).mean().iloc[-1]
        ewma_hum = hum_series.ewm(alpha=0.3).mean().iloc[-1]
        ewma_iaq = iaq_series.ewm(alpha=0.3).mean().iloc[-1]
        
        # Calculate rolling means
        roll_mean_temp = temp_series.rolling(window=min(3, len(temp_series)), min_periods=1).mean().iloc[-1]
        roll_mean_pres = pres_series.rolling(window=min(3, len(pres_series)), min_periods=1).mean().iloc[-1]
        roll_mean_hum = hum_series.rolling(window=min(3, len(hum_series)), min_periods=1).mean().iloc[-1]
        roll_mean_iaq = iaq_series.rolling(window=min(3, len(iaq_series)), min_periods=1).mean().iloc[-1]
        
        # Create feature vector
        features = {
            'lag1_temp': lag1_temp,
            'lag2_temp': lag2_temp,
            'lag1_pres': lag1_pres,
            'lag2_pres': lag2_pres,
            'lag1_hum': lag1_hum,
            'lag2_hum': lag2_hum,
            'lag1_iaq': lag1_iaq,
            'lag2_iaq': lag2_iaq,
            'lag1_time_diff': lag1_time_diff,
            'lag2_time_diff': lag2_time_diff,
            'ewma_temp': ewma_temp,
            'ewma_pres': ewma_pres,
            'ewma_hum': ewma_hum,
            'ewma_iaq': ewma_iaq,
            'roll_mean_temp': roll_mean_temp,
            'roll_mean_pres': roll_mean_pres,
            'roll_mean_hum': roll_mean_hum,
            'roll_mean_iaq': roll_mean_iaq,
            'delta_t': delta_t
        }
        
        # Create feature vector with feature names for scikit-learn
        X_dict = {
            'lag1_temp': lag1_temp, 'lag2_temp': lag2_temp,
            'lag1_pres': lag1_pres, 'lag2_pres': lag2_pres,
            'lag1_hum': lag1_hum, 'lag2_hum': lag2_hum,
            'lag1_iaq': lag1_iaq, 'lag2_iaq': lag2_iaq,
            'lag1_time_diff': lag1_time_diff, 'lag2_time_diff': lag2_time_diff,
            'ewma_temp': ewma_temp, 'ewma_pres': ewma_pres,
            'ewma_hum': ewma_hum, 'ewma_iaq': ewma_iaq,
            'roll_mean_temp': roll_mean_temp, 'roll_mean_pres': roll_mean_pres,
            'roll_mean_hum': roll_mean_hum, 'roll_mean_iaq': roll_mean_iaq,
            'delta_t': delta_t
        }
        
        return features, pd.DataFrame([X_dict])
    except Exception as e:
        return None, None


@st.cache_resource
def load_model():
    model_path = "model/model.bin"
    if os.path.exists(model_path):
        return TrainedModel(model_path)
    return None

@st.cache_resource
def load_model():
    path = "model/model.bin"
    if os.path.exists(path):
        return TrainedModel(path)
    return None

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### Filters")

    csv_path = st.text_input("CSV File Path", value="sensor_data.csv")
    auto_refresh = st.checkbox("Auto-refresh (2s)", value=True)

# ==================== SINGLE CSV LOAD ====================
updated = file_updated(csv_path)

if "df_all" not in st.session_state or updated:
    if os.path.exists(csv_path):
        st.session_state.df_all = load_csv(csv_path)
    else:
        st.session_state.df_all = pd.DataFrame()

df_all = st.session_state.df_all

# ==================== DEVICE SELECT ====================
with st.sidebar:
    if not df_all.empty and "device_id" in df_all.columns:
        devs = sorted(df_all.device_id.dropna().unique())
        labels = ["All Devices"] + [f"Device {d}" for d in devs]
        sel = st.selectbox("Device", labels)
        device_id = None if sel == "All Devices" else devs[labels.index(sel)-1]
    else:
        device_id = None

# ==================== LAYOUT ====================
main_col, right_col = st.columns([3, 1])

# ==================== MAIN ====================
with main_col:
    st.title("Sensor Monitor")

    if df_all.empty:
        st.info("No data available.")
    else:
        df = df_all.copy()
        if device_id is not None:
            df = df[df.device_id == device_id]

        df = df.sort_values("timestamp")
        latest = df.iloc[-1]

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Temperature", f"{latest.temperature:.1f}°C")
        c2.metric("Humidity", f"{latest.humidity:.1f}%")
        c3.metric("Pressure", f"{latest.pressure:.2f}")
        c4.metric("IAQ", f"{latest.iaq:.0f}")
        c5.metric("CO₂", f"{latest.co2_ppm:.0f}")
        c6.metric("VOC", f"{latest.voc_ppm:.2f}")

        st.markdown("---")
        st.markdown("### Recent Data")

        cols = [
            "timestamp","device_id","temperature","humidity",
            "pressure","iaq","co2_ppm","voc_ppm"
        ]
        st.dataframe(
            df[cols].tail(10).sort_values("timestamp", ascending=False),
            use_container_width=True,
            hide_index=True
        )

# ==================== RIGHT SIDEBAR (RESTORED) ====================
with right_col:
    st.markdown("### Schema & Predictions")
    st.markdown("---")

    if not df.empty and len(df) >= 3:
        features, X_df = calculate_features(df)
        model = load_model()

        if features is not None:
            st.markdown("#### Features")

            st.markdown("**Lag Values**")
            st.write(f"Temp (t-1): {features['lag1_temp']:.2f}")
            st.write(f"Temp (t-2): {features['lag2_temp']:.2f}")
            st.write(f"Pressure (t-1): {features['lag1_pres']:.2f}")
            st.write(f"Humidity (t-1): {features['lag1_hum']:.2f}")
            st.write(f"IAQ (t-1): {features['lag1_iaq']:.0f}")
            st.write(f"Δ Time: {features['delta_t']:.2f} min")

            st.markdown("---")
            st.markdown("#### Predictions")

            if model is None:
                st.warning("Model not loaded")
            else:
                try:
                    preds = model.predict(X_df)[0]
                    st.success("Next Reading Prediction")
                    st.write(f"🌡 Temperature: {preds[0]:.2f} °C")
                    st.write(f"📈 Pressure: {preds[1]:.2f}")
                    st.write(f"💧 Humidity: {preds[2]:.2f} %")
                    st.write(f"🌫 IAQ: {preds[3]:.0f}")
                except Exception as e:
                    st.error(f"Prediction error: {e}")
    else:
        st.info("Need at least 3 readings for prediction")

# ==================== FOOTER ====================
st.markdown("---")
st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if auto_refresh:
    time.sleep(2)
    st.rerun()
