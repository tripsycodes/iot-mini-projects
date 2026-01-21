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
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== STYLING ====================
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: #0e1117;
    }
    
    /* Right sidebar background matching left sidebar */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    
    h1 {
        font-size: 1.75rem;
        font-weight: 500;
        color: #ffffff;
        letter-spacing: -0.02em;
        margin-bottom: 2rem;
    }
    
    h2, h3 {
        font-size: 1.25rem;
        font-weight: 500;
        color: #ffffff;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 500;
        color: #ffffff !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.875rem;
        font-weight: 400;
        color: #ffffff !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 1px solid #3a3a3a;
    }
    
    .dataframe {
        font-size: 0.875rem;
        color: #ffffff;
    }
    
    p, div, span, .stMarkdown, .stText, .stCaption {
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

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
    try:
        df = pd.read_csv(path)
        
        # Handle single-column CSV (quoted rows)
        if len(df.columns) == 1:
            with open(path, 'r', encoding='utf-8-sig') as f:
                lines = f.readlines()
            
            if not lines:
                return pd.DataFrame()
            
            header_line = lines[0].strip().strip('"').strip('\ufeff')
            columns = [col.strip() for col in header_line.split(',')]
            
            data_rows = []
            for line in lines[1:]:
                line = line.strip().strip('"')
                if line:
                    values = [val.strip() for val in line.split(',')]
                    if len(values) == len(columns):
                        data_rows.append(values)
            
            if data_rows:
                df = pd.DataFrame(data_rows, columns=columns)
            else:
                return pd.DataFrame()

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
        
        bool_cols = ['stabilized', 'run_in_complete']
        for col in bool_cols:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].str.lower().map({'true': True, 'false': False})
                else:
                    df[col] = df[col].astype(bool)

        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

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
        
        temp_series = df_sorted.iloc[:last_idx+1]['temperature']
        pres_series = df_sorted.iloc[:last_idx+1]['pressure']
        hum_series = df_sorted.iloc[:last_idx+1]['humidity']
        iaq_series = df_sorted.iloc[:last_idx+1]['iaq']
        
        ewma_temp = temp_series.ewm(alpha=0.3).mean().iloc[-1]
        ewma_pres = pres_series.ewm(alpha=0.3).mean().iloc[-1]
        ewma_hum = hum_series.ewm(alpha=0.3).mean().iloc[-1]
        ewma_iaq = iaq_series.ewm(alpha=0.3).mean().iloc[-1]
        
        roll_mean_temp = temp_series.rolling(window=min(3, len(temp_series)), min_periods=1).mean().iloc[-1]
        roll_mean_pres = pres_series.rolling(window=min(3, len(pres_series)), min_periods=1).mean().iloc[-1]
        roll_mean_hum = hum_series.rolling(window=min(3, len(hum_series)), min_periods=1).mean().iloc[-1]
        roll_mean_iaq = iaq_series.rolling(window=min(3, len(iaq_series)), min_periods=1).mean().iloc[-1]
        
        features = {
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
        
        X_dict = features.copy()
        
        return features, pd.DataFrame([X_dict])
    except Exception as e:
        return None, None

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
    
    time_range = st.selectbox(
        "Time Range",
        ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days", "All Data"],
        index=2
    )
    
    st.markdown("---")

# ==================== DATA LOADING ====================
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
        st.caption("No devices found in data")

# ==================== LAYOUT ====================
main_col, right_col = st.columns([3, 1])

# ==================== MAIN ====================
with main_col:
    st.title("Sensor Monitor")

    if df_all.empty:
        st.info("No data available. Please ensure the CSV file exists and contains sensor data.")
    else:
        df = df_all.copy()
        if device_id is not None:
            df = df[df.device_id == device_id]
        
        # Apply time filtering
        if time_range != "All Data" and 'timestamp' in df.columns and df['timestamp'].notna().any():
            ranges = {
                "Last Hour": timedelta(hours=1),
                "Last 6 Hours": timedelta(hours=6),
                "Last 24 Hours": timedelta(hours=24),
                "Last 7 Days": timedelta(days=7)
            }
            
            # Get the range delta
            range_delta = ranges.get(time_range, timedelta(hours=24))
            
            # Get the most recent timestamp in the data
            max_timestamp = df['timestamp'].max()
            
            # Calculate start time from the most recent data point
            start_datetime = max_timestamp - range_delta
            
            # Filter data
            mask = df['timestamp'] >= start_datetime
            filtered_df = df[mask].copy()
            
            if not filtered_df.empty:
                df = filtered_df
            else:
                st.warning(f"No data available for {time_range}. Showing all available data.")

        df = df.sort_values("timestamp" if 'timestamp' in df.columns else df.index)
        latest = df.iloc[-1]

        # Metrics
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        with c1:
            temp = latest.get('temperature', None)
            st.metric("Temperature", f"{temp:.1f}°C" if pd.notna(temp) else "—")
        with c2:
            humidity = latest.get('humidity', None)
            st.metric("Humidity", f"{humidity:.1f}%" if pd.notna(humidity) else "—")
        with c3:
            pressure = latest.get('pressure', None)
            if pd.notna(pressure):
                if pressure < 10:
                    st.metric("Pressure", f"{pressure*100:.2f} kPa")
                else:
                    st.metric("Pressure", f"{pressure/1000:.2f} kPa")
            else:
                st.metric("Pressure", "—")
        with c4:
            iaq = latest.get('iaq', None)
            st.metric("IAQ", f"{iaq:.0f}" if pd.notna(iaq) else "—")
        with c5:
            co2 = latest.get('co2_ppm', None)
            st.metric("CO₂", f"{co2:.0f} ppm" if pd.notna(co2) else "—")
        with c6:
            voc = latest.get('voc_ppm', None)
            st.metric("VOC", f"{voc:.2f} ppm" if pd.notna(voc) else "—")

        st.markdown("---")
        st.markdown("### Sensor Readings")
        
        # Charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            if 'temperature' in df.columns and df['temperature'].notna().any():
                x_data = df['timestamp'] if 'timestamp' in df.columns else df.index
                valid_mask = df['temperature'].notna()
                x_valid = x_data[valid_mask]
                y_valid = df['temperature'][valid_mask]
                
                fig_temp = go.Figure()
                fig_temp.add_trace(go.Scatter(
                    x=x_valid, y=y_valid,
                    mode='lines+markers',
                    name='Temperature',
                    line=dict(color='#ffffff', width=1.5),
                    marker=dict(size=4, color='#ffffff'),
                    hovertemplate='%{y:.1f}°C<extra></extra>'
                ))
                fig_temp.update_layout(
                    xaxis_title="", yaxis_title="°C", height=280,
                    margin=dict(l=40, r=20, t=10, b=40),
                    plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                    font=dict(size=11, color='#ffffff'),
                    xaxis=dict(showgrid=True, gridcolor='#3a3a3a'),
                    yaxis=dict(showgrid=True, gridcolor='#3a3a3a'),
                    hovermode='x unified'
                )
                st.plotly_chart(fig_temp, use_container_width=True, config=dict(displayModeBar=False))
        
        with chart_col2:
            if 'humidity' in df.columns and df['humidity'].notna().any():
                x_data = df['timestamp'] if 'timestamp' in df.columns else df.index
                valid_mask = df['humidity'].notna()
                x_valid = x_data[valid_mask]
                y_valid = df['humidity'][valid_mask]
                
                fig_hum = go.Figure()
                fig_hum.add_trace(go.Scatter(
                    x=x_valid, y=y_valid,
                    mode='lines+markers',
                    name='Humidity',
                    line=dict(color='#ffffff', width=1.5),
                    marker=dict(size=4, color='#ffffff'),
                    hovertemplate='%{y:.1f}%<extra></extra>'
                ))
                fig_hum.update_layout(
                    xaxis_title="", yaxis_title="%", height=280,
                    margin=dict(l=40, r=20, t=10, b=40),
                    plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                    font=dict(size=11, color='#ffffff'),
                    xaxis=dict(showgrid=True, gridcolor='#3a3a3a'),
                    yaxis=dict(showgrid=True, gridcolor='#3a3a3a'),
                    hovermode='x unified'
                )
                st.plotly_chart(fig_hum, use_container_width=True, config=dict(displayModeBar=False))
        
        chart_col3, chart_col4 = st.columns(2)
        
        with chart_col3:
            if 'pressure' in df.columns and df['pressure'].notna().any():
                x_data = df['timestamp'] if 'timestamp' in df.columns else df.index
                valid_mask = df['pressure'].notna()
                x_valid = x_data[valid_mask]
                pressure_values = df['pressure'][valid_mask].copy()
                
                if pressure_values.max() < 10:
                    pressure_values = pressure_values * 100
                else:
                    pressure_values = pressure_values / 1000
                
                fig_press = go.Figure()
                fig_press.add_trace(go.Scatter(
                    x=x_valid, y=pressure_values,
                    mode='lines+markers',
                    name='Pressure',
                    line=dict(color='#ffffff', width=1.5),
                    marker=dict(size=4, color='#ffffff'),
                    hovertemplate='%{y:.2f} kPa<extra></extra>'
                ))
                fig_press.update_layout(
                    xaxis_title="", yaxis_title="kPa", height=280,
                    margin=dict(l=40, r=20, t=10, b=40),
                    plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                    font=dict(size=11, color='#ffffff'),
                    xaxis=dict(showgrid=True, gridcolor='#3a3a3a'),
                    yaxis=dict(showgrid=True, gridcolor='#3a3a3a'),
                    hovermode='x unified'
                )
                st.plotly_chart(fig_press, use_container_width=True, config=dict(displayModeBar=False))
        
        with chart_col4:
            if 'iaq' in df.columns and df['iaq'].notna().any():
                x_data = df['timestamp'] if 'timestamp' in df.columns else df.index
                valid_mask = df['iaq'].notna()
                x_valid = x_data[valid_mask]
                y_valid = df['iaq'][valid_mask]
                
                fig_iaq = go.Figure()
                fig_iaq.add_trace(go.Scatter(
                    x=x_valid, y=y_valid,
                    mode='lines+markers',
                    name='IAQ',
                    line=dict(color='#ffffff', width=1.5),
                    marker=dict(size=4, color='#ffffff'),
                    hovertemplate='IAQ: %{y:.0f}<extra></extra>'
                ))
                fig_iaq.update_layout(
                    xaxis_title="", yaxis_title="IAQ", height=280,
                    margin=dict(l=40, r=20, t=10, b=40),
                    plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                    font=dict(size=11, color='#ffffff'),
                    xaxis=dict(showgrid=True, gridcolor='#3a3a3a'),
                    yaxis=dict(showgrid=True, gridcolor='#3a3a3a'),
                    hovermode='x unified'
                )
                st.plotly_chart(fig_iaq, use_container_width=True, config=dict(displayModeBar=False))
        
        chart_col5, chart_col6 = st.columns(2)
        
        with chart_col5:
            if 'co2_ppm' in df.columns and df['co2_ppm'].notna().any():
                x_data = df['timestamp'] if 'timestamp' in df.columns else df.index
                valid_mask = df['co2_ppm'].notna()
                x_valid = x_data[valid_mask]
                y_valid = df['co2_ppm'][valid_mask]
                
                fig_co2 = go.Figure()
                fig_co2.add_trace(go.Scatter(
                    x=x_valid, y=y_valid,
                    mode='lines+markers',
                    name='CO₂',
                    line=dict(color='#ffffff', width=1.5),
                    marker=dict(size=4, color='#ffffff'),
                    hovertemplate='CO₂: %{y:.0f} ppm<extra></extra>'
                ))
                fig_co2.update_layout(
                    xaxis_title="", yaxis_title="ppm", height=280,
                    margin=dict(l=40, r=20, t=10, b=40),
                    plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                    font=dict(size=11, color='#ffffff'),
                    xaxis=dict(showgrid=True, gridcolor='#3a3a3a'),
                    yaxis=dict(showgrid=True, gridcolor='#3a3a3a'),
                    hovermode='x unified'
                )
                st.plotly_chart(fig_co2, use_container_width=True, config=dict(displayModeBar=False))
        
        with chart_col6:
            if 'voc_ppm' in df.columns and df['voc_ppm'].notna().any():
                x_data = df['timestamp'] if 'timestamp' in df.columns else df.index
                valid_mask = df['voc_ppm'].notna()
                x_valid = x_data[valid_mask]
                y_valid = df['voc_ppm'][valid_mask]
                
                fig_voc = go.Figure()
                fig_voc.add_trace(go.Scatter(
                    x=x_valid, y=y_valid,
                    mode='lines+markers',
                    name='VOC',
                    line=dict(color='#ffffff', width=1.5),
                    marker=dict(size=4, color='#ffffff'),
                    hovertemplate='VOC: %{y:.2f} ppm<extra></extra>'
                ))
                fig_voc.update_layout(
                    xaxis_title="", yaxis_title="ppm", height=280,
                    margin=dict(l=40, r=20, t=10, b=40),
                    plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                    font=dict(size=11, color='#ffffff'),
                    xaxis=dict(showgrid=True, gridcolor='#3a3a3a'),
                    yaxis=dict(showgrid=True, gridcolor='#3a3a3a'),
                    hovermode='x unified'
                )
                st.plotly_chart(fig_voc, use_container_width=True, config=dict(displayModeBar=False))

        st.markdown("---")
        st.markdown("### Recent Data")

        cols = ["timestamp","device_id","temperature","humidity","pressure","iaq","co2_ppm","voc_ppm"]
        available_cols = [c for c in cols if c in df.columns]
        display_df = df[available_cols].tail(100).copy()
        
        if 'timestamp' in display_df.columns:
            display_df = display_df.sort_values('timestamp', ascending=False)
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)

# ==================== RIGHT SIDEBAR ====================
with right_col:
    st.markdown("### Model Predictions")
    st.markdown("---")

    if not df.empty and len(df) >= 3:
        features, X_df = calculate_features(df)
        model = load_model()

        if features is not None:
            st.markdown("#### Input Features")
            st.write(f"Temp (t-1): {features['lag1_temp']:.2f}")
            st.write(f"Temp (t-2): {features['lag2_temp']:.2f}")
            st.write(f"Pressure (t-1): {features['lag1_pres']:.2f}")
            st.write(f"Humidity (t-1): {features['lag1_hum']:.2f}")
            st.write(f"IAQ (t-1): {features['lag1_iaq']:.0f}")
            st.write(f"Time Delta: {features['delta_t']:.2f} min")

            st.markdown("---")
            st.markdown("#### Next Reading")

            if model is None:
                st.warning("Model not loaded")
            else:
                try:
                    preds = model.predict(X_df)[0]
                    st.write(f"Temperature: {preds[0]:.2f} °C")
                    st.write(f"Pressure: {preds[1]:.2f}")
                    st.write(f"Humidity: {preds[2]:.2f} %")
                    st.write(f"IAQ: {preds[3]:.0f}")
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
        else:
            st.info("Insufficient data for features")
    else:
        st.info("Need at least 3 readings")

# ==================== FOOTER ====================
st.markdown("---")
st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if auto_refresh:
    time.sleep(2)
    st.rerun()
