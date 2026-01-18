"""
Environmental Sensor Monitoring Dashboard
Minimalist, professional interface for IoT sensor data visualization
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# Page configuration
st.set_page_config(
    page_title="Sensor Monitor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimalist CSS styling with visible text
st.markdown("""
    <style>
    /* Remove default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Clean typography */
    .main {
        padding-top: 3rem;
        padding-bottom: 2rem;
        background-color: #0e1117;
    }
    
    h1 {
        font-size: 1.75rem;
        font-weight: 500;
        color: #ffffff;
        letter-spacing: -0.02em;
        margin-bottom: 2rem;
    }
    
    h2 {
        font-size: 1.25rem;
        font-weight: 500;
        color: #ffffff;
        margin-top: 2.5rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        font-size: 1rem;
        font-weight: 500;
        color: #ffffff;
        margin-bottom: 0.75rem;
    }
    
    /* Metric styling - white text */
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
    
    [data-testid="stMetricDelta"] {
        font-size: 0.75rem;
        font-weight: 400;
        color: #ffffff !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* Clean dividers */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 1px solid #3a3a3a;
    }
    
    /* Table styling */
    .dataframe {
        font-size: 0.875rem;
        color: #ffffff;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #1a1a1a;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1.5rem;
        font-weight: 400;
        transition: background-color 0.2s;
    }
    
    .stButton>button:hover {
        background-color: #333;
    }
    
    /* Selectbox styling */
    .stSelectbox label {
        font-size: 0.875rem;
        font-weight: 500;
        color: #ffffff;
    }
    
    /* Caption styling */
    .stCaption {
        color: #ffffff !important;
    }
    
    /* Info box styling */
    [data-baseweb="notification"] {
        color: #ffffff;
    }
    
    /* Text elements */
    p, div, span {
        color: #ffffff;
    }
    
    /* Streamlit text elements */
    .stMarkdown {
        color: #ffffff;
    }
    
    .stText {
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# Load CSV data
@st.cache_data(ttl=5)  # Cache for 5 seconds to allow auto-refresh
def load_csv_data(csv_path='sensor_data.csv'):
    """Load sensor data from CSV file"""
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    
    try:
        # First, try reading normally
        df = pd.read_csv(csv_path)
        
        # Check if CSV was read as single column (common when entire rows are quoted)
        if len(df.columns) == 1:
            # Read the file and split manually
            with open(csv_path, 'r', encoding='utf-8-sig') as f:  # utf-8-sig handles BOM
                lines = f.readlines()
            
            if not lines:
                return pd.DataFrame()
            
            # Parse header - remove quotes and split by comma
            header_line = lines[0].strip().strip('"').strip('\ufeff')  # Remove BOM
            columns = [col.strip() for col in header_line.split(',')]
            
            # Parse data rows
            data_rows = []
            for line in lines[1:]:
                line = line.strip().strip('"')
                if line:
                    values = [val.strip() for val in line.split(',')]
                    if len(values) == len(columns):
                        data_rows.append(values)
            
            # Create DataFrame
            if data_rows:
                df = pd.DataFrame(data_rows, columns=columns)
            else:
                return pd.DataFrame()
        
        # Debug: show what columns we have
        if df.empty:
            return pd.DataFrame()
        
        # Convert timestamp to datetime if it exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Ensure numeric columns are properly typed
        numeric_cols = ['device_id', 'sequence', 'uptime', 'temperature', 'humidity', 
                       'pressure', 'iaq', 'iaq_accuracy', 'static_iaq', 'co2_ppm', 
                       'voc_ppm', 'gas_percent', 'rssi', 'snr']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert boolean columns - handle string 'true'/'false' and actual booleans
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
        import traceback
        st.error(traceback.format_exc())
        return pd.DataFrame()

# Sidebar
with st.sidebar:
    st.markdown("### Filters")
    
    # CSV file selector
    csv_path = st.text_input("CSV File Path", value="sensor_data.csv", help="Path to the CSV file containing sensor data")
    
    # Time range selector
    time_range = st.selectbox(
        "Time Range",
        ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days", "All Data", "Custom"],
        index=2,
        label_visibility="visible"
    )
    
    if time_range == "Custom":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start", value=datetime.now().date() - timedelta(days=1))
        with col2:
            start_time = st.time_input("", value=datetime.now().time())
        col3, col4 = st.columns(2)
        with col3:
            end_date = st.date_input("End", value=datetime.now().date())
        with col4:
            end_time = st.time_input(" ", value=datetime.now().time())
        start_datetime = datetime.combine(start_date, start_time)
        end_datetime = datetime.combine(end_date, end_time)
    elif time_range == "All Data":
        start_datetime = None
        end_datetime = None
    else:
        ranges = {
            "Last Hour": timedelta(hours=1),
            "Last 6 Hours": timedelta(hours=6),
            "Last 24 Hours": timedelta(hours=24),
            "Last 7 Days": timedelta(days=7)
        }
        start_datetime = datetime.now() - ranges.get(time_range, timedelta(hours=24))
        end_datetime = datetime.now()
    
    st.markdown("---")
    
    # Device selector
    df_all = load_csv_data(csv_path)
    if not df_all.empty and 'device_id' in df_all.columns:
        device_ids = sorted(df_all['device_id'].unique())
        device_options = [f"Device {did}" for did in device_ids]
        selected_device = st.selectbox("Device", ["All Devices"] + device_options)
        if selected_device != "All Devices":
            device_id = device_ids[device_options.index(selected_device)]
        else:
            device_id = None
    else:
        device_id = None
        st.caption("No devices found in data")

# Main content
st.title("Sensor Monitor")

# Load and filter data
df_all = load_csv_data(csv_path)

# Debug info (moved to after filtering)

if df_all.empty:
    st.info("No data available. Please ensure the CSV file exists and contains sensor data.")
    if os.path.exists(csv_path):
        st.warning(f"CSV file '{csv_path}' exists but appears to be empty or couldn't be parsed.")
else:
    # Filter by device if selected
    if device_id is not None:
        df = df_all[df_all['device_id'] == device_id].copy()
    else:
        df = df_all.copy()
    
    # Filter by time range if specified
    if start_datetime is not None and end_datetime is not None and 'timestamp' in df.columns:
        # Only filter if we have valid timestamps
        if df['timestamp'].notna().any():
            mask = (df['timestamp'] >= start_datetime) & (df['timestamp'] <= end_datetime)
            filtered_df = df[mask].copy()
            if filtered_df.empty:
                # If filtering removes all data, show all data instead
                st.warning(f"No data in selected time range. Showing all available data.")
            else:
                df = filtered_df
    
    # Sort by timestamp
    if 'timestamp' in df.columns and df['timestamp'].notna().any():
        df = df.sort_values('timestamp', ascending=True)
    elif not df.empty:
        # If no timestamp, sort by index
        df = df.sort_index(ascending=True)
    
    # Debug: Show data info
    if st.sidebar.checkbox("Show Debug Info", value=False):
        st.sidebar.write(f"CSV Path: {csv_path}")
        st.sidebar.write(f"File exists: {os.path.exists(csv_path)}")
        st.sidebar.write(f"Raw DataFrame shape: {df_all.shape}")
        st.sidebar.write(f"Filtered DataFrame shape: {df.shape}")
        st.sidebar.write(f"Columns: {list(df.columns) if not df.empty else 'None'}")
        if 'timestamp' in df.columns:
            st.sidebar.write(f"Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        if 'temperature' in df.columns:
            st.sidebar.write(f"Temperature values: {df['temperature'].tolist()}")
            st.sidebar.write(f"Temperature valid count: {df['temperature'].notna().sum()}")
    
    # Get latest reading
    if not df.empty:
        latest = df.iloc[-1]
        
        # Current status metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            temp = latest.get('temperature', None)
            st.metric("Temperature", f"{temp:.1f}°C" if pd.notna(temp) else "—")
        
        with col2:
            humidity = latest.get('humidity', None)
            st.metric("Humidity", f"{humidity:.1f}%" if pd.notna(humidity) else "—")
        
        with col3:
            pressure = latest.get('pressure', None)
            if pd.notna(pressure):
                # Pressure might be in bar (1.005) or Pa - convert to kPa
                if pressure < 10:  # Likely in bar
                    st.metric("Pressure", f"{pressure*100:.2f} kPa")
                else:  # Likely in Pa
                    st.metric("Pressure", f"{pressure/1000:.2f} kPa")
            else:
                st.metric("Pressure", "—")
        
        with col4:
            iaq = latest.get('iaq', None)
            iaq_label = latest.get('iaq_label', None)
            if pd.notna(iaq):
                st.metric("IAQ", f"{iaq:.0f}", delta=iaq_label.title() if pd.notna(iaq_label) else None)
            else:
                st.metric("IAQ", "—")
        
        with col5:
            co2 = latest.get('co2_ppm', None)
            st.metric("CO₂", f"{co2:.0f} ppm" if pd.notna(co2) else "—")
        
        with col6:
            voc = latest.get('voc_ppm', None)
            st.metric("VOC", f"{voc:.2f} ppm" if pd.notna(voc) else "—")
        
        st.markdown("---")
        
        # Charts section
        st.markdown("### Sensor Readings")
        
        # Temperature and Humidity
        col1, col2 = st.columns(2)
        
        with col1:
            if 'temperature' in df.columns and df['temperature'].notna().any():
                # Prepare x-axis data
                if 'timestamp' in df.columns and df['timestamp'].notna().any():
                    x_data = df['timestamp']
                else:
                    x_data = df.index
                
                # Filter out NaN values
                valid_mask = df['temperature'].notna()
                x_valid = x_data[valid_mask]
                y_valid = df['temperature'][valid_mask]
                
                fig_temp = go.Figure()
                if len(x_valid) > 0 and len(y_valid) > 0:
                    fig_temp.add_trace(go.Scatter(
                        x=x_valid,
                        y=y_valid,
                        mode='lines+markers',
                        name='Temperature',
                        line=dict(color='#ffffff', width=1.5),
                        marker=dict(size=4, color='#ffffff'),
                        hovertemplate='%{y:.1f}°C<extra></extra>'
                    ))
                    fig_temp.update_layout(
                        xaxis_title="",
                        yaxis_title="°C",
                        height=280,
                        margin=dict(l=40, r=20, t=10, b=40),
                        plot_bgcolor='#0e1117',
                        paper_bgcolor='#0e1117',
                        font=dict(size=11, color='#ffffff'),
                        xaxis=dict(showgrid=True, gridcolor='#3a3a3a', showline=True, linecolor='#3a3a3a'),
                        yaxis=dict(showgrid=True, gridcolor='#3a3a3a', showline=True, linecolor='#3a3a3a'),
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_temp, use_container_width=True, config=dict(displayModeBar=False))
                else:
                    st.info("No valid temperature data to display")
        
        with col2:
            if 'humidity' in df.columns and df['humidity'].notna().any():
                # Prepare x-axis data
                if 'timestamp' in df.columns and df['timestamp'].notna().any():
                    x_data = df['timestamp']
                else:
                    x_data = df.index
                
                # Filter out NaN values
                valid_mask = df['humidity'].notna()
                x_valid = x_data[valid_mask]
                y_valid = df['humidity'][valid_mask]
                
                fig_hum = go.Figure()
                if len(x_valid) > 0 and len(y_valid) > 0:
                    fig_hum.add_trace(go.Scatter(
                        x=x_valid,
                        y=y_valid,
                        mode='lines+markers',
                        name='Humidity',
                        line=dict(color='#ffffff', width=1.5),
                        marker=dict(size=4, color='#ffffff'),
                        hovertemplate='%{y:.1f}%<extra></extra>'
                    ))
                fig_hum.update_layout(
                    xaxis_title="",
                    yaxis_title="%",
                    height=280,
                    margin=dict(l=40, r=20, t=10, b=40),
                    plot_bgcolor='#0e1117',
                    paper_bgcolor='#0e1117',
                    font=dict(size=11, color='#ffffff'),
                    xaxis=dict(showgrid=True, gridcolor='#3a3a3a', showline=True, linecolor='#3a3a3a'),
                    yaxis=dict(showgrid=True, gridcolor='#3a3a3a', showline=True, linecolor='#3a3a3a'),
                    hovermode='x unified'
                )
                st.plotly_chart(fig_hum, use_container_width=True, config=dict(displayModeBar=False))
        
        # Pressure and IAQ
        col3, col4 = st.columns(2)
        
        with col3:
            if 'pressure' in df.columns and df['pressure'].notna().any():
                # Prepare x-axis data
                if 'timestamp' in df.columns and df['timestamp'].notna().any():
                    x_data = df['timestamp']
                else:
                    x_data = df.index
                
                # Filter out NaN values
                valid_mask = df['pressure'].notna()
                x_valid = x_data[valid_mask]
                pressure_values = df['pressure'][valid_mask].copy()
                
                # Convert to kPa if needed
                if pressure_values.max() < 10:  # Likely in bar
                    pressure_values = pressure_values * 100
                else:  # Likely in Pa
                    pressure_values = pressure_values / 1000
                
                fig_press = go.Figure()
                if len(x_valid) > 0 and len(pressure_values) > 0:
                    fig_press.add_trace(go.Scatter(
                        x=x_valid,
                        y=pressure_values,
                        mode='lines+markers',
                        name='Pressure',
                        line=dict(color='#ffffff', width=1.5),
                        marker=dict(size=4, color='#ffffff'),
                        hovertemplate='%{y:.2f} kPa<extra></extra>'
                    ))
                fig_press.update_layout(
                    xaxis_title="",
                    yaxis_title="kPa",
                    height=280,
                    margin=dict(l=40, r=20, t=10, b=40),
                    plot_bgcolor='#0e1117',
                    paper_bgcolor='#0e1117',
                    font=dict(size=11, color='#ffffff'),
                    xaxis=dict(showgrid=True, gridcolor='#3a3a3a', showline=True, linecolor='#3a3a3a'),
                    yaxis=dict(showgrid=True, gridcolor='#3a3a3a', showline=True, linecolor='#3a3a3a'),
                    hovermode='x unified'
                )
                st.plotly_chart(fig_press, use_container_width=True, config=dict(displayModeBar=False))
        
        with col4:
            if 'iaq' in df.columns and df['iaq'].notna().any():
                # Prepare x-axis data
                if 'timestamp' in df.columns and df['timestamp'].notna().any():
                    x_data = df['timestamp']
                else:
                    x_data = df.index
                
                # Filter out NaN values
                valid_mask = df['iaq'].notna()
                x_valid = x_data[valid_mask]
                y_valid = df['iaq'][valid_mask]
                
                fig_iaq = go.Figure()
                if len(x_valid) > 0 and len(y_valid) > 0:
                    fig_iaq.add_trace(go.Scatter(
                        x=x_valid,
                        y=y_valid,
                        mode='lines+markers',
                        name='IAQ',
                        line=dict(color='#ffffff', width=1.5),
                        marker=dict(size=4, color='#ffffff'),
                        hovertemplate='IAQ: %{y:.0f}<extra></extra>'
                    ))
                fig_iaq.update_layout(
                    xaxis_title="",
                    yaxis_title="IAQ",
                    height=280,
                    margin=dict(l=40, r=20, t=10, b=40),
                    plot_bgcolor='#0e1117',
                    paper_bgcolor='#0e1117',
                    font=dict(size=11, color='#ffffff'),
                    xaxis=dict(showgrid=True, gridcolor='#3a3a3a', showline=True, linecolor='#3a3a3a'),
                    yaxis=dict(showgrid=True, gridcolor='#3a3a3a', showline=True, linecolor='#3a3a3a'),
                    hovermode='x unified'
                )
                st.plotly_chart(fig_iaq, use_container_width=True, config=dict(displayModeBar=False))
        
        # CO2 and VOC
        col5, col6 = st.columns(2)
        
        with col5:
            if 'co2_ppm' in df.columns and df['co2_ppm'].notna().any():
                # Prepare x-axis data
                if 'timestamp' in df.columns and df['timestamp'].notna().any():
                    x_data = df['timestamp']
                else:
                    x_data = df.index
                
                # Filter out NaN values
                valid_mask = df['co2_ppm'].notna()
                x_valid = x_data[valid_mask]
                y_valid = df['co2_ppm'][valid_mask]
                
                fig_co2 = go.Figure()
                if len(x_valid) > 0 and len(y_valid) > 0:
                    fig_co2.add_trace(go.Scatter(
                        x=x_valid,
                        y=y_valid,
                        mode='lines+markers',
                        name='CO₂',
                        line=dict(color='#ffffff', width=1.5),
                        marker=dict(size=4, color='#ffffff'),
                        hovertemplate='CO₂: %{y:.0f} ppm<extra></extra>'
                    ))
                fig_co2.update_layout(
                    xaxis_title="",
                    yaxis_title="ppm",
                    height=280,
                    margin=dict(l=40, r=20, t=10, b=40),
                    plot_bgcolor='#0e1117',
                    paper_bgcolor='#0e1117',
                    font=dict(size=11, color='#ffffff'),
                    xaxis=dict(showgrid=True, gridcolor='#3a3a3a', showline=True, linecolor='#3a3a3a'),
                    yaxis=dict(showgrid=True, gridcolor='#3a3a3a', showline=True, linecolor='#3a3a3a'),
                    hovermode='x unified'
                )
                st.plotly_chart(fig_co2, use_container_width=True, config=dict(displayModeBar=False))
        
        with col6:
            if 'voc_ppm' in df.columns and df['voc_ppm'].notna().any():
                # Prepare x-axis data
                if 'timestamp' in df.columns and df['timestamp'].notna().any():
                    x_data = df['timestamp']
                else:
                    x_data = df.index
                
                # Filter out NaN values
                valid_mask = df['voc_ppm'].notna()
                x_valid = x_data[valid_mask]
                y_valid = df['voc_ppm'][valid_mask]
                
                fig_voc = go.Figure()
                if len(x_valid) > 0 and len(y_valid) > 0:
                    fig_voc.add_trace(go.Scatter(
                        x=x_valid,
                        y=y_valid,
                        mode='lines+markers',
                        name='VOC',
                        line=dict(color='#ffffff', width=1.5),
                        marker=dict(size=4, color='#ffffff'),
                        hovertemplate='VOC: %{y:.2f} ppm<extra></extra>'
                    ))
                fig_voc.update_layout(
                    xaxis_title="",
                    yaxis_title="ppm",
                    height=280,
                    margin=dict(l=40, r=20, t=10, b=40),
                    plot_bgcolor='#0e1117',
                    paper_bgcolor='#0e1117',
                    font=dict(size=11, color='#ffffff'),
                    xaxis=dict(showgrid=True, gridcolor='#3a3a3a', showline=True, linecolor='#3a3a3a'),
                    yaxis=dict(showgrid=True, gridcolor='#3a3a3a', showline=True, linecolor='#3a3a3a'),
                    hovermode='x unified'
                )
                st.plotly_chart(fig_voc, use_container_width=True, config=dict(displayModeBar=False))
        
        # Data table
        st.markdown("### Recent Data")
        display_cols = ['timestamp', 'device_id', 'temperature', 'humidity', 'pressure', 'iaq', 'iaq_label', 'co2_ppm', 'voc_ppm']
        available_cols = [col for col in display_cols if col in df.columns]
        
        display_df = df[available_cols].tail(100).copy()
        if 'timestamp' in display_df.columns:
            display_df = display_df.sort_values('timestamp', ascending=False)
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        
        # Format numeric columns
        for col in ['temperature', 'humidity', 'pressure', 'iaq', 'co2_ppm', 'voc_ppm']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        
        # Format pressure
        if 'pressure' in display_df.columns:
            def format_pressure(x):
                if x == "—":
                    return "—"
                val = float(x)
                if val < 10:  # Likely in bar
                    return f"{val*100:.2f}"
                else:  # Likely in Pa
                    return f"{val/1000:.2f}"
            display_df['pressure'] = display_df['pressure'].apply(format_pressure)
        
        st.dataframe(
            display_df,
            width='stretch',
            hide_index=True,
            height=400
        )

# Footer
st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
