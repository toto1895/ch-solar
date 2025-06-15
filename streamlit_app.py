import streamlit as st
import pandas as pd
import os
import numpy as np 
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import re
from google.oauth2 import service_account
from st_files_connection import FilesConnection
import gc
import io
import os

# ——— Page setup & state ———
st.set_page_config(
    page_title="Swiss Solar Dashboard",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "login"
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# ——— Auth Helpers ———
def user_obj():
    return getattr(st, "user", None)

def user_is_logged_in() -> bool:
    u = user_obj()
    return bool(getattr(u, "is_logged_in", False)) if u else False

def user_name() -> str:
    u = user_obj()
    return getattr(u, "name", "Guest") if u else "Guest"

def user_email() -> str:
    u = user_obj()
    return getattr(u, "email", "") if u else ""

# Set dark theme for the app
st.markdown(
    """
    <style>
    body {
        color: #fff;
        background-color: #1e1e1e;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


#import streamlit as st
from google.cloud import firestore
from google.cloud import logging as cloud_logging
import datetime
import hashlib
import json
import time
# Initialize Google Cloud services
def init_google_cloud():
    """Initialize Google Cloud Firestore and Logging clients"""
    try:
        st_time = time.time()
        # Initialize Firestore client
        project_id = 'gridalert-c48ee'
        db = firestore.Client(project=project_id)
        
        # Initialize Cloud Logging client
        logging_client = cloud_logging.Client(project=project_id)
        logging_client.setup_logging()

        print(time.time()-st_time)
        
        return db, logging_client
    except Exception as e:
        st.error(f"Failed to initialize Google Cloud services: {e}")
        return None, None

def get_user_info():
    """Extract comprehensive user information for logging"""
    user_info = {
        'email': user_name(),  # Your existing function
        'timestamp': datetime.datetime.utcnow(),
        'session_id': st.session_state.get('session_id', generate_session_id()),
        'user_agent': st.context.headers.get('User-Agent', 'Unknown') if hasattr(st, 'context') else 'Unknown',
        'ip_hash': hash_ip_address(),  # Privacy-friendly IP logging
    }
    return user_info

def generate_session_id():
    """Generate a unique session ID"""
    import uuid
    session_id = str(uuid.uuid4())
    st.session_state.session_id = session_id
    return session_id

def hash_ip_address():
    """Hash IP address for privacy compliance"""
    try:
        # Get IP from headers (works with most deployments)
        ip = st.context.headers.get('X-Forwarded-For', 
             st.context.headers.get('X-Real-IP', 'unknown')).split(',')[0].strip()
        
        # Hash the IP for privacy
        return hashlib.sha256(f"{ip}_salt_key".encode()).hexdigest()[:16]
    except:
        return "unknown"

def log_user_signin(db, logging_client, user_info):
    """Log user sign-in to both Firestore and Cloud Logging"""
    
    # 1. Store in Firestore for structured queries and analytics
    try:
        # Create user document in 'user_logins' collection
        doc_ref = db.collection('user_logins').document()
        
        # Prepare data for Firestore
        firestore_data = {
            'email': user_info['email'],
            'timestamp': user_info['timestamp'],
            'session_id': user_info['session_id'],
            'user_agent': user_info['user_agent'],
            'ip_hash': user_info['ip_hash'],
            'app_version': '1.0',  # Add your app version
            'platform': 'streamlit_web'
        }
        
        doc_ref.set(firestore_data)
        
        # Update user profile with last login
        user_profile_ref = db.collection('user_profiles').document(user_info['email'])
        user_profile_ref.set({
            'email': user_info['email'],
            'last_login': user_info['timestamp'],
            'total_logins': firestore.Increment(1),
            'last_session_id': user_info['session_id']
        }, merge=True)
        
    except Exception as e:
        st.error(f"Firestore logging failed: {e}")
    
    # 2. Log to Cloud Logging for monitoring and alerts
    try:
        import logging
        logger = logging.getLogger('solar_dashboard_auth')
        
        log_entry = {
            'event_type': 'user_signin_success',
            'user_email': user_info['email'],
            'session_id': user_info['session_id'],
            'timestamp': user_info['timestamp'].isoformat(),
            'ip_hash': user_info['ip_hash']
        }
        
        logger.info(f"User sign-in successful", extra={
            'json_fields': log_entry,
            'labels': {
                'component': 'authentication',
                'severity': 'INFO'
            }
        })
        
    except Exception as e:
        st.error(f"Cloud Logging failed: {e}")

def get_user_analytics(db, email):
    """Get user analytics from Firestore"""
    try:
        user_doc = db.collection('user_profiles').document(email).get()
        if user_doc.exists:
            data = user_doc.to_dict()
            return {
                'total_logins': data.get('total_logins', 1),
                'last_login': data.get('last_login'),
                'first_seen': data.get('first_seen', datetime.datetime.utcnow())
            }
    except Exception as e:
        st.error(f"Failed to get user analytics: {e}")
    return None

def login_page():
    st.markdown("### Secure Access Portal")
    
    # Initialize Google Cloud services
    db, logging_client = init_google_cloud()
    
    #col1, col2 = st.columns([1, 2])
    
    #with col1:
    st.markdown("""
    Welcome to the Swiss Solar Dashboard. This platform provides:
    
    - ☀️ **Real-time solar generation forecasts**
    - 📊 **Interactive data visualizations**
    - 🗺️ **Geographic power plant mapping**
    - 🌦️ **Weather forecast integration**
    
    Please log in to access the dashboard.
    """)
    
    st.info("Use the sidebar to authenticate with your Google account.")
    
    if user_is_logged_in():
        user_email = user_name()
        
        # Check if this login was already recorded in this session
        if not st.session_state.get('login_recorded', False):
            
            # Get user information
            user_info = get_user_info()
            
            # Log the successful sign-in
            if db and logging_client:
                log_user_signin(db, logging_client, user_info)
                
                # Get user analytics for display
                analytics = get_user_analytics(db, user_email)
                
                # Mark login as recorded for this session
                st.session_state.login_recorded = True
                
                # Display user analytics
                if analytics:
                    st.sidebar.success(f"Welcome back! Login #{analytics['total_logins']}")
                    if analytics['total_logins'] > 1:
                        st.sidebar.info(f"Last visit: {analytics['last_login'].strftime('%Y-%m-%d %H:%M UTC')}")
        
        st.success(f"✅ Logged in as: {user_email}")
        st.balloons()
        
        # Auto-redirect to home after successful login
        st.session_state.page = "home"
        st.rerun()



# ——— Original Functions (unchanged) ———
def get_connection():
    """Get the GCS connection instance"""
    return st.connection('gcs', type=FilesConnection)

def fetch_files(conn, prefix, pattern=None):
    """Fetch files from a bucket prefix with optional pattern matching"""
    try:
        conn._instance.invalidate_cache(prefix)
        files = conn._instance.ls(prefix, max_results=100)
        
        if pattern:
            regex = re.compile(pattern)
            files = [f for f in files if regex.search(f)]
            
        return sorted(files, reverse=True)
    except Exception as e:
        st.error(f"Error listing files: {e}")
        return []

def get_latest_parquet_file(conn):
    """Get the latest parquet file with format %Y-%m.parquet"""
    prefix = "oracle_predictions/swiss_solar/datasets/capa_timeseries"
    pattern = r'(\d{4}-\d{2})\.parquet$'
    
    files = fetch_files(conn, prefix, pattern)
    
    if not files:
        return None
    
    date_pattern = re.compile(r'(\d{4}-\d{2})\.parquet$')
    return sorted(files, key=lambda x: date_pattern.search(x).group(1), reverse=True)[0]

def load_data(file_path, input_format, conn):
    """Load data from a file using the connection"""
    try:
        return conn.read(file_path, input_format=input_format)
    except Exception as e:
        st.error(f"Error loading file {file_path}: {e}")
        return None

def get_forecast_files(model, cluster, conn):
    """Get list of available forecast files for the selected model and cluster"""
    if model in ['ICON-CH1','ICON-CH2']:
        prefix = f"icon-ch/ch{model.replace('ICON-CH','')}/ch-prod"
    else:
        prefix = f"oracle_predictions/swiss_solar/canton_forecasts_factor/{model}/{cluster}"
    return fetch_files(conn, prefix, r'\.parquet$'), conn

def load_and_concat_parquet_files(conn, date_str, time_str=None):
    """Load and concatenate parquet files from a specific date and optional time"""
    prefix = "dwd-solar-sat/daily_agg_asset_level_prod/"
    
    if time_str:
        if isinstance(time_str, list):
            time_patterns = '|'.join([f"{date_str}{t}" for t in time_str])
            pattern = f"({time_patterns})\.parquet$"
        else:
            pattern = f"{date_str}{time_str}\.parquet$"
    else:
        pattern = f"{date_str}\.parquet$"
    
    files = fetch_files(conn, prefix, pattern)
    
    if not files:
        return None
    
    dataframes = []
    for file_path in files:
        try:
            with conn._instance.open(file_path, mode='rb') as f:
                df = pd.read_parquet(io.BytesIO(f.read()))
                dataframes.append(df)
        except Exception as e:
            st.error(f"Error reading {file_path}: {e}")
    
    if not dataframes:
        st.error("No dataframes could be loaded successfully")
        return None
    
    concatenated_df = pd.concat(dataframes)
    return concatenated_df

def create_forecast_chart(filtered_df, pronovo_f, nowcast, filter_type, selected_cantons=None, selected_operators=None):
    """Create a forecast chart based on filtered data"""
    fig = go.Figure()
    plot_df = filtered_df.copy()
    
    if filter_type == "Canton" and selected_cantons:
        for canton in selected_cantons:
            total_df = plot_df[plot_df['Canton'] == canton]
            total_df = total_df.sort_values('datetime')

            canton_now = nowcast[nowcast['Canton'] == canton]
            canton_now = canton_now.sort_values('datetime')

            pronovo_now = pronovo_f[pronovo_f['Canton'] == canton]
            pronovo_now = pronovo_now.sort_values('datetime')
            
            total_df = total_df.groupby(['datetime']).agg({
                'p0.5_operator': 'sum',
                'p0.1_operator': 'sum',
                'p0.9_operator': 'sum'
            }).reset_index()

            canton_now = canton_now.groupby(['datetime']).agg({
                'SolarProduction':'sum'
            }).reset_index()

            pronovo_now = pronovo_now.groupby(['datetime']).agg({
                'Pronovo_f':'sum'
            }).reset_index()
            
    elif filter_type == "Operator" and 'operator' in filtered_df.columns and selected_operators:
        for operator in selected_operators:
            total_df = plot_df[plot_df['operator'] == operator]
            total_df = total_df.sort_values('datetime')

            canton_now = nowcast[nowcast['operator'] == operator]
            canton_now = canton_now.sort_values('datetime')

            pronovo_now = pronovo_f[pronovo_f['operator'] == operator]
            pronovo_now = pronovo_now.sort_values('datetime')

            canton_now = canton_now.groupby(['datetime']).agg({
                'SolarProduction':'sum'
            }).reset_index()

            pronovo_now = pronovo_now.groupby(['datetime']).agg({
                'Pronovo_f':'sum'
            }).reset_index()
            
            total_df = total_df.groupby(['datetime']).agg({
                'p0.5_operator': 'sum',
                'p0.1_operator': 'sum',
                'p0.9_operator': 'sum'
            }).reset_index()
    else:
        total_df = filtered_df.copy().sort_values('datetime')
        total_df = total_df.groupby(['datetime']).agg({
            'p0.5_operator': 'sum',
            'p0.1_operator': 'sum',
            'p0.9_operator': 'sum'
        }).reset_index()

        canton_now = nowcast.groupby(['datetime']).agg({
                'SolarProduction':'sum'
            }).reset_index()
        
        pronovo_now = pronovo_f.groupby(['datetime']).agg({
                'Pronovo_f':'sum'
            }).reset_index()

    try:
        multiple_selections = (selected_operators and len(selected_operators) > 1) or (selected_cantons and len(selected_cantons) > 1)
    except:
        multiple_selections = False
        
    if multiple_selections:
        if filter_type == "Canton" and selected_cantons:
            total_df = plot_df[plot_df['Canton'].isin(selected_cantons)]
        else:
            total_df = plot_df[plot_df['operator'].isin(selected_operators)]
            
        total_df = total_df.groupby(['datetime']).agg({
            'p0.5_operator': 'sum',
            'p0.1_operator': 'sum',
            'p0.9_operator': 'sum'
        }).reset_index()

        canton_now = nowcast.groupby(['datetime']).agg({
                'SolarProduction':'sum'
            })

    add_forecast_traces(fig, total_df, "Total", color='red')
    try:
        add_forecast_traces(fig, canton_now, "Nowcast", color='white')
    except:
        pass
    add_forecast_traces(fig, pronovo_now, "Pronovo", color='white')
    
    fig.update_layout(
        title="Solar Generation Forecast",
        xaxis_title="Date and Time",
        yaxis_title="Power (MW)",
        legend_title="Legend",
        template="plotly_dark",
        height=600,
        hovermode="x unified"
    )
    
    return fig

def add_forecast_traces(fig, df, name, line_width=2, color=None):
    """Add forecast traces to the figure"""
    line_style = dict(width=line_width)
    dash_style = dict(width=max(1, line_width-1), dash='dash')
    
    if color:
        line_style['color'] = color
        dash_style['color'] = color

    df['datetime']= pd.to_datetime(df['datetime'])

    try:
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['p0.5_operator'],
            mode='lines',
            name=f'{name} - Median (P50)',
            line=line_style
        ))
        
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['p0.1_operator'],
            mode='lines',
            name=f'{name} - Lower Bound (P10)',
            line=dash_style
        ))
        
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['p0.9_operator'],
            mode='lines',
            name=f'{name} - Upper Bound (P90)',
            line=dash_style
        ))
    except:
        try:
            df.set_index('datetime', inplace=True)
            df = df.asfreq('15min')
            df.reset_index(inplace=True)
            df.loc[:,'SolarProduction'] = df.loc[:,'SolarProduction'].fillna(0.0)

            fig.add_trace(go.Scatter(
                x=df['datetime'],
                y=df['SolarProduction'],
                mode='lines',
                name=f'{name} - Meteosat',
                line=line_style
            ))
        except:
            fig.add_trace(go.Scatter(
                x=df['datetime'],
                y=df['Pronovo_f'],
                mode='lines',
                name=f'{name} - Actual',
                line=dash_style
            ))

def create_heatmap(merged_plants):
    """Create a heatmap visualization for plant locations"""
    merged_plants['TotalPower_x'] = round(merged_plants['TotalPower_x']/1000,1)

    fig = px.density_map(
        merged_plants,
        lat="latitude",
        lon="longitude",
        z="TotalPower_x",
        hover_name="operator",
        hover_data={
            "Canton": True,
            "operator": True,
            "TotalPower_x": True,
        },
        color_continuous_scale="Jet",
        radius=15,
        zoom=8,
        title="Solar Power Plant Density",
        center={"lat": merged_plants['latitude'].mean(), "lon": merged_plants['longitude'].mean()},
        opacity=0.9
    )
    
    fig.update_layout(
        height=600,
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        coloraxis_colorbar=dict(
            title="Installed CAPA (MW)",
            tickformat=",.1f"
        )
    )
    
    return fig

# ——— Modified home_page with user info ———
def home_page():
    # Display user info in header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Swiss Solar Forecasts")
    with col2:
        st.markdown(f"**User:** {user_name()}")
        if user_email():
            st.caption(user_email())
    
    # Initialize connection
    conn = get_connection()

    # Define available models and clusters
    available_models = ["ICON-CH1","ICON-CH2"]
    available_clusters = ["cluster0", "cluster1", "cluster2"]
    
    # Create selection widgets in columns
    col1, col2 = st.columns(2)
    
    with col1:
        selected_model = st.selectbox(
            "Select Model:",
            options=available_models,
            index=0
        )
    
    if selected_model in ['ICON-CH1','ICON-CH2']:
        forecast_files, _ = get_forecast_files(selected_model, '', conn)
    else:
        with col2:
            selected_cluster = st.selectbox(
                "Select Cluster:",
                options=available_clusters,
                index=0
            )
        forecast_files, _ = get_forecast_files(selected_model, selected_cluster, conn)
    
    if not forecast_files:
        st.warning(f"No forecast files found for {selected_model}/{selected_cluster}")
        return
    
    selected_file = st.selectbox(
        "Select Forecast File:",
        options=forecast_files,
        index=0
    )
    
    latest_file = get_latest_parquet_file(conn)
    
    powerplants = load_data('oracle_predictions/swiss_solar/datasets/solar_mstr_data.csv', 'csv', conn)
    pronovo = load_data('oracle_predictions/swiss_solar/datasets/solar_part_0.csv', 'csv', conn)
    
    if powerplants is not None:
        powerplants = powerplants[['Canton', 'operator', 'longitude', 'latitude', 'TotalPower']]
    
    if not latest_file:
        st.warning("No capacity data files found")
        return
    
    with st.spinner("Downloading and processing capacity data..."):
        capa_df = load_data(latest_file, 'parquet', conn)
        
        if capa_df is None:
            st.error("Failed to load capacity data")
            return
            
        latest_mastr_date = capa_df.date.max()
        capa_df = capa_df.loc[capa_df.date == latest_mastr_date].drop(columns='date').reset_index(drop=True)
        
        st.warning(f"Master data latest update {latest_mastr_date.strftime('%Y-%m-%d')}")
        full_capa = load_data('oracle_predictions/swiss_solar/datasets/capa_timeseries/full_dataset.parquet', 'parquet', conn)
        
        with st.spinner(f"Downloading solar forecast data from {selected_file}..."):
            forecast_df = load_data(selected_file, 'parquet', conn)
            
            if forecast_df is None:
                st.error("Failed to load forecast data")
                return
                
            if selected_model == 'icon_d2':
                percentile_cols = ['p0.05', 'p0.1', 'p0.2', 'p0.3', 'p0.4', 'p0.5', 
                                'p0.6', 'p0.7', 'p0.8', 'p0.9', 'p0.95']
                max_idx = forecast_df.index.unique()[-1:]
                forecast_df = forecast_df.loc[forecast_df.index != max_idx[0]]
            
            try:
                merged_df = pd.merge(forecast_df.reset_index(), capa_df, on="Canton", how="left")
                merged_df.drop_duplicates(['datetime', 'Canton', 'operator'], inplace=True)
            except Exception as e:
                merged_df = pd.merge(forecast_df, capa_df, on=["Canton","operator"], how="left")
                merged_df.drop_duplicates(['datetime', 'Canton', 'operator'], inplace=True)

            dt = merged_df['datetime'].min().tz_convert('CET')
            
            h = []
            for ddt in pd.date_range(start=dt.strftime("%Y%m%d"),freq='D', periods=4):
                try:
                    nowcast = load_and_concat_parquet_files(conn, ddt.strftime("%Y%m%d"))
                    h.append(nowcast)
                except:
                    nowcast = pd.DataFrame(columns=['datetime','Canton','operator','SolarProduction'])
            nowcast = pd.concat(h)

            del forecast_df
            gc.collect()
            
            st.subheader("Filter Data")
            
            filter_col1, filter_col2 = st.columns([1, 3])
            
            with filter_col1:
                filter_type = st.selectbox(
                    "Filter by:",
                    options=["Canton", "Operator"],
                    index=0
                )
            
            selected_cantons = []
            selected_operators = []
            
            with filter_col2:
                filtered_df = merged_df.copy()
                
                if filter_type == "Canton":
                    all_cantons = sorted(merged_df["Canton"].unique().tolist())
                    
                    selected_cantons = st.multiselect(
                        "Select Cantons:",
                        options=all_cantons
                    )
                    
                    if selected_cantons:
                        filtered_df = merged_df[merged_df["Canton"].isin(selected_cantons)]
                        full_capa = full_capa[full_capa["Canton"].isin(selected_cantons)]
                        try:
                            nowcast = nowcast[nowcast["Canton"].isin(selected_cantons)]
                        except:
                            pass
                    
                elif filter_type == "Operator":
                    if 'operator' in merged_df.columns:
                        all_operators = sorted(merged_df["operator"].unique().tolist())
                        
                        selected_operators = st.multiselect(
                            "Select Operators:",
                            options=all_operators
                        )
                        
                        if selected_operators:
                            filtered_df = merged_df[merged_df["operator"].isin(selected_operators)]
                            
                            try:
                                nowcast = nowcast[nowcast["operator"].isin(selected_operators)]
                            except:
                                pass
                            
                            full_capa = full_capa[full_capa["operator"].isin(selected_operators)]
                    else:
                        st.warning("No 'operator' column found in the data. Please use Canton filtering instead.")
            
            del merged_df
            gc.collect()
            
            try:
                filtered_df = filtered_df[['datetime', 'p0.5', 'p0.1', 'p0.9', 'Canton', 'operator',
                                        'cum_canton', 'cum_operator','cum_ratio','year_month','TotalPower']]
            except:
                filtered_df = filtered_df[['datetime', 'SolarProduction', 'Canton', 'operator',
                                        'cum_canton', 'cum_operator','cum_ratio','year_month','TotalPower']]

            filtered_df.drop_duplicates(['datetime','Canton','operator'], inplace=True)

            try:
                nowcast.drop_duplicates(['datetime','Canton','operator'], inplace=True)
                nowcast['SolarProduction'] = 1.1*nowcast['SolarProduction']/1000.0
            except:
                nowcast = pd.DataFrame(columns=['datetime','Canton','operator','SolarProduction'])

            capa_installed =filtered_df.loc[filtered_df.datetime == filtered_df.datetime.max()
                                                   ].groupby('datetime')['cum_operator'].sum().values[0]
            st.success(f"Declared installed capacity: {round(capa_installed/1000):,.0f} MW  ( Today ~{1.15*round(capa_installed/1000):,.0f} MW) ")
            
            pronovo_long = pd.melt(
                pronovo,
                id_vars=['datetime'],
                value_vars=pronovo.drop(columns='datetime').columns,
                var_name='Canton',
                value_name='Pronovo'
            )
            pronovo_long['datetime'] = pd.to_datetime(pronovo_long['datetime'])
            pronovo_long = pronovo_long.sort_values('datetime').reset_index(drop=True)
            
            try:
                filtered_df['p0.5_canton'] = 1.1*filtered_df['p0.5'] * filtered_df['cum_canton'] / 1000
                filtered_df['p0.1_canton'] = 1.1*filtered_df['p0.1'] * filtered_df['cum_canton'] / 1000
                filtered_df['p0.9_canton'] = 1.1*filtered_df['p0.9'] * filtered_df['cum_canton'] / 1000
                
                filtered_df['p0.5_operator'] = 1.1*filtered_df['p0.5'] * filtered_df['cum_operator'] / 1000
                filtered_df['p0.1_operator'] = 1.1*filtered_df['p0.1'] * filtered_df['cum_operator'] / 1000
                filtered_df['p0.9_operator'] = 1.1*filtered_df['p0.9'] * filtered_df['cum_operator'] / 1000
            except:
                filtered_df['p0.5_canton'] = 1.05*filtered_df['SolarProduction'] / 1000
                filtered_df['p0.1_canton'] = np.nan 
                filtered_df['p0.9_canton'] = np.nan

                filtered_df['p0.5_operator'] = 1.05*filtered_df['SolarProduction'] / 1000
                filtered_df['p0.1_operator'] = np.nan 
                filtered_df['p0.9_operator'] = np.nan 

                filtered_df.loc[:,'datetime'] = filtered_df.loc[:,'datetime'] - pd.Timedelta(minutes=45) 
            
            pronovo_long['datetime'] = pd.to_datetime(pronovo_long['datetime'])
            filtered_df['datetime'] = pd.to_datetime(filtered_df['datetime'])
            
            pronovo_f = pd.merge(pronovo_long,filtered_df, on=['datetime',"Canton"], how="left")
            pronovo_f['Pronovo_f'] = 2 * pronovo_f['cum_ratio'] * pronovo_f['Pronovo'] 
            pronovo_f = pronovo_f.loc[pronovo_f.datetime>=filtered_df['datetime'].min(),:]
            
            if filter_type == "Canton" and selected_cantons:
                pronovo_f = pronovo_f[pronovo_f["Canton"].isin(selected_cantons)]
            elif filter_type == "Operator" and selected_operators:
                pronovo_f = pronovo_f[pronovo_f["operator"].isin(selected_operators)]

            chart_type = st.radio(
                "Select visualization type:",
                options=["Forecast Chart", 'Monthly installed capacity',"Powerplant Location Heatmap"],
                horizontal=True
            )
            
            if chart_type == "Forecast Chart":
                fig = create_forecast_chart(filtered_df,pronovo_f,nowcast, filter_type, selected_cantons, selected_operators)
                st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type =='Monthly installed capacity':
                full_capa = full_capa.groupby('year_month')['TotalPower'].sum()

                st.subheader('Monthly added capacity [MW]')
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=full_capa.index,
                    y=full_capa.values/1000,
                    name='Added Cap a'
                ))
                st.plotly_chart(fig, use_container_width=True)

            else:  # Powerplant Location Heatmap
                if powerplants is None:
                    st.error("Powerplant data is not available for the heatmap visualization")
                    return
                    
                latest_datetime = filtered_df['datetime'].max()
                latest_forecast = filtered_df[filtered_df['datetime'] == latest_datetime].copy()
                
                merge_conditions = ["Canton", "operator"]
                merged_plants = pd.merge(powerplants, latest_forecast, on=merge_conditions, how="inner")
                
                if filter_type == "Canton" and selected_cantons:
                    merged_plants = merged_plants[merged_plants['Canton'].isin(selected_cantons)]
                elif filter_type == "Operator" and selected_operators:
                    merged_plants = merged_plants[merged_plants['operator'].isin(selected_operators)]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Plants", f"{len(merged_plants):,}")
                with col2:
                    st.metric("Total Capacity", f"{merged_plants['TotalPower_x'].sum()/1000:,.2f} MW")
                
                fig = create_heatmap(merged_plants)
                st.plotly_chart(fig, use_container_width=True)

def about_page():
    st.title("About Swiss Solar Dashboard")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### About This Platform
        
        The Swiss Solar Dashboard provides comprehensive insights into solar energy generation across Switzerland. 
        This platform combines real-time data, advanced forecasting models, and interactive visualizations to help 
        users understand and analyze solar power production patterns.
        
        #### Key Features:
        - **Real-time Forecasting**: Access to multiple weather prediction models including ICON-CH1 and ICON-CH2
        - **Interactive Visualizations**: Dynamic charts and geographic heatmaps for data exploration
        - **Canton & Operator Filtering**: Detailed analysis by region or energy provider
        - **Weather Integration**: Near real-time satellite imagery and weather forecasts
        
        #### Data Sources:
        - Solar production data from Swiss energy operators
        - Weather forecasts from MeteoSwiss ICON models
        - Satellite imagery from Meteosat
        - Power plant registry data
        """)
    
    with col2:
        st.info("""
        **Current User:**  
        {0}  
        {1}
        """.format(user_name(), user_email() if user_email() else ""))
    
    st.markdown("""
    ### Contact
    For more information or support, please contact aminedev1895@gmail.com.
    
    ---
    
    **Version:** 1.0.0  
    **Last Updated:** January 2025
    """)

# Import animation functions (assuming these exist in your project)
from satAnimation import generate_sat_rad_anim
from satAnimation_icon import display_png_ch1, display_png_ch2

def sat_anim():
    fig_anim = generate_sat_rad_anim()
    st.plotly_chart(fig_anim, use_container_width=True, theme=None)

# ——— Main function with authentication ———
def main():
    # Sidebar navigation
    st.sidebar.title("☀️ Swiss Solar Dashboard")
    
    # Authentication section
    st.sidebar.markdown("### 🔐 Authentication")
    
    if user_is_logged_in():
        st.sidebar.success(f"Logged in as: {user_name()}")
        if st.sidebar.button("🚪 Logout"):
            st.logout()
            st.session_state.page = "login"
            st.rerun()
    else:
        st.sidebar.info("Please log in to access the dashboard")
        if st.sidebar.button("🔑 Login with Google"):
            st.login("google")
            st.rerun()
    
    # Show login page if not authenticated
    if not user_is_logged_in():
        login_page()
        return
        
    

    # Navigation menu (only shown when logged in)
    #st.sidebar.markdown("### 📊 Navigation")
    
    page_choice = st.sidebar.radio("Select Page:", [
        "Home",
        "Weather Near-Realtime (MeteoSat 5km)",
        "Weather Forecast (ICON-CH1 1km)",
        "Weather Forecast (ICON-CH2 2.1km)",
        "About"
    ])
    
    # Page routing
    if page_choice == "Home":
        home_page()
    elif page_choice == 'Weather Near-Realtime (MeteoSat 5km)':
        sat_anim()
    elif page_choice == "Weather Forecast (ICON-CH1 1km)":
        selected = st.selectbox(
            "Weather parameter:",
            options=['solar','precipitation','cloud','temperature'],
            index=0
        )
        with st.spinner("Downloading ..."):
            display_png_ch1(selected)
    elif page_choice == "Weather Forecast (ICON-CH2 2.1km)":
        selected = st.selectbox(
            "Weather parameter:",
            options=['solar','precipitation','cloud','temperature'],
            index=0
        )
        with st.spinner("Downloading ..."):
            display_png_ch2(selected)


if __name__ == "__main__":
    main()