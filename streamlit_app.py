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
import json
from pathlib import Path

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

import socket
def _is_private_ip(ip: str) -> bool:
    """Check if an IP address is private/local"""
    private_ranges = [
        '10.',
        '172.16.', '172.17.', '172.18.', '172.19.',
        '172.20.', '172.21.', '172.22.', '172.23.',
        '172.24.', '172.25.', '172.26.', '172.27.',
        '172.28.', '172.29.', '172.30.', '172.31.',
        '192.168.',
        '127.'
    ]
    return any(ip.startswith(range) for range in private_ranges)

def get_user_ip() -> str:
    """
    Get user's IP address with multiple fallback methods.
    Tries multiple approaches to ensure IP detection works in various environments.
    
    Returns:
        str: The user's IP address, or "Unknown" if unable to determine
    """
    
    # Method 1: Try Streamlit's context headers (current API)
    try:
        headers = st.context.headers
        if headers:
            print(f"Headers found: {dict(headers)}")  # Debug: print all headers
            
            # Check common forwarded IP headers (in order of preference)
            forwarded_headers = [
                'x-forwarded-for',
                'x-real-ip',
                'cf-connecting-ip',  # Cloudflare
                'x-client-ip',
                'true-client-ip',    # Cloudflare Enterprise
                'x-cluster-client-ip'
            ]
            
            for header in forwarded_headers:
                ip_value = headers.get(header)
                if ip_value:
                    # X-Forwarded-For can contain multiple IPs, take the first
                    print(f"Found header {header}: {ip_value}")
                    ip = ip_value.split(',')[0].strip()
                    if not _is_private_ip(ip):
                        return ip
        else:
            print("No headers found in st.context")
    except Exception as e:
        print(f"Error in method 1: {e}")
    
    # Method 2: Try alternative Streamlit session approach
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        
        if ctx and hasattr(ctx, 'session_id'):
            print(f"Found session context with ID: {ctx.session_id}")
            
            # Try to get session info
            from streamlit.runtime import get_instance
            runtime = get_instance()
            if runtime:
                session_info = runtime._session_mgr.get_session_info(ctx.session_id)
                if session_info and hasattr(session_info, 'client'):
                    client = session_info.client
                    if hasattr(client, 'request'):
                        request = client.request
                        if hasattr(request, 'headers'):
                            print(f"Found request headers: {dict(request.headers)}")
                            for header in ['x-forwarded-for', 'x-real-ip']:
                                if header in request.headers:
                                    ip = request.headers[header].split(',')[0].strip()
                                    if not _is_private_ip(ip):
                                        return ip
    except Exception as e:
        print(f"Error in method 2: {e}")

    # Method 3: Socket approach (usually returns local IP)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        print(f"Socket method returned: {local_ip}")
        # This will almost always be a private IP
    except Exception as e:
        print(f"Error in socket method: {e}")
    
    return "Unknown"


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

# ——— User Logging Functions ———
def log_user_signin_simple(user_email):
    """Simple, fast user login tracking"""
    try:
        # Create logs directory if it doesn't exist
        logs_dir = Path("user_logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Log file path
        log_file = logs_dir / "user_logins.jsonl"
        
        # Create login record
        login_record = {
            "timestamp": pd.Timestamp.now('UTC').isoformat(),
            "email": user_email,
            "ip_addresses": get_user_ip(),
            "session_id": st.session_state.get('session_id', 'unknown'),
            "date": pd.Timestamp.now('UTC').date().isoformat()
        }
        
        # Append to log file (JSONL format - one JSON per line)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(login_record) + "\n")
        
        return True
        
    except Exception as e:
        st.error(f"Logging failed: {e}")
        return False

def get_user_stats(user_email, ip_address=None):
    """Get user login statistics including IP address tracking"""
    try:
        log_file = Path("user_logs/user_logins.jsonl")
        if not log_file.exists():
            return {
                "total_logins": 1, 
                "first_login": True,
                "ip_addresses": [ip_address] if ip_address else [],
                "unique_ips": 1 if ip_address else 0
            }
        
        total_logins = 0
        ip_addresses = set()
        
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    if record.get("email") == user_email:
                        total_logins += 1
                        # Collect IP addresses from previous logins
                        if "ip_address" in record:
                            ip_addresses.add(record["ip_address"])
                except:
                    continue
        
        # Add current IP if provided
        if ip_address:
            ip_addresses.add(ip_address)
        
        return {
            "total_logins": total_logins + 1,  # +1 for current login
            "first_login": total_logins == 0,
            "ip_addresses": list(ip_addresses),
            "unique_ips": len(ip_addresses)
        }
    except:
        return {
            "total_logins": 1, 
            "first_login": True,
            "ip_addresses": [ip_address] if ip_address else [],
            "unique_ips": 1 if ip_address else 0
        }


from google.cloud import storage

def upload_logs_to_gcs():
    """Upload local logs to Google Cloud Storage"""
    print("=== Starting upload_logs_to_gcs function ===")
    
    try:
        import json
        import pandas as pd
        from pathlib import Path
        from google.cloud import storage
        from google.oauth2 import service_account
        import streamlit as st
        
        print("Imports successful")
        
        # Get bucket name
        bucket_name = "ch-solar-dash-logs"
        print(f"Bucket name: {bucket_name}")
 
        # Check if log file exists
        log_file = Path("user_logs/user_logins.jsonl")
        print(f"Checking for log file: {log_file}")
        print(f"Log file exists: {log_file.exists()}")
        print(f"Current working directory: {Path.cwd()}")
        
        if not log_file.exists():
            print('pas de logs à uploader')
            st.warning("No log file found to upload")
            return
            
        print(f"Log file size: {log_file.stat().st_size} bytes")
            
        # Create blob name with date structure
        blob_name = f"user_logins/{pd.Timestamp.now('UTC').strftime('%Y_%m_%d')}/logins.jsonl"
        print(f"Blob name: {blob_name}")
        
        print("Getting service account credentials...")
        service_account_json = st.secrets.secrets.service_account_json
        #service_account_json = service_account_json.replace('\\n', '\n')

        service_account_info = json.loads(service_account_json)
        import os
        #os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_info
        print("Service account info parsed successfully")

        # Create credentials object properly
        credentials = service_account.Credentials.from_service_account_info(service_account_info)
        print("Credentials created successfully")

        # Upload using Google Cloud Storage client
        print("Creating GCS client...")
        client = storage.Client(project=st.secrets.get("GOOGLE_CLOUD_PROJECT_ID"),
                               credentials=credentials)
        
        print("Getting bucket...")
        bucket = client.bucket(bucket_name)
        
        print("Creating blob...")
        blob = bucket.blob(blob_name)
        blob.content_type = 'application/jsonl'
        
        print("Starting upload...")
        blob.upload_from_filename(str(log_file))
        
        success_msg = f"Successfully uploaded to gs://{bucket_name}/{blob_name}"
        print(success_msg)
        #st.success(success_msg)
        
    except Exception as e:
        error_msg = f"Cloud upload failed: {str(e)}"
        print(error_msg)
        st.error(error_msg)
        # Print the full traceback for debugging
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())


def upload_df_to_gcs(df, filename="uuid.csv"):
    try:
        import json
        import pandas as pd
        from io import StringIO
        from google.cloud import storage
        from google.oauth2 import service_account
        import streamlit as st

        # Get bucket name
        bucket_name = "solar-api-user"
        print(f"Bucket name: {bucket_name}")
        
        blob_name = filename
        print(f"Blob name: {blob_name}")
        
        print("Getting service account credentials...")
        service_account_json = st.secrets.secrets.service_account_json
        
        service_account_info = json.loads(service_account_json)
        print("Service account info parsed successfully")

        # Create credentials object
        credentials = service_account.Credentials.from_service_account_info(service_account_info)
        print("Credentials created successfully")

        # Upload using Google Cloud Storage client
        print("Creating GCS client...")
        client = storage.Client(project=st.secrets.get("GOOGLE_CLOUD_PROJECT_ID"),
                               credentials=credentials)
        
        print("Getting bucket...")
        bucket = client.bucket(bucket_name)
        
        print("Creating blob...")
        blob = bucket.blob(blob_name)
        blob.content_type = 'text/csv'
        
        print("Converting DataFrame to CSV...")
        # Convert DataFrame to CSV string
        csv_string = df.to_csv(index=False)
        
        print("Starting upload...")
        # Upload CSV string directly
        blob.upload_from_string(csv_string, content_type='text/csv')
        
        success_msg = f"Successfully uploaded DataFrame to gs://{bucket_name}/{blob_name}"
        print(success_msg)
        #st.success(success_msg)
        
    except Exception as e:
        error_msg = f"Cloud upload failed: {str(e)}"
        print(error_msg)
        st.error(error_msg)
        # Print the full traceback for debugging
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())



def download_logs_from_gcs(date_filter=None):
    """Download logs from Google Cloud Storage
    
    Args:
        date_filter (str, optional): Date in format 'YYYY/MM/DD' to download specific date.
                                   If None, downloads the most recent file.
    """
    try:
        import json
        import pandas as pd
        from pathlib import Path
        from google.cloud import storage
        from google.oauth2 import service_account
        import streamlit as st
        
        print("=== Starting download_logs_from_gcs function ===")
        
        # Get bucket name
        bucket_name = "ch-solar-dash-logs"
        
        # Setup credentials
        service_account_json = st.secrets.secrets.service_account_json
        service_account_info = json.loads(service_account_json)
        credentials = service_account.Credentials.from_service_account_info(service_account_info)
        
        # Create GCS client
        client = storage.Client(project="gridalert-c48ee", credentials=credentials)
        bucket = client.bucket(bucket_name)
        
        if date_filter:
            # Download specific date
            blob_name = f"user_logins/{date_filter}/logins.jsonl"
            blob = bucket.blob(blob_name)
            
            if not blob.exists():
                st.error(f"No logs found for date {date_filter}")
                return None
                
            print(f"Downloading: {blob_name}")
            
        else:
            # Find the most recent file
            print("Finding most recent log file...")
            blobs = list(client.list_blobs(bucket_name, prefix="user_logins/"))
            
            if not blobs:
                st.error("No log files found in bucket")
                return None
            
            # Sort by creation time to get the most recent
            blobs.sort(key=lambda x: x.time_created, reverse=True)
            blob = blobs[0]
            print(f"Most recent file: {blob.name}")
        
        # Create local directory if it doesn't exist
        local_dir = Path("downloaded_logs")
        local_dir = Path("user_logs")
        local_dir.mkdir(exist_ok=True)
        
        # Create local filename with timestamp
        if date_filter:
            local_filename = f"logins_{date_filter.replace('/', '_')}.jsonl"
        else:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            local_filename = f"logins_{timestamp}.jsonl"
            local_filename = 'user_logins.jsonl'  # Always save as user_logins.jsonl for consistency

        local_path = local_dir / local_filename
        
        # Download the file
        print(f"Downloading to: {local_path}")
        blob.download_to_filename(str(local_path))
        
        success_msg = f"Successfully downloaded to {local_path}"
        print(success_msg)
        st.success(success_msg)
        
        # Return the local path for further processing if needed
        return str(local_path)
        
    except Exception as e:
        error_msg = f"Download failed: {str(e)}"
        print(error_msg)
        st.error(error_msg)
        import traceback
        print(traceback.format_exc())
        return None

def list_available_log_dates():
    """List all available log dates in GCS bucket"""
    try:
        import json
        from google.cloud import storage
        from google.oauth2 import service_account
        import streamlit as st
        
        # Setup credentials
        service_account_json = st.secrets["service_account_json"]
        service_account_json = service_account_json.replace('\\n', '\n')
        service_account_info = json.loads(service_account_json)
        credentials = service_account.Credentials.from_service_account_info(service_account_info)
        
        # Create GCS client
        client = storage.Client(project="gridalert-c48ee", credentials=credentials)
        bucket_name = "ch-solar-dash-logs"
        
        # List all blobs with the user_logins prefix
        blobs = client.list_blobs(bucket_name, prefix="user_logins/")
        
        # Extract dates from blob names
        dates = set()
        for blob in blobs:
            # Extract date from path like "user_logins/2025/06/15/logins.jsonl"
            parts = blob.name.split('/')
            if len(parts) >= 4:
                date_str = f"{parts[1]}/{parts[2]}/{parts[3]}"
                dates.add(date_str)
        
        return sorted(list(dates), reverse=True)  # Most recent first
        
    except Exception as e:
        print(f"Failed to list dates: {e}")
        st.error(f"Failed to list available dates: {e}")
        return []

def show_login_analytics():
    """Simple analytics from local log files"""
    st.title("📊 User Login Analytics")

    #st.info(get_user_ip())

    download_logs_from_gcs()
    
    log_file = Path("user_logs/user_logins.jsonl")
    
    if not log_file.exists():
        st.info("No login data available yet.")
        return
    
    
    # Read all login records
    records = []
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    records.append(record)
                except:
                    continue
    except Exception as e:
        st.error(f"Error reading log file: {e}")
        return
    
    if not records:
        st.info("No valid login records found.")
        return
    
    # Basic statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_logins = len(records)
        st.metric("Total Logins", total_logins)
    
    with col2:
        unique_users = len(set(r['email'] for r in records))
        st.metric("Unique Users", unique_users)
    
    with col3:
        today = pd.Timestamp.now('UTC').date().isoformat()
        today_logins = len([r for r in records if r.get('date') == today])
        st.metric("Logins Today", today_logins)
    
    # Recent logins table
    st.subheader("Recent Logins")
    
    # Show last 20 logins
    recent_records = records[:]
    recent_records.reverse() # Show newest first
    
    if recent_records:
        df = pd.DataFrame(recent_records)
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'],format='ISO8601').dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            st.error(f"Error formatting timestamps: {e}")
        
        df = df.loc[df.email !='aminedev1895@gmail.com',:]
        df = df.loc[df.email !='amineweibel01@gmail.com',:]
        # Display table
        st.dataframe(
            df[['timestamp', 'email','ip_addresses']].rename(columns={
                'timestamp': 'Login Time',
                'email': 'User Email'
            }),
            use_container_width=True
        )
        # Download option
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download Full Log CSV",
            csv,
            f"user_logins_{pd.Timestamp.now('UTC').strftime('%Y%m%d')}.csv",
            "text/csv"
        )
        if st.button("Upload Logs to Cloud"):
            upload_logs_to_gcs()
        
        if st.button("← Back to Dashboard"):
            st.session_state.page = "home"
            #st.rerun()
            return

def login_page():
    st.markdown("### Secure Access Portal")
    
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
        user_email_str = user_name()
        
        # Show immediate login success
        st.success(f"✅ Logged in as: {user_email_str}")
        
        # Only log once per session
        if not st.session_state.get('login_logged', False):
            
            # Simple, fast logging
            try:
                if log_user_signin_simple(user_email_str):
                    st.session_state.login_logged = True
                    
                    print(get_user_ip())
                    # Get user stats
                    ip=get_user_ip()
                    stats = get_user_stats(user_email_str, ip)
                    
                    # Show welcome message
                    if stats["first_login"]:
                        st.sidebar.success("🎉 Welcome to Solar Dashboard!")
                    else:
                        st.sidebar.success(f"Welcome back! Visit #{stats['total_logins']}")
                    
                    # Upload to cloud in background (optional)
                    #if st.secrets.get("GOOGLE_CLOUD_PROJECT_ID"):
                    upload_logs_to_gcs()
                    
                    #st.balloons()
                    st.info("🔄 Redirecting to dashboard...")
                    # Small delay to ensure logging completes

            except Exception as e:
                st.error(f"Logging error: {e}")
        import time
        time.sleep(2)
                    
        # Auto-redirect to home after successful login
        if st.button("Continue to Dashboard") or st.session_state.get('login_logged', False):
            st.session_state.page = "home"
            import time
            time.sleep(2)
            st.rerun()

# ——— Original Functions (unchanged) ———
def get_connection():
    """Get the GCS connection instance"""
    return st.connection('gcs', type=FilesConnection)

def read_parquet_gcs(path: str, columns=None, ttl=600):
    conn = get_connection()
    return conn.read(path, input_format="parquet", ttl=ttl, columns=columns)


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

def get_latest_parquet_file(conn, prefix = "oracle_predictions/swiss_solar/datasets/capa_timeseries",pattern = r'(\d{4}-\d{2})\.parquet$'):
    """Get the latest parquet file with format %Y-%m.parquet"""
    #prefix = "oracle_predictions/swiss_solar/datasets/capa_timeseries"
    #pattern = r'(\d{4}-\d{2})\.parquet$'
    try:
        files = fetch_files(conn, prefix, pattern)
    except:
        st.error("Error fetching files from GCS")
        return None

    if not files:
        return None
    
    date_pattern = re.compile(pattern)
    matching_files = [f for f in files if date_pattern.search(f)]

    return sorted(matching_files, key=lambda x: date_pattern.search(x).group(1), reverse=True)[0]

def load_data(file_path, input_format, conn):
    """Load data from a file using the connection"""
    try:
        return conn.read(file_path, input_format=input_format)
    except Exception as e:
        st.error(f"Error loading file {file_path}: {e}")
        return None

def get_forecast_files(model, cluster, conn):
    """Get list of available forecast files for the selected model and cluster"""
    if model in ["ICON-d2-ruc","FastCloudML-001",'ICON-CH1','ICON-CH2','FastCloud']:
        prefix = f"icon-ch/ch{model.replace('ICON-CH','')}/ch-prod"
        if model == "ICON-d2-ruc":
            prefix = f"icon-d2-ruc/ch-prod"
        if model =='FastCloud':
            prefix = f"icon-ch/cloud-rad/ch-prod"
        elif model=='FastCloudML-001':
            prefix = f"icon-ch/fastcloudml001/ch-prod"
    else:
        prefix = f"oracle_predictions/swiss_solar/canton_forecasts_factor/{model}/{cluster}"
    return fetch_files(conn, prefix, r'\.parquet$'), conn

def load_and_concat_parquet_files(conn, date_str, time_str=None, prefix = "dwd-solar-sat/daily_agg_asset_level_prod/", pattern=None):
    """Load and concatenate parquet files from a specific date and optional time"""
    #prefix = "dwd-solar-sat/daily_agg_asset_level_prod/"
    
    if pattern is None:
        if time_str:
            if isinstance(time_str, list):
                time_patterns = '|'.join([f"{date_str}{t}" for t in time_str])
                pattern = rf"({time_patterns})\.parquet$"
            else:
                pattern = rf"{date_str}{time_str}\.parquet$"
        else:
            pattern = rf"{date_str}\.parquet$"
    else:
        pattern = rf"{pattern}_{date_str}\.parquet$"
    
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

def create_forecast_chart(selected_model,filtered_df, pronovo_f, nowcast, stationprod, filter_type, selected_cantons=None, selected_operators=None):
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

    total_df['datetime'] = pd.to_datetime(total_df['datetime'], utc=True)
    canton_now['datetime'] = pd.to_datetime(canton_now['datetime'], utc=True)

    if selected_model in ['ICON-CH1','ICON-CH2']:
        total_df['datetime'] = pd.to_datetime(total_df['datetime']) - pd.Timedelta(hours=1)


    total_df.loc[total_df['datetime'].dt.hour == 3, total_df.columns != 'datetime'] = 0
    add_forecast_traces(selected_model,fig, total_df.round(1), "Total", color='red')
    try:
        add_forecast_traces(selected_model,fig, canton_now.round(1), "Nowcast", color='white')
    except:
        pass

    try:
        r = stationprod.sum(axis=1).to_frame('Solar')
        #r.index = pd.to_datetime(r.index) + pd.Timedelta(minutes=5)
        r.index = pd.to_datetime(r.index,utc=True).tz_convert('CET')
        #r = r.tz_convert('UTC')
        r = r.resample('15min').mean()
        #r['datetime'] = r.index - pd.Timedelta(minutes=15)
        r['datetime'] = r.index 
        add_forecast_traces(selected_model,fig, r.round(1), "Nowcast", color='green',line_width=2)
    except Exception as e:
        st.write(e)

    try:
        add_forecast_traces(selected_model,fig, pronovo_now, "Pronovo", color='white')
    except Exception as e:
        pass
    
    now = pd.Timestamp.now('utc')
    if (r['datetime'].max() - now) <= pd.Timedelta(hours=36) and \
            (now - r['datetime'].min()) <= pd.Timedelta(hours=36):
        fig.add_vline(
            x=now,
            line=dict(color="grey", dash="dash"),
            #annotation_text="",
            #annotation_position="top"
        )

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

def add_forecast_traces(selected_model,fig, df, name, line_width=2, color=None):
    """Add forecast traces to the figure"""
    line_style = dict(width=line_width)
    dash_style = dict(width=max(1, line_width-1), dash='dash')
    
    if color:
        line_style['color'] = color
        dash_style['color'] = color

    try:
        df.loc[:,'datetime']= df['datetime'].dt.tz_convert('CET')
    except Exception as e:
        print(e)
        df.loc[:,'datetime']= pd.to_datetime(df['datetime'],utc=True).dt.tz_convert('CET')

    try:
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['p0.5_operator'],
            mode='lines',
            name=f'{name} forecast',
            line=line_style
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
            try:
                fig.add_trace(go.Scatter(
                    x=df['datetime'],
                    y=df['Pronovo_f'],
                    mode='lines',
                    name=f'{name} - Actual',
                    line=dash_style
                ))
            except:
                
                fig.add_trace(go.Scatter(
                    x=df['datetime'],
                    y=df['Solar'],
                    mode='lines',
                    name=f'{name} - Ground Station',
                    line=line_style
                ))

def create_heatmap(merged_plants):
    """Create a heatmap visualization for plant locations"""
    merged_plants['TotalPower'] = round(merged_plants['TotalPower']/1000,1)

    fig = px.density_map(
        merged_plants,
        lat="latitude",
        lon="longitude",
        z="TotalPower",
        hover_name="operator",
        hover_data={
            "Canton": True,
            "operator": True,
            "TotalPower": True,
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

def download_tmp_parquet(blob_name, credentials=None):
    client = storage.Client(project="gridalert-c48ee", credentials=credentials)
    bucket = client.bucket('icon-ch')
    blob = bucket.blob(blob_name)
    blob.download_to_filename('tmp.parquet')


# ——— Modified home_page with user info ———
def home_page():

    service_account_json = st.secrets.secrets.service_account_json
    #service_account_json = service_account_json.replace('\\n', '\n')
    service_account_info = json.loads(service_account_json)
    import os
    print("Service account info parsed successfully")
    credentials = service_account.Credentials.from_service_account_info(service_account_info)
    print("Credentials created successfully")

    # Display user info in header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Swiss Solar Forecasts")

    conn = get_connection()

    fcst = fetch_files(conn, "icon-ch/all_models_ch_prod", r'(\d{6})\.parquet$')
    fcst = load_data(fcst[-1], 'parquet', conn)
    st.dataframe(fcst.head(5))
  
    powerplants = load_data('oracle_predictions/swiss_solar/datasets/solar_mstr_data.csv', 'csv', conn)
    
    if powerplants is not None:
        powerplants = powerplants[['BeginningOfOperation','Canton','operator', 'longitude', 'latitude', 'TotalPower']]
    
    with st.spinner("Downloading and processing capacity data..."):

        st.warning(f"Master data latest update {powerplants['BeginningOfOperation'].max()}")

        dt = pd.to_datetime(fcst.index.min(),utc=True).tz_convert('CET')
        h = []
        for ddt in pd.date_range(start=dt.strftime("%Y-%m-%d"),freq='D', periods=5):
            try:
                nowcast = read_parquet_gcs(f'gcs://dwd-solar-sat/daily_agg_asset_level_prod/{ddt.strftime("%Y%m%d")}.parquet')
                
                h.append(nowcast)
            except:
                nowcast = pd.DataFrame(columns=['datetime','Canton','operator','SolarProduction'])
        nowcast = pd.concat(h)

        gc.collect()
        
        h = []
        for ddt in pd.date_range(start=dt.strftime("%Y%m%d"), freq='D', periods=5):
            try:
                stationprod = read_parquet_gcs(f'gcs://icon-ch/groundstations/ch-prod/cantons_{ddt.strftime("%Y%m%d")}.parquet',engine='pyarrow')
                h.append(stationprod)
            except:
                stationprod = pd.DataFrame()
        try:
            stationprod = pd.concat(h)
        except Exception as e:
            stationprod = pd.DataFrame()

        gc.collect()

        try:
            nowcast.drop_duplicates(['datetime','Canton','operator'], inplace=True)
            nowcast['SolarProduction'] = 1.15*nowcast['SolarProduction']/1000.0
        except:
            nowcast = pd.DataFrame(columns=['datetime','Canton','operator','SolarProduction'])

        st.info('nowcast')
        st.dataframe(nowcast.head(5))

        st.info('stationprod')
        st.dataframe(stationprod.head(5))


        chart_type = st.radio(
            "Select visualization type:",
            options=["Forecast Chart", 'Monthly installed capacity',"Powerplant Location Heatmap"],
            horizontal=True
        )
        
        if chart_type == "Forecast Chart":
            #fig = create_forecast_chart(selected_model,filtered_df,pronovo_f,nowcast,stationprod, filter_type, selected_cantons, selected_operators)
            #st.plotly_chart(fig, use_container_width=True)
            print('oh')
        
        elif chart_type =='Monthly installed capacity':
            full_capa = load_data('oracle_predictions/swiss_solar/datasets/capa_timeseries/full_dataset.parquet', 'parquet', conn)
            col1, col2 = st.columns([3, 1])
            with col1:
                filter_type = st.selectbox("Filter by:", options=["Canton", "Operator"])
            with col2:
                if filter_type == "Canton":
                    selected_cantons = st.multiselect("Select Cantons:", options=sorted(full_capa['Canton'].dropna().unique().tolist()), default=['BE'])
                    selected_operators = None
                else:
                    selected_operators = st.multiselect("Select Operators:", options=sorted(full_capa['operator'].dropna().unique().tolist()),default=['BKW Energie AG'])
                    selected_cantons = None
   
            if filter_type == "Canton" and selected_cantons:
                full_capa_ = full_capa[full_capa['Canton'].isin(selected_cantons)]
            elif filter_type == "Operator" and selected_operators:
                full_capa_ = full_capa[full_capa['operator'].isin(selected_operators)]
            else:
                full_capa_ = full_capa.copy()
            
            full_capa_ = full_capa_.groupby('year_month')['TotalPower'].sum()

            st.subheader('Monthly added capacity [MW]')
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=full_capa_.index,
                y=full_capa_.values/1000,
                name='Added Capacity'
            ))
            st.plotly_chart(fig, use_container_width=True)

        else:  # Powerplant Location Heatmap
            if powerplants is None:
                st.error("Powerplant data is not available for the heatmap visualization")
                return
            
            col1, col2 = st.columns([3, 1])
            with col1:
                filter_type = st.selectbox("Filter by:", options=["Canton", "Operator"])
            with col2:
                if filter_type == "Canton":
                    selected_cantons = st.multiselect("Select Cantons:", options=sorted(powerplants['Canton'].dropna().unique().tolist()), default=['BE'])
                    selected_operators = None
                else:
                    selected_operators = st.multiselect("Select Operators:", options=sorted(powerplants['operator'].dropna().unique().tolist()),default=['BKW Energie AG'])
                    selected_cantons = None
   
            if filter_type == "Canton" and selected_cantons:
                merged_plants = powerplants[powerplants['Canton'].isin(selected_cantons)]
            elif filter_type == "Operator" and selected_operators:
                merged_plants = powerplants[powerplants['operator'].isin(selected_operators)]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Plants", f"{len(merged_plants):,}")
            with col2:
                st.metric("Total Capacity", f"{merged_plants['TotalPower'].sum()/1000:,.2f} MW")
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

from google.cloud import parametermanager_v1
from google.cloud import parametermanager
import os
import uuid

def data_api_page():
    """Simple analytics from local log files"""
    def get_version(client, parent, version_id):
        try:
            return client.get_parameter_version(name=f"{parent}/versions/{version_id}")
        except Exception:
            return None

    service_account_json = st.secrets.secrets.service_account_json
    service_account_info = json.loads(service_account_json)
    print("Service account info parsed successfully")
    credentials = service_account.Credentials.from_service_account_info(service_account_info)

    
    param_id = "solar-dashboard-users"
    project_id = "gridalert-c48ee"
    client = parametermanager_v1.ParameterManagerClient(credentials=credentials)
    parent = client.parameter_path(project_id, "global", param_id)

    email = user_email()
    version_id = email.replace('@','-arobase-').replace('.','_')
    
    existing_version = get_version(client, parent, version_id)

    st.title("📊 Data API page")
    st.markdown('Endpoint')
    st.markdown("[Swagger API documentation](https://ch-solar-api-59139140460.europe-west1.run.app/docs/)", unsafe_allow_html=True)

    # Initialize session state for show/hide toggle if not exists
    if 'show_api_key' not in st.session_state:
        st.session_state.show_api_key = False

    if existing_version:
        data = json.loads(existing_version.payload.data.decode("utf-8"))
        api_key = data.get('uuid')
        
        # Create columns for the API key display and toggle button
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.session_state.show_api_key:
                st.text(f"API Key:")
                st.code(f"{api_key}", language="text")
            else:
                # Show masked version
                masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "****"
                st.text(f"API Key: {masked_key}")
        
        with col2:
            # Toggle button
            if st.button("👁️ " + ("Hide" if st.session_state.show_api_key else "Show")):
                st.session_state.show_api_key = not st.session_state.show_api_key
                st.rerun()
                
    else:
        if st.button("Create API Key"):
            data_dict = {"email": email, "uuid": str(uuid.uuid4()), "credits":100}
            data_json = json.dumps(data_dict)
            request = parametermanager_v1.CreateParameterVersionRequest(
                parent=parent,
                parameter_version_id=version_id,
                parameter_version=parametermanager_v1.ParameterVersion(
                    payload=parametermanager_v1.ParameterVersionPayload(
                        data=data_json.encode("utf-8")
                    )
                ),
            )
            client.create_parameter_version(request=request)
            
            # Show the newly created API key
            st.session_state.show_api_key = True
            api_key = data_dict.get('uuid')

            data_df = pd.DataFrame([data_dict])

            data_df['datetime'] = pd.Timestamp.now('CET').round('1min')
            data_df['action'] = 'key_created'

            upload_df_to_gcs(data_df, api_key + '.csv')
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"API Key: {api_key}")
            with col2:
                if st.button("👁️ Hide"):
                    st.session_state.show_api_key = False
                    st.rerun()

    if st.button("← Back to Dashboard"):
        st.session_state.page = "home"
        #st.rerun()
        return



# Check if animation modules exist and import safely
from satAnimation import generate_sat_rad_anim
from satAnimation_icon import display_png_ch1, display_png_ch2
ANIMATION_AVAILABLE = True



def sat_anim():
    """Satellite animation function"""
    if ANIMATION_AVAILABLE:
        fig_anim = generate_sat_rad_anim()
        st.plotly_chart(fig_anim, use_container_width=True, theme=None)
    else:
        st.error("Satellite animation module not available")

def is_admin_user():
    """Check if current user is admin"""
    admin_emails = ["aminedev1895@gmail.com"]  # Configure admin emails
    return user_name() in admin_emails or user_email() in admin_emails




def main_():
    # Sidebar navigation
    st.sidebar.title("☀️ Swiss Solar Dashboard")

    # Show maintenance message
    st.warning("🚧 The dashboard is currently under maintenance. Please check back later.")
    return

    # ---- Original code below (commented) ----
    """
    def main():
        # Sidebar navigation
        st.sidebar.title("☀️ Swiss Solar Dashboard")
        
        # Authentication section
        st.sidebar.markdown("### 🔐 Authentication")
        
        if user_is_logged_in():
            st.sidebar.success(f"Logged in as: {user_name()}")
            
            # Log user signin if not already logged in this session
            if not st.session_state.get('login_logged', False):
                try:
                    user_email_str = user_email()
                    if log_user_signin_simple(user_email_str):
                        st.session_state.login_logged = True
                        print(get_user_ip())
                        # Get user stats
                        ip = get_user_ip()
                        stats = get_user_stats(user_email_str, ip)
                        
                        # Show welcome message
                        if stats["first_login"]:
                            st.sidebar.info("🎉 Welcome to Solar Dashboard!")
                        else:
                            st.sidebar.info(f"Welcome back! Visit #{stats['total_logins']}")
                        
                        # Upload to cloud in background (optional)
                        upload_logs_to_gcs()
                except Exception as e:
                    print(f"Logging error: {e}")
            
            if st.sidebar.button("🚪 Logout"):
                # Clear login session state
                st.session_state.login_logged = False
                st.logout()
                st.session_state.page = "login"
                return
        else:
            st.sidebar.info("Please log in to access the dashboard")
            if st.sidebar.button("🔑 Login with Google"):
                st.login("google")
        
        # Show login page if not authenticated
        if not user_is_logged_in():
            st.session_state.page = "login"
            login_page()
            return
        
        # Admin section (only for admin users)
        if is_admin_user():
            st.sidebar.markdown("### 👨‍💼 Admin")
            if st.sidebar.button("📊 View Login Analytics"):
                st.session_state.page = "admin"
                st.rerun()

                
        if st.sidebar.button("DATA API Access"):
            st.session_state.page = "dataApi"
            st.rerun()
        
        # Handle admin pages first (before navigation menu)
        if st.session_state.get('page') == 'admin':
            show_login_analytics()
            return
            
        if st.session_state.get('page') == 'dataApi':
            data_api_page()
            return
        
        # Handle login page routing
        if st.session_state.get('page') == 'login':
            login_page()
            return
        
        # Navigation menu (only shown when not on admin pages)
        st.sidebar.markdown("### 📊 Navigation")
        
        page_choice = st.sidebar.radio("Select Page:", [
            "Home",
            "Weather Realtime (MeteoSat 5km)",
            "Weather Forecast (ICON-CH1 1km)",
            "Weather Forecast (ICON-CH2 2.1km)",
            "About"
        ])
        
        # Update page state based on selection
        page_mapping = {
            "Home": "home",
            "Weather Realtime (MeteoSat 5km)": "sat_anim",
            "Weather Forecast (ICON-CH1 1km)": "icon_ch1",
            "Weather Forecast (ICON-CH2 2.1km)": "icon_ch2",
            "About": "about"
        }
        
        st.session_state.page = page_mapping.get(page_choice, "home")
        
        # Page routing
        if page_choice == "Home":
            home_page()
        elif page_choice == 'Weather Realtime (MeteoSat 5km)':
            if ANIMATION_AVAILABLE:
                sat_anim()
            else:
                st.error("Weather animation feature is not available. Please check if satAnimation module is installed.")
        elif page_choice == "Weather Forecast (ICON-CH1 1km)":
            if ANIMATION_AVAILABLE:
                selected = st.selectbox(
                    "Weather parameter:",
                    options=['solar','precipitation','cloud','temperature'],
                    index=0
                )
                with st.spinner("Downloading ..."):
                    display_png_ch1(selected)
            else:
                st.error("Weather forecast feature is not available. Please check if satAnimation_icon module is installed.")
        elif page_choice == "Weather Forecast (ICON-CH2 2.1km)":
            if ANIMATION_AVAILABLE:
                selected = st.selectbox(
                    "Weather parameter:",
                    options=['solar','precipitation','cloud','temperature'],
                    index=0
                )
                with st.spinner("Downloading ..."):
                    display_png_ch2(selected)
            else:
                st.error("Weather forecast feature is not available. Please check if satAnimation_icon module is installed.")
        elif page_choice == "About":
            about_page()
    """

  

def main():
        # Sidebar navigation
        st.sidebar.title("☀️ Swiss Solar Dashboard")
        
        # Authentication section
        st.sidebar.markdown("### 🔐 Authentication")
        
        if user_is_logged_in():
            st.sidebar.success(f"Logged in as: {user_name()}")
            
            # Log user signin if not already logged in this session
            if not st.session_state.get('login_logged', False):
                try:
                    user_email_str = user_email()
                    if log_user_signin_simple(user_email_str):
                        st.session_state.login_logged = True
                        print(get_user_ip())
                        # Get user stats
                        ip = get_user_ip()
                        stats = get_user_stats(user_email_str, ip)
                        
                        # Show welcome message
                        if stats["first_login"]:
                            st.sidebar.info("🎉 Welcome to Solar Dashboard!")
                        else:
                            st.sidebar.info(f"Welcome back! Visit #{stats['total_logins']}")
                        
                        # Upload to cloud in background (optional)
                        upload_logs_to_gcs()
                except Exception as e:
                    print(f"Logging error: {e}")
            
            if st.sidebar.button("🚪 Logout"):
                # Clear login session state
                st.session_state.login_logged = False
                st.logout()
                st.session_state.page = "login"
                return
        else:
            st.sidebar.info("Please log in to access the dashboard")
            if st.sidebar.button("🔑 Login with Google"):
                st.login("google")
        
        # Show login page if not authenticated
        if not user_is_logged_in():
            st.session_state.page = "login"
            login_page()
            return
        
        # Admin section (only for admin users)
        if is_admin_user():
            st.sidebar.markdown("### 👨‍💼 Admin")
            if st.sidebar.button("📊 View Login Analytics"):
                st.session_state.page = "admin"
                st.rerun()

                
        #if st.sidebar.button("DATA API Access"):
        #    st.session_state.page = "dataApi"
        #    st.rerun()
        
        # Handle admin pages first (before navigation menu)
        if st.session_state.get('page') == 'admin':
            show_login_analytics()
            return
            
        if st.session_state.get('page') == 'dataApi':
            data_api_page()
            return
        
        # Handle login page routing
        if st.session_state.get('page') == 'login':
            login_page()
            return
        
        # Navigation menu (only shown when not on admin pages)
        st.sidebar.markdown("### 📊 Navigation")
        
        page_choice = st.sidebar.radio("Select Page:", [
            "Home",
            "Weather Realtime (MeteoSat 5km)",
            "Weather Forecast (ICON-CH1 1km)",
            "Weather Forecast (ICON-CH2 2.1km)",
            "About"
        ])
        
        # Update page state based on selection
        page_mapping = {
            "Home": "home",
            "Weather Realtime (MeteoSat 5km)": "sat_anim",
            "Weather Forecast (ICON-CH1 1km)": "icon_ch1",
            "Weather Forecast (ICON-CH2 2.1km)": "icon_ch2",
            "About": "about"
        }
        
        st.session_state.page = page_mapping.get(page_choice, "home")
        
        # Page routing
        if page_choice == "Home":
            home_page()
        elif page_choice == 'Weather Realtime (MeteoSat 5km)':
            if ANIMATION_AVAILABLE:
                sat_anim()
            else:
                st.error("Weather animation feature is not available. Please check if satAnimation module is installed.")
        elif page_choice == "Weather Forecast (ICON-CH1 1km)":
            if ANIMATION_AVAILABLE:
                selected = st.selectbox(
                    "Weather parameter:",
                    options=['solar','precipitation','cloud','temperature'],
                    index=0
                )
                with st.spinner("Downloading ..."):
                    display_png_ch1(selected)
            else:
                st.error("Weather forecast feature is not available. Please check if satAnimation_icon module is installed.")
        elif page_choice == "Weather Forecast (ICON-CH2 2.1km)":
            if ANIMATION_AVAILABLE:
                selected = st.selectbox(
                    "Weather parameter:",
                    options=['solar','precipitation','cloud','temperature'],
                    index=0
                )
                with st.spinner("Downloading ..."):
                    display_png_ch2(selected)
            else:
                st.error("Weather forecast feature is not available. Please check if satAnimation_icon module is installed.")
        elif page_choice == "About":
            about_page()

if __name__ == "__main__":
    main()
