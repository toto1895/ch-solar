import streamlit as st
import pandas as pd
import os
import numpy as np 
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from google.oauth2 import service_account
from st_files_connection import FilesConnection
from google.cloud import storage
import tempfile

# Page configuration
st.set_page_config(
    page_title="Swiss Solar Forecasts",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Environment variables
GCLOUD = os.getenv("service_account_json")

def get_forecast_files():
    """Get list of forecast files from GCS bucket"""
    conn = st.connection('gcs', type=FilesConnection)
    prefix = "oracle_predictions/swiss_solar/forecasts"
    
    # Invalidate the cache to refresh the bucket listing
    conn._instance.invalidate_cache(prefix)
    
    # Retrieve all files
    all_files = conn._instance.ls(prefix, max_results=1000)
    
    return sorted(all_files, reverse=True), conn

def load_entsoe_data(start_date, end_date=None):
    """
    Load ENTSOE data for the given date range from GCS bucket.
    """
    try:
        if end_date is None:
            end_date = start_date + timedelta(days=3)
            
        # Initialize storage client
        service_account_info = json.loads(GCLOUD)
        credentials = service_account.Credentials.from_service_account_info(service_account_info)
        storage_client = storage.Client(credentials=credentials)
        bucket = storage_client.bucket("oracle_predictions")
        prefix = "swiss_solar/solar_actual/switzerland"
        
        # Convert dates to strings for comparison
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # List all blobs with the given prefix
        blobs = list(bucket.list_blobs(prefix=prefix))
        
        # Find files within date range
        matching_blobs = []
        
        for blob in blobs:
            filename = blob.name.split('/')[-1]
            if filename.endswith('.parquet') and len(filename) >= 10:
                date_part = filename[:10]  # Extract YYYY-MM-DD part
                if start_str <= date_part <= end_str:
                    matching_blobs.append(blob)
        
        if not matching_blobs:
            st.warning(f"No ENTSOE data files found for date range {start_str} to {end_str}")
            return pd.DataFrame()
        
        # Load and concatenate all matching files
        dfs = []
        for blob in matching_blobs:
            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=True) as temp_file:
                blob.download_to_filename(temp_file.name)
                df = pd.read_parquet(temp_file.name)
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
            
        combined_df = pd.concat(dfs)
        
        # Ensure datetime index is timezone-aware
        if combined_df.index.tz is None:
            combined_df.index = combined_df.index.tz_localize('UTC')
            
        return combined_df
        
    except Exception as e:
        st.error(f"Error loading ENTSOE data: {e}")
        return pd.DataFrame()

def home_page():
    st.title("Swiss Solar Forecasts")
    
    try:
        sorted_files, conn = get_forecast_files()
        if not sorted_files:
            st.error("No forecast files found.")
            return

        # Get a list of unique dates from filenames
        file_dates = [s.split('/')[-1][:-8] for s in sorted_files]
        
        # Select date for the forecast
        selected_dt = st.selectbox("Forecast generation time:", file_dates)
        
        # Load forecast data
        file_path = f"oracle_predictions/swiss_solar/forecasts/{selected_dt}.parquet"
        forecast_df = conn.read(file_path, input_format="parquet").round(2)
        
        # Extract the date range from the forecast data
        start_date = forecast_df.index.min()
        end_date = forecast_df.index.max()
        
        # Load ENTSOE data for the same period
        st.info("Loading ENTSOE actual data from GCS bucket...")
        entsoe_df = load_entsoe_data(start_date, end_date)
            
        # Combine forecast and ENTSOE data
        if not entsoe_df.empty:
            # Align indexes and combine data
            if entsoe_df.index.tz != forecast_df.index.tz and entsoe_df.index.tz is not None:
                # Convert to common timezone
                entsoe_df = entsoe_df.tz_convert(forecast_df.index.tz)
                
            # If needed, resample to match frequency
            if len(entsoe_df) != len(forecast_df):
                if len(forecast_df.index) > 0 and len(entsoe_df.index) > 0:
                    freq = pd.infer_freq(forecast_df.index)
                    if freq:
                        entsoe_df = entsoe_df.resample(freq).mean()
            
            # Rename ENTSOE columns before merging
            if 'Solar' in entsoe_df.columns:
                entsoe_df = entsoe_df.rename(columns={'Solar': 'actual_entsoe'})
                
            # Merge dataframes
            st.success("ENTSOE data loaded successfully!")
        else:
            st.warning("No ENTSOE data available for the selected period in the GCS bucket.")

        # Create visualization
        st.subheader("Solar Power Forecast vs Actual")
        
        # Create a figure
        fig = go.Figure()
        
        # Add forecast data
        forecast_models = ["metno_0.5", "knmi_0.5", "icon_0.5", "meteofrance_0.5", "avg_0.5"]
        colors = {
            "metno_0.5": "blue",
            "knmi_0.5": "red",
            "icon_0.5": "green",
            "meteofrance_0.5": "orange",
            "avg_0.5": "purple"
        }
        
        for model in forecast_models:
            if model in forecast_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=forecast_df.index,
                        y=forecast_df[model],
                        name=model,
                        mode="lines",
                        line=dict(color=colors.get(model, "blue"))
                    )
                )
        
        # Add ENTSOE actual data if available
        if not entsoe_df.empty:
            if 'actual_entsoe' in entsoe_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=entsoe_df.index,
                        y=entsoe_df['actual_entsoe'],
                        name="ENTSOE Actual",
                        mode="lines",
                        line=dict(color="white", width=3)
                    )
                )
            elif 'Solar' in entsoe_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=entsoe_df.index,
                        y=entsoe_df['Solar'],
                        name="ENTSOE Actual",
                        mode="lines",
                        line=dict(color="white", width=3)
                    )
                )
        
        # Update layout
        fig.update_layout(
            title="Solar Power Forecast vs Actual",
            xaxis_title="Date/Time",
            yaxis_title="Power (MW)",
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display forecast data in table
        with st.expander("View Forecast Data"):
            st.dataframe(forecast_df)
            
        # Display ENTSOE data in table if available
        if not entsoe_df.empty:
            with st.expander("View ENTSOE Data"):
                st.dataframe(entsoe_df)
        
        # Calculate and display error metrics if actual data is available
        if not entsoe_df.empty:
            st.subheader("Forecast Performance Metrics")
            
            # Prepare dataframe for metrics calculation
            eval_data = forecast_df.copy()
            
            # Add actual data
            if 'actual_entsoe' in entsoe_df.columns:
                eval_data['actual'] = entsoe_df['actual_entsoe']
            elif 'Solar' in entsoe_df.columns:
                eval_data['actual'] = entsoe_df['Solar']
                
            # Calculate metrics only if actual column exists and has data
            if 'actual' in eval_data.columns and not eval_data['actual'].isna().all():
                # Define metrics to calculate
                metrics = {}
                
                # Calculate RMSE and MAE for each forecast model
                for model in forecast_models:
                    if model in eval_data.columns:
                        # Make sure we only use rows where both forecast and actual values exist
                        valid_data = eval_data.dropna(subset=[model, 'actual'])
                        
                        if len(valid_data) > 0:
                            # Calculate RMSE
                            rmse = np.sqrt(np.mean((valid_data[model] - valid_data['actual'])**2))
                            metrics[f"{model}_RMSE"] = round(rmse, 2)
                            
                            # Calculate MAE
                            mae = np.mean(np.abs(valid_data[model] - valid_data['actual']))
                            metrics[f"{model}_MAE"] = round(mae, 2)
                
                # Display metrics in a dataframe
                metrics_df = pd.DataFrame([metrics])
                
                # Split into RMSE and MAE columns for display
                rmse_cols = [col for col in metrics_df.columns if 'RMSE' in col]
                mae_cols = [col for col in metrics_df.columns if 'MAE' in col]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("RMSE")
                    st.dataframe(metrics_df[rmse_cols])
                    
                with col2:
                    st.subheader("MAE")
                    st.dataframe(metrics_df[mae_cols])
    
    except Exception as e:
        st.error(f"Error in Home function: {e}")
        import traceback
        st.error(traceback.format_exc())

def main():
    st.sidebar.title("Navigation")
    
    # Cache clearing button
    if st.sidebar.button("Clear Cache"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.sidebar.success("Cache cleared!")
    
    # Page navigation
    page_choice = st.sidebar.radio("Go to page:", ["Home", "About"])
    
    if page_choice == "Home":
        home_page()
    elif page_choice == "About":
        st.title("About This App")
        st.write("This application displays solar power forecasts for Switzerland along with ENTSOE actual data.")
        st.write("It compares different forecasting models and calculates performance metrics.")

if __name__ == "__main__":
    main()