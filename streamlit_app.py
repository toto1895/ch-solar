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
import re

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

def get_latest_capa():
    """Get list of forecast files from GCS bucket"""
    conn = st.connection('gcs', type=FilesConnection)
    prefix = "oracle_predictions/swiss_solar/datasets/capa_timeseries"
    
    # Invalidate the cache to refresh the bucket listing
    conn._instance.invalidate_cache(prefix)
    
    # Retrieve all files
    all_files = conn._instance.ls(prefix, max_results=100)
    
    return sorted(all_files, reverse=True), conn

def get_latest_parquet_file():
    """Get the latest parquet file with format %Y-%m.parquet"""
    conn = st.connection('gcs', type=FilesConnection)
    prefix = "oracle_predictions/swiss_solar/datasets/capa_timeseries"
    
    # Invalidate the cache to refresh the bucket listing
    conn._instance.invalidate_cache(prefix)
    
    # Retrieve all files
    all_files = conn._instance.ls(prefix, max_results=100)
    
    # Filter for parquet files with the format YYYY-MM.parquet
    date_pattern = re.compile(r'(\d{4}-\d{2})\.parquet$')
    parquet_files = [f for f in all_files if date_pattern.search(f)]
    
    if not parquet_files:
        return None, conn
    
    # Sort files by date in filename (newest first)
    latest_file = sorted(parquet_files, key=lambda x: date_pattern.search(x).group(1), reverse=True)[0]
    
    return latest_file, conn

def download_and_load_parquet(file_path, conn):
    """Download and load the parquet file into a pandas DataFrame"""
    try:
        # Use the connection to read the parquet file directly
        df = conn.read(file_path, input_format="parquet")
        return df
    except Exception as e:
        st.error(f"Error loading parquet file: {e}")
        return None

def get_available_forecast_files(model, cluster):
    """Get list of available forecast files for the selected model and cluster"""
    conn = st.connection('gcs', type=FilesConnection)
    prefix = f"oracle_predictions/swiss_solar/canton_forecasts_factor/{model}/{cluster}"
    
    # Invalidate the cache to refresh the bucket listing
    conn._instance.invalidate_cache(prefix)
    
    # Retrieve all files
    try:
        all_files = conn._instance.ls(prefix, max_results=100)
        # Filter for parquet files only
        parquet_files = [f for f in all_files if f.endswith('.parquet')]
        return sorted(parquet_files, reverse=True), conn
    except Exception as e:
        st.error(f"Error listing forecast files: {e}")
        return [], conn

def get_solar_forecast(forecast_path):
    """Get the specific solar forecast parquet file"""
    conn = st.connection('gcs', type=FilesConnection)
    
    try:
        # Download and load the forecast
        forecast_df = conn.read(forecast_path, input_format="parquet")
        return forecast_df, conn
    except Exception as e:
        st.error(f"Error loading forecast parquet file: {e}")
        return None, conn

def home_page():
    st.title("Swiss Solar Forecasts")
    
    # Define available models and clusters
    available_models = ["dmi_seamless", "metno_seamless", "icon_d2", "meteofrance_seamless"]
    available_clusters = ["cluster0", "cluster1", "cluster2"]
    
    # Create selection widgets in columns
    col1, col2 = st.columns(2)
    
    with col1:
        selected_model = st.selectbox(
            "Select Model:",
            options=available_models,
            index=0  # Default to dmi_seamless
        )
    
    with col2:
        selected_cluster = st.selectbox(
            "Select Cluster:",
            options=available_clusters,
            index=0  # Default to cluster0
        )
    
    # Get available forecast files for the selected model and cluster
    forecast_files, conn = get_available_forecast_files(selected_model, selected_cluster)
    
    if not forecast_files:
        st.warning(f"No forecast files found for {selected_model}/{selected_cluster}")
        return
    
    # Create a dropdown to select the forecast file
    selected_file = st.selectbox(
        "Select Forecast File:",
        options=forecast_files,
        index=0  # Default to the most recent file
    )
    
    # Get the latest parquet file for capacity data
    latest_file, conn = get_latest_parquet_file()
    
    if latest_file:
        with st.spinner("Downloading and processing capacity data..."):
            # Load capacity data
            capa_df = download_and_load_parquet(latest_file, conn)
            if capa_df is not None:
                # Get the latest date's capacity data
                capa = capa_df.loc[capa_df.datetime == capa_df.datetime.max()].drop(columns='datetime').reset_index(drop=True)
                
                # Download the selected solar forecast data
                with st.spinner(f"Downloading solar forecast data from {selected_file}..."):
                    forecast_df, _ = get_solar_forecast(selected_file)
                    
                    if forecast_df is not None:
                        # Merge forecast with capacity data on Canton
                        merged_df = pd.merge(forecast_df.reset_index(), capa, on="Canton", how="left")
                        
                        # Add filter section
                        st.subheader("Filter Data")
                        
                        # Create two columns for filter type selection and the actual filter
                        filter_col1, filter_col2 = st.columns([1, 3])
                        
                        with filter_col1:
                            # Dropdown to select filter type
                            filter_type = st.selectbox(
                                "Filter by:",
                                options=["Canton", "Operator"],
                                index=0
                            )
                        
                        with filter_col2:
                            # Initialize filtered_df
                            filtered_df = merged_df.copy()
                            
                            if filter_type == "Canton":
                                # Get all unique cantons
                                all_cantons = sorted(merged_df["Canton"].unique().tolist())
                                
                                # Create a multiselect widget for cantons
                                selected_cantons = st.multiselect(
                                    "Select Cantons:",
                                    options=all_cantons,
                                    #default=all_cantons  # By default, select all cantons
                                )
                                
                                # Filter the dataframe based on selected cantons
                                if selected_cantons:
                                    filtered_df = merged_df[merged_df["Canton"].isin(selected_cantons)]
                                
                            elif filter_type == "Operator":
                                # Check if 'operator' column exists in merged_df
                                if 'operator' in merged_df.columns:
                                    # Get all unique operators
                                    all_operators = sorted(merged_df["operator"].unique().tolist())
                                    
                                    # Create a multiselect widget for operators
                                    selected_operators = st.multiselect(
                                        "Select Operators:",
                                        options=all_operators,
                                        #default=all_operators  # By default, select all operators
                                    )
                                    
                                    # Filter the dataframe based on selected operators
                                    if selected_operators:
                                        filtered_df = merged_df[merged_df["operator"].isin(selected_operators)]
                                else:
                                    st.warning("No 'operator' column found in the data. Please use Canton filtering instead.")
                        
                        # Display the filtered dataframe
                        #st.subheader("Solar Forecast with Capacity Data")
                        capa_installed = round(filtered_df.loc[filtered_df.datetime==filtered_df.datetime.max()]['cum_operator'].sum())
                        st.success(f"{round(capa_installed/1000):,.0f} MW")

                        filtered_df['p0.5_canton'] = filtered_df['p0.5'] * filtered_df['cum_canton']
                        filtered_df['p0.2_canton'] = filtered_df['p0.2'] * filtered_df['cum_canton']
                        filtered_df['p0.8_canton'] = filtered_df['p0.8'] * filtered_df['cum_canton']

                        filtered_df['p0.5_operator'] = filtered_df['p0.5'] * filtered_df['cum_operator']
                        filtered_df['p0.2_operator'] = filtered_df['p0.2'] * filtered_df['cum_operator']
                        filtered_df['p0.8_operator'] = filtered_df['p0.8'] * filtered_df['cum_operator']

                        st.dataframe(filtered_df)
                        
                        # Create scatter plot based on filter type
                        st.subheader("Forecast Visualization")
                        
                        if filter_type == "Canton" and selected_cantons:
                            # Group by datetime and Canton, then sum the values
                            plot_df = filtered_df.copy()
                            
                            # Create the plot
                            fig = go.Figure()
                            
                            # Add traces for each canton
                            for canton in selected_cantons:
                                canton_df = plot_df[plot_df['Canton'] == canton]
                                canton_df = canton_df.sort_values('datetime')

                                canton_df = canton_df.groupby(['datetime','operator']).agg({
                                'p0.5_canton': 'sum',
                                'p0.2_canton': 'sum',
                                'p0.8_canton': 'sum'
                                }).reset_index()
                                
                                # Add median forecast line
                                fig.add_trace(go.Scatter(
                                    x=canton_df['datetime'],
                                    y=canton_df['p0.5_canton'],
                                    mode='lines',
                                    name=f'{canton} - Median (P50)',
                                    line=dict(width=2)
                                ))
                                
                                # Add lower bound
                                fig.add_trace(go.Scatter(
                                    x=canton_df['datetime'],
                                    y=canton_df['p0.2_canton'],
                                    mode='lines',
                                    name=f'{canton} - Lower Bound (P20)',
                                    line=dict(width=1, dash='dash')
                                ))
                                
                                # Add upper bound
                                fig.add_trace(go.Scatter(
                                    x=canton_df['datetime'],
                                    y=canton_df['p0.8_canton'],
                                    mode='lines',
                                    name=f'{canton} - Upper Bound (P80)',
                                    line=dict(width=1, dash='dash')
                                ))
                            
                            
                        elif filter_type == "Operator" and 'operator' in filtered_df.columns and selected_operators:
                            # Group by datetime and Operator, then sum the values
                            plot_df = filtered_df.copy()
                            
                            # Create the plot
                            fig = go.Figure()
                            
                            # Add traces for each operator
                            for operator in selected_operators:
                                operator_df = plot_df[plot_df['operator'] == operator]
                                operator_df = operator_df.sort_values('datetime')
                                
                                # Add median forecast line
                                fig.add_trace(go.Scatter(
                                    x=operator_df['datetime'],
                                    y=operator_df['p0.5_operator'],
                                    mode='lines',
                                    name=f'{operator} - Median (P50)',
                                    line=dict(width=2)
                                ))
                                
                                # Add lower bound
                                fig.add_trace(go.Scatter(
                                    x=operator_df['datetime'],
                                    y=operator_df['p0.2_operator'],
                                    mode='lines',
                                    name=f'{operator} - Lower Bound (P20)',
                                    line=dict(width=1, dash='dash')
                                ))
                                
                                # Add upper bound
                                fig.add_trace(go.Scatter(
                                    x=operator_df['datetime'],
                                    y=operator_df['p0.8_operator'],
                                    mode='lines',
                                    name=f'{operator} - Upper Bound (P80)',
                                    line=dict(width=1, dash='dash')
                                ))
                            
                            # Also add a total line that sums all selected operators
                            agg_df = filtered_df.groupby(['datetime','operator']).agg({
                                'p0.5_operator': 'sum',
                                'p0.2_operator': 'sum',
                                'p0.8_operator': 'sum'
                            }).reset_index()
                            
                            # Add total lines with distinct styling
                            fig.add_trace(go.Scatter(
                                x=agg_df['datetime'],
                                y=agg_df['p0.5_operator'],
                                mode='lines',
                                name='Total - Median (P50)',
                                line=dict(color='black', width=3)
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=agg_df['datetime'],
                                y=agg_df['p0.2_operator'],
                                mode='lines',
                                name='Total - Lower Bound (P20)',
                                line=dict(color='black', width=2, dash='dash')
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=agg_df['datetime'],
                                y=agg_df['p0.8_operator'],
                                mode='lines',
                                name='Total - Upper Bound (P80)',
                                line=dict(color='black', width=2, dash='dash')
                            ))
                        else:
                            # Create an empty figure if no selections are made
                            fig = go.Figure()
                            fig.update_layout(
                                annotations=[dict(
                                    text="Please select at least one Canton or Operator to display the forecast.",
                                    showarrow=False,
                                    xref="paper",
                                    yref="paper",
                                    x=0.5,
                                    y=0.5
                                )]
                            )
                        
                        # Update layout for all plots
                        fig.update_layout(
                            title="Solar Power Forecast",
                            xaxis_title="Date and Time",
                            yaxis_title="Power (W)",
                            legend_title="Legend",
                            template="plotly_dark",
                            height=600,
                            hovermode="x unified"
                        )
                        
                        # Display the plot
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display selection information
                        st.info(f"Displaying data for Model: {selected_model}, Cluster: {selected_cluster}, File: {os.path.basename(selected_file)}")
                        
                        # Display filtering stats
                        if len(filtered_df) < len(merged_df):
                            st.success(f"Filtered data: {len(filtered_df)} of {len(merged_df)} records shown based on current filters.")
                        
                    else:
                        st.error("Failed to load solar forecast data.")
            else:
                st.error("Failed to load capacity data.")
    else:
        st.warning("No parquet files with format YYYY-MM.parquet found in the bucket.")

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