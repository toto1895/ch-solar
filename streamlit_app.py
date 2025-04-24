import streamlit as st
import pandas as pd
import os
import numpy as np 
import plotly.graph_objects as go
import plotly.express as px
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

def download_and_load_parquet(file_path,format, conn):
    """Download and load the parquet file into a pandas DataFrame"""
    try:
        # Use the connection to read the parquet file directly
        df = conn.read(file_path, input_format=format)
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
    
import gc

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

    powerplants = download_and_load_parquet('oracle_predictions/swiss_solar/datasets/solar_mstr_data.csv','csv', conn)[
        ['Canton','operator','longitude','latitude','TotalPower']
    ]
    #st.dataframe(powerplants.head(10))
    
    if latest_file:
        with st.spinner("Downloading and processing capacity data..."):
            # Load capacity data
            capa_df = download_and_load_parquet(latest_file,'parquet', conn)
            if capa_df is not None:
                # Get the latest date's capacity data
                
                #print(capa_df)
                latest_mastr_date =capa_df.Date.max()
                capa = capa_df.loc[capa_df.Date == latest_mastr_date].drop(columns='Date').reset_index(drop=True)
                
                # Download the selected solar forecast data
                st.warning(f"Master data latest update {latest_mastr_date.strftime('%Y-%m-%d')}")

                
                with st.spinner(f"Downloading solar forecast data from {selected_file}..."):
                    forecast_df, _ = get_solar_forecast(selected_file)

                    if selected_model == 'icon_d2':
                        
                        max_idx = forecast_df.index.unique()[-1:]
                        
                        percentile_cols = ['p0.05', 'p0.1', 'p0.2', 'p0.3', 'p0.4', 'p0.5', 
                                        'p0.6', 'p0.7', 'p0.8', 'p0.9', 'p0.95']
                        
                        # Set these columns to NaN for the row with max index
                        forecast_df = forecast_df.loc[forecast_df.index != max_idx[0]]
                        #forecast_df.loc[forecast_df.index == max_idx[1], percentile_cols] = np.nan


                        #st.dataframe(forecast_df)


                    
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
                        
                        del merged_df
                        del capa
                        del forecast_df
                        gc.collect()
                        # Display the filtered dataframe
                        filtered_df = filtered_df[['datetime','p0.5','p0.1','p0.9','Canton','operator','CumulativePower_canton','CumulativePower_operator']].copy()

                        

                        capa_installed = round(filtered_df.loc[filtered_df.datetime==filtered_df.datetime.max()].drop_duplicates('operator')['CumulativePower_operator'].sum())
                        
                        st.success(f"Installed capacity: {round(capa_installed/1000):,.0f} MW")

                        filtered_df['p0.5_canton'] = filtered_df['p0.5'] * filtered_df['CumulativePower_canton']/1000
                        filtered_df['p0.1_canton'] = filtered_df['p0.1'] * filtered_df['CumulativePower_canton']/1000
                        filtered_df['p0.9_canton'] = filtered_df['p0.9'] * filtered_df['CumulativePower_canton']/1000

                        filtered_df['p0.5_operator'] = filtered_df['p0.5'] * filtered_df['CumulativePower_operator']/1000
                        filtered_df['p0.1_operator'] = filtered_df['p0.1'] * filtered_df['CumulativePower_operator']/1000
                        filtered_df['p0.9_operator'] = filtered_df['p0.9'] * filtered_df['CumulativePower_operator']/1000

                        # Add a radio button for chart type selection
                        chart_type = st.radio(
                            "Select visualization type:",
                            options=["Forecast Chart", "Powerplant Location Heatmap"],
                            horizontal=True
                        )
                        
                        if chart_type == "Forecast Chart":
                            # Original chart visualization code
                            if filter_type == "Canton" and selected_cantons:
                                # Group by datetime and Canton, then sum the values
                                plot_df = filtered_df.copy()
                                
                                # Create the plot
                                fig = go.Figure()
                                
                                for canton in selected_cantons:
                                    canton_df = plot_df[plot_df['Canton'] == canton]
                                    canton_df = canton_df.sort_values('datetime')

                                    canton_df = canton_df.groupby(['datetime']).agg({
                                    'p0.5_operator': 'sum',
                                    'p0.1_operator': 'sum',
                                    'p0.9_operator': 'sum'
                                    }).reset_index()
                                    
                                    # Add median forecast line
                                    fig.add_trace(go.Scatter(
                                        x=canton_df['datetime'],
                                        y=canton_df['p0.5_operator'],
                                        mode='lines',
                                        name=f'{canton} - Median (P50)',
                                        line=dict(width=2)
                                    ))
                                    
                                    # Add lower bound
                                    fig.add_trace(go.Scatter(
                                        x=canton_df['datetime'],
                                        y=canton_df['p0.1_operator'],
                                        mode='lines',
                                        name=f'{canton} - Lower Bound (P10)',
                                        line=dict(width=1, dash='dash')
                                    ))
                                    
                                    # Add upper bound
                                    fig.add_trace(go.Scatter(
                                        x=canton_df['datetime'],
                                        y=canton_df['p0.9_operator'],
                                        mode='lines',
                                        name=f'{canton} - Upper Bound (P90)',
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

                                    operator_df = operator_df.groupby(['datetime']).agg({
                                    'p0.5_operator': 'sum',
                                    'p0.1_operator': 'sum',
                                    'p0.9_operator': 'sum'
                                    }).reset_index()
                                    
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
                                        y=operator_df['p0.1_operator'],
                                        mode='lines',
                                        name=f'{operator} - Lower Bound (P10)',
                                        line=dict(width=1, dash='dash')
                                    ))
                                    
                                    # Add upper bound
                                    fig.add_trace(go.Scatter(
                                        x=operator_df['datetime'],
                                        y=operator_df['p0.9_operator'],
                                        mode='lines',
                                        name=f'{operator} - Upper Bound (P90)',
                                        line=dict(width=1, dash='dash')
                                    ))  
                                
                            else:
                                # Create an empty figure if no selections are made
                                operator_df = filtered_df.copy().sort_values('datetime')

                                operator_df = operator_df.groupby(['datetime']).agg({
                                'p0.5_operator': 'sum',
                                'p0.1_operator': 'sum',
                                'p0.9_operator': 'sum'
                                }).reset_index()
                                
                                fig = go.Figure()
                                # Add median forecast line
                                fig.add_trace(go.Scatter(
                                    x=operator_df['datetime'],
                                    y=operator_df['p0.5_operator'],
                                    mode='lines',
                                    name=f'Total - Median (P50)',
                                    line=dict(width=2, color='white' )
                                ))
                                
                                # Add lower bound
                                fig.add_trace(go.Scatter(
                                    x=operator_df['datetime'],
                                    y=operator_df['p0.1_operator'],
                                    mode='lines',
                                    name=f'Total - Lower Bound (P10)',
                                    line=dict(width=1, dash='dash', color='white')
                                ))
                                
                                # Add upper bound
                                fig.add_trace(go.Scatter(
                                    x=operator_df['datetime'],
                                    y=operator_df['p0.9_operator'],
                                    mode='lines',
                                    name=f'Total - Upper Bound (P90)',
                                    line=dict(width=1, dash='dash', color='white')
                                ))
                            

                            try:
                                len(selected_operators)>1
                            except:
                                selected_operators = [0]
                            
                            try:
                                len(selected_cantons)>1
                            except:
                                selected_cantons = [0]

                            if (len(selected_operators)>1) or (len(selected_cantons)>1): 
                                    if len(selected_operators)>1:
                                        total_df = plot_df[plot_df['operator'].isin(selected_operators)]
                                    elif (len(selected_cantons)>1):
                                        total_df = plot_df[plot_df['Canton'].isin(selected_cantons)]
                                    # Group by datetime and sum
                                    total_df = total_df.groupby(['datetime']).agg({
                                        'p0.5_operator': 'sum',
                                        'p0.1_operator': 'sum',
                                        'p0.9_operator': 'sum'
                                    }).reset_index()

                                    # Add median forecast line for Total
                                    fig.add_trace(go.Scatter(
                                        x=total_df['datetime'],
                                        y=total_df['p0.5_operator'],
                                        mode='lines',
                                        name='Total - Median (P50)',
                                        line=dict(width=3, color='white')  # Making Total line thicker and black for emphasis
                                    ))

                                    # Add lower bound for Total
                                    fig.add_trace(go.Scatter(
                                        x=total_df['datetime'],
                                        y=total_df['p0.1_operator'],
                                        mode='lines',
                                        name='Total - Lower Bound (P10)',
                                        line=dict(width=2, dash='dash', color='white')
                                    ))

                                    # Add upper bound for Total
                                    fig.add_trace(go.Scatter(
                                        x=total_df['datetime'],
                                        y=total_df['p0.9_operator'],
                                        mode='lines',
                                        name='Total - Upper Bound (P90)',
                                        line=dict(width=2, dash='dash', color='white')
                                    ))
                            # Update layout for all plots
                            fig.update_layout(
                                title="Solar Generation Forecast",
                                xaxis_title="Date and Time",
                                yaxis_title="Power (MW)",
                                legend_title="Legend",
                                template="plotly_dark",
                                height=600,
                                hovermode="x unified"
                            )
                            
                            # Display the plot
                            st.plotly_chart(fig, use_container_width=True)
                            
                        else:  # Powerplant Location Heatmap
                            # Merge powerplants with filtered data based on Canton and operator
                            # First extract the latest datetime for forecast
                            latest_datetime = filtered_df['datetime'].max()
                            latest_forecast = filtered_df[filtered_df['datetime'] == latest_datetime].copy()
                            
                            # Merge with powerplants data
                            if filter_type == "Canton" and selected_cantons:
                                merged_plants = pd.merge(
                                    powerplants, 
                                    latest_forecast, 
                                    on=["Canton","operator"], 
                                    how="inner"
                                )
                                if selected_cantons:
                                    merged_plants = merged_plants[merged_plants['Canton'].isin(selected_cantons)]
                            elif filter_type == "Operator" and 'operator' in filtered_df.columns and selected_operators:
                                merged_plants = pd.merge(
                                    powerplants, 
                                    latest_forecast, 
                                    on=["Canton", "operator"], 
                                    how="inner"
                                )
                                if selected_operators:
                                    merged_plants = merged_plants[merged_plants['operator'].isin(selected_operators)]
                            else:
                                merged_plants = pd.merge(
                                    powerplants, 
                                    latest_forecast, 
                                    on=["Canton","operator"], 
                                    how="inner"
                                )
                            

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Plants", f"{len(merged_plants):,}")
                            with col2:
                                st.metric("Total Capacity", f"{merged_plants['TotalPower'].sum()/1000:,.2f} MW")
                            # Create the heatmap using plotly
                            fig = px.density_map(
                                merged_plants,
                                lat="latitude",
                                lon="longitude",
                                z="TotalPower",  # Color intensity based on forecast power
                                hover_name="operator",
                                hover_data={
                                    "Canton": True,
                                    "operator":True,
                                    "TotalPower": True,
                                    #"forecast_power": ":.2f",
                                    #"latitude": False,
                                    #"longitude": False
                                },
                                color_continuous_scale="Jet",
                                radius=10,
                                zoom=6,
                                #mapbox_style="carto-darkmatter",
                                title="Solar Power Plant Density",
                                center={"lat": 46.8, "lon": 8.2},  # Center of Switzerland
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
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display additional statistics
                            
                           # with col3:
                            #    st.metric("Forecast Power", f"{merged_plants['forecast_power'].sum():,.2f} MW")
                            
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
    
    page_choice = st.sidebar.radio("Go to page:", ["Home", "About"])
    
    if page_choice == "Home":
        home_page()
    elif page_choice == "About":
        st.title("About This App")
        st.write("This application displays solar power forecasts for Switzerland based on PRONOVO data.")


if __name__ == "__main__":
    main()