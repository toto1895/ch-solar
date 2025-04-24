import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import re
import gc
from datetime import datetime
from st_files_connection import FilesConnection

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

# Cache data loading functions
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_latest_parquet_file():
    """Get the latest parquet file with format %Y-%m.parquet"""
    conn = st.connection('gcs', type=FilesConnection)
    prefix = "oracle_predictions/swiss_solar/datasets/capa_timeseries"
    
    # Retrieve all files - fixed to use the correct API
    try:
        all_files = conn.fs.ls(prefix, max_results=100)
        
        # Filter for parquet files with the format YYYY-MM.parquet
        date_pattern = re.compile(r'(\d{4}-\d{2})\.parquet

@st.cache_data(ttl=3600)  # Cache for 1 hour
def download_and_load_parquet(file_path, format, conn):
    """Download and load the parquet file into a pandas DataFrame"""
    try:
        # Use the connection to read the parquet file directly
        df = conn.read(file_path, input_format=format)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        # Try alternative method if the first fails
        try:
            if format == 'parquet':
                with conn.fs.open(file_path, 'rb') as f:
                    df = pd.read_parquet(f)
                return df
            elif format == 'csv':
                with conn.fs.open(file_path, 'rb') as f:
                    df = pd.read_csv(f)
                return df
        except Exception as e2:
            st.error(f"Error loading file (alternative method): {e2}")
            return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_solar_forecast(forecast_path, conn):
    """Get the specific solar forecast parquet file"""
    try:
        # Download and load the forecast
        forecast_df = conn.read(forecast_path, input_format="parquet")
        return forecast_df
    except Exception as e:
        st.error(f"Error loading forecast parquet file: {e}")
        # Try alternative method if the first fails
        try:
            with conn.fs.open(forecast_path, 'rb') as f:
                forecast_df = pd.read_parquet(f)
            return forecast_df
        except Exception as e2:
            st.error(f"Error loading forecast parquet file (alternative method): {e2}")
            return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_powerplants_data(conn):
    """Get power plants master data"""
    try:
        powerplants_df = download_and_load_parquet(
            'oracle_predictions/swiss_solar/datasets/solar_mstr_data.csv', 
            'csv', 
            conn
        )
        if powerplants_df is not None:
            return powerplants_df[['Canton', 'operator', 'longitude', 'latitude', 'TotalPower']]
        else:
            st.error("Failed to load power plants data")
            return pd.DataFrame(columns=['Canton', 'operator', 'longitude', 'latitude', 'TotalPower'])
    except Exception as e:
        st.error(f"Error loading power plants data: {e}")
        return pd.DataFrame(columns=['Canton', 'operator', 'longitude', 'latitude', 'TotalPower'])

def process_forecast_data(forecast_df, selected_model):
    """Process forecast data based on model"""
    if forecast_df is None:
        return None
        
    if selected_model == 'icon_d2':
        max_idx = forecast_df.index.unique()[-1:]
        # Remove the last index which might contain incomplete data
        forecast_df = forecast_df.loc[forecast_df.index != max_idx[0]]
    
    return forecast_df

def create_forecast_chart(filtered_df, filter_type, selected_cantons=None, selected_operators=None):
    """Create forecast chart based on filtering"""
    fig = go.Figure()
    plot_df = filtered_df.copy()
    
    # Handle canton filtering
    if filter_type == "Canton" and selected_cantons:
        for canton in selected_cantons:
            canton_df = plot_df[plot_df['Canton'] == canton]
            canton_df = canton_df.sort_values('datetime')
            
            # Aggregate by datetime
            canton_df = canton_df.groupby(['datetime']).agg({
                'p0.5_operator': 'sum',
                'p0.1_operator': 'sum',
                'p0.9_operator': 'sum'
            }).reset_index()
            
            # Add traces
            fig.add_trace(go.Scatter(
                x=canton_df['datetime'],
                y=canton_df['p0.5_operator'],
                mode='lines',
                name=f'{canton} - Median (P50)',
                line=dict(width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=canton_df['datetime'],
                y=canton_df['p0.1_operator'],
                mode='lines',
                name=f'{canton} - Lower Bound (P10)',
                line=dict(width=1, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=canton_df['datetime'],
                y=canton_df['p0.9_operator'],
                mode='lines',
                name=f'{canton} - Upper Bound (P90)',
                line=dict(width=1, dash='dash')
            ))
    
    # Handle operator filtering
    elif filter_type == "Operator" and 'operator' in filtered_df.columns and selected_operators:
        for operator in selected_operators:
            operator_df = plot_df[plot_df['operator'] == operator]
            operator_df = operator_df.sort_values('datetime')
            
            # Aggregate by datetime
            operator_df = operator_df.groupby(['datetime']).agg({
                'p0.5_operator': 'sum',
                'p0.1_operator': 'sum',
                'p0.9_operator': 'sum'
            }).reset_index()
            
            # Add traces
            fig.add_trace(go.Scatter(
                x=operator_df['datetime'],
                y=operator_df['p0.5_operator'],
                mode='lines',
                name=f'{operator} - Median (P50)',
                line=dict(width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=operator_df['datetime'],
                y=operator_df['p0.1_operator'],
                mode='lines',
                name=f'{operator} - Lower Bound (P10)',
                line=dict(width=1, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=operator_df['datetime'],
                y=operator_df['p0.9_operator'],
                mode='lines',
                name=f'{operator} - Upper Bound (P90)',
                line=dict(width=1, dash='dash')
            ))
    
    # Default case - show total
    else:
        operator_df = filtered_df.copy().sort_values('datetime')
        
        # Aggregate by datetime
        operator_df = operator_df.groupby(['datetime']).agg({
            'p0.5_operator': 'sum',
            'p0.1_operator': 'sum',
            'p0.9_operator': 'sum'
        }).reset_index()
        
        # Add traces
        fig.add_trace(go.Scatter(
            x=operator_df['datetime'],
            y=operator_df['p0.5_operator'],
            mode='lines',
            name='Total - Median (P50)',
            line=dict(width=2, color='white')
        ))
        
        fig.add_trace(go.Scatter(
            x=operator_df['datetime'],
            y=operator_df['p0.1_operator'],
            mode='lines',
            name='Total - Lower Bound (P10)',
            line=dict(width=1, dash='dash', color='white')
        ))
        
        fig.add_trace(go.Scatter(
            x=operator_df['datetime'],
            y=operator_df['p0.9_operator'],
            mode='lines',
            name='Total - Upper Bound (P90)',
            line=dict(width=1, dash='dash', color='white')
        ))
    
    # Add total line for multiple selections
    has_multiple_selections = ((selected_operators and len(selected_operators) > 1) or 
                              (selected_cantons and len(selected_cantons) > 1))
    
    if has_multiple_selections:
        if selected_operators and len(selected_operators) > 1:
            total_df = plot_df[plot_df['operator'].isin(selected_operators)]
        elif selected_cantons and len(selected_cantons) > 1:
            total_df = plot_df[plot_df['Canton'].isin(selected_cantons)]
            
        # Group by datetime and sum
        total_df = total_df.groupby(['datetime']).agg({
            'p0.5_operator': 'sum',
            'p0.1_operator': 'sum',
            'p0.9_operator': 'sum'
        }).reset_index()
        
        # Add total traces
        fig.add_trace(go.Scatter(
            x=total_df['datetime'],
            y=total_df['p0.5_operator'],
            mode='lines',
            name='Total - Median (P50)',
            line=dict(width=3, color='white')
        ))
        
        fig.add_trace(go.Scatter(
            x=total_df['datetime'],
            y=total_df['p0.1_operator'],
            mode='lines',
            name='Total - Lower Bound (P10)',
            line=dict(width=2, dash='dash', color='white')
        ))
        
        fig.add_trace(go.Scatter(
            x=total_df['datetime'],
            y=total_df['p0.9_operator'],
            mode='lines',
            name='Total - Upper Bound (P90)',
            line=dict(width=2, dash='dash', color='white')
        ))
    
    # Update layout
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

def create_heatmap(powerplants, filtered_df, filter_type, selected_cantons=None, selected_operators=None):
    """Create heatmap of power plant locations"""
    # Extract latest datetime for forecast
    latest_datetime = filtered_df['datetime'].max()
    latest_forecast = filtered_df[filtered_df['datetime'] == latest_datetime].copy()
    
    # Merge with powerplants data
    merged_plants = pd.merge(
        powerplants,
        latest_forecast,
        on=["Canton", "operator"],
        how="inner"
    )
    
    # Apply filters
    if filter_type == "Canton" and selected_cantons:
        merged_plants = merged_plants[merged_plants['Canton'].isin(selected_cantons)]
    elif filter_type == "Operator" and selected_operators:
        merged_plants = merged_plants[merged_plants['operator'].isin(selected_operators)]
    
    # Create heatmap
    fig = px.density_mapbox(
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
        radius=10,
        zoom=6,
        mapbox_style="carto-darkmatter",
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
    
    return fig, merged_plants

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
    
    if not latest_file:
        st.warning("No parquet files with format YYYY-MM.parquet found in the bucket.")
        return
    
    # Load capacity data
    with st.spinner("Loading data..."):
        # Get power plants data - load only once
        powerplants = get_powerplants_data(conn)
        
        # Load capacity data
        capa_df = download_and_load_parquet(latest_file, 'parquet', conn)
        
        # Get the latest date's capacity data
        if capa_df is not None:
            latest_mastr_date = capa_df.Date.max()
            st.warning(f"Master data latest update {latest_mastr_date.strftime('%Y-%m-%d')}")
            
            # Only keep the latest date data to save memory
            capa = capa_df.loc[capa_df.Date == latest_mastr_date].drop(columns='Date').reset_index(drop=True)
            
            # Clean up memory
            del capa_df
            gc.collect()
            
            # Download and process forecast data
            forecast_df = get_solar_forecast(selected_file, conn)
            forecast_df = process_forecast_data(forecast_df, selected_model)
            
            if forecast_df is not None:
                # Merge forecast with capacity data on Canton - more efficient with only needed columns
                forecast_columns = ['index', 'datetime', 'Canton', 'p0.5', 'p0.1', 'p0.9']
                
                # Ensure all needed columns exist
                forecast_columns = [col for col in forecast_columns if col in forecast_df.columns or col == 'index']
                
                # Merge dataframes
                merged_df = pd.merge(
                    forecast_df.reset_index()[forecast_columns], 
                    capa, 
                    on="Canton", 
                    how="left"
                )
                
                # Remove duplicates
                merged_df.drop_duplicates(['datetime', 'Canton', 'operator'], inplace=True)
                
                # Clean up memory
                del forecast_df
                gc.collect()
                
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
                    # Initialize filtered_df and selected items
                    filtered_df = merged_df.copy()
                    selected_cantons = []
                    selected_operators = []
                    
                    if filter_type == "Canton":
                        # Get all unique cantons
                        all_cantons = sorted(merged_df["Canton"].unique().tolist())
                        
                        # Create a multiselect widget for cantons
                        selected_cantons = st.multiselect(
                            "Select Cantons:",
                            options=all_cantons
                        )
                        
                        # Filter the dataframe based on selected cantons
                        if selected_cantons:
                            filtered_df = merged_df[merged_df["Canton"].isin(selected_cantons)].copy()
                        
                    elif filter_type == "Operator":
                        # Check if 'operator' column exists in merged_df
                        if 'operator' in merged_df.columns:
                            # Get all unique operators
                            all_operators = sorted(merged_df["operator"].unique().tolist())
                            
                            # Create a multiselect widget for operators
                            selected_operators = st.multiselect(
                                "Select Operators:",
                                options=all_operators
                            )
                            
                            # Filter the dataframe based on selected operators
                            if selected_operators:
                                filtered_df = merged_df[merged_df["operator"].isin(selected_operators)].copy()
                        else:
                            st.warning("No 'operator' column found in the data. Please use Canton filtering instead.")
                
                # Keep only needed columns to save memory
                filtered_df = filtered_df[['datetime', 'p0.5', 'p0.1', 'p0.9', 'Canton', 'operator', 
                                         'CumulativePower_canton', 'CumulativePower_operator']].copy()
                
                # Calculate installed capacity
                capa_installed = round(filtered_df.loc[filtered_df.datetime == filtered_df.datetime.max()]
                                      .drop_duplicates('operator')['CumulativePower_operator'].sum())
                
                st.success(f"Installed capacity: {round(capa_installed/1000):,.0f} MW")
                
                # Calculate power values - vectorized operations
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
                    # Create forecast chart
                    fig = create_forecast_chart(filtered_df, filter_type, selected_cantons, selected_operators)
                    st.plotly_chart(fig, use_container_width=True)
                else:  # Powerplant Location Heatmap
                    # Create heatmap
                    fig, merged_plants = create_heatmap(powerplants, filtered_df, filter_type, 
                                                       selected_cantons, selected_operators)
                    
                    # Display metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Plants", f"{len(merged_plants):,}")
                    with col2:
                        st.metric("Total Capacity", f"{merged_plants['TotalPower'].sum()/1000:,.2f} MW")
                    
                    # Display heatmap
                    st.plotly_chart(fig, use_container_width=True)
                
                # Clean up memory
                del merged_df
                del filtered_df
                gc.collect()
            else:
                st.error("Failed to load solar forecast data.")
        else:
            st.error("Failed to load capacity data.")

def about_page():
    st.title("About This App")
    st.write("This application displays solar power forecasts for Switzerland based on PRONOVO data.")
    st.write("The app has been optimized for memory efficiency.")

def main():
    st.sidebar.title("Navigation")
    
    # Cache clearing button
    if st.sidebar.button("Clear Cache"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.experimental_memo.clear()  # For older Streamlit versions
        st.sidebar.success("Cache cleared!")
    
    # Add memory usage info
    st.sidebar.subheader("Memory Management")
    if st.sidebar.button("Garbage Collect"):
        n = gc.collect()
        st.sidebar.success(f"Freed {n} objects from memory")
    
    page_choice = st.sidebar.radio("Go to page:", ["Home", "About"])
    
    if page_choice == "Home":
        home_page()
    elif page_choice == "About":
        about_page()

if __name__ == "__main__":
    main()
)
        parquet_files = [f for f in all_files if date_pattern.search(f)]
        
        if not parquet_files:
            return None, conn
        
        # Sort files by date in filename (newest first)
        latest_file = sorted(parquet_files, key=lambda x: date_pattern.search(x).group(1), reverse=True)[0]
        
        return latest_file, conn
    except Exception as e:
        st.error(f"Error listing parquet files: {e}")
        return None, conn

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_available_forecast_files(model, cluster):
    """Get list of available forecast files for the selected model and cluster"""
    conn = st.connection('gcs', type=FilesConnection)
    prefix = f"oracle_predictions/swiss_solar/canton_forecasts_factor/{model}/{cluster}"
    
    # Retrieve all files - fixed to use the correct API
    try:
        all_files = conn.fs.ls(prefix, max_results=100)
        # Filter for parquet files only
        parquet_files = [f for f in all_files if f.endswith('.parquet')]
        return sorted(parquet_files, reverse=True), conn
    except Exception as e:
        st.error(f"Error listing forecast files: {e}")
        return [], conn

@st.cache_data(ttl=3600)  # Cache for 1 hour
def download_and_load_parquet(file_path, format, conn):
    """Download and load the parquet file into a pandas DataFrame"""
    try:
        # Use the connection to read the parquet file directly
        df = conn.read(file_path, input_format=format)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_solar_forecast(forecast_path, conn):
    """Get the specific solar forecast parquet file"""
    try:
        # Download and load the forecast
        forecast_df = conn.read(forecast_path, input_format="parquet")
        return forecast_df
    except Exception as e:
        st.error(f"Error loading forecast parquet file: {e}")
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_powerplants_data(conn):
    """Get power plants master data"""
    return download_and_load_parquet(
        'oracle_predictions/swiss_solar/datasets/solar_mstr_data.csv', 
        'csv', 
        conn
    )[['Canton', 'operator', 'longitude', 'latitude', 'TotalPower']]

def process_forecast_data(forecast_df, selected_model):
    """Process forecast data based on model"""
    if forecast_df is None:
        return None
        
    if selected_model == 'icon_d2':
        max_idx = forecast_df.index.unique()[-1:]
        # Remove the last index which might contain incomplete data
        forecast_df = forecast_df.loc[forecast_df.index != max_idx[0]]
    
    return forecast_df

def create_forecast_chart(filtered_df, filter_type, selected_cantons=None, selected_operators=None):
    """Create forecast chart based on filtering"""
    fig = go.Figure()
    plot_df = filtered_df.copy()
    
    # Handle canton filtering
    if filter_type == "Canton" and selected_cantons:
        for canton in selected_cantons:
            canton_df = plot_df[plot_df['Canton'] == canton]
            canton_df = canton_df.sort_values('datetime')
            
            # Aggregate by datetime
            canton_df = canton_df.groupby(['datetime']).agg({
                'p0.5_operator': 'sum',
                'p0.1_operator': 'sum',
                'p0.9_operator': 'sum'
            }).reset_index()
            
            # Add traces
            fig.add_trace(go.Scatter(
                x=canton_df['datetime'],
                y=canton_df['p0.5_operator'],
                mode='lines',
                name=f'{canton} - Median (P50)',
                line=dict(width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=canton_df['datetime'],
                y=canton_df['p0.1_operator'],
                mode='lines',
                name=f'{canton} - Lower Bound (P10)',
                line=dict(width=1, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=canton_df['datetime'],
                y=canton_df['p0.9_operator'],
                mode='lines',
                name=f'{canton} - Upper Bound (P90)',
                line=dict(width=1, dash='dash')
            ))
    
    # Handle operator filtering
    elif filter_type == "Operator" and 'operator' in filtered_df.columns and selected_operators:
        for operator in selected_operators:
            operator_df = plot_df[plot_df['operator'] == operator]
            operator_df = operator_df.sort_values('datetime')
            
            # Aggregate by datetime
            operator_df = operator_df.groupby(['datetime']).agg({
                'p0.5_operator': 'sum',
                'p0.1_operator': 'sum',
                'p0.9_operator': 'sum'
            }).reset_index()
            
            # Add traces
            fig.add_trace(go.Scatter(
                x=operator_df['datetime'],
                y=operator_df['p0.5_operator'],
                mode='lines',
                name=f'{operator} - Median (P50)',
                line=dict(width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=operator_df['datetime'],
                y=operator_df['p0.1_operator'],
                mode='lines',
                name=f'{operator} - Lower Bound (P10)',
                line=dict(width=1, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=operator_df['datetime'],
                y=operator_df['p0.9_operator'],
                mode='lines',
                name=f'{operator} - Upper Bound (P90)',
                line=dict(width=1, dash='dash')
            ))
    
    # Default case - show total
    else:
        operator_df = filtered_df.copy().sort_values('datetime')
        
        # Aggregate by datetime
        operator_df = operator_df.groupby(['datetime']).agg({
            'p0.5_operator': 'sum',
            'p0.1_operator': 'sum',
            'p0.9_operator': 'sum'
        }).reset_index()
        
        # Add traces
        fig.add_trace(go.Scatter(
            x=operator_df['datetime'],
            y=operator_df['p0.5_operator'],
            mode='lines',
            name='Total - Median (P50)',
            line=dict(width=2, color='white')
        ))
        
        fig.add_trace(go.Scatter(
            x=operator_df['datetime'],
            y=operator_df['p0.1_operator'],
            mode='lines',
            name='Total - Lower Bound (P10)',
            line=dict(width=1, dash='dash', color='white')
        ))
        
        fig.add_trace(go.Scatter(
            x=operator_df['datetime'],
            y=operator_df['p0.9_operator'],
            mode='lines',
            name='Total - Upper Bound (P90)',
            line=dict(width=1, dash='dash', color='white')
        ))
    
    # Add total line for multiple selections
    has_multiple_selections = ((selected_operators and len(selected_operators) > 1) or 
                              (selected_cantons and len(selected_cantons) > 1))
    
    if has_multiple_selections:
        if selected_operators and len(selected_operators) > 1:
            total_df = plot_df[plot_df['operator'].isin(selected_operators)]
        elif selected_cantons and len(selected_cantons) > 1:
            total_df = plot_df[plot_df['Canton'].isin(selected_cantons)]
            
        # Group by datetime and sum
        total_df = total_df.groupby(['datetime']).agg({
            'p0.5_operator': 'sum',
            'p0.1_operator': 'sum',
            'p0.9_operator': 'sum'
        }).reset_index()
        
        # Add total traces
        fig.add_trace(go.Scatter(
            x=total_df['datetime'],
            y=total_df['p0.5_operator'],
            mode='lines',
            name='Total - Median (P50)',
            line=dict(width=3, color='white')
        ))
        
        fig.add_trace(go.Scatter(
            x=total_df['datetime'],
            y=total_df['p0.1_operator'],
            mode='lines',
            name='Total - Lower Bound (P10)',
            line=dict(width=2, dash='dash', color='white')
        ))
        
        fig.add_trace(go.Scatter(
            x=total_df['datetime'],
            y=total_df['p0.9_operator'],
            mode='lines',
            name='Total - Upper Bound (P90)',
            line=dict(width=2, dash='dash', color='white')
        ))
    
    # Update layout
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

def create_heatmap(powerplants, filtered_df, filter_type, selected_cantons=None, selected_operators=None):
    """Create heatmap of power plant locations"""
    # Extract latest datetime for forecast
    latest_datetime = filtered_df['datetime'].max()
    latest_forecast = filtered_df[filtered_df['datetime'] == latest_datetime].copy()
    
    # Merge with powerplants data
    merged_plants = pd.merge(
        powerplants,
        latest_forecast,
        on=["Canton", "operator"],
        how="inner"
    )
    
    # Apply filters
    if filter_type == "Canton" and selected_cantons:
        merged_plants = merged_plants[merged_plants['Canton'].isin(selected_cantons)]
    elif filter_type == "Operator" and selected_operators:
        merged_plants = merged_plants[merged_plants['operator'].isin(selected_operators)]
    
    # Create heatmap
    fig = px.density_mapbox(
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
        radius=10,
        zoom=6,
        mapbox_style="carto-darkmatter",
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
    
    return fig, merged_plants

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
    
    if not latest_file:
        st.warning("No parquet files with format YYYY-MM.parquet found in the bucket.")
        return
    
    # Load capacity data
    with st.spinner("Loading data..."):
        # Get power plants data - load only once
        powerplants = get_powerplants_data(conn)
        
        # Load capacity data
        capa_df = download_and_load_parquet(latest_file, 'parquet', conn)
        
        # Get the latest date's capacity data
        if capa_df is not None:
            latest_mastr_date = capa_df.Date.max()
            st.warning(f"Master data latest update {latest_mastr_date.strftime('%Y-%m-%d')}")
            
            # Only keep the latest date data to save memory
            capa = capa_df.loc[capa_df.Date == latest_mastr_date].drop(columns='Date').reset_index(drop=True)
            
            # Clean up memory
            del capa_df
            gc.collect()
            
            # Download and process forecast data
            forecast_df = get_solar_forecast(selected_file, conn)
            forecast_df = process_forecast_data(forecast_df, selected_model)
            
            if forecast_df is not None:
                # Merge forecast with capacity data on Canton - more efficient with only needed columns
                forecast_columns = ['index', 'datetime', 'Canton', 'p0.5', 'p0.1', 'p0.9']
                
                # Ensure all needed columns exist
                forecast_columns = [col for col in forecast_columns if col in forecast_df.columns or col == 'index']
                
                # Merge dataframes
                merged_df = pd.merge(
                    forecast_df.reset_index()[forecast_columns], 
                    capa, 
                    on="Canton", 
                    how="left"
                )
                
                # Remove duplicates
                merged_df.drop_duplicates(['datetime', 'Canton', 'operator'], inplace=True)
                
                # Clean up memory
                del forecast_df
                gc.collect()
                
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
                    # Initialize filtered_df and selected items
                    filtered_df = merged_df.copy()
                    selected_cantons = []
                    selected_operators = []
                    
                    if filter_type == "Canton":
                        # Get all unique cantons
                        all_cantons = sorted(merged_df["Canton"].unique().tolist())
                        
                        # Create a multiselect widget for cantons
                        selected_cantons = st.multiselect(
                            "Select Cantons:",
                            options=all_cantons
                        )
                        
                        # Filter the dataframe based on selected cantons
                        if selected_cantons:
                            filtered_df = merged_df[merged_df["Canton"].isin(selected_cantons)].copy()
                        
                    elif filter_type == "Operator":
                        # Check if 'operator' column exists in merged_df
                        if 'operator' in merged_df.columns:
                            # Get all unique operators
                            all_operators = sorted(merged_df["operator"].unique().tolist())
                            
                            # Create a multiselect widget for operators
                            selected_operators = st.multiselect(
                                "Select Operators:",
                                options=all_operators
                            )
                            
                            # Filter the dataframe based on selected operators
                            if selected_operators:
                                filtered_df = merged_df[merged_df["operator"].isin(selected_operators)].copy()
                        else:
                            st.warning("No 'operator' column found in the data. Please use Canton filtering instead.")
                
                # Keep only needed columns to save memory
                filtered_df = filtered_df[['datetime', 'p0.5', 'p0.1', 'p0.9', 'Canton', 'operator', 
                                         'CumulativePower_canton', 'CumulativePower_operator']].copy()
                
                # Calculate installed capacity
                capa_installed = round(filtered_df.loc[filtered_df.datetime == filtered_df.datetime.max()]
                                      .drop_duplicates('operator')['CumulativePower_operator'].sum())
                
                st.success(f"Installed capacity: {round(capa_installed/1000):,.0f} MW")
                
                # Calculate power values - vectorized operations
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
                    # Create forecast chart
                    fig = create_forecast_chart(filtered_df, filter_type, selected_cantons, selected_operators)
                    st.plotly_chart(fig, use_container_width=True)
                else:  # Powerplant Location Heatmap
                    # Create heatmap
                    fig, merged_plants = create_heatmap(powerplants, filtered_df, filter_type, 
                                                       selected_cantons, selected_operators)
                    
                    # Display metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Plants", f"{len(merged_plants):,}")
                    with col2:
                        st.metric("Total Capacity", f"{merged_plants['TotalPower'].sum()/1000:,.2f} MW")
                    
                    # Display heatmap
                    st.plotly_chart(fig, use_container_width=True)
                
                # Clean up memory
                del merged_df
                del filtered_df
                gc.collect()
            else:
                st.error("Failed to load solar forecast data.")
        else:
            st.error("Failed to load capacity data.")

def about_page():
    st.title("About This App")
    st.write("This application displays solar power forecasts for Switzerland based on PRONOVO data.")
    st.write("The app has been optimized for memory efficiency.")

def main():
    st.sidebar.title("Navigation")
    
    # Cache clearing button
    if st.sidebar.button("Clear Cache"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.experimental_memo.clear()  # For older Streamlit versions
        st.sidebar.success("Cache cleared!")
    
    # Add memory usage info
    st.sidebar.subheader("Memory Management")
    if st.sidebar.button("Garbage Collect"):
        n = gc.collect()
        st.sidebar.success(f"Freed {n} objects from memory")
    
    page_choice = st.sidebar.radio("Go to page:", ["Home", "About"])
    
    if page_choice == "Home":
        home_page()
    elif page_choice == "About":
        about_page()

if __name__ == "__main__":
    main()