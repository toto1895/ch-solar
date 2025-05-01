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
if not os.path.exists('.streamlit'):
    os.makedirs('.streamlit')

with open('.streamlit/config.toml', 'w') as f:
    f.write('''
[theme]
base = "dark"
    ''')

# Page configuration
st.set_page_config(
    page_title="Swiss solar dashboard",
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

# Create a connection instance once
def get_connection():
    """Get the GCS connection instance"""
    return st.connection('gcs', type=FilesConnection)

def fetch_files(conn, prefix, pattern=None):
    """Fetch files from a bucket prefix with optional pattern matching"""
    try:
        # Invalidate the cache to refresh the bucket listing
        conn._instance.invalidate_cache(prefix)
        # Retrieve all files
        files = conn._instance.ls(prefix, max_results=100)
        
        # Apply pattern filtering if provided
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
    
    # Sort files by date in filename (newest first)
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
    prefix = f"oracle_predictions/swiss_solar/canton_forecasts_factor/{model}/{cluster}"
    return fetch_files(conn, prefix, r'\.parquet$'), conn

def load_and_concat_parquet_files(conn, date_str, time_str=None):
    """
    Load and concatenate parquet files from a specific date and optional time
    
    Parameters:
    -----------
    date_str : str
        Date string in format YYYYMMDD (e.g., '20250428')
    time_str : str or list, optional
        Time string(s) in format HHMM (e.g., '0445'), or list of time strings
        
    Returns:
    --------
    pd.DataFrame
        Concatenated dataframe from all matching parquet files
    """
    # Get connection
    #conn = get_connection()
    
    # Set up the prefix to look in
    prefix = "dwd-solar-sat/daily_agg_asset_level_prod/"
    
    # Create pattern based on date and optional time
    if time_str:
        if isinstance(time_str, list):
            # Create a pattern for multiple specific times
            time_patterns = '|'.join([f"{date_str}{t}" for t in time_str])
            pattern = f"({time_patterns})\.parquet$"
        else:
            # Pattern for a single specific time
            pattern = f"{date_str}{time_str}\.parquet$"
    else:
        # Pattern for any time on the specified date
        pattern = f"{date_str}\.parquet$"
    
    # Fetch matching files
    files = fetch_files(conn, prefix, pattern)
    
    if not files:
        st.warning(f"No files found matching the pattern: {pattern}")
        return None
    
    # Load and concatenate files
    dataframes = []
    for file_path in files:
        try:
            # Read file from GCS
            with conn._instance.open(file_path, mode='rb') as f:
                # Read parquet content
                df = pd.read_parquet(io.BytesIO(f.read()))
                dataframes.append(df)
                #st.info(f"Loaded: {file_path}")
        except Exception as e:
            st.error(f"Error reading {file_path}: {e}")
    
    if not dataframes:
        st.error("No dataframes could be loaded successfully")
        return None
    
    # Concatenate all dataframes
    concatenated_df = pd.concat(dataframes)
    #st.success(f"Successfully concatenated {len(dataframes)} files. " 
    #           f"Total rows: {len(concatenated_df)}")
    
    return concatenated_df


def create_forecast_chart(filtered_df, nowcast, filter_type, selected_cantons=None, selected_operators=None):
    """Create a forecast chart based on filtered data"""
    fig = go.Figure()
    plot_df = filtered_df.copy()
    
    # Case 1: Canton filtering
    if filter_type == "Canton" and selected_cantons:
        for canton in selected_cantons:
            canton_df = plot_df[plot_df['Canton'] == canton]
            canton_df = canton_df.sort_values('datetime')

            canton_now = nowcast[nowcast['Canton'] == canton]
            canton_now = canton_now.sort_values('datetime')
            
            canton_df = canton_df.groupby(['datetime']).agg({
                'p0.5_operator': 'sum',
                'p0.1_operator': 'sum',
                'p0.9_operator': 'sum'
            }).reset_index()

            canton_now = canton_now.groupby(['datetime']).agg({
                'SolarProduction':'sum'
            })
            
            # Add traces for this canton
            add_forecast_traces(fig, canton_df, canton)
            
    # Case 2: Operator filtering
    elif filter_type == "Operator" and 'operator' in filtered_df.columns and selected_operators:
        for operator in selected_operators:
            operator_df = plot_df[plot_df['operator'] == operator]
            operator_df = operator_df.sort_values('datetime')
            
            operator_df = operator_df.groupby(['datetime']).agg({
                'p0.5_operator': 'sum',
                'p0.1_operator': 'sum',
                'p0.9_operator': 'sum'
            }).reset_index()
            
            # Add traces for this operator
            add_forecast_traces(fig, operator_df, operator)

    
    # Case 3: No specific filtering
    else:
        operator_df = filtered_df.copy().sort_values('datetime')
        operator_df = operator_df.groupby(['datetime']).agg({
            'p0.5_operator': 'sum',
            'p0.1_operator': 'sum',
            'p0.9_operator': 'sum'
        }).reset_index()

        canton_now = canton_now.groupby(['datetime']).agg({
                'SolarProduction':'sum'
            })
        
        # Add traces for the total
        add_forecast_traces(fig, operator_df, "Total", color='red')
        add_forecast_traces(fig, canton_now, "Nowcast", color='white')
    
    # Add total line if multiple selections
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

        canton_now = canton_now.groupby(['datetime']).agg({
                'SolarProduction':'sum'
            })
        
        # Add traces for the total
        add_forecast_traces(fig, total_df, "Total", line_width=3, color='red')
        add_forecast_traces(fig, canton_now, "Nowcast", color='white')
    
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

def add_forecast_traces(fig, df, name, line_width=2, color=None):
    """Add forecast traces to the figure"""
    # Base style settings
    line_style = dict(width=line_width)
    dash_style = dict(width=max(1, line_width-1), dash='dash')
    
    # Apply color if specified
    if color:
        line_style['color'] = color
        dash_style['color'] = color
    
    # Add median forecast line
    fig.add_trace(go.Scatter(
        x=df['datetime'],
        y=df['p0.5_operator'],
        mode='lines',
        name=f'{name} - Median (P50)',
        line=line_style
    ))
    
    # Add lower bound
    fig.add_trace(go.Scatter(
        x=df['datetime'],
        y=df['p0.1_operator'],
        mode='lines',
        name=f'{name} - Lower Bound (P10)',
        line=dash_style
    ))
    
    # Add upper bound
    fig.add_trace(go.Scatter(
        x=df['datetime'],
        y=df['p0.9_operator'],
        mode='lines',
        name=f'{name} - Upper Bound (P90)',
        line=dash_style
    ))

def create_heatmap(merged_plants):
    """Create a heatmap visualization for plant locations"""
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
        radius=10,
        zoom=7,
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
    
    return fig

def download_parquet_from_gcs(bucket_name, prefix, date_pattern):
    """
    Download parquet files from GCS bucket matching a specific date pattern
    """
    # Initialize GCS client
    storage_client = storage.Client()
    
    # Get bucket
    bucket = storage_client.get_bucket(bucket_name)
    
    # List files with the prefix
    blobs = list(bucket.list_blobs(prefix=prefix))
    
    # Filter blobs by date pattern
    filtered_blobs = [blob for blob in blobs if date_pattern in blob.name]
    
    # Print the files being downloaded
    print(f"Downloading {len(filtered_blobs)} files:")
    for blob in filtered_blobs:
        print(f"- {blob.name}")
    
    # Download and read each file
    dataframes = []
    for blob in filtered_blobs:
        file_content = blob.download_as_bytes()
        dataframe = pd.read_parquet(io.BytesIO(file_content))
        dataframes.append(dataframe)
        print(f"Downloaded and read {blob.name}")
    
    return dataframes

def home_page():
    st.title("Swiss Solar Forecasts")
    
    # Initialize connection
    conn = get_connection()

    


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
    forecast_files, _ = get_forecast_files(selected_model, selected_cluster, conn)

    
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
    latest_file = get_latest_parquet_file(conn)
    
    # Load the power plants data
    powerplants = load_data('oracle_predictions/swiss_solar/datasets/solar_mstr_data.csv', 'csv', conn)
    
    if powerplants is not None:
        powerplants = powerplants[['Canton', 'operator', 'longitude', 'latitude', 'TotalPower']]
    
    if not latest_file:
        st.warning("No capacity data files found")
        return
    
    #map = powerplants.drop_duplicates(['Canton','operator'])[['Canton','operator']].reset_index(drop=True)
    # Main data loading and processing
    with st.spinner("Downloading and processing capacity data..."):
        # Load capacity data
        capa_df = load_data(latest_file, 'parquet', conn)

        #unique_pairs = set(zip(map['Canton'], map['operator']))
        #capa_df = capa_d[capa_d.apply(lambda row: (row['Canton'], row['operator']) in unique_pairs, axis=1)]
        
        if capa_df is None:
            st.error("Failed to load capacity data")
            return
            
        # Get the latest date's capacity data
        latest_mastr_date = capa_df.date.max()
        capa_df = capa_df.loc[capa_df.date == latest_mastr_date].drop(columns='date').reset_index(drop=True)
        
        # Status notification
        st.warning(f"Master data latest update {latest_mastr_date.strftime('%Y-%m-%d')}")
        full_capa = load_data('oracle_predictions/swiss_solar/datasets/capa_timeseries/full_dataset.parquet', 'parquet', conn)
        # Download the selected solar forecast data
        with st.spinner(f"Downloading solar forecast data from {selected_file}..."):
            forecast_df = load_data(selected_file, 'parquet', conn)
            
            if forecast_df is None:
                st.error("Failed to load forecast data")
                return
                
            # Special handling for icon_d2 model
            if selected_model == 'icon_d2':
                percentile_cols = ['p0.05', 'p0.1', 'p0.2', 'p0.3', 'p0.4', 'p0.5', 
                                'p0.6', 'p0.7', 'p0.8', 'p0.9', 'p0.95']
                max_idx = forecast_df.index.unique()[-1:]
                forecast_df = forecast_df.loc[forecast_df.index != max_idx[0]]
            
            # Merge forecast with capacity data
            merged_df = pd.merge(forecast_df.reset_index(), capa_df, on="Canton", how="left")
            merged_df.drop_duplicates(['datetime', 'Canton', 'operator'], inplace=True)

            dt = merged_df['datetime'].min()
            nowcast = load_and_concat_parquet_files(conn, dt.strftime("%Y%m%d"),
            #                                         ['0445', '0500']
                                                     )
    
            #merged_df = pd.merge(merged_df, nowcast, on=["datetime","Canton",'operator'], how="left")
            #st.dataframe(merged_df.tail())
            
            # Clean up to free memory
            del capa_df
            del forecast_df
            gc.collect()
            
            # Add filter section
            st.subheader("Filter Data")
            
            # Create columns for filter selection
            filter_col1, filter_col2 = st.columns([1, 3])
            
            with filter_col1:
                filter_type = st.selectbox(
                    "Filter by:",
                    options=["Canton", "Operator"],
                    index=0
                )
            
            # Initialize variables
            selected_cantons = []
            selected_operators = []
            
            with filter_col2:
                # Initialize filtered_df
                filtered_df = merged_df.copy()
                
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
                        filtered_df = merged_df[merged_df["Canton"].isin(selected_cantons)]
                        full_capa = full_capa[full_capa["Canton"].isin(selected_cantons)]
                        nowcast = nowcast[nowcast["Canton"].isin(selected_cantons)]
                    
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
                            filtered_df = merged_df[merged_df["operator"].isin(selected_operators)]
                            nowcast = nowcast[nowcast["operator"].isin(selected_operators)]
                            full_capa = full_capa[full_capa["operator"].isin(selected_operators)]
                    else:
                        st.warning("No 'operator' column found in the data. Please use Canton filtering instead.")
            
            # Clean up merged_df to free memory
            del merged_df
            gc.collect()
            
            # Prepare the filtered dataframe for visualization
            filtered_df = filtered_df[['datetime', 'p0.5', 'p0.1', 'p0.9', 'Canton', 'operator',
                                       'cum_canton', 'cum_operator','year_month','TotalPower']]
            filtered_df.drop_duplicates(['datetime','Canton','operator'], inplace=True)
            nowcast.drop_duplicates(['datetime','Canton','operator'], inplace=True)
            #st.dataframe(filtered_df)

            capa_installed =filtered_df.loc[filtered_df.datetime == filtered_df.datetime.max()
                                                   ].groupby('datetime')['cum_operator'].sum().values[0]
            st.success(f"Declared installed capacity: {round(capa_installed/1000):,.0f} MW  ( Today ~{1.1*round(capa_installed/1000):,.0f} MW) ")
            
            # Calculate power metrics
            filtered_df['p0.5_canton'] = 1.1*filtered_df['p0.5'] * filtered_df['cum_canton'] / 1000
            filtered_df['p0.1_canton'] = 1.1*filtered_df['p0.1'] * filtered_df['cum_canton'] / 1000
            filtered_df['p0.9_canton'] = 1.1*filtered_df['p0.9'] * filtered_df['cum_canton'] / 1000
            
            filtered_df['p0.5_operator'] = 1.1*filtered_df['p0.5'] * filtered_df['cum_operator'] / 1000
            filtered_df['p0.1_operator'] = 1.1*filtered_df['p0.1'] * filtered_df['cum_operator'] / 1000
            filtered_df['p0.9_operator'] = 1.1*filtered_df['p0.9'] * filtered_df['cum_operator'] / 1000
            
            # Add a radio button for chart type selection
            chart_type = st.radio(
                "Select visualization type:",
                options=["Forecast Chart", 'Monthly installed capacity',"Powerplant Location Heatmap"],
                horizontal=True
            )
            
            if chart_type == "Forecast Chart":
                # Create forecast chart
                fig = create_forecast_chart(filtered_df,nowcast, filter_type, selected_cantons, selected_operators)
                st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type =='Monthly installed capacity':
                full_capa = full_capa.groupby('year_month')['TotalPower'].sum()

                st.subheader('Monthly added capacity [MW]')
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=full_capa.index,  # Use the index of the grouped Series
                    y=full_capa.values/1000,
                    name='Added Cap a'
                    # Remove the mode='lines' parameter as it's not applicable for bar charts
                ))
                st.plotly_chart(fig, use_container_width=True)


            else:  # Powerplant Location Heatmap
                if powerplants is None:
                    st.error("Powerplant data is not available for the heatmap visualization")
                    return
                    
                # Extract the latest datetime for forecast
                latest_datetime = filtered_df['datetime'].max()
                latest_forecast = filtered_df[filtered_df['datetime'] == latest_datetime].copy()
                
                # Merge with powerplants data
                merge_conditions = ["Canton", "operator"]
                merged_plants = pd.merge(powerplants, latest_forecast, on=merge_conditions, how="inner")
                
                # Apply filters
                if filter_type == "Canton" and selected_cantons:
                    merged_plants = merged_plants[merged_plants['Canton'].isin(selected_cantons)]
                elif filter_type == "Operator" and selected_operators:
                    merged_plants = merged_plants[merged_plants['operator'].isin(selected_operators)]
                
                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Plants", f"{len(merged_plants):,}")
                with col2:
                    st.metric("Total Capacity", f"{merged_plants['TotalPower_x'].sum()/1000:,.2f} MW")
                
                # Create heatmap
                fig = create_heatmap(merged_plants)
                st.plotly_chart(fig, use_container_width=True)

def about_page():
    st.title("About This App")
    st.write("This application displays solar power forecasts for Switzerland based on PRONOVO data.")
    
    st.markdown("""
    ### Data Sources
    - Solar installation data from the PRONOVO registry
    - Weather forecast data from multiple meteorological models
    
    ### Features
    - View forecasts by canton or operator
    - Compare multiple forecast scenarios
    - Visualize solar plant locations across Switzerland
    
    ### Contact
    For more information or support, please contact the development team.
    """)

def main():
    
    st.sidebar.title("Navigation")

    # col1, col2, col3, col4 = st.columns(4)
    # with col4:
    #    st.components.v1.html("""
    #    <script type="text/javascript" src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js" data-name="bmc-button" data-slug="wamine" data-color="#FFDD00" data-emoji=""  data-font="Cookie" data-text="Buy me a coffee" data-outline-color="#000000" data-font-color="#000000" data-coffee-color="#ffffff" ></script>
    #    """, height=70)
   
    if st.sidebar.button("Clear Cache"):  
        st.cache_resource.clear()
        st.cache_data.clear()
        st.sidebar.success("Cache cleared!")

    page_choice = st.sidebar.radio("Go to page:", ["Home", "About"])

    if page_choice == "Home":
        home_page()
    elif page_choice == "About":
        about_page()

if __name__ == "__main__":
    
    main()

    #add_google_analytics('G-NKZVTQPKS5')
