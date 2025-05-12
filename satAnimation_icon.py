import xarray as xr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google.cloud import storage
import os
import tempfile
from datetime import datetime, timedelta
import pandas as pd
import json
import streamlit as st
from st_files_connection import FilesConnection

def concat_datasets(datasets):
    # Sort datasets by time
    datasets.sort(key=lambda ds: ds.time.values[0])
    
    # Concatenate along the time dimension
    combined_dataset = xr.concat(datasets, dim='time')
    
    # Ensure time values are datetime objects
    if not np.issubdtype(combined_dataset.time.dtype, np.datetime64):
        combined_dataset = combined_dataset.assign_coords(
            time=pd.to_datetime(combined_dataset.time.values)
        )
    
    # Convert time coordinate to pandas DatetimeIndex with UTC timezone
    time_index = pd.DatetimeIndex(combined_dataset.time.values).tz_localize('UTC')
    
    # Replace the time coordinate
    combined_dataset = combined_dataset.assign_coords(time=time_index)
    
    return combined_dataset


def load_geojson(file_path):
    """
    Load a GeoJSON file.
    
    Parameters:
    -----------
    file_path : str
        Path to the GeoJSON file
        
    Returns:
    --------
    dict
        GeoJSON data
    """
    with open(file_path, 'r') as f:
        geojson_data = json.load(f)
    
    return geojson_data

def create_boundary_traces(geojson_data):
    """
    Create traces for boundary lines from GeoJSON data.
    
    Parameters:
    -----------
    geojson_data : dict
        GeoJSON data
        
    Returns:
    --------
    list
        List of Scattergeo traces
    """
    traces = []
    
    for feature in geojson_data['features']:
        # Get the geometry
        geometry = feature['geometry']
        
        # Process MultiPolygon
        if geometry['type'] == 'MultiPolygon':
            for polygon in geometry['coordinates']:
                for ring in polygon:
                    lons, lats = zip(*ring)
                    traces.append(
                        go.Scattergeo(
                            lon=lons,
                            lat=lats,
                            mode='lines',
                            line=dict(width=2.5, color='white'),
                            showlegend=False,
                            hoverinfo='skip'
                        )
                    )
        
        # Process Polygon
        elif geometry['type'] == 'Polygon':
            for ring in geometry['coordinates']:
                lons, lats = zip(*ring)
                traces.append(
                    go.Scattergeo(
                        lon=lons,
                        lat=lats,
                        mode='lines',
                        line=dict(width=2.5, color='white'),
                        showlegend=False,
                        hoverinfo='skip'
                    )
                )
    
    return traces

def get_latest_nc_files(conn, prefix, count=12):
    """
    Get the latest count nc files from the specified prefix.
    
    Parameters:
    -----------
    conn : FilesConnection
        The connection to GCS
    prefix : str
        Prefix for the objects to list
    count : int, optional
        Number of latest files to return
        
    Returns:
    --------
    list
        List of file paths sorted by date (newest first)
    """
    try:
        # Invalidate the cache to refresh the bucket listing
        conn._instance.invalidate_cache(prefix)
        
        # List all files in the prefix
        files = conn._instance.ls(prefix, max_results=50)
        
        # Filter for .nc files
        nc_files = [f for f in files if f.endswith('.nc')]
        
        # Sort files by name (which should contain date information)
        nc_files.sort(reverse=True)
        
        # Return the latest count
        return nc_files[:count]
    except Exception as e:
        print(f"Error listing files: {e}")
        return []

def download_and_open_nc_files(conn, file_paths):
    """
    Download nc files from GCS and open them with xarray.
    
    Parameters:
    -----------
    conn : FilesConnection
        The connection to GCS
    file_paths : list
        List of file paths to download
        
    Returns:
    --------
    list
        List of xarray datasets
    """
    datasets = []
    
    # Create a temporary directory to store the downloaded files
    temp_dir = tempfile.mkdtemp()
    
    for file_path in file_paths:
        try:
            # Extract filename from path
            file_name = os.path.basename(file_path)
            temp_file_path = os.path.join(temp_dir, file_name)
            
            # Download the file to the temporary location
            conn._instance.get(file_path, temp_file_path)
            
            # Now open the local file with xarray
            ds = xr.open_dataset(temp_file_path)
            
            # Extract the timestamp from the filename
            timestamp_str = file_name.split('.')[0]
            timestamp = datetime.strptime(timestamp_str, '%Y%m%d%H%M')
            
            # Set the time coordinate
            ds = ds.assign_coords(time=[timestamp])
            datasets.append(ds)
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    return datasets


def get_connection():
    """Get the GCS connection instance"""
    return st.connection('gcs', type=FilesConnection)


def plot_multiple_solar_radiation_plots(xr_dataset, geojson_path=None, min_value=0, max_value=700, 
                                      downsample_factor=1, num_plots=12, num_columns=3):
    """
    Create multiple solar radiation plots arranged in a grid with a single colorbar at the top.
    
    Parameters:
    -----------
    xr_dataset : xarray.Dataset
        The dataset containing the solar radiation data
    geojson_path : str, optional
        Path to the GeoJSON file for boundaries
    min_value : float, optional
        Minimum value for the color scale
    max_value : float, optional
        Maximum value for the color scale
    downsample_factor : int, optional
        Factor to downsample the spatial resolution
    num_plots : int, optional
        Number of plots to create
    num_columns : int, optional
        Number of columns in the grid layout
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The figure containing multiple plots arranged in a grid with a single colorbar
    """
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    import json
    import numpy as np
    import math
    
    # Debug: Print dataset structure to understand its format
    print(f"Dataset dimensions: {xr_dataset.dims}")
    print(f"Dataset variables: {list(xr_dataset.variables)}")
    print(f"Dataset coordinates: {list(xr_dataset.coords)}")
    
    # Get the variable name for solar radiation
    var_name = 'SID' if 'SID' in xr_dataset.variables else list(xr_dataset.data_vars)[0]
    print(f"Using variable: {var_name}")
    
    # Get the coordinates properly - handle different naming conventions
    if 'lat' in xr_dataset.dims:
        lats = xr_dataset.lat.values
        lons = xr_dataset.lon.values
    elif 'latitude' in xr_dataset.dims:
        lats = xr_dataset.latitude.values
        lons = xr_dataset.longitude.values
    elif 'rlat' in xr_dataset.dims and 'rlon' in xr_dataset.dims:
        # For rotated lat/lon grids
        lats = xr_dataset.rlat.values
        lons = xr_dataset.rlon.values
    else:
        # If we can't find dimension names, try looking for coordinate variables
        possible_lat_names = ['lat', 'latitude', 'rlat', 'y']
        possible_lon_names = ['lon', 'longitude', 'rlon', 'x']
        
        for lat_name in possible_lat_names:
            if lat_name in xr_dataset.coords:
                lats = xr_dataset[lat_name].values
                break
        else:
            raise ValueError("Could not find latitude coordinate")
            
        for lon_name in possible_lon_names:
            if lon_name in xr_dataset.coords:
                lons = xr_dataset[lon_name].values
                break
        else:
            raise ValueError("Could not find longitude coordinate")
    
    # Print the shape of the coordinate arrays
    print(f"Latitude shape: {lats.shape}")
    print(f"Longitude shape: {lons.shape}")
    
    # Downsample spatial resolution for better performance
    # Ensure we don't downsample too much for small arrays
    downsample_factor = min(downsample_factor, max(1, len(lats)//20))
    lats_downsampled = lats[::downsample_factor]
    lons_downsampled = lons[::downsample_factor]
    
    # Load the boundary data more efficiently
    boundary_traces = []
    if geojson_path:
        try:
            # Check if the geojson file exists
            import os
            if not os.path.exists(geojson_path):
                print(f"Warning: GeoJSON file not found: {geojson_path}")
            else:
                # Load the boundary data once and create traces
                with open(geojson_path, 'r') as f:
                    geojson_data = json.load(f)
                
                # Extract coordinates based on GeoJSON type
                if geojson_data['type'] == 'FeatureCollection':
                    features = geojson_data['features']
                elif geojson_data['type'] == 'Feature':
                    features = [geojson_data]
                else:
                    # If it's a direct geometry
                    geometry = geojson_data
                    features = [{'geometry': geometry}]
                
                for feature in features:
                    geometry = feature.get('geometry', {})
                    if not geometry:
                        continue
                        
                    # Process different geometry types
                    if geometry['type'] == 'Polygon':
                        rings = geometry['coordinates']
                        # Add the outer ring of the polygon
                        for ring in rings:
                            lons_poly, lats_poly = zip(*ring)
                            boundary_traces.append(
                                go.Scatter(
                                    x=lons_poly,
                                    y=lats_poly,
                                    mode='lines',
                                    line=dict(color='white', width=2.5),
                                    hoverinfo='skip',
                                    showlegend=False
                                )
                            )
                    elif geometry['type'] == 'MultiPolygon':
                        for polygon in geometry['coordinates']:
                            for ring in polygon:
                                lons_poly, lats_poly = zip(*ring)
                                boundary_traces.append(
                                    go.Scatter(
                                        x=lons_poly,
                                        y=lats_poly,
                                        mode='lines',
                                        line=dict(color='white', width=2.5),
                                        hoverinfo='skip',
                                        showlegend=False
                                    )
                                )
        except Exception as e:
            print(f"Error loading GeoJSON: {e}")
            import traceback
            traceback.print_exc()
    
    # Get the time values - handle the case where valid_time might not be the name
    time_dim = 'valid_time' if 'valid_time' in xr_dataset.dims else 'time'
    time_values = xr_dataset[time_dim].values
    print(f"Number of time steps: {len(time_values)}")
    
    # Select time indices at regular intervals
    if len(time_values) <= num_plots:
        time_indices = list(range(len(time_values)))
    else:
        # Select frames at regular intervals
        step = max(1, len(time_values) // num_plots)
        time_indices = list(range(0, len(time_values), step))
        # Always include the last frame
        if len(time_values) - 1 not in time_indices:
            time_indices.append(len(time_values) - 1)
    
    # Sort time indices to ensure they're in order
    time_indices.sort()
    time_indices = time_indices[:num_plots]  # Limit to requested number of plots
    
    # Create time labels for each frame
    time_labels = []
    for t_idx in time_indices:
        if hasattr(xr_dataset[time_dim][t_idx], 'dt'):
            # If it's a pandas/numpy datetime
            ts = xr_dataset[time_dim][t_idx].dt.strftime('%Y-%m-%d %H:%M').values
            time_str = str(ts)
        else:
            # Try to convert from numpy datetime64
            try:
                time_str = pd.to_datetime(xr_dataset[time_dim][t_idx].values).strftime('%Y-%m-%d %H:%M')
            except:
                time_str = f"Frame {t_idx+1}"
        time_labels.append(time_str)
    
    # Calculate number of rows needed
    num_rows = math.ceil(len(time_indices) / num_columns)
    
    # Create a subplot with multiple rows and columns
    fig = make_subplots(
        rows=num_rows, 
        cols=num_columns,
        subplot_titles=[f"Solar Radiation at {time_str} CET" for time_str in time_labels],
        vertical_spacing=0.1,
        horizontal_spacing=0.05,
        specs=[[{'type': 'xy'} for _ in range(num_columns)] for _ in range(num_rows)]
    )
    
    # Create plot for each selected time
    for i, (t_idx, time_str) in enumerate(zip(time_indices, time_labels)):
        # Calculate row and column for this plot
        row = (i // num_columns) + 1
        col = (i % num_columns) + 1
        
        try:
            # Get data for this time and downsample, handling different dimension names
            if time_dim == 'valid_time':
                data_slice = xr_dataset[var_name].isel(valid_time=t_idx).values
            else:
                data_slice = xr_dataset[var_name].isel(time=t_idx).values
            
            # For 3D arrays, we might need to select a specific level
            if data_slice.ndim > 2:
                # Take first level if there are multiple levels
                data_slice = data_slice[0]
            
            # Downsample the data with proper dimension handling
            if data_slice.shape[0] == len(lats) and data_slice.shape[1] == len(lons):
                # Regular grid
                data_downsampled = data_slice[::downsample_factor, ::downsample_factor]
            else:
                # Irregular grid or other shape - reshape data
                print(f"Warning: Data shape {data_slice.shape} doesn't match coordinates: lats {len(lats)}, lons {len(lons)}")
                import scipy.ndimage
                zoom_factors = (len(lats_downsampled)/data_slice.shape[0], len(lons_downsampled)/data_slice.shape[1])
                data_downsampled = scipy.ndimage.zoom(data_slice, zoom_factors, order=1)
            
            # Debug: check data values
            print(f"Plot {i} data min: {np.nanmin(data_downsampled)}, max: {np.nanmax(data_downsampled)}")
            
            # Replace NaN values with a default value if necessary
            data_downsampled = np.nan_to_num(data_downsampled, nan=-999)
            
            # Add contour plot WITHOUT a colorbar (we'll add a single colorbar later)
            contour = go.Contour(
                z=data_downsampled,
                x=lons_downsampled,
                y=lats_downsampled,
                colorscale='turbo',
                zmin=min_value,
                zmax=max_value,
                ncontours=50,
                contours=dict(
                    start=min_value,
                    end=max_value,
                    size=(max_value-min_value)/20,
                    showlabels=True,
                    labelfont=dict(
                        size=8,
                        color='white',
                    ),
                ),
                line=dict(width=0.),
                showscale=False,  # Hide individual colorbars
                hoverinfo='none',
            )
            
            # Add the contour to the subplot
            fig.add_trace(contour, row=row, col=col)
            
            # Add boundary traces for this subplot
            for boundary in boundary_traces:
                fig.add_trace(
                    go.Scatter(
                        x=boundary.x,
                        y=boundary.y,
                        mode='lines',
                        line=dict(color='white', width=1.5),  # Thinner lines for grid layout
                        hoverinfo='skip',
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
        except Exception as e:
            print(f"Error creating plot {i}: {e}")
            import traceback
            traceback.print_exc()
            
            # Add error annotation to the subplot
            fig.add_annotation(
                text=f"Error: Could not create plot for {time_str}",
                xref=f"x{i+1}", yref=f"y{i+1}",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color="white")
            )
    
    # Calculate an appropriate height based on number of rows
    # Each plot is approximately 300px tall, but we need less when arranged in a grid
    plot_height = 240  # Height per plot in pixels
    
    # Add a single colorbar at the top
    # Create a dummy contour trace just for the colorbar
    dummy_z = np.linspace(min_value, max_value, 10).reshape(2, 5)
    colorbar_trace = go.Contour(
        z=dummy_z,
        x=[0, 1],
        y=[0, 1],
        zmin=min_value,
        zmax=max_value,
        colorscale='turbo',
        showscale=True,
        colorbar=dict(
            title='W/m²',
            title_side='right',
            thickness=15,
            len=0.6,
            yanchor="top",
            y=1.02,  # Position above the plots
            xanchor="center",
            x=0.5,   # Center horizontally
            orientation="h",  # Horizontal colorbar
            tickmode='auto',
            nticks=8,
        ),
        hoverinfo='none',
    )
    
    # Create a new empty subplot for the colorbar
    # Add the dummy trace to a new row at the top
    fig.add_trace(
        colorbar_trace,
        row=1, col=1  # Adding to first subplot but will be hidden
    )
    # Hide the dummy trace completely
    dummy_trace_index = len(fig.data) - 1
    fig.data[dummy_trace_index].visible = True
    fig.data[dummy_trace_index].opacity = 0
    
    # Update layout for all subplots
    fig.update_layout(
        height=plot_height * num_rows + 80,  # Add extra space for the colorbar at top
        width=300 * num_columns,  # Width per column
        template="plotly_dark",
        margin=dict(l=50, r=80, t=120, b=50),  # Increased top margin for colorbar
    )
    
    # Update x and y axes for all subplots to maintain aspect ratio and labels
    for i in range(1, len(time_indices) + 1):
        row = ((i-1) // num_columns) + 1
        col = ((i-1) % num_columns) + 1
        
        # Only add x-axis title to the bottom row
        show_x_title = (row == num_rows) or (i + num_columns > len(time_indices))
        
        # Only add y-axis title to the leftmost column
        show_y_title = (col == 1)
        
        fig.update_xaxes(
            title='Longitude' if show_x_title else '',
            constrain='domain',
            autorange=True,
            row=row, col=col,
            tickfont=dict(size=10),  # Smaller tick labels for grid layout
            title_font=dict(size=12)  # Smaller axis titles
        )
        fig.update_yaxes(
            title='Latitude' if show_y_title else '',
            scaleanchor=f'x{i}',
            scaleratio=1,
            autorange=True,
            row=row, col=col,
            tickfont=dict(size=10),  # Smaller tick labels
            title_font=dict(size=12)  # Smaller axis titles
        )
    
    # Make subplot titles smaller
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=12)
    
    return fig


def generate_sat_rad_multi_plots():
    """
    Generate multiple solar radiation plots arranged in a grid layout.
    """
    # Set the prefix
    prefix = "icon-ch/ch1/radiation/"
    
    # Get the connection using FilesConnection
    conn = get_connection()
    
    # Get the latest nc files - reduced from original count
    files = get_latest_nc_files(conn, prefix, count=1)
    
    # Download and open the files
    datasets = download_and_open_nc_files(conn, files)
    
    # Concatenate the datasets
    combined_dataset = concat_datasets(datasets)

    min_lon, max_lon = 5.8, 10.5
    min_lat, max_lat = 45.8, 48

    combined_dataset = combined_dataset.where((combined_dataset['lon'] >= min_lon) & 
                      (combined_dataset['lon'] <= max_lon) & 
                      (combined_dataset['lat'] >= min_lat) & 
                      (combined_dataset['lat'] <= max_lat), 
                      drop=True)

    # Rename variables
    ds_renamed_var = combined_dataset.rename({'GLOBAL_SW': 'SID'})[['SID']]
    
    # Convert time zones
    time_index = pd.DatetimeIndex(ds_renamed_var.valid_time.values).tz_localize('UTC')
    ds_renamed_var = ds_renamed_var.assign_coords(valid_time=time_index.tz_convert('CET'))
    
    # Path to the Swiss cantonal boundaries GeoJSON
    geojson_path = 'swissBOUNDARIES3D_1_3_TLM_KANTONSGEBIET.geojson'
    
    # Create multiple plots in a grid layout with 3 columns
    fig = plot_multiple_solar_radiation_plots(
        ds_renamed_var, 
        geojson_path, 
        min_value=0, 
        max_value=1100,
        downsample_factor=1,
        num_plots=32,       # Create 12 plots (can be adjusted)
        num_columns=3       # Arrange in 3 columns
    )
    
    return fig

