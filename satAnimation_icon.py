# -*- coding: utf-8 -*-
"""
Created on Sat May  3 12:07:54 2025

@author: weibe
"""
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
                            line=dict(width=1, color='white'),
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
                        line=dict(width=1, color='white'),
                        showlegend=False,
                        hoverinfo='skip'
                    )
                )
    
    return traces


def plot_solar_radiation_animation_optimized(xr_dataset, geojson_path=None, min_value=0, max_value=700, 
                                 downsample_factor=2, max_frames=12):
    """
    Create an optimized animation of solar radiation data from an xarray dataset with Swiss boundaries.
    
    Parameters:
    -----------
    xr_dataset : xarray.Dataset
        Dataset containing solar radiation data with dimensions (time, lat, lon)
    geojson_path : str, optional
        Path to the GeoJSON file with boundary data
    min_value : float, optional
        Minimum value for the color scale
    max_value : float, optional
        Maximum value for the color scale
    downsample_factor : int, optional
        Factor by which to downsample the spatial resolution
    max_frames : int, optional
        Maximum number of frames to include in the animation
        
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    import pandas as pd
    import json
    import numpy as np
    
    # Get the variable name for solar radiation (assuming it's SID)
    var_name = 'SID' if 'SID' in xr_dataset.variables else list(xr_dataset.data_vars)[0]
    
    # Create figure
    fig = go.Figure()
    
    # Get the coordinates properly
    if 'lat' in xr_dataset.dims:
        lats = xr_dataset.lat.values
        lons = xr_dataset.lon.values
    elif 'latitude' in xr_dataset.dims:
        lats = xr_dataset.latitude.values
        lons = xr_dataset.longitude.values
    else:
        raise ValueError("Could not find latitude/longitude dimensions in the dataset")
    
    # Downsample spatial resolution for better performance
    lats_downsampled = lats[::downsample_factor]
    lons_downsampled = lons[::downsample_factor]
    
    # Load the boundary data more efficiently
    boundary_traces = []
    if geojson_path:
        try:
            # Load the boundary data once and create traces
            with open(geojson_path, 'r') as f:
                geojson_data = json.load(f)
            
            # Extract simplified boundary coordinates
            # We'll just create a single trace for better performance
            if geojson_data['type'] == 'FeatureCollection':
                all_lons = []
                all_lats = []
                for feature in geojson_data['features']:
                    geometry = feature['geometry']
                    if geometry['type'] == 'Polygon':
                        for ring in geometry['coordinates']:
                            # Add None to create discontinuity between polygons
                            if all_lons:  # Not the first polygon
                                all_lons.append(None)
                                all_lats.append(None)
                            # Downsample the coordinates for better performance
                            coords = ring[::3]  # Take every 3rd coordinate
                            lons_poly, lats_poly = zip(*coords)
                            all_lons.extend(lons_poly)
                            all_lats.extend(lats_poly)
                    elif geometry['type'] == 'MultiPolygon':
                        for polygon in geometry['coordinates']:
                            for ring in polygon:
                                if all_lons:  # Not the first polygon
                                    all_lons.append(None)
                                    all_lats.append(None)
                                coords = ring[::3]  # Take every 3rd coordinate
                                lons_poly, lats_poly = zip(*coords)
                                all_lons.extend(lons_poly)
                                all_lats.extend(lats_poly)
                
                # Create single boundary trace
                if all_lons:
                    boundary_traces.append(
                        go.Scatter(
                            x=all_lons,
                            y=all_lats,
                            mode='lines',
                            line=dict(color='white', width=1),
                            hoverinfo='skip',
                            showlegend=False
                        )
                    )
        except Exception as e:
            print(f"Error loading GeoJSON: {e}")
    
    # Limit the number of frames
    time_values = xr_dataset.valid_time.values
    if len(time_values) > max_frames:
        # Select frames at regular intervals
        step = len(time_values) // max_frames
        time_indices = list(range(0, len(time_values), step))
        # Always include the last frame
        if len(time_values) - 1 not in time_indices:
            time_indices.append(len(time_values) - 1)
    else:
        time_indices = list(range(len(time_values)))
    
    # Get the last time index (always included)
    last_t_idx = time_indices[-1]
    
    # Create the animation frames
    frames = []
    for i, t_idx in enumerate(time_indices):
        # Get data for this time and downsample
        data_slice = xr_dataset[var_name].isel(valid_time=t_idx).values[::downsample_factor, ::downsample_factor]
        time_str = pd.to_datetime(xr_dataset.valid_time[t_idx].values).tz_localize('UTC').tz_convert('CET').strftime('%Y-%m-%d %H:%M')
        
        # Create optimized heatmap 
        # Using heatmap instead of contour for better performance
        frame_data = [
            go.Heatmap(
                z=data_slice,
                x=lons_downsampled,
                y=lats_downsampled,
                colorscale='turbo',
                zmin=min_value,
                zmax=max_value,
                colorbar=dict(
                    title='W/m²',
                    title_side='right',
                    orientation='h',     
                    y=-0.15,             
                    len=0.6,             
                    thickness=20,        
                    tickmode='auto',     
                    nticks=8            
                ),
                hovertemplate='Lon: %{x:.2f}<br>Lat: %{y:.2f}<br>Solar Radiation: %{z:.1f} W/m²<extra></extra>'
            )
        ]
        
        # Create frame
        frame = go.Frame(
            data=frame_data + boundary_traces,  # Add boundary traces to each frame
            name=f'frame{i}',
            layout=go.Layout(
                title=dict(
                    text=f"Solar Radiation at {time_str} CET",
                    x=0.4,
                    y=0.865,
                    xanchor='left',
                    yanchor='top'
                )
            )
        )
        frames.append(frame)
    
    # Initial data for the figure - use last frame
    initial_data = [
        go.Heatmap(
            z=xr_dataset[var_name].isel(valid_time=last_t_idx).values[::downsample_factor, ::downsample_factor],
            x=lons_downsampled,
            y=lats_downsampled,
            colorscale='turbo',
            zmin=min_value,
            zmax=max_value,
            colorbar=dict(
                title='W/m²',
                title_side='right',
                orientation='h',
                y=-0.15,
                len=0.6,
                thickness=20,
                tickmode='auto',
                nticks=8
            ),
            hovertemplate='Lon: %{x:.2f}<br>Lat: %{y:.2f}<br>Solar Radiation: %{z:.1f} W/m²<extra></extra>'
        )
    ]
    
    # Add boundary traces to initial data
    for trace in boundary_traces:
        initial_data.append(trace)
    
    # Add traces to figure
    for trace in initial_data:
        fig.add_trace(trace)
    
    # Compute the time string for the last time index
    last_time_str = pd.to_datetime(xr_dataset.valid_time[last_t_idx].values).tz_localize('UTC').tz_convert('CET').strftime('%Y-%m-%d %H:%M')
    
    # Update layout with optimized settings
    fig.update_layout(
        title=dict(
            text=f"Solar Radiation at {last_time_str} CET",
            x=0.0,
            y=0.95,
            xanchor='left',
            yanchor='top'
        ),
        xaxis=dict(
            title='Longitude',
            constrain='domain',
            autorange=True
        ),
        yaxis=dict(
            title='Latitude',
            scaleanchor='x',
            scaleratio=1,
            autorange=True
        ),
        margin=dict(l=0, r=0, t=90, b=80),
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 10},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 1.05,
                "yanchor": "bottom"
            }
        ],
        sliders=[
            {
                "active": len(frames) - 1,  # Set to last frame
                "yanchor": "bottom",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 16},
                    "prefix": "Time: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 10},
                "len": 0.9,
                "x": 0.1,
                "y": 1.05,
                "steps": [
                    {
                        "args": [
                            [f"frame{i}"],
                            {
                                "frame": {"duration": 300, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 300}
                            }
                        ],
                        "label": pd.to_datetime(xr_dataset.valid_time[time_indices[i]].values).tz_localize('UTC').tz_convert('CET').strftime('%H:%M CET'),
                        "method": "animate"
                    }
                    for i in range(len(frames))
                ]
            }
        ],
        height=700,  # Reduced height for better performance
        width=700,   # Reduced width for better performance
        template="plotly_dark",
    )
    
    fig.frames = frames
    return fig


def generate_sat_rad_anim_ch1_optimized():
    """
    Optimized version of the original function to generate the solar radiation animation.
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

    # Rename variables
    ds_renamed_var = combined_dataset.rename({'GLOBAL_SW': 'SID'})['SID']
    
    # Convert time zones
    time_index = pd.DatetimeIndex(ds_renamed_var.valid_time.values).tz_localize('UTC')
    ds_renamed_var = ds_renamed_var.assign_coords(valid_time=time_index.tz_convert('CET'))
    
    # Path to the Swiss cantonal boundaries GeoJSON
    geojson_path = 'swissBOUNDARIES3D_1_3_TLM_KANTONSGEBIET.geojson'
    
    # Create the animation with optimized settings
    # Use downsampling and limit frames for better performance
    fig = plot_solar_radiation_animation_optimized(
        ds_renamed_var, 
        geojson_path, 
        min_value=0, 
        max_value=900,
        downsample_factor=3,  # Downsample spatial resolution
        max_frames=8          # Limit number of frames
    )
    
    return fig



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


def generate_sat_rad_anim_ch1():
    # Set the prefix
    prefix = "icon-ch/ch1/radiation/"
    
    # Get the connection using FilesConnection
    conn = get_connection()
    
    # Get the latest nc files
    files = get_latest_nc_files(conn, prefix, count=1)
    # Download and open the files
    datasets = download_and_open_nc_files(conn, files)
    
    # Concatenate the datasets
    combined_dataset = concat_datasets(datasets)


    ds_renamed_var = combined_dataset.rename({'GLOBAL_SW': 'SID'})[['SID']]
    # Convert time zones
    time_index = pd.DatetimeIndex(ds_renamed_var.valid_time.values).tz_localize('UTC')
    ds_renamed_var = ds_renamed_var.assign_coords(valid_time=time_index.tz_convert('CET'))
    
    st.write(ds_renamed_var)
    # Path to the Swiss cantonal boundaries GeoJSON
    geojson_path = 'swissBOUNDARIES3D_1_3_TLM_KANTONSGEBIET.geojson'
    
    # Create the animation
    fig = plot_solar_radiation_animation_optimized(ds_renamed_var, geojson_path, min_value=0, max_value=900)
    
    return fig