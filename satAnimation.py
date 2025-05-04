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

import xarray as xr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json

def plot_solar_radiation_animation(xr_dataset, geojson_path=None, min_value=0, max_value=900):
    """
    Create an animation of solar radiation data from an xarray dataset with Swiss boundaries.
    
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
        
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    # Get the variable name for solar radiation (assuming it's SID)
    var_name = 'SID' if 'SID' in xr_dataset.variables else list(xr_dataset.data_vars)[0]
    
    # Create figure
    fig = go.Figure()
    
    frames = []
    
    # Get the coordinates properly
    if 'lat' in xr_dataset.dims:
        lats = xr_dataset.lat.values
        lons = xr_dataset.lon.values
    elif 'latitude' in xr_dataset.dims:
        lats = xr_dataset.latitude.values
        lons = xr_dataset.longitude.values
    else:
        raise ValueError("Could not find latitude/longitude dimensions in the dataset")
    
    # Load the GeoJSON file
    border_coords = []
    if geojson_path:
        try:
            with open(geojson_path, 'r') as f:
                geojson_data = json.load(f)
                
            # Extract coordinates from the GeoJSON
            # Handle different GeoJSON structures
            if geojson_data['type'] == 'FeatureCollection':
                features = geojson_data['features']
                for feature in features:
                    geometry = feature['geometry']
                    if geometry['type'] == 'Polygon':
                        # Get the outer ring of the polygon
                        border_coords.append(geometry['coordinates'][0])
                    elif geometry['type'] == 'MultiPolygon':
                        # Get all polygons
                        for polygon in geometry['coordinates']:
                            border_coords.append(polygon[0])  # Outer ring of each polygon
            elif geojson_data['type'] == 'Feature':
                geometry = geojson_data['geometry']
                if geometry['type'] == 'Polygon':
                    border_coords.append(geometry['coordinates'][0])
                elif geometry['type'] == 'MultiPolygon':
                    for polygon in geometry['coordinates']:
                        border_coords.append(polygon[0])
        except Exception as e:
            print(f"Error loading GeoJSON: {e}")
            import traceback
            traceback.print_exc()
    
    # Get the last time index
    last_t_idx = len(xr_dataset.time) - 1
    
    # Define proper turbo colorscale
    # This is an implementation of Google's Turbo colormap
    # Reference: https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html
    turbo_colorscale = [
        [0.0, 'rgb(48,18,59)'],      # Dark purple
        [0.1, 'rgb(86,15,105)'],     # Purple
        [0.2, 'rgb(127,11,126)'],    # Magenta
        [0.3, 'rgb(166,27,120)'],    # Pink
        [0.4, 'rgb(200,47,96)'],     # Dark pink/red
        [0.5, 'rgb(229,84,68)'],     # Red-orange
        [0.6, 'rgb(248,130,48)'],    # Orange
        [0.7, 'rgb(253,184,46)'],    # Yellow-orange
        [0.8, 'rgb(235,229,52)'],    # Yellow
        [0.9, 'rgb(190,245,99)'],    # Light green
        [1.0, 'rgb(252,255,191)']    # Light yellow/white
    ]
    
    # Custom colorscale with black for values below threshold
    threshold = 100
    threshold_ratio = threshold / max_value
    
    custom_turbo = [
        [0, 'rgb(0,0,0)'],             # Black for 0
        [threshold_ratio, 'rgb(0,0,0)']  # Black up to threshold
    ]
    
    # Add the turbo colors above the threshold
    for i, [pos, color] in enumerate(turbo_colorscale):
        if i > 0:  # Skip the first color to avoid duplicating the threshold color
            # Scale the position to be between threshold_ratio and 1.0
            scaled_pos = threshold_ratio + (1 - threshold_ratio) * pos
            custom_turbo.append([scaled_pos, color])
    
    for t_idx in range(len(xr_dataset.time)):
        # Get data for this time
        data_slice = xr_dataset[var_name].isel(time=t_idx).values
        time_str = pd.to_datetime(xr_dataset.time[t_idx].values).strftime('%Y-%m-%d %H:%M')
        
        # Create contour plot
        frame_data = [
            go.Contour(  # Using Contour with coloring='fill' for filled contours
                z=data_slice,
                x=lons,
                y=lats,
                colorscale=custom_turbo,
                zmin=min_value,
                zmax=max_value,
                colorbar=dict(
                    title="Solar Radiation (W/m²)",
                    titleside="right",
                    tickvals=[0, threshold, max_value*0.25, max_value*0.5, max_value*0.75, max_value],
                    ticktext=["0", str(threshold), str(int(max_value*0.25)), str(int(max_value*0.5)), 
                             str(int(max_value*0.75)), str(int(max_value))],
                    ticks="outside"
                ),
                contours=dict(
                    showlabels=False,
                    coloring='fill',
                ),
                hovertemplate='Lon: %{x:.2f}<br>Lat: %{y:.2f}<br>Solar Radiation: %{z:.1f} W/m²<extra></extra>'
            )
        ]
        
        # Create frame
        frame = go.Frame(
            data=frame_data,
            name=f'frame{t_idx}',
            layout=go.Layout(
                title_text=f"Solar Radiation at {time_str} CET"
            )
        )
        frames.append(frame)
    
    # Initial data for the figure - use last frame instead of first
    initial_data = [
        go.Contour(  # Using Contour with coloring='fill'
            z=xr_dataset[var_name].isel(time=last_t_idx).values,
            x=lons,
            y=lats,
            colorscale=custom_turbo,
            zmin=min_value,
            zmax=max_value,
            colorbar=dict(
                title="Solar Radiation (W/m²)",
                titleside="right",
                tickvals=[0, threshold, max_value*0.25, max_value*0.5, max_value*0.75, max_value],
                ticktext=["0", str(threshold), str(int(max_value*0.25)), str(int(max_value*0.5)), 
                         str(int(max_value*0.75)), str(int(max_value))],
                ticks="outside"
            ),
            contours=dict(
                showlabels=False,
                coloring='fill',
            ),
            hovertemplate='Lon: %{x:.2f}<br>Lat: %{y:.2f}<br>Solar Radiation: %{z:.1f} W/m²<extra></extra>'
        )
    ]
    
    # Add boundary traces to initial data
    for coords in border_coords:
        lons_boundary = [point[0] for point in coords]
        lats_boundary = [point[1] for point in coords]
        initial_data.append(
            go.Scatter(
                x=lons_boundary,
                y=lats_boundary,
                mode='lines',
                line=dict(color='white', width=1.5),
                hoverinfo='skip',
                showlegend=False
            )
        )
    
    # Add traces to figure
    for trace in initial_data:
        fig.add_trace(trace)
    
    # Calculate appropriate axis ranges
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)
    
    # If we have boundary data, adjust the plot bounds to include all boundaries
    if border_coords:
        all_boundary_lons = [point[0] for coords in border_coords for point in coords]
        all_boundary_lats = [point[1] for coords in border_coords for point in coords]
        
        if all_boundary_lons and all_boundary_lats:
            lon_min = min(lon_min, min(all_boundary_lons))
            lon_max = max(lon_max, max(all_boundary_lons))
            lat_min = min(lat_min, min(all_boundary_lats))
            lat_max = max(lat_max, max(all_boundary_lats))
    
    # Add a small buffer for visibility
    lon_buffer = (lon_max - lon_min) * 0.0
    lat_buffer = (lat_max - lat_min) * 0.0
    
    # Compute the time string for the last time index
    last_time_str = pd.to_datetime(xr_dataset.time[last_t_idx].values).strftime('%Y-%m-%d %H:%M')
    
    # Update layout
    fig.update_layout(
        title_text=f"Solar Radiation at {last_time_str} CET",
        xaxis=dict(
            title='Longitude',
            range=[lon_min - lon_buffer, lon_max + lon_buffer],
            constrain='domain'
        ),
        yaxis=dict(
            title='Latitude',
            range=[lat_min - lat_buffer, lat_max + lat_buffer],
            scaleanchor='x',
            scaleratio=1,
        ),
        margin=dict(l=0, r=0, t=50, b=0),
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
                "y": 0,
                "yanchor": "top"
            }
        ],
        sliders=[
            {
                "active": last_t_idx,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 16},
                    "prefix": "Time: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [
                            [f"frame{k}"],
                            {
                                "frame": {"duration": 300, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 300}
                            }
                        ],
                        "label": pd.to_datetime(xr_dataset.time[k].values).strftime('%H:%M CET'),
                        "method": "animate"
                    }
                    for k in range(len(xr_dataset.time))
                ]
            }
        ],
        height=800,
        width=800,
        template="plotly_dark"  # Dark theme for better visualization
    )
    
    fig.frames = frames
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
        files = conn._instance.ls(prefix, max_results=100)
        
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


def generate_sat_rad_anim():
    # Set the prefix
    prefix = "dwd-solar-sat/radiation_sid/"
    
    # Get the connection using FilesConnection
    conn = get_connection()
    
    # Get the latest nc files
    files = get_latest_nc_files(conn, prefix, count=12)

    # Download and open the files
    datasets = download_and_open_nc_files(conn, files)
    

    # Concatenate the datasets
    combined_dataset = concat_datasets(datasets)
    
    # Convert time zones
    time_index = pd.DatetimeIndex(combined_dataset.time.values).tz_localize('UTC')
    combined_dataset = combined_dataset.assign_coords(time=time_index.tz_convert('CET'))
        
    # Path to the Swiss cantonal boundaries GeoJSON
    geojson_path = 'swissBOUNDARIES3D_1_3_TLM_KANTONSGEBIET.geojson'
    
    # Create the animation
    fig = plot_solar_radiation_animation(combined_dataset, geojson_path, min_value=0,max_value=900)
    
    return fig