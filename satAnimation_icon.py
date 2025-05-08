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


def plot_solar_radiation_animation(xr_dataset, geojson_path=None, min_value=0, max_value=700):
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
    import plotly.graph_objects as go
    import pandas as pd
    import json
    import numpy as np
    
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
                        
            print(f"Loaded {len(border_coords)} boundary segments from GeoJSON")
        except Exception as e:
            print(f"Error loading GeoJSON: {e}")
            import traceback
            traceback.print_exc()
    
    # Get the last time index
    last_t_idx = len(xr_dataset.valid_time) - 1
    
    # Using the default 'turbo' colorscale with 50 steps but keeping min/max values
    import plotly.colors
    colorscale = plotly.colors.sample_colorscale('turbo', 50)
    
    for t_idx in range(len(xr_dataset.valid_time)):
        # Get data for this time
        data_slice = xr_dataset[var_name].isel(valid_time=t_idx).values
        time_str = pd.to_datetime(xr_dataset.valid_time[t_idx].values).tz_localize('UTC').tz_convert('CET').strftime('%Y-%m-%d %H:%M')
        
        # Create heatmap instead of image
        frame_data = [
            go.Contour(
                z=data_slice,
                x=lons,
                y=lats,
                colorscale=colorscale,  # Use turbo colorscale with 50 steps
                zmin=min_value,         # Keep fixed min value
                zmax=max_value,         # Keep fixed max value
                contours=dict(
                    coloring='fill',
                    showlabels=False,
                ),
                ncontours=50,  # Set number of contour levels to 50
                line=dict(width=0),
                connectgaps=True,
                hovertemplate='Lon: %{x:.2f}<br>Lat: %{y:.2f}<br>Solar Radiation: %{z:.1f} W/m²<extra></extra>',
                colorbar=dict(
                    title='W/m²',
                    title_side='right',
                    orientation='h',     # Horizontal colorbar
                    y=-0.15,             # Position below the plot
                    len=0.6,             # Length of the colorbar (60% of plot width)
                    thickness=20,        # Thickness of the colorbar
                    tickmode='auto',     # Automatic tick marks
                    nticks=10            # Number of tick marks
                )
            )
        ]
        
        # Create frame
        frame = go.Frame(
            data=frame_data,
            name=f'frame{t_idx}',
            layout=go.Layout(
                title=dict(
                    text=f"Solar Radiation at {time_str} CET",
                    x=0.4,  # Position at left
                    y=0.865, # Position near top
                    xanchor='left',
                    yanchor='top'
                )
            )
        )
        frames.append(frame)
    
    # Initial data for the figure - use last frame instead of first
    initial_data = [
        go.Contour(
            z=xr_dataset[var_name].isel(valid_time=last_t_idx).values,
            x=lons,
            y=lats,
            colorscale=colorscale,  # Use turbo colorscale with 50 steps
            zmin=min_value,         # Keep fixed min value
            zmax=max_value,         # Keep fixed max value
            contours=dict(
                coloring='fill',
                showlabels=False,
            ),
            ncontours=50,  # Set number of contour levels to 50
            line=dict(width=0),
            connectgaps=True,
            hovertemplate='Lon: %{x:.2f}<br>Lat: %{y:.2f}<br>Solar Radiation: %{z:.1f} W/m²<extra></extra>',
            colorbar=dict(
                title='W/m²',
                title_side='right',
                orientation='h',     # Horizontal colorbar
                y=-0.15,             # Position below the plot
                len=0.6,             # Length of the colorbar (60% of plot width)
                thickness=20,        # Thickness of the colorbar
                tickmode='auto',     # Automatic tick marks
                nticks=10            # Number of tick marks
            )
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
    
    # Let the plot autoscale for axis ranges - remove manual range setting
    # We'll still calculate the bounds for reference
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)
    
    # If we have boundary data, include those bounds as well
    if border_coords:
        all_boundary_lons = [point[0] for coords in border_coords for point in coords]
        all_boundary_lats = [point[1] for coords in border_coords for point in coords]
        
        if all_boundary_lons and all_boundary_lats:
            lon_min = min(lon_min, min(all_boundary_lons))
            lon_max = max(lon_max, max(all_boundary_lons))
            lat_min = min(lat_min, min(all_boundary_lats))
            lat_max = max(lat_max, max(all_boundary_lats))
    
    # Compute the time string for the last time index
    last_time_str = pd.to_datetime(xr_dataset.valid_time[last_t_idx].values).tz_localize('UTC').tz_convert('CET').strftime('%Y-%m-%d %H:%M')
    
    # Update layout with title at top left and add space for slider and colorbar
    fig.update_layout(
        title=dict(
            text=f"Solar Radiation at {last_time_str} CET",
            x=0.0,  # Position at left
            y=0.95, # Position near top
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
        # Adjusted margins to accommodate top slider and bottom colorbar
        margin=dict(l=0, r=0, t=90, b=80),  # Increased top margin for slider
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
                "y": 1.05,  # Move buttons to top to be near the slider
                "yanchor": "bottom"
            }
        ],
        sliders=[
            {
                "active": last_t_idx,  # Set the active slider position to the last time index
                "yanchor": "bottom",   # Anchor to bottom (will be at top of chart)
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 16},
                    "prefix": "Time: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 10},  # Reduced top padding
                "len": 0.9,
                "x": 0.1,
                "y": 1.05,  # Position above the plot (>1.0 means above the plot area)
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
                        "label": pd.to_datetime(xr_dataset.valid_time[k].values).tz_localize('UTC').tz_convert('CET').strftime('%H:%M CET'),
                        "method": "animate"
                    }
                    for k in range(len(xr_dataset.valid_time))
                ]
            }
        ],
        height=800,
        width=800,
        template="plotly_dark"  # Use dark theme for better visualization of solar data
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
    
    st.write(ds_renamed_var.drop_vars('time'))
    # Path to the Swiss cantonal boundaries GeoJSON
    geojson_path = 'swissBOUNDARIES3D_1_3_TLM_KANTONSGEBIET.geojson'
    
    # Create the animation
    fig = plot_solar_radiation_animation(ds_renamed_var, geojson_path, min_value=0,max_value=900)
    
    return fig