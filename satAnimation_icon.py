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


def plot_solar_radiation_animation_optimized(xr_dataset, geojson_path=None, min_value=0, max_value=700, 
                                 downsample_factor=1, max_frames=48):
    
    import plotly.graph_objects as go
    import pandas as pd
    import json
    import numpy as np
    
    # Debug: Print dataset structure to understand its format
    print(f"Dataset dimensions: {xr_dataset.dims}")
    print(f"Dataset variables: {list(xr_dataset.variables)}")
    print(f"Dataset coordinates: {list(xr_dataset.coords)}")
    
    # Get the variable name for solar radiation
    var_name = 'SID' if 'SID' in xr_dataset.variables else list(xr_dataset.data_vars)[0]
    print(f"Using variable: {var_name}")
    
    # Create figure
    fig = go.Figure()
    
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
                
                # Create simplified boundary trace
                # We'll create a single trace with Nones between features for efficiency
                all_lons = []
                all_lats = []
                
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
                        if all_lons and all_lons[-1] is not None:
                            all_lons.append(None)
                            all_lats.append(None)
                        coords = rings[0][::3]  # Downsample coordinates
                        lons_poly, lats_poly = zip(*coords)
                        all_lons.extend(lons_poly)
                        all_lats.extend(lats_poly)
                    
                    elif geometry['type'] == 'MultiPolygon':
                        for polygon in geometry['coordinates']:
                            if all_lons and all_lons[-1] is not None:
                                all_lons.append(None)
                                all_lats.append(None)
                            coords = polygon[0][::3]  # Outer ring, downsampled
                            lons_poly, lats_poly = zip(*coords)
                            all_lons.extend(lons_poly)
                            all_lats.extend(lats_poly)
                
                # Create single boundary trace if we have coordinates
                if all_lons:
                    boundary_traces.append(
                        go.Scatter(
                            x=all_lons,
                            y=all_lats,
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
    
    # Limit the number of frames
    if len(time_values) > max_frames:
        # Select frames at regular intervals
        step = max(1, len(time_values) // max_frames)
        time_indices = list(range(0, len(time_values), step))
        # Always include the last frame
        if len(time_values) - 1 not in time_indices:
            time_indices.append(len(time_values) - 1)
    else:
        time_indices = list(range(len(time_values)))
    
    # Sort time indices to ensure they're in order
    time_indices.sort()
    
    # Get the last time index (always included)
    last_t_idx = time_indices[-1]
    
    # Create time labels for each frame for slider
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
    
    # Create the animation frames
    frames = []
    for i, t_idx in enumerate(time_indices):
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
            
            # Format time string
            time_str = time_labels[i]
            
            # Debug: check data values
            print(f"Frame {i} data min: {np.nanmin(data_downsampled)}, max: {np.nanmax(data_downsampled)}")
            
            # Replace NaN values with a default value if necessary
            data_downsampled = np.nan_to_num(data_downsampled, nan=-999)
            
            # Calculate contour levels
            contour_levels = np.linspace(min_value, max_value, 20)
            
            # Create frame with contour plot instead of heatmap
            frame_data = [
                go.Contour(
                    z=data_downsampled,
                    x=lons_downsampled,
                    y=lats_downsampled,
                    colorscale='turbo',
                    zmin=min_value,
                    zmax=max_value,
                    ncontours=50,  # Set number of contour levels to 50
                line=dict(width=0),
                connectgaps=True,
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
                    line=dict(width=0.5),
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
                    hovertemplate='Lon: %{x:.2f}<br>Lat: %{y:.2f}<br>Solar Radiation: %{z:.1f} W/m²<extra></extra>',
                )
            ]
            
            # Create frame - boundary traces go after contour so they're visible on top
            frame = go.Frame(
                data=frame_data + boundary_traces,
                name=f'frame{i}',
                layout=go.Layout(
                    title=dict(
                        text=f"Solar Radiation at {time_str}",
                        x=0.4,
                        y=0.95,
                        xanchor='left',
                        yanchor='top'
                    )
                )
            )
            frames.append(frame)
        except Exception as e:
            print(f"Error creating frame {i}: {e}")
            import traceback
            traceback.print_exc()
    
    # If we couldn't create any frames, show a message
    if not frames:
        fig.add_annotation(
            text="Error: Could not create animation frames from the dataset",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="white")
        )
        return fig
    
    # Initial data for the figure - use first frame as fallback if last frame fails
    try:
        # Get data for the initial display (last time index)
        if time_dim == 'valid_time':
            initial_data_slice = xr_dataset[var_name].isel(valid_time=last_t_idx).values
        else:
            initial_data_slice = xr_dataset[var_name].isel(time=last_t_idx).values
        
        # For 3D arrays, we might need to select a specific level
        if initial_data_slice.ndim > 2:
            initial_data_slice = initial_data_slice[0]
        
        # Downsample the data
        if initial_data_slice.shape[0] == len(lats) and initial_data_slice.shape[1] == len(lons):
            initial_data_downsampled = initial_data_slice[::downsample_factor, ::downsample_factor]
        else:
            import scipy.ndimage
            zoom_factors = (len(lats_downsampled)/initial_data_slice.shape[0], 
                           len(lons_downsampled)/initial_data_slice.shape[1])
            initial_data_downsampled = scipy.ndimage.zoom(initial_data_slice, zoom_factors, order=1)
        
        # Replace NaN values with a default value if necessary
        initial_data_downsampled = np.nan_to_num(initial_data_downsampled, nan=-999)
        
        # Calculate contour levels
        contour_levels = np.linspace(min_value, max_value, 20)
        
        # Create contour plot instead of heatmap
        initial_data = [
            go.Contour(
                z=initial_data_downsampled,
                x=lons_downsampled,
                y=lats_downsampled,
                colorscale='turbo',
                zmin=min_value,
                zmax=max_value,
                ncontours=50,  # Set number of contour levels to 50
                line=dict(width=0),
                connectgaps=True,
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
                line=dict(width=0.5),
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
                hovertemplate='Lon: %{x:.2f}<br>Lat: %{y:.2f}<br>Solar Radiation: %{z:.1f} W/m²<extra></extra>',
            )
        ]
    except Exception as e:
        print(f"Error creating initial frame: {e}")
        # Use first frame data as fallback
        initial_data = frames[0].data[:1]  # Just use the contour from the first frame
    
    # Add boundary traces to initial data
    initial_data.extend(boundary_traces)
    
    # Add traces to figure
    for trace in initial_data:
        fig.add_trace(trace)
    
    # Get the time string for the last time index
    last_time_str = time_labels[-1]
    
    # Update layout with optimized settings
    fig.update_layout(
        title=dict(
            text=f"Solar Radiation at {last_time_str}",
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
                        "args": [None, {"frame": {"duration": 300, "redraw": True}, "fromcurrent": True}],
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
                        "label": time_labels[i],  # Use time as the label instead of frame number
                        "method": "animate"
                    }
                    for i in range(len(frames))
                ]
            }
        ],
        height=700,
        width=700,
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
    ds_renamed_var = combined_dataset.rename({'GLOBAL_SW': 'SID'})[['SID']]
    
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
        max_value=1100,
        downsample_factor=1,  # Downsample spatial resolution
        max_frames=96          # Limit number of frames
    )
    
    return fig