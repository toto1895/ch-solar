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

def get_latest_nc_files(bucket_name, prefix, count=12):
    """
    Get the latest count nc files from the specified bucket and prefix.
    
    Parameters:
    -----------
    bucket_name : str
        Name of the Google Cloud Storage bucket
    prefix : str
        Prefix for the objects to list
    count : int, optional
        Number of latest files to return
        
    Returns:
    --------
    list
        List of blob names sorted by date (newest first)
    """
    # Initialize the storage client
    storage_client = storage.Client()
    
    # Get the bucket
    bucket = storage_client.get_bucket(bucket_name)
    
    # List all blobs with the specified prefix
    blobs = list(bucket.list_blobs(prefix=prefix))
    
    # Filter for .nc files
    nc_blobs = [blob for blob in blobs if blob.name.endswith('.nc')]
    
    # Sort by name (which should be by date if they follow the format in the example)
    nc_blobs.sort(key=lambda blob: blob.name, reverse=True)
    
    # Return the latest count
    return [blob.name for blob in nc_blobs[:count]]

def download_and_open_nc_files(bucket_name, blob_names):
    """
    Download nc files from GCS and open them with xarray.
    
    Parameters:
    -----------
    bucket_name : str
        Name of the Google Cloud Storage bucket
    blob_names : list
        List of blob names to download
        
    Returns:
    --------
    list
        List of xarray datasets
    """
    # Initialize the storage client
    storage_client = storage.Client()
    
    # Get the bucket
    bucket = storage_client.get_bucket(bucket_name)
    
    datasets = []
    temp_dir = tempfile.mkdtemp()
    
    for blob_name in blob_names:
        # Get the blob
        blob = bucket.blob(blob_name)
        
        # Download the file to a temporary location
        temp_file = os.path.join(temp_dir, os.path.basename(blob_name))
        blob.download_to_filename(temp_file)
        
        # Open the file with xarray
        ds = xr.open_dataset(temp_file)
        
        # Add the file to the list
        datasets.append(ds)
        
        # Extract the timestamp from the filename
        timestamp_str = os.path.basename(blob_name).split('.')[0]
        timestamp = datetime.strptime(timestamp_str, '%Y%m%d%H%M')
        
        # Set the time coordinate
        ds = ds.assign_coords(time=[timestamp])
        datasets[-1] = ds
    
    return datasets




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
    # Get the variable name for solar radiation (assuming it's SID)
    var_name = 'SID' if 'SID' in xr_dataset.variables else list(xr_dataset.data_vars)[0]
    
    # Create figure
    fig = go.Figure()
    
    # Create frames for animation
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
    
    for t_idx in range(len(xr_dataset.time)):
        # Get data for this time
        data_slice = xr_dataset[var_name].isel(time=t_idx).values
        time_str = pd.to_datetime(xr_dataset.time[t_idx].values).tz_localize('UTC').tz_convert('CET').strftime('%Y-%m-%d %H:%M')
        
        # Create heatmap instead of image
        frame_data = [
            go.Contour(
                z=data_slice,
                x=lons,
                y=lats,
                colorscale='magma',
                zmin=min_value,
                zmax=max_value,
                colorbar=dict(title='W/m²'),
                contours=dict(
                    coloring='fill',
                    showlabels=False,
                ),
                line=dict(width=0),
                connectgaps=True,
                hovertemplate='Lon: %{x:.2f}<br>Lat: %{y:.2f}<br>Solar Radiation: %{z:.1f} W/m²<extra></extra>'
            )
        ]
        
        # Add boundary traces
        # for coords in border_coords:
        #     lons_boundary = [point[0] for point in coords]
        #     lats_boundary = [point[1] for point in coords]
        #     frame_data.append(
        #         go.Scatter(
        #             x=lons_boundary,
        #             y=lats_boundary,
        #             mode='lines',
        #             line=dict(color='white', width=1.5),
        #             hoverinfo='skip',
        #             showlegend=False
        #         )
        #     )
        
        # Create frame
        frame = go.Frame(
            data=frame_data,
            name=f'frame{t_idx}',
            layout=go.Layout(
                title_text=f"Solar Radiation at {time_str} CET"
            )
        )
        frames.append(frame)
    
    # Initial data for the figure - first frame
    initial_data = [
        go.Contour(
            z=xr_dataset[var_name].isel(time=0).values,
            x=lons,
            y=lats,
            colorscale='magma',
            zmin=min_value,
            zmax=max_value,
            colorbar=dict(title='W/m²'),
            contours=dict(
                coloring='fill',
                showlabels=False,
            ),
            line=dict(width=0),
            connectgaps=True,
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
    # Use the bounds of the data plus a small buffer
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
    lon_buffer = (lon_max - lon_min) * 0.05
    lat_buffer = (lat_max - lat_min) * 0.05
    
    for coords in border_coords:
        lons_boundary = [point[0] for point in coords]
        lats_boundary = [point[1] for point in coords]
        
        fig.add_trace(go.Scatter(
                x=lons_boundary,
                y=lats_boundary,
                mode='lines',
                line=dict(color='white', width=1.5),
                hoverinfo='skip',
                showlegend=False
            )
        )
    
    # Update layout
    fig.update_layout(
        title_text=f"Solar Radiation at {pd.to_datetime(xr_dataset.time[0].values).tz_localize('UTC').tz_convert('CET').strftime('%Y-%m-%d %H:%M')} CET",
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
                "active": 0,
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
                        "label": pd.to_datetime(xr_dataset.time[k].values).tz_localize('UTC').tz_convert('CET').strftime('%H:%M CET'),
                        "method": "animate"
                    }
                    for k in range(len(xr_dataset.time))
                ]
            }
        ],
        height=900,
        width=900,
        margin=dict(l=50, r=50, t=80, b=50),
        template="plotly_dark"  # Use dark theme for better visualization of solar data
    )
    
    # Add frames to the figure
    fig.frames = frames
    
    return fig


from GCSconnection import GCSConnection


def get_connection(bucket_name):
    """Get the GCS connection instance"""
    return GCSConnection(
        "gcs",
        bucket_name=bucket_name  # Replace with your actual bucket name
    )

def generate_sat_rad_anim():
    # Set the bucket name and prefix
    
    bucket_name = "dwd-solar-sat"  # Replace with your actual bucket name
    prefix = "radiation_sid/"
    
    
    conn = get_connection(bucket_name)
    files = conn.get_latest_nc_files(prefix, count=3*4)
    datasets = conn.download_and_open_nc_files(files)
   
    # Get the latest 12 nc files
    #blob_names = get_latest_nc_files(bucket_name, prefix, 3*6)
    
    # Download and open the files
    #datasets = download_and_open_nc_files(bucket_name, blob_names)
    
    # Concatenate the datasets
    combined_dataset = concat_datasets(datasets)
    # In main(), remove these lines:
    time_index = pd.DatetimeIndex(combined_dataset.time.values).tz_localize('UTC')
    combined_dataset = combined_dataset.assign_coords(time=time_index.tz_convert('CET'))
        
    # Path to the Swiss cantonal boundaries GeoJSON
    geojson_path = 'swissBOUNDARIES3D_1_3_TLM_KANTONSGEBIET.geojson'
    
    # First create a test plot to verify GeoJSON data
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)
        
    # Extract coordinates from the GeoJSON
    border_coords = []
    
    # Handle different GeoJSON structures
    if geojson_data['type'] == 'FeatureCollection':
        features = geojson_data['features']
        for feature in features:
            geometry = feature['geometry']
            if geometry['type'] == 'Polygon':
                border_coords.append(geometry['coordinates'][0])
            elif geometry['type'] == 'MultiPolygon':
                for polygon in geometry['coordinates']:
                    border_coords.append(polygon[0])
    elif geojson_data['type'] == 'Feature':
        geometry = geojson_data['geometry']
        if geometry['type'] == 'Polygon':
            border_coords.append(geometry['coordinates'][0])
        elif geometry['type'] == 'MultiPolygon':
            for polygon in geometry['coordinates']:
                border_coords.append(polygon[0])

    
    
    # Create the animation
    fig = plot_solar_radiation_animation(combined_dataset, geojson_path,max_value=900)
    
    # Save the animation to an HTML file
    output_file = "solar_radiation_animation.html"
    #fig.write_html(output_file)
    
    #print(f"Animation saved to {output_file}")
    return fig
