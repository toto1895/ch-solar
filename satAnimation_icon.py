import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
import pandas as pd
import json
import os
import tempfile
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
import io
from matplotlib.figure import Figure
import streamlit as st

def concat_datasets(datasets):
    """Concatenate datasets along the time dimension."""
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
    """Load a GeoJSON file."""
    with open(file_path, 'r') as f:
        geojson_data = json.load(f)
    
    return geojson_data

def plot_solar_radiation_subplots(xr_dataset, geojson_path=None, min_value=0, max_value=700, 
                                 downsample_factor=1, time_indices=None, num_cols=32, figsize=(16, 10)):
    """
    Create multiple solar radiation plots using Matplotlib with a shared colorbar.
    
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
    time_indices : list, optional
        List of time indices to plot. If None, plots first 9 time steps
    num_cols : int, optional
        Number of columns in the subplot grid
    figsize : tuple, optional
        Figure size (width, height) in inches
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the contour plots
    """
    # Get the variable name for solar radiation
    var_name = 'SID' if 'SID' in xr_dataset.variables else list(xr_dataset.data_vars)[0]
    
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
    
    # Downsample spatial resolution for better performance
    downsample_factor = min(downsample_factor, max(1, len(lats)//20))
    lats_downsampled = lats[::downsample_factor]
    lons_downsampled = lons[::downsample_factor]
    
    # Get the time values - handle the case where valid_time might not be the name
    time_dim = 'valid_time' if 'valid_time' in xr_dataset.dims else 'time'
    time_values = xr_dataset[time_dim].values
    
    # Default to first 9 time steps if not specified
    if time_indices is None:
        time_indices = list(range(min(9, len(time_values))))
    
    # Calculate grid dimensions
    num_plots = len(time_indices)
    num_rows = (num_plots + num_cols - 1) // num_cols  # Ceiling division
    
    # Create a figure with GridSpec to control layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(num_rows + 1, num_cols, height_ratios=[0.08] + [1] * num_rows)
    
    # Create a meshgrid for the contour plot
    lon_mesh, lat_mesh = np.meshgrid(lons_downsampled, lats_downsampled)
    
    # Define levels for contours
    levels = np.linspace(min_value, max_value, 20)
    
    # Prepare a list to store contour objects for the colorbar
    contour_filled_list = []
    
    # Create all the individual subplots
    for i, time_idx in enumerate(time_indices):
        # Calculate row and column in the grid
        row = i // num_cols + 1  # +1 because first row is for colorbar
        col = i % num_cols
        
        # Create subplot with Cartopy projection
        ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
        
        # Get time string for the title
        if hasattr(xr_dataset[time_dim][time_idx], 'dt'):
            # If it's a pandas/numpy datetime
            ts = xr_dataset[time_dim][time_idx].dt.strftime('%Y-%m-%d %H:%M').values
            time_str = str(ts)
        else:
            # Try to convert from numpy datetime64
            try:
                time_str = pd.to_datetime(xr_dataset[time_dim][time_idx].values).strftime('%Y-%m-%d %H:%M')
            except:
                time_str = f"Frame {time_idx+1}"
        
        # Get data for this time and downsample
        if time_dim == 'valid_time':
            data_slice = xr_dataset[var_name].isel(valid_time=time_idx).values
        else:
            data_slice = xr_dataset[var_name].isel(time=time_idx).values
        
        # For 3D arrays, we might need to select a specific level
        if data_slice.ndim > 2:
            # Take first level if there are multiple levels
            data_slice = data_slice[0]
        
        # Downsample the data with proper dimension handling
        if data_slice.shape[0] == len(lats) and data_slice.shape[1] == len(lons):
            # Regular grid
            data_downsampled = data_slice[::downsample_factor, ::downsample_factor]
        else:
            # Reshape data if dimensions don't match
            print(f"Warning: Data shape {data_slice.shape} doesn't match coordinates: lats {len(lats)}, lons {len(lons)}")
            import scipy.ndimage
            zoom_factors = (len(lats_downsampled)/data_slice.shape[0], len(lons_downsampled)/data_slice.shape[1])
            data_downsampled = scipy.ndimage.zoom(data_slice, zoom_factors, order=1)
        
        # Add borders and coastlines for context
        ax.coastlines(resolution='10m', linewidth=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.7)
        
        # Create contour plot
        contour_filled = ax.contourf(lon_mesh, lat_mesh, data_downsampled, 
                                   levels=levels, 
                                   cmap='turbo', 
                                   extend='both',
                                   transform=ccrs.PlateCarree())
        
        # Store for colorbar (only need one)
        if i == 0:
            contour_filled_list.append(contour_filled)
        
        # Add contour lines with labels (fewer labels for subplots)
        contour_lines = ax.contour(lon_mesh, lat_mesh, data_downsampled, 
                                  levels=levels[::5],  # Fewer levels for contour lines
                                  colors='white', 
                                  linewidths=0.5,
                                  transform=ccrs.PlateCarree())
        
        # Only add contour labels to some plots to avoid clutter
        if num_plots <= 6 or i % 3 == 0:
            plt.clabel(contour_lines, inline=True, fontsize=7, fmt='%1.0f')
        
        # Add GeoJSON boundaries if provided
        if geojson_path and os.path.exists(geojson_path):
            try:
                geojson_data = load_geojson(geojson_path)
                
                # Extract features
                if 'features' in geojson_data:
                    features = geojson_data['features']
                    
                    for feature in features:
                        geometry = feature.get('geometry', {})
                        
                        # Process polygon geometries
                        if geometry['type'] == 'Polygon':
                            for ring in geometry['coordinates']:
                                lons_poly, lats_poly = zip(*ring)
                                ax.plot(lons_poly, lats_poly, '-', color='white', 
                                       linewidth=1.5, transform=ccrs.PlateCarree())
                        
                        # Process multipolygon geometries
                        elif geometry['type'] == 'MultiPolygon':
                            for polygon in geometry['coordinates']:
                                for ring in polygon:
                                    lons_poly, lats_poly = zip(*ring)
                                    ax.plot(lons_poly, lats_poly, '-', color='white', 
                                           linewidth=1.5, transform=ccrs.PlateCarree())
            except Exception as e:
                print(f"Error loading GeoJSON: {e}")
        
        # Set extent to focus on the area of interest
        ax.set_extent([min(lons), max(lons), min(lats), max(lats)], crs=ccrs.PlateCarree())
        
        # Add gridlines (simplified for subplots)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                         linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Only show y labels on the leftmost column
        if col != 0:
            gl.left_labels = False
        
        # Only show x labels on the bottom row
        if row != num_rows:
            gl.bottom_labels = False
        
        # Add title
        ax.set_title(f'{time_str} UTC', fontsize=10)
    
    # Add shared colorbar at the top
    cbar_ax = fig.add_subplot(gs[0, :])
    cbar = plt.colorbar(contour_filled_list[0], cax=cbar_ax, orientation='horizontal', pad=0.05)
    cbar.set_label('Solar Radiation (W/m²)', fontsize=12)
    
    # Add main title
    fig.suptitle('Solar Radiation Forecast', fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig

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



@st.cache_data(ttl=3600)  # Cache for 1 hour
def download_png(conn, file_paths):
    """
    Download PNG files from GCS with caching support.
    
    Parameters:
    -----------
    conn : FilesConnection
        The connection to GCS
    file_paths : list
        List of file paths to download
        
    Returns:
    --------
    str or None
        Path to the downloaded file or None if error
    """
    # Create a temporary directory to store the downloaded files
    temp_dir = tempfile.mkdtemp()
    
    for file_path in file_paths:
        try:
            # Extract filename from path to use in the cached path
            file_name = os.path.basename(file_path)
            temp_file_path = os.path.join(temp_dir, file_name)
            
            # Download the file to the temporary location
            conn._instance.get(file_path, temp_file_path)
            return temp_file_path
            
        except Exception as e:
            st.error(f"Error downloading file {file_path}: {e}")
            return None
    
    return None

# Also improve the display_png function with caching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_latest_png_file_cached(conn, prefix, filename_prefix=None):
    """
    Get the latest PNG file path with caching.
    
    Parameters:
    -----------
    conn : FilesConnection
        The connection to GCS
    prefix : str
        Prefix for the objects to list (directory path)
    filename_prefix : str, optional
        Optional prefix for the filenames
        
    Returns:
    --------
    list
        List containing the most recent file path
    """
    try:
        # Invalidate the cache to refresh the bucket listing
        conn._instance.invalidate_cache(prefix)
        
        # List all files in the prefix
        files = conn._instance.ls(prefix, max_results=100)
        
        # Filter for .png files
        png_files = [f for f in files if f.endswith('.png')]
        
        # Apply the filename prefix filter if provided
        if filename_prefix:
            png_files = [f for f in png_files if f.split('/')[-1].startswith(filename_prefix)]
        
        # Sort files by name (which should contain date information)
        png_files.sort(reverse=True)
        
        # Return only the latest file
        return png_files[:1] if png_files else []
    
    except Exception as e:
        st.error(f"Error listing files: {e}")
        return []


@st.cache_resource
def initialize_connection():
    """Initialize and return the GCS connection."""
    return st.connection('gcs', type=FilesConnection)


def generate_solar_radiation_plots(data_path=None, geojson_path=None, num_plots=32):
    """
    Generate multiple solar radiation plots using sample data or provided data.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to the netCDF file containing solar radiation data
    geojson_path : str, optional
        Path to the GeoJSON file for boundaries
    num_plots : int, optional
        Number of plots to generate
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the contour plots
    """
    # Load data - for demonstration using a sample dataset
        # If actual data is provided
    prefix = "icon-ch/ch1/radiation/"

# Get the connection using FilesConnection
    conn = get_connection()
    
    # Get the latest nc files - reduced from original count
    files = get_latest_nc_files(conn, prefix, count=1)
    
    # Download and open the files
    datasets = download_and_open_nc_files(conn, files)
    
    # Concatenate the datasets
    ds = concat_datasets(datasets)

    ds_renamed_var = ds.rename({'GLOBAL_SW': 'SID'})[['SID']]

    # Convert time zones
    time_index = pd.DatetimeIndex(ds_renamed_var.valid_time.values).tz_localize('UTC')
    ds = ds_renamed_var.assign_coords(valid_time=time_index.tz_convert('CET'))

    # Define plot parameters
    min_value = 0
    max_value = 1100
    
    # Get appropriate time indices
    time_dim = 'valid_time' if 'valid_time' in ds.dims else 'time'
    available_times = len(ds[time_dim])
    time_indices = list(range(min(num_plots, available_times)))
    
    # Create plot
    fig = plot_solar_radiation_subplots(
        ds, 
        geojson_path=geojson_path, 
        min_value=min_value, 
        max_value=max_value,
        downsample_factor=1,
        time_indices=time_indices,
        num_cols=3,
        figsize=(16, 40)
    )
    
    return fig


from PIL import Image

def display_png_streamlit(image_path):
    """
    Display a PNG image in a Streamlit app.
    
    Parameters:
    -----------
    image_path : str
        Path to the PNG image file
    """
    try:
        st.success(f"MODEL RUN {image_path.split('/')[-1][:-4]}")
        
        # Check if file exists
        if not os.path.exists(image_path):
            st.error(f"Image file not found: {image_path}")
            return
        
        # Load and display the image directly with streamlit
        img = Image.open(image_path)
        st.image(img, caption="", use_container_width =True)

    except Exception as e:
        st.error(f"Error loading or displaying image: {e}")



def display_png(param):
    
    if param=='solar':
        prefix = "icon-ch/ch1/rad-png"
        filename_prefix = None
        
    elif param=='precipitation':
        prefix = "icon-ch/ch1/other_png"
        filename_prefix = 'TOT_PREC'

    elif param=='cloud':
        prefix = "icon-ch/ch1/other_png"
        filename_prefix = 'CLCT'
    
    elif param=='temperature':
        prefix = "icon-ch/ch1/other_png"
        filename_prefix = 'T_2M'

    
    conn = get_connection()
    files = get_latest_png_file_cached(conn, prefix, filename_prefix, count=1)
    png_path = download_png(conn, files)
    #datasets = download_and_open_nc_files(conn, files)
    display_png_streamlit(png_path)
