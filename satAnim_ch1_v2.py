import xarray as xr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import tempfile
from datetime import datetime, timedelta
import pandas as pd
import json
import streamlit as st
from st_files_connection import FilesConnection
import matplotlib.pyplot as plt
import io
from PIL import Image
import base64
import time
import concurrent.futures
from functools import lru_cache

# Keep existing functions as they are
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
    """Load a GeoJSON file."""
    with open(file_path, 'r') as f:
        geojson_data = json.load(f)
    return geojson_data

def get_latest_nc_files(conn, prefix, count=12):
    """Get the latest count nc files from the specified prefix."""
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
    """Download nc files from GCS and open them with xarray."""
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

# New functions for generating pre-rendered images

def generate_single_frame_image(xr_dataset, t_idx, time_dim, var_name, lats, lons, 
                               lats_downsampled, lons_downsampled, downsample_factor,
                               min_value, max_value, geojson_path=None,
                               width=700, height=700, dpi=100):
    """
    Generate a single frame as a matplotlib image and return it as a PIL Image.
    """
    # Create a new matplotlib figure
    fig, ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi)
    
    try:
        # Get data for this time and downsample
        if time_dim == 'valid_time':
            data_slice = xr_dataset[var_name].isel(valid_time=t_idx).values
        else:
            data_slice = xr_dataset[var_name].isel(time=t_idx).values
        
        # For 3D arrays, select specific level if needed
        if data_slice.ndim > 2:
            data_slice = data_slice[0]
        
        # Downsample the data with proper dimension handling
        if data_slice.shape[0] == len(lats) and data_slice.shape[1] == len(lons):
            # Regular grid
            data_downsampled = data_slice[::downsample_factor, ::downsample_factor]
        else:
            # Irregular grid or other shape - reshape data
            import scipy.ndimage
            zoom_factors = (len(lats_downsampled)/data_slice.shape[0], 
                           len(lons_downsampled)/data_slice.shape[1])
            data_downsampled = scipy.ndimage.zoom(data_slice, zoom_factors, order=1)
        
        # Replace NaN values
        data_downsampled = np.nan_to_num(data_downsampled, nan=-999)
        
        # Create contour plot
        contour = ax.contourf(lons_downsampled, lats_downsampled, data_downsampled, 
                             levels=np.linspace(min_value, max_value, 50),
                             cmap='turbo', vmin=min_value, vmax=max_value)
        
        # Add colorbar
        cbar = fig.colorbar(contour, ax=ax, orientation='horizontal', 
                           label='W/m²', pad=0.1)
        
        # Add boundaries if geojson is provided
        if geojson_path and os.path.exists(geojson_path):
            try:
                from matplotlib.path import Path
                from matplotlib.patches import PathPatch
                
                with open(geojson_path, 'r') as f:
                    geojson_data = json.load(f)
                
                # Process GeJSON features
                if geojson_data['type'] == 'FeatureCollection':
                    features = geojson_data['features']
                elif geojson_data['type'] == 'Feature':
                    features = [geojson_data]
                else:
                    features = [{'geometry': geojson_data}]
                
                for feature in features:
                    geometry = feature.get('geometry', {})
                    if not geometry:
                        continue
                        
                    # Process different geometry types
                    if geometry['type'] == 'Polygon':
                        for ring in geometry['coordinates']:
                            coords = ring
                            x, y = zip(*coords)
                            ax.plot(x, y, color='white', linewidth=1, alpha=0.7)
                    
                    elif geometry['type'] == 'MultiPolygon':
                        for polygon in geometry['coordinates']:
                            for ring in polygon:
                                coords = ring
                                x, y = zip(*coords)
                                ax.plot(x, y, color='white', linewidth=1, alpha=0.7)
            except Exception as e:
                print(f"Error adding boundaries: {e}")
        
        # Format time string
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
        
        # Set title and labels
        ax.set_title(f"Solar Radiation at {time_str} CET")
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        # Set background color to match dark theme
        fig.patch.set_facecolor('#111111')
        ax.set_facecolor('#111111')
        
        # Adjust text and tick colors for dark theme
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        cbar.ax.xaxis.label.set_color('white')
        cbar.ax.tick_params(axis='x', colors='white')
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert matplotlib figure to PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), dpi=dpi)
        buf.seek(0)
        img = Image.open(buf)
        
        # Close the matplotlib figure to free memory
        plt.close(fig)
        
        return img, time_str
        
    except Exception as e:
        print(f"Error generating frame {t_idx}: {e}")
        import traceback
        traceback.print_exc()
        plt.close(fig)
        return None, None

def generate_all_frame_images(xr_dataset, time_dim, var_name, max_frames=48, downsample_factor=1,
                             min_value=0, max_value=700, geojson_path=None):
    """
    Generate all frame images and return them as a list of PIL Images with time labels.
    """
    # Get the coordinates
    if 'lat' in xr_dataset.dims:
        lats = xr_dataset.lat.values
        lons = xr_dataset.lon.values
    elif 'latitude' in xr_dataset.dims:
        lats = xr_dataset.latitude.values
        lons = xr_dataset.longitude.values
    elif 'rlat' in xr_dataset.dims and 'rlon' in xr_dataset.dims:
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
    
    # Get the time values
    time_values = xr_dataset[time_dim].values
    
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
    
    frames = []
    time_labels = []
    
    # Generate images in parallel for better performance
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_idx = {
            executor.submit(
                generate_single_frame_image, 
                xr_dataset, t_idx, time_dim, var_name, 
                lats, lons, lats_downsampled, lons_downsampled, 
                downsample_factor, min_value, max_value, geojson_path
            ): t_idx for t_idx in time_indices
        }
        
        for future in concurrent.futures.as_completed(future_to_idx):
            t_idx = future_to_idx[future]
            try:
                img, time_str = future.result()
                if img and time_str:
                    # Keep track of original index position
                    idx_position = time_indices.index(t_idx)
                    frames.append((idx_position, img))
                    time_labels.append((idx_position, time_str))
            except Exception as e:
                print(f"Error processing frame {t_idx}: {e}")
    
    # Sort frames and time_labels by original position
    frames.sort(key=lambda x: x[0])
    time_labels.sort(key=lambda x: x[0])
    
    # Extract just the images and labels
    frames = [f[1] for f in frames]
    time_labels = [t[1] for t in time_labels]
    
    return frames, time_labels

def create_image_slider_animation(frames, time_labels):
    """
    Create a Streamlit animation using pre-rendered images with a slider control.
    """
    # Create a container for the animation
    animation_container = st.empty()
    
    # Create a slider for selecting frames
    frame_index = st.slider(
        "Time", 
        min_value=0, 
        max_value=len(frames)-1, 
        value=len(frames)-1,  # Default to last frame
        format=lambda i: time_labels[i][-5:] + " CET"  # Show time in slider
    )
    
    # Display the selected frame
    animation_container.image(frames[frame_index], use_column_width=True)
    
    # Add a title showing the full datetime
    st.markdown(f"## Solar Radiation at {time_labels[frame_index]} CET")
    
    # Add play button
    if st.button("Play Animation"):
        # Show animation frames
        progress_bar = st.progress(0)
        for i in range(len(frames)):
            # Update slider (this will update the image)
            st.session_state['slider_value'] = i
            frame_index = i
            
            # Update progress bar
            progress_bar.progress((i+1)/len(frames))
            
            # Display the current frame
            animation_container.image(frames[i], use_column_width=True)
            
            # Update title
            st.markdown(f"## Solar Radiation at {time_labels[i]} CET")
            
            # Small delay between frames
            time.sleep(0.3)
        
        # Reset to last frame after animation finishes
        st.session_state['slider_value'] = len(frames)-1

# Cache the frame generation to avoid regenerating on every interaction
@lru_cache(maxsize=1)
def generate_cached_frames(dataset_json):
    """
    Generate and cache frames to avoid regeneration on every interaction.
    We use a JSON string of the dataset as the cache key.
    """
    # Recreate dataset from JSON
    dataset = xr.Dataset.from_dict(json.loads(dataset_json))
    
    # Get variable name
    var_name = 'SID' if 'SID' in dataset.variables else list(dataset.data_vars)[0]
    
    # Get time dimension name
    time_dim = 'valid_time' if 'valid_time' in dataset.dims else 'time'
    
    # Generate all frame images
    geojson_path = 'swissBOUNDARIES3D_1_3_TLM_KANTONSGEBIET.geojson'
    frames, time_labels = generate_all_frame_images(
        dataset, time_dim, var_name,
        max_frames=96,  # Higher frame count for smoother animation
        min_value=0, max_value=1100,
        geojson_path=geojson_path
    )
    
    return frames, time_labels

def generate_image_based_animation():
    """
    Generate an image-based animation instead of using Plotly.
    """
    # Set the prefix
    prefix = "icon-ch/ch1/radiation/"
    
    # Get the connection using FilesConnection
    conn = get_connection()
    
    # Show a loading message
    with st.spinner("Loading data and generating frames..."):
        # Get the latest nc files
        files = get_latest_nc_files(conn, prefix, count=1)
        
        # Download and open the files
        datasets = download_and_open_nc_files(conn, files)
        
        # Concatenate the datasets
        combined_dataset = concat_datasets(datasets)
    
        # Filter region
        min_lon, max_lon = 5.8, 10.5
        min_lat, max_lat = 45.8, 48
    
        combined_dataset = combined_dataset.where(
            (combined_dataset['lon'] >= min_lon) & 
            (combined_dataset['lon'] <= max_lon) & 
            (combined_dataset['lat'] >= min_lat) & 
            (combined_dataset['lat'] <= max_lat), 
            drop=True
        )
    
        # Rename variables
        ds_renamed_var = combined_dataset.rename({'GLOBAL_SW': 'SID'})[['SID']]
        
        # Convert time zones
        time_index = pd.DatetimeIndex(ds_renamed_var.valid_time.values).tz_localize('UTC')
        ds_renamed_var = ds_renamed_var.assign_coords(valid_time=time_index.tz_convert('CET'))
        
        # Convert dataset to JSON for caching
        dataset_json = ds_renamed_var.to_dict().__str__()
        
        # Generate frames (cached)
        frames, time_labels = generate_cached_frames(dataset_json)
        
        # Create the slider-based animation
        create_image_slider_animation(frames, time_labels)
    
    return "Animation completed"

# Main streamlit app function
def main():
    # Generate image-based animation
    result = generate_image_based_animation()

    
