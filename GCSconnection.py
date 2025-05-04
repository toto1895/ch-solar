import streamlit as st
import os
import re
import tempfile
from datetime import datetime
import xarray as xr
from st_files_connection import FilesConnection
from google.cloud import storage

class GCSConnection(FilesConnection):
    """A custom connection for Google Cloud Storage with NetCDF file handling capabilities."""
    
    def __init__(self, connection_name="gcs", bucket_name=None, **kwargs):
        super().__init__(connection_name, **kwargs)
        self.bucket_name = bucket_name or self._secrets.get("bucket_name")
        self._client = None
        
    @property
    def client(self):
        """Lazy-load the GCS client."""
        if self._client is None:
            self._client = storage.Client()
        return self._client
    
    def get_bucket(self, bucket_name=None):
        """Get a bucket object."""
        bucket_name = bucket_name or self.bucket_name
        if not bucket_name:
            raise ValueError("Bucket name must be provided")
        return self.client.get_bucket(bucket_name)
    
    def get_latest_nc_files(self, prefix, count=12, bucket_name=None):
        """
        Get the latest count nc files from the specified bucket and prefix.
        
        Parameters:
        -----------
        prefix : str
            Prefix for the objects to list
        count : int, optional
            Number of latest files to return
        bucket_name : str, optional
            Name of the bucket (overrides the default)
            
        Returns:
        --------
        list
            List of blob names sorted by date (newest first)
        """
        bucket = self.get_bucket(bucket_name)
        
        # Use caching based on the prefix and count
        @st.cache_data(ttl="1h")
        def fetch_nc_files(prefix, count):
            # List all blobs with the specified prefix
            blobs = list(bucket.list_blobs(prefix=prefix))
            
            # Filter for .nc files
            nc_blobs = [blob for blob in blobs if blob.name.endswith('.nc')]
            
            # Sort by name (assuming timestamp format)
            nc_blobs.sort(key=lambda blob: blob.name, reverse=True)
            
            # Return the latest count
            return [blob.name for blob in nc_blobs[:count]]
        
        return fetch_nc_files(prefix, count)
    
    def download_and_open_nc_files(self, blob_names, bucket_name=None):
        """
        Download nc files from GCS and open them with xarray.
        
        Parameters:
        -----------
        blob_names : list
            List of blob names to download
        bucket_name : str, optional
            Name of the bucket (overrides the default)
            
        Returns:
        --------
        list
            List of xarray datasets with time coordinates
        """
        bucket = self.get_bucket(bucket_name)
        
        # Cache the dataset loading for performance
        @st.cache_data(ttl="1h")
        def load_dataset(blob_name):
            # Get the blob
            blob = bucket.blob(blob_name)
            
            # Create a temporary directory
            temp_dir = tempfile.mkdtemp()
            
            # Download the file to a temporary location
            temp_file = os.path.join(temp_dir, os.path.basename(blob_name))
            blob.download_to_filename(temp_file)
            
            # Open the file with xarray
            ds = xr.open_dataset(temp_file)
            
            # Extract the timestamp from the filename
            timestamp_str = os.path.basename(blob_name).split('.')[0]
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y%m%d%H%M')
                # Set the time coordinate
                ds = ds.assign_coords(time=[timestamp])
            except ValueError:
                # If timestamp parsing fails, don't modify the dataset
                st.warning(f"Could not parse timestamp from filename: {blob_name}")
            
            return ds
        
        # Load each dataset
        datasets = [load_dataset(blob_name) for blob_name in blob_names]
        return datasets
    
    def fetch_files(self, prefix, pattern=None, max_results=100):
        """
        Fetch files from a bucket prefix with optional pattern matching
        
        Parameters:
        -----------
        prefix : str
            Prefix path in the bucket
        pattern : str, optional
            Regex pattern to filter files
        max_results : int, optional
            Maximum number of results to return
            
        Returns:
        --------
        list
            Sorted list of file paths
        """
        try:
            # Invalidate the cache to refresh the bucket listing
            self.invalidate_cache(prefix)
            
            # List files in the bucket with the given prefix
            files = self.ls(prefix, max_results=max_results)
            
            # Apply pattern filtering if provided
            if pattern:
                regex = re.compile(pattern)
                files = [f for f in files if regex.search(f)]
                
            return sorted(files, reverse=True)
        except Exception as e:
            st.error(f"Error listing files: {e}")
            return []
    
    def combine_datasets(self, datasets, dim='time'):
        """
        Combine multiple datasets along a dimension (typically time)
        
        Parameters:
        -----------
        datasets : list
            List of xarray datasets to combine
        dim : str, optional
            Dimension to combine along
            
        Returns:
        --------
        xarray.Dataset
            Combined dataset
        """
        if not datasets:
            return None
        
        # Combine the datasets using xarray.concat
        combined = xr.concat(datasets, dim=dim)
        return combined