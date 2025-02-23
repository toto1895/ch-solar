import streamlit as st
import pandas as pd
import requests
import os
import numpy as np 


st.set_page_config(
    page_title="beta swiss solar forecasts",
    layout="wide",  # Enable wide mode
    initial_sidebar_state="expanded"  # Sidebar is expanded by default
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


GCLOUD = os.getenv("service_account_json")



import os, sys
import json
from google.oauth2 import service_account
from st_files_connection import FilesConnection



def Home():
    st.title("Forecasts")
    conn = st.connection('gcs', type=FilesConnection)

    try:
        all_files = []
        token = None
        while True:
            res = conn._instance.ls(
                "oracle_predictions/swiss_solar/forecasts",
                max_results=50,
                page_token=token
            )
            if isinstance(res, tuple):
                files, token = res[0], (res[1] if len(res) > 1 else None)
            else:
                files, token = res, None
            all_files.extend(files)
            if not token:
                break
    except Exception as e:
        st.error(f"Error retrieving files: {e}")
        return

    sorted_files = sorted(all_files)
    # Create a mapping: display datetime -> full file path
    file_map = {
        pd.to_datetime(os.path.basename(f).split('_q50_fcst')[0], format='%Y_%m_%d_%H_%M_%S'): f for f in sorted_files
    }
    # Dropdown displays only the datetime strings
    selected_dt = st.selectbox("Select a file (datetime):", list(file_map.keys()))

    selected_file = file_map[selected_dt]

    # Load and display the parquet file as a DataFrame
    df = conn.read(selected_file, input_format="parquet")
    st.dataframe(df)

    # Extract start and end timestamps from the DataFrame.
    # Assumes a 'timestamp' column exists.
    start_ts = pd.to_datetime(df.index.min())
    end_ts = pd.to_datetime(df.index.max())

    # Format timestamps to ISO with milliseconds and 'Z'
    start_str = start_ts.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    end_str = end_ts.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    # Build payload for the request
    payload = {
        "dateTimeRange": {"from": start_str, "to": end_str},
        "areaList": ["BZN|10YCH-SWISSGRIDZ"],
        "filterMap": {},
        "timeZone": "CET"
    }

    # Request additional data
    url = "https://newtransparency.entsoe.eu/generation/forecast/windAndSolar/solar/loadpayload"
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        additional_data = response.json()
        # Convert additional data to a DataFrame
        additional_df = pd.DataFrame(additional_data)
        st.subheader("Additional Data")
        st.dataframe(additional_df)
        # Optionally, merge or join with the loaded DataFrame if applicable.
        # combined_df = pd.merge(df, additional_df, on="timestamp", how="outer")
        # st.dataframe(combined_df)
    else:
        st.error("Error fetching additional data")












# ---------------- Main App with Navigation ----------------
def main():
    st.sidebar.title("Navigation")
    page_choice = st.sidebar.radio("Go to page:", ["Home", "Portfolio"])
    if page_choice == "Home":
        Home()
    elif page_choice== 'Portfolio':
        print()



if __name__ == "__main__":
    main()