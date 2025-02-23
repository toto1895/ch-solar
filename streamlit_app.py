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

    # Define request headers based on the captured network headers
    headers = {
        "accept": "application/json",
        "content-type": "application/json; charset=utf-8",
        "origin": "https://newtransparency.entsoe.eu",
        "referer": "https://newtransparency.entsoe.eu/generation/forecast/windAndSolar/solar",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
    }

    # Make the POST request to fetch additional data
    url = "https://newtransparency.entsoe.eu/generation/forecast/windAndSolar/solar/load"
    response = requests.post(url, headers=headers, json=payload)


    if response.status_code == 200:
        additional_data = response.json()
        try:
            # Extract the base time from the API response
            base_time_str = (additional_data.get('instanceList', [{}])[0]
                             .get('curveData', {})
                             .get('periodList', [{}])[0]
                             .get('timeInterval', {})
                             .get('from', None))
            if base_time_str is None:
                st.error("Base time not found in API response.")
                return

            base_time = pd.to_datetime(base_time_str)
            # Extract pointMap data
            point_map = (additional_data.get('instanceList', [{}])[0]
                                        .get('curveData', {})
                                        .get('periodList', [{}])[0]
                                        .get('pointMap', {}))
            # Build a list of points with computed timestamps.
            # Assuming each key in pointMap represents an offset in hours.
            points = []
            for offset_str, value in point_map.items():
                try:
                    offset_hours = float(offset_str)
                    ts = base_time + pd.Timedelta(hours=offset_hours)
                    points.append({"timestamp": ts, "value": value})
                except Exception as e:
                    st.error(f"Error processing offset '{offset_str}': {e}")

            additional_df = pd.DataFrame(points)
            st.subheader("Additional Data")
            st.dataframe(additional_df)
        except Exception as ex:
            st.error(f"Error processing additional data: {ex}")
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