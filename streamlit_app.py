import streamlit as st
import pandas as pd
import requests
import os
import numpy as np 
import plotly.express as px


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

def get_entsoe(df):
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
            
            additional_df = pd.DataFrame(additional_data['instanceList'][0]['curveData']['periodList'][0]['pointMap']).T
            additional_df = additional_df[[0,3]]
            additional_df.columns = ['DA_entsoe','actual']
            additional_df['DA_entsoe'] = pd.to_numeric(additional_df['DA_entsoe'])
            additional_df['actual'] = pd.to_numeric(additional_df['actual'])
            st.dataframe(additional_df)
        except Exception as e:
            print()
      
    else:
        st.error("Error fetching additional data")

def Home():
    st.title("Forecasts")
    conn = st.connection('gcs', type=FilesConnection)
    all_files = []
    try:
        
        token = None
        while True:
            res = conn._instance.ls(
                "oracle_predictions/swiss_solar/forecasts",
                max_results=30,
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

    sorted_files = sorted(all_files)[::-1]
    print(sorted_files)
    # Create a mapping: display datetime -> full file path
   # file_map = {
   #     pd.to_datetime(str(f).split('_q50_fcst')[0], format='%Y_%m_%d_%H_%M_%S'): f for f in sorted_files
   # }
    # Dropdown displays only the datetime strings
    #selected_dt = st.selectbox("Generated at :", list(file_map.keys()))
    selected_dt = st.selectbox("Generated at :", sorted_files)
    #selected_file = file_map[selected_dt]

    # Load and display the parquet file as a DataFrame
    df = conn.read(selected_file, input_format="parquet").round(2)

    # Reset index to have 'time' as a column and melt the DataFrame to long format
    df_long = df.reset_index().melt(id_vars="time", 
                                    value_vars=["metno_0.5", "knmi_0.5", "icon_0.5", "meteofrance_0.5", "avg"],
                                    var_name="source", value_name="value")

    # Define custom colors for each column
    color_map = {
        "metno_0.5": "blue",
        "knmi_0.5": "red",
        "icon_0.5": "green",
        "meteofrance_0.5": "orange",
        "avg": "purple"
    }

    # Create the line plot with assigned colors
    fig = px.line(df_long, x="time", y="value", color="source",
                title="Solar Forecast", color_discrete_map=color_map)
    fig.update_layout(xaxis_title="Time", yaxis_title="Forecast Value")
    st.plotly_chart(fig)

 
        
    st.dataframe(df)


    











import time
# ---------------- Main App with Navigation ----------------
def main():
    st.sidebar.title("Navigation")
    if st.button("Clear Cache"):
        st.cache_resource.clear()
        st.write("Cache cleared!")
        time.sleep(10)
    page_choice = st.sidebar.radio("Go to page:", ["Home", "Portfolio"])
    if page_choice == "Home":
        Home()
    elif page_choice== 'Portfolio':
        print()



if __name__ == "__main__":
    main()