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

def get_files():
    conn = st.connection('gcs', type=FilesConnection)
    all_files = []
    token = None
    while True:
        res = conn._instance.ls(
            "oracle_predictions/swiss_solar/forecasts",
            max_results=100,
            page_token=token
        )
        st.write(f"Current token: {token} | Response: {res}")  # Debug info

        if isinstance(res, tuple):
            files, token = res[0], (res[1] if len(res) > 1 else None)
        else:
            files, token = res, None

        all_files.extend(files)
        if not token:
            break
    return sorted(all_files, reverse=True), conn


def get_entsoe(df):
    start_ts = pd.to_datetime(df.index.min())
    end_ts = pd.to_datetime(df.index.max())
    start_str = start_ts.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    end_str = end_ts.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    payload = {
        "dateTimeRange": {"from": start_str, "to": end_str},
        "areaList": ["BZN|10YCH-SWISSGRIDZ"],
        "filterMap": {},
        "timeZone": "CET"
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json; charset=utf-8",
        "origin": "https://newtransparency.entsoe.eu",
        "referer": "https://newtransparency.entsoe.eu/generation/forecast/windAndSolar/solar",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
    }

    url = "https://newtransparency.entsoe.eu/generation/forecast/windAndSolar/solar/load"
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        try:
            additional_data = response.json()
            additional_df = pd.DataFrame(
                additional_data['instanceList'][0]['curveData']['periodList'][0]['pointMap']
            ).T[[0, 3]]
            additional_df.columns = ['DA_entsoe', 'actual']
            additional_df = additional_df.apply(pd.to_numeric)
            st.dataframe(additional_df)
        except Exception as e:
            st.error(f"Error processing additional data: {e}")
    else:
        st.error("Error fetching additional data")

def Home():
    st.title("Forecasts")
    sorted_files, conn = get_files()
    if not sorted_files:
        st.error("No files found.")
        return

    selected_dt = st.selectbox("Generated at :", [s.split('/')[-1][:-8] for s in sorted_files])
    file_path = f"oracle_predictions/swiss_solar/forecasts/{selected_dt}.parquet"
    df = conn.read(file_path, input_format="parquet").round(2)

    df_long = df.reset_index().melt(
        id_vars="time",
        value_vars=["metno_0.5", "knmi_0.5", "icon_0.5", "meteofrance_0.5", "avg"],
        var_name="source",
        value_name="value"
    )

    color_map = {
        "metno_0.5": "blue",
        "knmi_0.5": "red",
        "icon_0.5": "green",
        "meteofrance_0.5": "orange",
        "avg": "purple"
    }

    fig = px.line(
        df_long,
        x="time",
        y="value",
        color="source",
        title="Solar Forecast",
        color_discrete_map=color_map
    )
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
        #time.sleep(10)
    page_choice = st.sidebar.radio("Go to page:", ["Home", "Portfolio"])
    if page_choice == "Home":
        Home()
    elif page_choice== 'Portfolio':
        print()



if __name__ == "__main__":
    main()