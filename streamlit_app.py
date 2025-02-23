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

def get_latest_da_fcst_file(selected_date,files):
    selected_str = pd.to_datetime(selected_date).strftime("%Y_%m_%d")
    files_time = []
    for f in files:
        if not f.endswith(".parquet"):
            continue
        basename = f.split("/")[-1].split('_')
        date_part = basename[0]+'_'+basename[1]+'_'+basename[2]
        hour = basename[3] 
        #if (date_part == selected_str) and (int(hour) < 10):
        files_time.append(f)
    if  len(files_time)==0:
        #st.warning("No files found for the selected date before 10:00.")
        return
    selected_file = sorted(files_time)
    return selected_file

def Home():


    st.title("Benchmark Models")
    conn = st.connection('gcs', type=FilesConnection)

    selected_date = st.date_input("Submission date", pd.to_datetime("today"))

    try:
        all_files = []
        token = None
        while True:
            res = conn._instance.ls(
                f"oracle_predictions/swiss_solar/forecasts",
                max_results=50,
                page_token=token
            )
            # If ls returns a tuple, take the first two elements; otherwise, treat it as files only.
            if isinstance(res, tuple):
                files = res[0]
                token = res[1] if len(res) > 1 else None
            else:
                files = res
                token = None

            all_files.extend(files)  # extend() flattens the list if files is a list
            if not token:
                break

        sel = get_latest_da_fcst_file(selected_date,all_files)
        #print(sel)
        
    except Exception as e:
            pass    
    print(sel)
    df = conn.read(sel, input_format="parquet")
    print(df)









# ---------------- Main App with Navigation ----------------
def main():
    st.sidebar.title("Navigation")
    page_choice = st.sidebar.radio("Go to page:", ["Home", "Portfolio"])
    if page_choice == "Home":
        print()
    elif page_choice== 'Portfolio':
        print()



if __name__ == "__main__":
    main()