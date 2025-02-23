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


    st.title("Benchmark Models")
    conn = st.connection('gcs', type=FilesConnection)

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

    except Exception as e:
            pass    
    print(all_files)
    
    #df = conn.read(all_files[0], input_format="parquet")









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