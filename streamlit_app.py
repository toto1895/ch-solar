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
    selected_file = st.selectbox("Select a file:", sorted_files)









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