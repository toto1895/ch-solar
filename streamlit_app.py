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