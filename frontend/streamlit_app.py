
import streamlit as st
import plotly.express as px
import pandas as pd
import requests
from dotenv import load_dotenv
import os

# load .env
load_dotenv(".env")
backend_url = os.getenv("BACKEND_URL") + "/health"
# print(backend_url)

# Page config
st.set_page_config(page_title="Frontend App", layout="wide")

st.title("RAG Document Assistant Frontend")
st.write("Hello!, this is application using Streamlit for frontend and FastAPI for backend.")

# Example: simple dataframe
df = pd.DataFrame({
    "Category": ["A", "B", "C", "D"],
    "Values": [10, 23, 17, 8]
})

# Plotly chart
fig = px.bar(df, x="Category", y="Values", title="Sample Bar Chart")
st.plotly_chart(fig)

# Example input
name = st.text_input("Enter your name:")
if name:
    st.success(f"Welcome, {name}!")
    
# Backend URL
# backend_url = "http://127.0.0.1:8000/health"

# Button to trigger request
if st.button("Check Backend Health"):
    st.write(f"Connecting to: {backend_url}")
    try:
        response = requests.get(backend_url)
        # print(response)
        # print(response.status_code)
        if response.status_code == 200:            
            data = response.json()
            st.success(f"Backend says: {data['status']}")
        else:
            st.error(f"Error: {response.status_code}")
    except Exception as e:
        st.error(f"Failed to connect: {e}")
        
