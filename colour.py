import streamlit as st

def apply_custom_css():
    st.markdown("""
        <style>
        # Sidebar colour
        [data-testid="stSidebar"] {
            background-color: #f0f8ff;
            border-right: 2px solid #3498db;
        }
       # Sidebar heading
        [data-testid="stSidebar"] h1 {
            color: #1f77b4;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
       # Colour for Button & Slider
        .stButton>button {
            background-color: #3498db;
            color: white;
            border-radius: 5px;
        }
        # Customize for Metric frame
        div[data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #3498db;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
        }
        # Adjust text colour of Metric label
        div[data-testid="stMetricLabel"] {
            color: #1f77b4 !important;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

def blue_header(text):
    st.markdown(f"<h2 style='color: #1f77b4;'>{text}</h2>", unsafe_allow_html=True)
