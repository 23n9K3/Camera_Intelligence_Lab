import logging

import streamlit as st

from src.storage.store import Store
from src.utils.config import ROOT
from src.visualization.workbench import PAGES, render

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="Camera Intelligence Lab", layout="wide", page_icon="📷")
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {background: #f4f7fa;}
[data-testid="stSidebar"] {background: #e7eff5;}
[data-testid="stMetric"] {
    background: #fff; border: 1px solid #dae6ef; border-radius: 12px;
    padding: 1rem 1.15rem; border-top: 3px solid #147d74;
}
[data-testid="stHeader"] {background: #f4f7fa;}
h1 {letter-spacing: -0.035em;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def local_store():
    return Store(ROOT / "artifacts")


st.sidebar.title("Camera Intelligence Lab")
page = st.sidebar.radio("工作台", PAGES, label_visibility="collapsed")
st.title(page)
render(local_store(), page)

