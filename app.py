import streamlit as st
import sys
import os

st.write("Python path:", sys.path)
st.write("Files:", os.listdir("."))

from engine_omr import process_pdf
