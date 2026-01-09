import streamlit as st
import sys
import os

st.write("Python path:", sys.path)
st.write("Files:", os.listdir("."))

import importlib
import traceback
import streamlit as st

try:
    m = importlib.import_module("engine_omr")
    st.success("engine_omr imported OK ✅")
    st.write("Attributes contain process_pdf?", hasattr(m, "process_pdf"))
    st.write("Available top-level names (first 30):", sorted([n for n in dir(m) if not n.startswith("_")])[:30])
except Exception as e:
    st.error("Failed to import engine_omr ❌")
    st.code(traceback.format_exc())
    st.stop()

process_pdf = getattr(m, "process_pdf", None)
if process_pdf is None:
    st.error("process_pdf is missing inside engine_omr.py ❌")
    st.stop()

