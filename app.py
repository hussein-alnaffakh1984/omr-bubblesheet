import os
import streamlit as st
import pandas as pd

from engine_omr import process_pdf

st.set_page_config(page_title="Smart Bubble Sheet Scanner", layout="wide")
st.title("📄 Smart Bubble Sheet Scanner (Template-Free)")

uploaded = st.file_uploader("Upload a Bubble Sheet PDF", type=["pdf"])

col1, col2 = st.columns(2)
with col1:
    zoom = st.slider("Render quality (zoom)", 1.5, 4.0, 2.5, 0.1)
with col2:
    save_csv = st.checkbox("Save results to CSV", value=True)

if uploaded:
    # Save upload to a temp file
    tmp_dir = "tmp_uploads"
    os.makedirs(tmp_dir, exist_ok=True)
    pdf_path = os.path.join(tmp_dir, uploaded.name)

    with open(pdf_path, "wb") as f:
        f.write(uploaded.read())

    st.success(f"Uploaded: {uploaded.name}")

    with st.spinner("Processing..."):
        out_csv = os.path.join(tmp_dir, "bubble_results.csv") if save_csv else "bubble_results.csv"
        df = process_pdf(pdf_path, out_csv=out_csv)

    st.subheader("✅ Results")
    st.dataframe(df, use_container_width=True)

    if save_csv and os.path.exists(out_csv):
        with open(out_csv, "rb") as f:
            st.download_button(
                "⬇️ Download CSV",
                data=f,
                file_name="bubble_results.csv",
                mime="text/csv"
            )
