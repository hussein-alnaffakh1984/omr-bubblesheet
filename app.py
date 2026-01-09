import os
import streamlit as st
import pandas as pd

from engine_omr import process_pdf


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Smart Bubble Sheet Scanner", layout="wide")
st.title("📄 Smart Bubble Sheet Scanner (Template-Free)")
st.caption("Reads student code + answers from different bubble sheet formats without fixed templates.")

with st.sidebar:
    st.header("⚙️ Settings")
    zoom = st.slider("PDF render quality (zoom)", 1.5, 4.0, 2.5, 0.1)
    save_csv = st.checkbox("Save CSV file", value=True)
    show_answers = st.checkbox("Show answers text", value=True)
    st.divider()
    st.markdown("**Tips**")
    st.write("- Use clear PDF scans (prefer 300 DPI).")
    st.write("- If too many AMB/MULTI, increase zoom or rescan with better lighting.")

uploaded = st.file_uploader("Upload a Bubble Sheet PDF", type=["pdf"])

if uploaded is None:
    st.info("⬆️ Upload a PDF to start.")
    st.stop()

# -----------------------------
# Save upload
# -----------------------------
tmp_dir = "tmp_uploads"
os.makedirs(tmp_dir, exist_ok=True)
pdf_path = os.path.join(tmp_dir, uploaded.name)

with open(pdf_path, "wb") as f:
    f.write(uploaded.read())

st.success(f"Uploaded: {uploaded.name}")

# -----------------------------
# Run processing
# -----------------------------
out_csv = os.path.join(tmp_dir, "bubble_results.csv")

with st.spinner("Processing PDF..."):
    df = process_pdf(pdf_path, out_csv=out_csv, zoom=zoom)

# -----------------------------
# Results
# -----------------------------
st.subheader("✅ Results")
st.dataframe(df, use_container_width=True)

# Summary
colA, colB, colC = st.columns(3)
with colA:
    st.metric("Pages", len(df))
with colB:
    st.metric("Avg. Questions/Page", round(df["num_questions"].mean(), 2) if len(df) else 0)
with colC:
    flags_count = int((df["flags"].astype(str) != "").sum()) if "flags" in df.columns else 0
    st.metric("Pages With Flags", flags_count)

# Show answers nicely
if show_answers and "answers" in df.columns:
    st.subheader("🧾 Answers per page")
    for _, row in df.iterrows():
        page = int(row["page"])
        code = row["student_code"]
        flags = str(row.get("flags", ""))
        answers = str(row.get("answers", ""))

        with st.expander(f"Page {page} | Code: {code} | Flags: {flags if flags else 'None'}"):
            st.write(answers if answers else "(no answers detected)")

# Download CSV
if save_csv and os.path.exists(out_csv):
    with open(out_csv, "rb") as f:
        st.download_button(
            "⬇️ Download CSV النتائج",
            data=f,
            file_name="bubble_results.csv",
            mime="text/csv"
        )

st.divider()
st.caption("If you see many 'AMB' or 'MULTI', increase zoom, rescan the PDF, or tune thresholds in engine_omr.py.")
