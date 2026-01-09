import os
import streamlit as st
from engine_omr import process_pdf

st.set_page_config(page_title="Smart Bubble Sheet Scanner", layout="wide")
st.title("📄 Smart Bubble Sheet Scanner (Template-Free)")

with st.sidebar:
    st.header("⚙️ Settings")
    zoom = st.slider("PDF render zoom (quality)", 1.5, 4.0, 2.5, 0.1)
    save_csv = st.checkbox("Save CSV", value=True)
    show_answers = st.checkbox("Show answers per page", value=True)

uploaded = st.file_uploader("Upload PDF", type=["pdf"])
if not uploaded:
    st.info("⬆️ ارفع ملف PDF للببل شيت")
    st.stop()

tmp_dir = "tmp_uploads"
os.makedirs(tmp_dir, exist_ok=True)

pdf_path = os.path.join(tmp_dir, uploaded.name)
with open(pdf_path, "wb") as f:
    f.write(uploaded.read())

out_csv = os.path.join(tmp_dir, "bubble_results.csv")

with st.spinner("Processing..."):
    df = process_pdf(pdf_path, out_csv=out_csv, zoom=zoom)

st.subheader("✅ Results")
st.dataframe(df, use_container_width=True)

# Download CSV
if save_csv and os.path.exists(out_csv):
    with open(out_csv, "rb") as f:
        st.download_button(
            "⬇️ Download CSV",
            data=f,
            file_name="bubble_results.csv",
            mime="text/csv",
        )

# Answers expanders
if show_answers and "answers" in df.columns:
    st.subheader("🧾 Answers per page")
    for _, row in df.iterrows():
        page = int(row.get("page", 0))
        code = row.get("student_code", "UNKNOWN")
        flags = str(row.get("flags", ""))
        answers = str(row.get("answers", ""))

        title = f"Page {page} | Code: {code}"
        if flags and flags != "nan":
            title += f" | Flags: {flags}"

        with st.expander(title):
            st.write(answers if answers else "(no answers detected)")
