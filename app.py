import io
import pandas as pd
import streamlit as st

st.set_page_config(page_title="OMR Bubble Sheet", layout="wide")
st.title("OMR Bubble Sheet (Test Deployment)")

st.subheader("1) ملف الطلاب (Roster)")
roster_file = st.file_uploader("Excel/CSV: student_code, student_name", type=["xlsx","xls","csv"])

roster = {}
if roster_file:
    name = roster_file.name.lower()
    df = pd.read_csv(roster_file) if name.endswith(".csv") else pd.read_excel(roster_file)
    df.columns = [c.strip().lower() for c in df.columns]
    if "student_code" not in df.columns or "student_name" not in df.columns:
        st.error("لازم الأعمدة: student_code و student_name")
        st.stop()
    df["student_code"] = df["student_code"].astype(str).str.strip()
    df["student_name"] = df["student_name"].astype(str).str.strip()
    roster = dict(zip(df["student_code"], df["student_name"]))
    st.success(f"تم تحميل {len(roster)} طالب")

st.subheader("2) رفع أي ملف PDF للتجربة (اختياري)")
pdf_file = st.file_uploader("PDF", type=["pdf"])

if st.button("تصدير Excel تجريبي"):
    if not roster:
        st.error("ارفع roster أولاً")
        st.stop()

    # Excel تجريبي: يطلع أول 10 طلاب بدرجة 0
    out = pd.DataFrame({
        "sheet_index": list(range(1, min(11, len(roster)+1))),
        "student_code": list(roster.keys())[:10],
        "student_name": list(roster.values())[:10],
        "score": [0]*min(10, len(roster))
    })

    buf = io.BytesIO()
    out.to_excel(buf, index=False)
    st.download_button("تحميل Excel", buf.getvalue(), "results.xlsx")
