import os
import io
import re
import pandas as pd
import streamlit as st

from engine_omr import process_pdf


st.set_page_config(page_title="University OMR System", layout="wide")
st.title("🎓 University OMR System (Strong + Template-Free)")
st.caption("PDF → student_code + answers → correct with key → export Excel results")


# -----------------------------
# Helpers
# -----------------------------
def normalize_code(x):
    if x is None:
        return ""
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)  # excel numeric ids
    return s

def parse_answers_str(s: str):
    s = ("" if s is None else str(s)).strip()
    return s.split() if s else []

def parse_key_text(key_text: str):
    key_text = (key_text or "").strip()
    key_text = re.sub(r"[\n\r\t]+", " ", key_text)
    key_text = re.sub(r"\s+", " ", key_text).strip()
    return key_text.split() if key_text else []

def read_roster(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    cols = {c.lower().strip(): c for c in df.columns}
    code_col = None
    name_col = None

    for k in cols:
        if k in ("student_code", "code", "id", "student_id", "studentid", "كود", "رقم", "رقم_الطالب"):
            code_col = cols[k]
        if k in ("student_name", "name", "full_name", "الاسم", "اسم", "اسم_الطالب"):
            name_col = cols[k]

    if code_col is None:
        code_col = df.columns[0]
    if name_col is None:
        name_col = df.columns[1] if len(df.columns) >= 2 else None

    out = pd.DataFrame()
    out["student_code"] = df[code_col].apply(normalize_code).astype(str).str.strip()
    out["student_name"] = df[name_col].astype(str).str.strip() if name_col is not None else ""
    out = out.dropna(subset=["student_code"])
    out["student_code"] = out["student_code"].astype(str).str.strip()
    return out

def load_key(key_mode, key_text, key_file):
    if key_mode == "Paste Key":
        return [x.strip().upper() for x in parse_key_text(key_text)]

    if key_file is None:
        return []

    fname = key_file.name.lower()
    if fname.endswith(".csv"):
        kdf = pd.read_csv(key_file)
    else:
        kdf = pd.read_excel(key_file)

    lower_cols = {c.lower().strip(): c for c in kdf.columns}
    if "answer" in lower_cols:
        ans_col = lower_cols["answer"]
        if "q" in lower_cols or "question" in lower_cols:
            qcol = lower_cols.get("q", lower_cols.get("question"))
            kdf = kdf.sort_values(qcol)
        return [str(x).strip().upper() for x in kdf[ans_col].tolist() if str(x).strip() not in ("", "nan", "None")]

    # fallback: first row
    first = kdf.iloc[0].tolist()
    return [str(x).strip().upper() for x in first if str(x).strip() not in ("", "nan", "None")]

def score_answers(student_ans, key_ans):
    n = min(len(student_ans), len(key_ans))
    correct = wrong = blank = amb = multi = 0
    details = []

    for i in range(n):
        a = student_ans[i] if i < len(student_ans) else ""
        k = key_ans[i]

        if a in ("", None):
            blank += 1; wrong += 1
            details.append((i+1, k, "", "BLANK"))
            continue
        if a == "AMB":
            amb += 1; wrong += 1
            details.append((i+1, k, "AMB", "AMB"))
            continue
        if a == "MULTI":
            multi += 1; wrong += 1
            details.append((i+1, k, "MULTI", "MULTI"))
            continue

        if str(a).upper() == str(k).upper():
            correct += 1
            details.append((i+1, k, a, "OK"))
        else:
            wrong += 1
            details.append((i+1, k, a, "WRONG"))

    return {"n": n, "correct": correct, "wrong": wrong, "blank": blank, "amb": amb, "multi": multi, "details": details}

def make_excel_download(df_results: pd.DataFrame, df_details: pd.DataFrame):
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df_results.to_excel(writer, index=False, sheet_name="Results")
        df_details.to_excel(writer, index=False, sheet_name="Details")
    bio.seek(0)
    return bio


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("1) PDF أوراق الببل شيت")
    pdfs = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

    st.divider()
    st.header("2) Answer Key")
    key_mode = st.radio("Key Input", ["Paste Key", "Upload Key Excel"], index=0)
    key_text = ""
    key_file = None
    if key_mode == "Paste Key":
        key_text = st.text_area("Paste key (مثال): C B B C B ...", height=120)
    else:
        key_file = st.file_uploader("Key Excel/CSV (columns: q, answer OR one-row)", type=["xlsx", "xls", "csv"])

    st.divider()
    st.header("3) Roster Excel (اختياري)")
    roster_file = st.file_uploader("Excel/CSV: student_code + student_name", type=["xlsx", "xls", "csv"])

    st.divider()
    st.header("4) إعدادات")
    zoom = st.slider("PDF zoom (quality)", 1.6, 4.0, 2.6, 0.1)
    strong_margin = st.slider("Prefer strongest mark (حل مشكلة Q5 / MULTI)", 0.06, 0.25, 0.12, 0.01)

    run_btn = st.button("🚀 Run Extraction + Correction + Export")


if not run_btn:
    st.info("⬅️ ارفع الملفات من اليسار واضغط Run")
    st.stop()

if not pdfs:
    st.error("لازم ترفع PDF واحد على الأقل.")
    st.stop()

key = load_key(key_mode, key_text, key_file)
if not key:
    st.error("لازم تدخل Answer Key.")
    st.stop()

roster = None
if roster_file is not None:
    roster = read_roster(roster_file)

tmp_dir = "tmp_uploads"
os.makedirs(tmp_dir, exist_ok=True)

# -----------------------------
# Run extraction
# -----------------------------
all_pages = []
with st.spinner("Extracting answers from PDFs..."):
    for up in pdfs:
        pdf_path = os.path.join(tmp_dir, up.name)
        with open(pdf_path, "wb") as f:
            f.write(up.read())

        df_pages = process_pdf(
            pdf_path,
            out_csv=os.path.join(tmp_dir, "bubble_pages.csv"),
            zoom=zoom,
            strong_margin=strong_margin
        )
        all_pages.append(df_pages)

df_pages = pd.concat(all_pages, ignore_index=True)

st.subheader("📄 Raw Extraction")
st.dataframe(df_pages, use_container_width=True)
st.write("Questions per page:", df_pages["num_questions"].tolist())

# -----------------------------
# Correction
# -----------------------------
results_rows = []
details_rows = []

for _, r in df_pages.iterrows():
    file_name = r.get("file", "")
    page = int(r.get("page", 0))
    student_code = normalize_code(r.get("student_code", "UNKNOWN"))
    answers = parse_answers_str(r.get("answers", ""))

    sc = score_answers(answers, key)

    student_name = ""
    if roster is not None and student_code and student_code != "UNKNOWN":
        m = roster[roster["student_code"] == student_code]
        if len(m) > 0:
            student_name = str(m.iloc[0]["student_name"])

    results_rows.append({
        "file": file_name,
        "page": page,
        "student_code": student_code,
        "student_name": student_name,
        "num_detected": int(r.get("num_questions", len(answers))),
        "num_key": len(key),
        "score": sc["correct"],
        "wrong": sc["wrong"],
        "blank": sc["blank"],
        "amb": sc["amb"],
        "multi": sc["multi"],
        "flags": r.get("flags", ""),
        "answers": " ".join(answers),
    })

    for (q, k, a, status) in sc["details"]:
        details_rows.append({
            "file": file_name,
            "page": page,
            "student_code": student_code,
            "student_name": student_name,
            "q": q,
            "key": k,
            "ans": a,
            "status": status
        })

df_results = pd.DataFrame(results_rows)
df_details = pd.DataFrame(details_rows)

st.subheader("✅ Corrected Results")
st.dataframe(df_results, use_container_width=True)

st.subheader("🔍 Details (first 300 rows)")
st.dataframe(df_details.head(300), use_container_width=True)

# -----------------------------
# Export Excel
# -----------------------------
excel_bytes = make_excel_download(df_results, df_details)
st.download_button(
    "⬇️ Download Excel النتائج النهائية",
    data=excel_bytes,
    file_name="OMR_Results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.success("تم ✅ استخراج + تصحيح + تصدير Excel")
