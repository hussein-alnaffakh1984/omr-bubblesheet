import os
import io
import re
import pandas as pd
import streamlit as st

from engine_omr import process_pdf


st.set_page_config(page_title="University OMR", layout="wide")
st.title("🎓 University OMR System")
st.caption("1) Key PDF  2) Roster Excel (names+codes)  3) Students PDF  → Correct + Export Excel")


# -----------------------------
# Helpers
# -----------------------------
def normalize_code(x):
    if x is None:
        return ""
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)
    return s

def parse_answers_str(s: str):
    s = ("" if s is None else str(s)).strip()
    return s.split() if s else []

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

def pick_best_key_from_pages(df_key_pages: pd.DataFrame) -> list:
    best = []
    for _, r in df_key_pages.iterrows():
        ans = parse_answers_str(r.get("answers", ""))
        if len(ans) > len(best):
            best = ans
    return best

def score_answers(student_ans, key_ans):
    n = min(len(student_ans), len(key_ans))
    correct = wrong = blank = amb = multi = 0
    details = []

    for i in range(n):
        a = student_ans[i] if i < len(student_ans) else ""
        k = key_ans[i]

        if a in ("", None):
            blank += 1; wrong += 1
            details.append((i+1, k, "", "BLANK")); continue
        if a == "AMB":
            amb += 1; wrong += 1
            details.append((i+1, k, "AMB", "AMB")); continue
        if a == "MULTI":
            multi += 1; wrong += 1
            details.append((i+1, k, "MULTI", "MULTI")); continue

        if str(a).upper() == str(k).upper():
            correct += 1
            details.append((i+1, k, a, "OK"))
        else:
            wrong += 1
            details.append((i+1, k, a, "WRONG"))

    return {"n": n, "correct": correct, "wrong": wrong, "blank": blank, "amb": amb, "multi": multi, "details": details}


# -----------------------------
# UI
# -----------------------------
with st.sidebar:
    st.header("1) ارفع PDF الانسر (Answer Key)")
    key_pdf = st.file_uploader("Key PDF", type=["pdf"])

    st.divider()
    st.header("2) ارفع Excel أسماء الطلاب + الأكواد")
    roster_file = st.file_uploader("Roster Excel/CSV", type=["xlsx", "xls", "csv"])

    st.divider()
    st.header("3) ارفع PDF الطلبة (كل الصفحات)")
    students_pdf = st.file_uploader("Students PDF", type=["pdf"])

    st.divider()
    st.header("إعدادات")
    zoom = st.slider("PDF zoom (quality)", 1.6, 4.0, 2.6, 0.1)
    strong_margin = st.slider("Prefer strongest mark (حل MULTI مثل سؤال 5)", 0.06, 0.25, 0.12, 0.01)

    run_btn = st.button("🚀 Run & Export")


if not run_btn:
    st.info("⬅️ ارفع الملفات الثلاثة ثم اضغط Run")
    st.stop()

if key_pdf is None:
    st.error("لازم ترفع PDF الانسر.")
    st.stop()

if roster_file is None:
    st.error("لازم ترفع Excel roster.")
    st.stop()

if students_pdf is None:
    st.error("لازم ترفع PDF الطلبة.")
    st.stop()


tmp_dir = "tmp_uploads"
os.makedirs(tmp_dir, exist_ok=True)

# -----------------------------
# 1) Key extraction
# -----------------------------
key_path = os.path.join(tmp_dir, "KEY_" + key_pdf.name)
with open(key_path, "wb") as f:
    f.write(key_pdf.read())

with st.spinner("1/3 استخراج الانسر من Key PDF..."):
    df_key_pages = process_pdf(key_path, out_csv=os.path.join(tmp_dir, "key_pages.csv"), zoom=zoom, strong_margin=strong_margin)

st.subheader("🗝️ Key PDF Extraction")
st.dataframe(df_key_pages, use_container_width=True)

key_answers = pick_best_key_from_pages(df_key_pages)
if not key_answers:
    st.error("ماكدرنا نستخرج الانسر من Key PDF. تأكد الفقاعات واضحة.")
    st.stop()

st.success(f"✅ Key detected: {len(key_answers)} answers")
st.write("Key preview (first 30):", " ".join(key_answers[:30]))


# -----------------------------
# 2) Roster
# -----------------------------
with st.spinner("2/3 قراءة roster Excel..."):
    roster = read_roster(roster_file)

st.subheader("👥 Roster Preview")
st.dataframe(roster.head(50), use_container_width=True)

roster_map = dict(zip(roster["student_code"].astype(str), roster["student_name"].astype(str)))


# -----------------------------
# 3) Students extraction
# -----------------------------
students_path = os.path.join(tmp_dir, "STUDENTS_" + students_pdf.name)
with open(students_path, "wb") as f:
    f.write(students_pdf.read())

with st.spinner("3/3 استخراج أكواد وإجابات الطلبة من Students PDF..."):
    df_pages = process_pdf(students_path, out_csv=os.path.join(tmp_dir, "students_pages.csv"), zoom=zoom, strong_margin=strong_margin)

st.subheader("📄 Students Extraction (Raw)")
st.dataframe(df_pages, use_container_width=True)
st.write("Questions per page:", df_pages["num_questions"].tolist())


# -----------------------------
# Matching + scoring
# -----------------------------
results_rows = []
details_rows = []

for _, r in df_pages.iterrows():
    page = int(r.get("page", 0))
    code = normalize_code(r.get("student_code", "UNKNOWN"))
    name = roster_map.get(code, "") if code not in ("", "UNKNOWN") else ""
    answers = parse_answers_str(r.get("answers", ""))

    sc = score_answers(answers, key_answers)

    results_rows.append({
        "student_code": code,
        "student_name": name,
        "page": page,
        "score": sc["correct"],
        "num_key": len(key_answers),
        "wrong": sc["wrong"],
        "blank": sc["blank"],
        "amb": sc["amb"],
        "multi": sc["multi"],
        "flags": r.get("flags", ""),
        "answers_sequence": " ".join(answers),
    })

    for (q, k, a, status) in sc["details"]:
        details_rows.append({
            "student_code": code,
            "student_name": name,
            "page": page,
            "q": q,
            "key": k,
            "ans": a,
            "status": status,
        })

df_results = pd.DataFrame(results_rows)
df_details = pd.DataFrame(details_rows)

st.subheader("✅ Results (Per Student Page)")
st.dataframe(df_results, use_container_width=True)

st.subheader("🔍 Details (first 300 rows)")
st.dataframe(df_details.head(300), use_container_width=True)


# -----------------------------
# Export Excel
# -----------------------------
bio = io.BytesIO()
with pd.ExcelWriter(bio, engine="openpyxl") as writer:
    pd.DataFrame({"q": list(range(1, len(key_answers) + 1)), "key": key_answers}).to_excel(writer, index=False, sheet_name="AnswerKey")
    df_results.to_excel(writer, index=False, sheet_name="Results")
    df_details.to_excel(writer, index=False, sheet_name="Details")
bio.seek(0)

st.download_button(
    "⬇️ Download Excel النتائج النهائية",
    data=bio,
    file_name="OMR_Results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.success("✅ تم: استخراج الانسر + مطابقة الطلبة + تصدير Excel")
