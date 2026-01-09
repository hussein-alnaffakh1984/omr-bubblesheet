"""
🤖 AI OMR - Scalable Version for Large Classes (500-700 students)
معالجة قابلة للتوسع للأعداد الكبيرة

✅ NEW (FIX):
- اقرأ كود الطالب بالـ AI لكن من ROI شبكة الكود فقط (Bubble grid) لمنع خلطه مع الأسئلة
- تحقق صارم للكود + إعادة محاولة + مرشحين واختيار مطابق لقائمة الطلاب
"""

import io, base64, time, gc, re, json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2, numpy as np, pandas as pd
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image
from datetime import datetime


# ----------------------------
# Helpers
# ----------------------------
def read_bytes(f):
    if not f:
        return b""
    try:
        return f.getbuffer().tobytes()
    except:
        try:
            return f.read()
        except:
            return b""


def load_pages(file_bytes, filename, dpi=170):
    """
    ✅ DPI متوسط لتوازن الدقة والذاكرة.
    - لا ترفع كثير حتى لا تتعب الذاكرة
    - ولا تنزل جدًا حتى لا تتدهور قراءة الفقاعات
    """
    if filename.lower().endswith(".pdf"):
        pages = convert_from_bytes(
            file_bytes,
            dpi=dpi,
            fmt="jpeg",
            jpegopt={"quality": 85, "optimize": True},
        )
        return [p.convert("RGB") for p in pages]
    return [Image.open(io.BytesIO(file_bytes)).convert("RGB")]


def pil_to_bgr(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def bgr_to_bytes(bgr):
    ok, buffer = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    return buffer.tobytes() if ok else b""


def normalize_code(x: str) -> str:
    return str(x).strip().replace(" ", "").replace("-", "")


# ----------------------------
# Data classes
# ----------------------------
@dataclass
class AIResult:
    answers: Dict
    confidence: str
    notes: List
    success: bool
    student_code: Optional[str] = None


@dataclass
class StudentRecord:
    student_id: str
    name: str
    code: str


@dataclass
class GradingResult:
    student_id: str
    name: str
    detected_code: str
    score: int
    total: int
    page_number: int = 0


# ----------------------------
# Student loading / matching
# ----------------------------
def load_students_from_excel(file_bytes):
    """Load students from Excel"""
    try:
        df = pd.read_excel(io.BytesIO(file_bytes))
        id_col = name_col = code_col = None

        for col in df.columns:
            cl = str(col).lower().strip()
            if ("id" in cl) or ("رقم" in cl):
                id_col = col
            elif ("name" in cl) or ("اسم" in cl):
                name_col = col
            elif ("code" in cl) or ("كود" in cl) or ("رمز" in cl):
                code_col = col

        if not all([id_col, name_col, code_col]):
            return []

        students = []
        for _, row in df.iterrows():
            students.append(
                StudentRecord(
                    str(row[id_col]),
                    str(row[name_col]),
                    str(row[code_col]),
                )
            )
        return students

    except Exception as e:
        st.error(f"Excel error: {e}")
        return []


def find_student_by_code(students, code):
    """Find student with flexible matching"""
    code_norm = normalize_code(code)

    for s in students:
        s_code = normalize_code(s.code)
        if s_code == code_norm:
            return s

    # Prefix match (لو AI رجّع أكثر من 4)
    if len(code_norm) > 4:
        for length in [4, 5, 6]:
            if len(code_norm) >= length:
                prefix = code_norm[:length]
                for s in students:
                    s_code = normalize_code(s.code)
                    if s_code == prefix:
                        return s
    return None


# ----------------------------
# Grading / Export
# ----------------------------
def grade_student(student_answers, answer_key):
    score = sum(1 for q in answer_key.keys() if student_answers.get(q) == answer_key[q])
    return score, len(answer_key)


def export_results(results):
    data = [
        {
            "Page": r.page_number,
            "ID": r.student_id,
            "Name": r.name,
            "Code": r.detected_code,
            "Score": r.score,
        }
        for r in results
    ]
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame(data).to_excel(writer, sheet_name="Results", index=False)
    return output.getvalue()


# ----------------------------
# ✅ NEW: Code ROI + AI for code only
# ----------------------------
def crop_code_grid_roi(bgr_image):
    """
    ✅ قص شبكة الكود (Bubble grid) فقط.
    IMPORTANT: عدّل هذه النسب مرة واحدة فقط حتى تقع على شبكة الكود بشكل مثالي.
    """
    h, w = bgr_image.shape[:2]

    # قيم بداية جيدة لقالبك (قد تحتاج تعديل بسيط)
    y1, y2 = int(0.12 * h), int(0.33 * h)
    x1, x2 = int(0.05 * w), int(0.50 * w)

    roi = bgr_image[y1:y2, x1:x2].copy()
    return roi


def ai_call(client, model, image_png_bytes, prompt, max_tokens=500):
    image_b64 = base64.b64encode(image_png_bytes).decode("utf-8")
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return msg.content[0].text


def parse_json_robust(txt: str) -> dict:
    # Extract JSON even if wrapped
    if "```" in txt:
        m = re.search(r"\{[\s\S]*\}", txt)
        if m:
            txt = m.group()
    else:
        m = re.search(r"\{[\s\S]*\}", txt)
        if m:
            txt = m.group()
    return json.loads(txt)


def is_valid_code(code: str, students: List[StudentRecord], min_code=1000, max_code=1999) -> bool:
    code = normalize_code(code)
    if not (code.isdigit() and len(code) == 4):
        return False
    v = int(code)
    if not (min_code <= v <= max_code):
        return False
    return find_student_by_code(students, code) is not None


def read_code_with_ai_strict(
    bgr_page: np.ndarray,
    api_key: str,
    students: List[StudentRecord],
    model: str = "claude-sonnet-4-20250514",
    min_code: int = 1000,
    max_code: int = 1057,   # ✅ نطاقك الحقيقي حسب كلامك
    debug_show: bool = False
) -> Tuple[Optional[str], dict]:
    """
    ✅ AI يقرأ كود الطالب من ROI شبكة الكود فقط + تحقق صارم + إعادة محاولة.
    Returns: (code or None, meta)
    """
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    roi = crop_code_grid_roi(bgr_page)

    # تكبير ROI لتحسين قراءة الفقاعات
    roi_big = cv2.resize(roi, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)
    roi_png = bgr_to_bytes(roi_big)

    prompt1 = """
أنت نظام OMR لقراءة كود الطالب من الفقاعات فقط.
الصورة تحتوي شبكة كود: 4 صفوف (row1..row4) وكل صف فيه الأرقام 0..9.
اقرأ كل صف وحده واختر الفقاعة الأغمق.
أعد JSON فقط (بدون شرح):

{
  "row1":"digit",
  "row2":"digit",
  "row3":"digit",
  "row4":"digit",
  "student_code":"dddd",
  "confidence":"high|medium|low"
}
"""

    try:
        txt1 = ai_call(client, model, roi_png, prompt1, max_tokens=350)
        data1 = parse_json_robust(txt1)
        code1 = normalize_code(data1.get("student_code", ""))

        if is_valid_code(code1, students, min_code=min_code, max_code=max_code):
            return code1, {"ok": True, "try": 1, "data": data1}

        # ✅ محاولة ثانية: تكبير أكثر + توجيه أقوى
        roi_big2 = cv2.resize(roi, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        roi_png2 = bgr_to_bytes(roi_big2)

        prompt2 = """
اقرأ شبكة كود الطالب (Bubble grid) بدقة عالية جداً.
- 4 صفوف فقط، كل صف اختيار واحد من 0..9
- تجاهل أي علامات خارج الفقاعات
- إذا كان هناك أكثر من فقاعة مظللة في الصف، اختر الأغمق
أعد JSON فقط:

{"student_code":"dddd","confidence":"high|medium|low"}
"""
        txt2 = ai_call(client, model, roi_png2, prompt2, max_tokens=250)
        data2 = parse_json_robust(txt2)
        code2 = normalize_code(data2.get("student_code", ""))

        if is_valid_code(code2, students, min_code=min_code, max_code=max_code):
            return code2, {"ok": True, "try": 2, "data": data2}

        # ✅ محاولة ثالثة: اطلب 3 مرشحين واختر المطابق لقائمة الطلاب
        prompt3 = """
اقرأ شبكة الكود (4 صفوف × 10 أرقام 0..9).
إذا غير متأكد أعط أفضل 3 احتمالات.
JSON فقط:

{"candidates":["dddd","dddd","dddd"],"confidence":"high|medium|low"}
"""
        txt3 = ai_call(client, model, roi_png2, prompt3, max_tokens=300)
        data3 = parse_json_robust(txt3)
        candidates = [normalize_code(x) for x in data3.get("candidates", [])]

        picked = None
        for c in candidates:
            if is_valid_code(c, students, min_code=min_code, max_code=max_code):
                picked = c
                break

        if picked:
            return picked, {"ok": True, "try": 3, "picked": picked, "data": data3}

        meta = {
            "ok": False,
            "try": 3,
            "first": {"code": code1, "data": data1},
            "second": {"code": code2, "data": data2},
            "candidates": candidates,
        }

        # اختياري: عرض ROI لتشخيص سريع
        if debug_show:
            meta["roi"] = roi
        return None, meta

    except Exception as e:
        return None, {"ok": False, "error": str(e)}


# ----------------------------
# AI analysis for Answer key / Answers (unchanged idea)
# ----------------------------
def analyze_answers_with_ai(image_bytes, api_key, is_answer_key=True):
    """
    ✅ هذه الدالة الآن مخصصة للإجابات فقط.
    - Answer Key: يرجع answers فقط
    - Student page: يرجع answers فقط (لا نطلب student_code هنا)
    """
    if not api_key or len(api_key) < 20:
        return AIResult({}, "no_api", ["API Key required"], False)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        if is_answer_key:
            prompt = "اقرأ Answer Key فقط. JSON فقط: {\"answers\": {\"1\": \"C\", ...}}"
            max_tokens = 800
        else:
            prompt = """
أنت نظام OMR خبير. اقرأ إجابات الطالب فقط (10 أسئلة).
قاعدة X:
- إذا على الفقاعة علامة X فهي ملغاة مهما كانت مظللة.
- من الباقي اختر الأكثر قتامة.
- إذا لا شيء: "?"

JSON فقط:
{
  "answers": {
    "1":"A|B|C|D|?",
    "2":"A|B|C|D|?",
    ...
    "10":"A|B|C|D|?"
  }
}
"""
            max_tokens = 1200

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        response_text = message.content[0].text

        # robust JSON parse
        try:
            result = parse_json_robust(response_text)
        except:
            # fallback
            match = re.search(r"\{[\s\S]*\}", response_text)
            if match:
                result = json.loads(match.group())
            else:
                raise ValueError("No JSON")

        answers = {int(k): v for k, v in result.get("answers", {}).items()}
        return AIResult(answers, result.get("confidence", "medium"), result.get("notes", []), True, None)

    except Exception as e:
        return AIResult({}, "error", [str(e)], False)


# ----------------------------
# MAIN APP
# ----------------------------
def main():
    st.set_page_config(page_title="🤖 AI OMR - Scalable", layout="wide")
    st.title("🤖 نظام OMR للأعداد الكبيرة")
    st.markdown("### 📊 500-700 طالب بدون مشاكل!")

    # Session state
    if "answer_key" not in st.session_state:
        st.session_state.answer_key = {}
    if "students" not in st.session_state:
        st.session_state.students = []
    if "results" not in st.session_state:
        st.session_state.results = []
    if "processed_pages" not in st.session_state:
        st.session_state.processed_pages = set()
    if "duplicate_warnings" not in st.session_state:
        st.session_state.duplicate_warnings = []
    if "allow_duplicates" not in st.session_state:
        st.session_state.allow_duplicates = False

    # Sidebar
    with st.sidebar:
        st.header("⚙️ الإعدادات")

        api_key = ""
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
            if api_key:
                st.success("✅ API Key")
        except:
            pass

        if not api_key:
            api_key = st.text_input("🔑 API Key", type="password")

        st.markdown("---")
        st.metric("Answer Key", f"{len(st.session_state.answer_key)} Q")
        st.metric("Students", len(st.session_state.students))
        st.metric("Graded", len(st.session_state.results))

        if st.session_state.results:
            avg = np.mean([r.score / r.total * 100 for r in st.session_state.results])
            st.metric("Average", f"{avg:.1f}%")

        if st.button("🔄 Reset All", type="secondary"):
            st.session_state.answer_key = {}
            st.session_state.results = []
            st.session_state.processed_pages = set()
            st.session_state.duplicate_warnings = []
            # لا نمسح الطلاب إلا إذا تريد
            st.rerun()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ Answer Key", "2️⃣ Students", "3️⃣ Grade", "4️⃣ Results"])

    # TAB 1: Answer Key
    with tab1:
        st.subheader("📝 Answer Key")
        key_file = st.file_uploader("Upload Answer Key", type=["pdf", "png", "jpg"], key="key")

        if key_file:
            if st.button("🤖 Analyze Key", type="primary"):
                if not api_key:
                    st.error("❌ Need API Key")
                else:
                    with st.spinner("Analyzing Answer Key..."):
                        b = read_bytes(key_file)
                        pages = load_pages(b, key_file.name, dpi=200)
                        if pages:
                            img = bgr_to_bytes(pil_to_bgr(pages[0]))
                            res = analyze_answers_with_ai(img, api_key, True)
                            if res.success:
                                st.session_state.answer_key = res.answers
                                st.success(f"✅ {len(res.answers)} questions")
                            else:
                                st.error("Failed")

        if st.session_state.answer_key:
            st.info(" | ".join([f"Q{q}: {a}" for q, a in sorted(st.session_state.answer_key.items())]))

    # TAB 2: Students
    with tab2:
        st.subheader("👥 Students")
        excel = st.file_uploader("Upload Excel (ID, Name, Code)", type=["xlsx", "xls"], key="excel")

        if excel and st.button("📊 Load"):
            students = load_students_from_excel(read_bytes(excel))
            if students:
                st.session_state.students = students
                st.success(f"✅ {len(students)} students")

        if st.session_state.students:
            st.info(f"Loaded: {len(st.session_state.students)} students")
            with st.expander("View Students"):
                df = pd.DataFrame(
                    [{"ID": s.student_id, "Name": s.name, "Code": s.code} for s in st.session_state.students[:50]]
                )
                st.dataframe(df)

    # TAB 3: Grading
    with tab3:
        st.subheader("✅ Grading - Optimized for Large Scale")

        if not st.session_state.answer_key:
            st.warning("⚠️ Load Answer Key first")
            return
        if not st.session_state.students:
            st.warning("⚠️ Load Students first")
            return

        st.info(
            """
💡 **للأعداد الكبيرة (500-700 طالب):**
- قسّم PDF إلى ملفات أصغر (30-50 ورقة لكل ملف)
- batch صغير 10-20
- الكود الآن يُقرأ من ROI شبكة الكود فقط (Fix)
"""
        )

        sheets = st.file_uploader(
            "ارفع ملف PDF (⚠️ **أقصى حد: 50 صفحة**)",
            type=["pdf"],
            accept_multiple_files=False,
            key="sheets",
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            batch_size = st.slider("📦 Batch size", 5, 20, 10)
        with col2:
            auto_continue = st.checkbox("🔄 Auto-continue", value=False)
        with col3:
            debug_roi = st.checkbox("🧪 Debug ROI (للتشخيص)", value=False)

        st.markdown("---")
        st.subheader("🔍 إدارة التكرارات")

        dup_mode = st.radio(
            "كيف تتعامل مع الأكواد المكررة؟",
            options=[
                "⚠️ تحذير فقط (صحح الجميع)",
                "🚫 تجاهل التكرارات (صحح الأول فقط)",
                "✅ لا تفحص التكرارات (صحح كل شيء)",
            ],
        )

        if st.session_state.duplicate_warnings:
            st.warning(f"⚠️ تم اكتشاف {len(st.session_state.duplicate_warnings)} كود مكرر!")
            with st.expander("عرض الأكواد المكررة"):
                for dup in st.session_state.duplicate_warnings:
                    st.error(f"الكود {dup['code']} - الصفحات: {', '.join(map(str, dup['pages']))}")

        if sheets and "current_file_pages" not in st.session_state:
            if st.button("🔍 Load File"):
                with st.spinner("Loading file..."):
                    b = read_bytes(sheets)
                    pages = load_pages(b, sheets.name, dpi=185)
                    st.session_state.current_file_pages = pages
                    st.session_state.current_file_idx = 0
                    st.success(f"✅ Loaded {len(pages)} pages from {sheets.name}")

        if "current_file_pages" in st.session_state:
            pages = st.session_state.current_file_pages
            current = st.session_state.current_file_idx
            total = len(pages)
            remaining = total - current

            st.metric("File Progress", f"{current}/{total} ({current/total*100:.0f}%)")

            if remaining > 0:
                if st.button(f"🚀 Process next {min(batch_size, remaining)}", type="primary") or auto_continue:
                    end = min(current + batch_size, total)

                    progress = st.progress(0)
                    status = st.empty()

                    processed_count = 0

                    for i in range(current, end):
                        rel = i - current
                        status.text(f"Page {i+1}/{total} ({rel+1}/{end-current})")
                        progress.progress((rel + 1) / (end - current))

                        if i in st.session_state.processed_pages:
                            status.text(f"⏭️ Page {i+1} already processed")
                            continue

                        page = pages[i]
                        bgr = pil_to_bgr(page)

                        # ✅ (1) اقرأ الكود من ROI فقط باستخدام AI + validation + retry/candidates
                        code, meta = read_code_with_ai_strict(
                            bgr,
                            api_key,
                            st.session_state.students,
                            min_code=1000,
                            max_code=1057,   # عدّلها حسب قائمتك لو لزم
                            debug_show=debug_roi
                        )

                        if not code:
                            st.warning(f"⚠️ Page {i+1}: Failed to read CODE (AI+ROI).")
                            if debug_roi and "roi" in meta:
                                st.image(meta["roi"], caption=f"Code ROI - Page {i+1}")
                                st.json({k: v for k, v in meta.items() if k != "roi"})
                            st.session_state.processed_pages.add(i)  # حتى لا يعيد الدوران على نفس الصفحة
                            del page, bgr
                            continue

                        student = find_student_by_code(st.session_state.students, code)
                        if not student:
                            st.warning(f"⚠️ Page {i+1}: Code {code} not found in student list")
                            st.session_state.processed_pages.add(i)
                            del page, bgr
                            continue

                        # ✅ (2) اقرأ الإجابات (AI) من الصفحة كاملة
                        img_page = bgr_to_bytes(bgr)
                        res_ans = analyze_answers_with_ai(img_page, api_key, is_answer_key=False)

                        # تحرير ذاكرة مبكر
                        del page, bgr, img_page

                        if not res_ans.success:
                            st.warning(f"⚠️ Page {i+1}: Failed to read answers")
                            st.session_state.processed_pages.add(i)
                            continue

                        # GC كل 10 صفحات
                        if (i - current) % 10 == 0:
                            gc.collect()

                        # Duplicate check
                        already_graded = any(r.detected_code == code for r in st.session_state.results)
                        if already_graded:
                            if "تجاهل" in dup_mode:
                                st.info(f"ℹ️ Page {i+1}: Code {code} ({student.name}) already graded - skipping")
                                st.session_state.processed_pages.add(i)
                                continue
                            elif "تحذير" in dup_mode:
                                st.warning(f"⚠️ Page {i+1}: Code {code} is DUPLICATE - grading anyway")
                                existing_dup = next(
                                    (d for d in st.session_state.duplicate_warnings if d["code"] == code), None
                                )
                                if existing_dup:
                                    existing_dup["pages"].append(i + 1)
                                else:
                                    st.session_state.duplicate_warnings.append(
                                        {"code": code, "name": student.name, "pages": [i + 1]}
                                    )
                            # Mode 3: no check

                        score, total_q = grade_student(res_ans.answers, st.session_state.answer_key)

                        st.session_state.results.append(
                            GradingResult(student.student_id, student.name, code, score, total_q, i + 1)
                        )

                        st.session_state.processed_pages.add(i)
                        processed_count += 1
                        status.text(f"✅ Page {i+1}: {code} - {student.name} ({score}/{total_q})")

                    st.session_state.current_file_idx = end

                    # Cleanup when file done
                    if end >= total:
                        del st.session_state.current_file_pages
                        del st.session_state.current_file_idx
                        gc.collect()

                    gc.collect()
                    st.success(f"✅ Processed {processed_count} pages")

                    if end >= total:
                        st.balloons()
                        st.success("🎉 File complete!")
                    elif auto_continue:
                        time.sleep(0.4)
                        st.rerun()

            else:
                st.success("File complete! Upload next file or go to Results.")

    # TAB 4: Results
    with tab4:
        st.subheader("📊 Results")

        if not st.session_state.results:
            st.info("No results yet")
            return

        # Duplicate warnings section
        if st.session_state.duplicate_warnings:
            st.error(f"⚠️ **تحذير: تم اكتشاف {len(st.session_state.duplicate_warnings)} كود مكرر!**")

            with st.expander("🔍 تفاصيل الأكواد المكررة", expanded=True):
                for dup in st.session_state.duplicate_warnings:
                    st.warning(
                        f"**الكود:** {dup['code']} - **الاسم:** {dup['name']}  \n"
                        f"**الصفحات:** {', '.join(map(str, dup['pages']))}  \n"
                        f"**عدد التكرارات:** {len(dup['pages'])} مرة"
                    )

                dup_codes = [d["code"] for d in st.session_state.duplicate_warnings]
                dup_results = [r for r in st.session_state.results if r.detected_code in dup_codes]
                if dup_results:
                    dup_df = pd.DataFrame(
                        [{"Page": r.page_number, "Code": r.detected_code, "Name": r.name, "Score": r.score}
                         for r in dup_results]
                    )
                    st.dataframe(dup_df, width="stretch")

            st.markdown("---")

        scores = [r.score / r.total * 100 for r in st.session_state.results]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Graded", len(scores))
        with col2:
            st.metric("Average", f"{np.mean(scores):.1f}%")
        with col3:
            st.metric("Max", f"{np.max(scores):.1f}%")
        with col4:
            st.metric("Min", f"{np.min(scores):.1f}%")

        df = pd.DataFrame(
            [
                {
                    "Page": r.page_number,
                    "ID": r.student_id,
                    "Name": r.name,
                    "Code": r.detected_code,
                    "Score": r.score,
                    "%": f"{r.score / r.total * 100:.0f}",
                }
                for r in st.session_state.results
            ]
        )
        st.dataframe(df, width="stretch")

        # Duplicate cleaning options
        if st.session_state.duplicate_warnings:
            st.markdown("---")
            st.subheader("🧹 تنظيف التكرارات")

            clean_method = st.radio(
                "كيف تريد التعامل مع التكرارات؟",
                options=[
                    "احتفظ بالأول فقط",
                    "احتفظ بالأعلى درجة",
                    "احتفظ بالأقل درجة (للمراجعة)",
                    "احتفظ بالجميع (Excel سيظهر كلهم)",
                ],
            )

            if st.button("🧹 إنشاء نتائج نظيفة"):
                if "الأول" in clean_method:
                    clean_results = []
                    seen_codes = set()
                    for r in sorted(st.session_state.results, key=lambda x: x.page_number):
                        if r.detected_code not in seen_codes:
                            clean_results.append(r)
                            seen_codes.add(r.detected_code)
                    st.session_state.clean_results = clean_results
                    st.success(
                        f"✅ تم! {len(clean_results)} نتيجة نظيفة (حذف {len(st.session_state.results) - len(clean_results)} تكرار)"
                    )

                elif "الأعلى" in clean_method:
                    from collections import defaultdict
                    by_code = defaultdict(list)
                    for r in st.session_state.results:
                        by_code[r.detected_code].append(r)
                    clean_results = [max(v, key=lambda x: x.score) for v in by_code.values()]
                    st.session_state.clean_results = clean_results
                    st.success(f"✅ تم! {len(clean_results)} نتيجة (أفضل درجة لكل كود)")

                elif "الأقل" in clean_method:
                    from collections import defaultdict
                    by_code = defaultdict(list)
                    for r in st.session_state.results:
                        by_code[r.detected_code].append(r)
                    clean_results = [min(v, key=lambda x: x.score) for v in by_code.values()]
                    st.session_state.clean_results = clean_results
                    st.success(f"✅ تم! {len(clean_results)} نتيجة (أقل درجة للمراجعة)")

                else:
                    st.session_state.clean_results = st.session_state.results
                    st.success("✅ تم! الاحتفاظ بالجميع")

        st.markdown("---")
        results_to_export = st.session_state.get("clean_results", st.session_state.results)

        if st.button("📥 Export Excel", type="primary"):
            excel = export_results(results_to_export)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            status_text = f"({len(results_to_export)} نتيجة"
            if "clean_results" in st.session_state and len(results_to_export) < len(st.session_state.results):
                status_text += f" - تم تنظيف {len(st.session_state.results) - len(results_to_export)} تكرار"
            status_text += ")"

            st.download_button(
                f"⬇️ Download {status_text}",
                excel,
                f"results_{ts}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


if __name__ == "__main__":
    main()
