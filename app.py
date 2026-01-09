"""
🤖 OMR Pro — Code from Bubbles ONLY + Answers via AI (optional)
✅ كود الطالب: OMR حتمي من الفقاعات فقط (لا OCR، لا تخمين، لا AI للكود)
✅ محاذاة الورقة (Perspective Alignment) لتثبيت مكان الشبكات
✅ وضع صارم: إذا الكود غير واضح → REVIEW بدل كود غلط
✅ نتائج + تصدير Excel بدون KeyError

متطلبات:
pip install streamlit opencv-python numpy pandas pdf2image pillow openpyxl
وعلى Linux تحتاج poppler لـ pdf2image.
"""

import io, base64, gc, re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image
from datetime import datetime


# =========================
# Basic IO
# =========================
def read_bytes(f):
    if not f:
        return b""
    try:
        return f.getbuffer().tobytes()
    except Exception:
        try:
            return f.read()
        except Exception:
            return b""


def load_pages(file_bytes: bytes, filename: str, dpi: int = 220) -> List[Image.Image]:
    """Load PDF pages (or single image) to PIL images."""
    if filename.lower().endswith(".pdf"):
        pages = convert_from_bytes(
            file_bytes,
            dpi=dpi,
            fmt="jpeg",
            jpegopt={"quality": 88, "optimize": True},
        )
        return [p.convert("RGB") for p in pages]
    return [Image.open(io.BytesIO(file_bytes)).convert("RGB")]


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def bgr_to_png_bytes(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        return b""
    return buf.tobytes()


# =========================
# Students
# =========================
@dataclass
class StudentRecord:
    student_id: str
    name: str
    code: str


def load_students_from_excel(file_bytes: bytes) -> List[StudentRecord]:
    try:
        df = pd.read_excel(io.BytesIO(file_bytes))
        id_col = name_col = code_col = None

        for col in df.columns:
            cl = str(col).lower().strip()
            if id_col is None and ("id" in cl or "رقم" in cl):
                id_col = col
            if name_col is None and ("name" in cl or "اسم" in cl):
                name_col = col
            if code_col is None and ("code" in cl or "كود" in cl or "رمز" in cl):
                code_col = col

        if not all([id_col, name_col, code_col]):
            return []

        students: List[StudentRecord] = []
        for _, row in df.iterrows():
            sid = str(row.get(id_col, "")).strip()
            nm = str(row.get(name_col, "")).strip()
            cd = str(row.get(code_col, "")).strip().replace(" ", "").replace("-", "")
            students.append(StudentRecord(sid, nm, cd))
        return students
    except Exception as e:
        st.error(f"Excel error: {e}")
        return []


def find_student_by_code(students: List[StudentRecord], code: str) -> Optional[StudentRecord]:
    code_norm = str(code).strip().replace(" ", "").replace("-", "")
    for s in students:
        s_code = str(s.code).strip().replace(" ", "").replace("-", "")
        if s_code == code_norm:
            return s
    return None


# =========================
# Perspective Alignment
# =========================
def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def align_page(bgr: np.ndarray, out_w: int = 1700, out_h: int = 2400) -> Tuple[np.ndarray, bool]:
    """
    Align scanned page using largest 4-point contour.
    If fails, return original.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return bgr, False

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    page_cnt = None

    for c in cnts[:6]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.20 * (h * w):
            page_cnt = approx
            break

    if page_cnt is None:
        return bgr, False

    pts = _order_points(page_cnt.reshape(4, 2))
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(bgr, M, (out_w, out_h))
    return warped, True


# =========================
# Code OMR (Bubbles ONLY)
# =========================
@dataclass
class CodeRead:
    code: Optional[str]
    ok: bool
    reason: str
    row_digits: List[int]
    row_scores: List[float]
    row_margins: List[float]


def _ink_score(cell_gray: np.ndarray) -> float:
    """
    Compute ink ratio inside center circle.
    Higher = darker.
    """
    g = cv2.GaussianBlur(cell_gray, (3, 3), 0)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)

    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    hh, ww = th.shape
    mask = np.zeros_like(th)
    cy, cx = hh // 2, ww // 2
    r = int(min(hh, ww) * 0.33)
    cv2.circle(mask, (cx, cy), r, 255, -1)

    ink = cv2.countNonZero(cv2.bitwise_and(th, mask))
    area = cv2.countNonZero(mask) + 1e-6
    return float(ink) / float(area)


def extract_code_roi(aligned_bgr: np.ndarray, roi_cfg: Dict[str, float]) -> np.ndarray:
    """
    Extract code grid ROI based on relative coordinates.
    roi_cfg = {x1,x2,y1,y2} in [0..1] relative to aligned size.
    """
    H, W = aligned_bgr.shape[:2]
    y1 = int(roi_cfg["y1"] * H)
    y2 = int(roi_cfg["y2"] * H)
    x1 = int(roi_cfg["x1"] * W)
    x2 = int(roi_cfg["x2"] * W)
    y1 = max(0, min(H - 1, y1))
    y2 = max(1, min(H, y2))
    x1 = max(0, min(W - 1, x1))
    x2 = max(1, min(W, x2))
    if y2 <= y1 or x2 <= x1:
        return aligned_bgr.copy()
    return aligned_bgr[y1:y2, x1:x2].copy()


def read_code_from_bubbles(aligned_bgr: np.ndarray, roi_cfg: Dict[str, float],
                           min_ink: float, min_margin: float) -> CodeRead:
    """
    Reads code grid (4 rows × 10 cols) deterministically.
    Reject if row faint or ambiguous.
    """
    roi = extract_code_roi(aligned_bgr, roi_cfg)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    rows, cols = 4, 10
    rh = gray.shape[0] / rows
    cw = gray.shape[1] / cols

    chosen: List[int] = []
    scores: List[float] = []
    margins: List[float] = []

    for r in range(rows):
        row_scores = []
        for c in range(cols):
            yA, yB = int(r * rh), int((r + 1) * rh)
            xA, xB = int(c * cw), int((c + 1) * cw)
            cell = gray[yA:yB, xA:xB]
            row_scores.append(_ink_score(cell))

        row_scores = np.array(row_scores, dtype=np.float32)
        idx = np.argsort(-row_scores)
        best = int(idx[0])
        second = int(idx[1])

        best_sc = float(row_scores[best])
        second_sc = float(row_scores[second])
        margin = best_sc - second_sc

        chosen.append(best)
        scores.append(best_sc)
        margins.append(margin)

        if best_sc < min_ink:
            return CodeRead(None, False, f"REVIEW: row {r+1} too faint (best={best_sc:.3f})",
                            chosen, scores, margins)

        if margin < min_margin:
            return CodeRead(None, False, f"REVIEW: row {r+1} ambiguous (margin={margin:.3f})",
                            chosen, scores, margins)

    code = "".join(str(d) for d in chosen)
    if not (code.isdigit() and len(code) == 4):
        return CodeRead(None, False, "REVIEW: invalid code format", chosen, scores, margins)

    return CodeRead(code, True, "OK", chosen, scores, margins)


# =========================
# AI Answers (optional)
# =========================
@dataclass
class AIResult:
    answers: Dict[int, str]
    success: bool
    notes: List[str]
    confidence: str = "medium"


def analyze_answers_with_ai(full_page_png_bytes: bytes, api_key: str) -> AIResult:
    """
    AI reads answers only.
    If you want pure OMR answers later, replace this function.
    """
    if not api_key or len(api_key) < 20:
        return AIResult({}, False, ["API Key required"], "no_api")

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        image_b64 = base64.b64encode(full_page_png_bytes).decode("utf-8")

        prompt = r"""
أنت نظام OMR خبير.
اقرأ إجابات الطالب (10 أسئلة) من الفقاعات فقط.
قواعد:
1) إذا يوجد X على/فوق فقاعة => تُلغى حتى لو مظللة.
2) بعد إلغاء فقاعات X: اختر الأغمق فقط.
3) إذا لا يوجد اختيار واضح => ضع "?".
أعد JSON فقط:
{"answers":{"1":"A","2":"B",...,"10":"D"}}
"""

        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=700,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )

        txt = msg.content[0].text
        import json
        m = re.search(r"\{[\s\S]*\}", txt)
        data = json.loads(m.group()) if m else {}

        ans = data.get("answers", {})
        answers: Dict[int, str] = {}
        for k, v in ans.items():
            try:
                q = int(k)
                answers[q] = str(v).strip().upper()
            except Exception:
                pass

        if not answers:
            return AIResult({}, False, ["AI returned no answers"], "low")

        return AIResult(answers, True, [], data.get("confidence", "medium"))

    except Exception as e:
        return AIResult({}, False, [str(e)], "error")


# =========================
# Grading / Export
# =========================
def grade_student(student_answers: Dict[int, str], answer_key: Dict[int, str]) -> Tuple[int, int]:
    total = len(answer_key)
    score = 0
    for q, a in answer_key.items():
        if student_answers.get(q) == a:
            score += 1
    return score, total


def export_results_df(df: pd.DataFrame) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Results", index=False)
    return out.getvalue()


# =========================
# Streamlit App
# =========================
def main():
    st.set_page_config(page_title="OMR Pro (Code bubbles) + AI Answers", layout="wide")
    st.title("✅ OMR Pro — كود من الفقاعات فقط + إجابات بالـ AI (اختياري)")

    # Session
    if "answer_key" not in st.session_state: st.session_state.answer_key = {}
    if "students" not in st.session_state: st.session_state.students = []
    if "results" not in st.session_state: st.session_state.results = []
    if "review_pages" not in st.session_state: st.session_state.review_pages = []
    if "processed_pages" not in st.session_state: st.session_state.processed_pages = set()
    if "current_pages" not in st.session_state: st.session_state.current_pages = None
    if "current_idx" not in st.session_state: st.session_state.current_idx = 0

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # API key for AI answers
        api_key = ""
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
            if api_key:
                st.success("✅ API Key from secrets")
        except Exception:
            pass
        if not api_key:
            api_key = st.text_input("🔑 Anthropic API Key (للإجابات)", type="password")

        st.markdown("---")

        # ROI config (relative coords) — adjust ONCE
        st.subheader("📌 Code ROI (ثابت)")
        st.caption("عدّلها مرة واحدة إذا احتجت (للشبكة أعلى اليمين).")

        # Defaults tuned for your sheet style (aligned 1700x2400)
        roi_cfg = {
            "y1": st.number_input("y1", 0.0, 1.0, 0.12, 0.01),
            "y2": st.number_input("y2", 0.0, 1.0, 0.32, 0.01),
            "x1": st.number_input("x1", 0.0, 1.0, 0.60, 0.01),
            "x2": st.number_input("x2", 0.0, 1.0, 0.88, 0.01),
        }

        st.markdown("---")
        st.subheader("🛡️ Strictness")
        min_ink = st.slider("MIN_INK (ضعيف=Review)", 0.01, 0.20, 0.06, 0.005)
        min_margin = st.slider("MIN_MARGIN (التباس=Review)", 0.005, 0.10, 0.018, 0.001)

        st.markdown("---")
        st.subheader("📊 Status")
        st.metric("Answer Key", f"{len(st.session_state.answer_key)} Q")
        st.metric("Students", len(st.session_state.students))
        st.metric("Graded", len(st.session_state.results))
        st.metric("Review", len(st.session_state.review_pages))

        if st.button("🔄 Reset All", type="secondary"):
            st.session_state.answer_key = {}
            st.session_state.students = []
            st.session_state.results = []
            st.session_state.review_pages = []
            st.session_state.processed_pages = set()
            st.session_state.current_pages = None
            st.session_state.current_idx = 0
            st.rerun()

    tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ Answer Key", "2️⃣ Students", "3️⃣ Grade", "4️⃣ Results"])

    # -------------------------
    # TAB 1 — Answer Key
    # -------------------------
    with tab1:
        st.subheader("📝 Answer Key")

        mode = st.radio("طريقة إدخال Answer Key:", ["Manual (يدوي)", "AI (من ورقة key)"], horizontal=True)

        if mode.startswith("Manual"):
            st.info("أدخل الإجابات يدويًا (10 أسئلة).")
            cols = st.columns(10)
            tmp = {}
            for i in range(10):
                with cols[i]:
                    tmp[i+1] = st.selectbox(f"Q{i+1}", ["A", "B", "C", "D"], index=0, key=f"k{i+1}")
            if st.button("✅ Save Answer Key", type="primary"):
                st.session_state.answer_key = {q: a for q, a in tmp.items()}
                st.success("✅ Saved")
        else:
            st.warning("AI هنا للـ Answer Key فقط. الكود لا يستخدم AI إطلاقًا.")
            key_file = st.file_uploader("Upload Answer Key sheet (PDF/IMG)", type=["pdf", "png", "jpg"])
            if key_file and st.button("🤖 Read Answer Key with AI", type="primary"):
                if not api_key:
                    st.error("❌ API Key required")
                else:
                    b = read_bytes(key_file)
                    pages = load_pages(b, key_file.name, dpi=250)
                    bgr = pil_to_bgr(pages[0])
                    aligned, _ = align_page(bgr)
                    ai = analyze_answers_with_ai(bgr_to_png_bytes(aligned), api_key)
                    if ai.success:
                        st.session_state.answer_key = ai.answers
                        st.success(f"✅ Loaded {len(ai.answers)} answers")
                    else:
                        st.error("Failed: " + " | ".join(ai.notes))

        if st.session_state.answer_key:
            st.markdown("**Current Key:** " + " | ".join([f"Q{q}:{a}" for q, a in sorted(st.session_state.answer_key.items())]))

    # -------------------------
    # TAB 2 — Students
    # -------------------------
    with tab2:
        st.subheader("👥 Students")
        excel = st.file_uploader("Upload Excel (ID, Name, Code)", type=["xlsx", "xls"])
        if excel and st.button("📊 Load Students", type="primary"):
            students = load_students_from_excel(read_bytes(excel))
            if students:
                st.session_state.students = students
                st.success(f"✅ Loaded {len(students)} students")
            else:
                st.error("❌ لم أستطع تحديد أعمدة ID/Name/Code تلقائيًا. تأكد من العناوين.")

        if st.session_state.students:
            st.info(f"Loaded: {len(st.session_state.students)} students")
            with st.expander("Preview (first 50)"):
                dfp = pd.DataFrame([{"ID": s.student_id, "Name": s.name, "Code": s.code} for s in st.session_state.students[:50]])
                st.dataframe(dfp, use_container_width=True)

    # -------------------------
    # TAB 3 — Grade
    # -------------------------
    with tab3:
        st.subheader("✅ Grading — Code via Bubbles ONLY (No Guessing)")
        if not st.session_state.answer_key:
            st.warning("⚠️ Load Answer Key first")
            return
        if not st.session_state.students:
            st.warning("⚠️ Load Students first")
            return

        colA, colB, colC = st.columns(3)
        with colA:
            dpi = st.slider("DPI", 150, 320, 220, help="220 مناسب. ارفع إذا الورق باهت.")
        with colB:
            batch_size = st.slider("Batch size", 5, 30, 10)
        with colC:
            strict_mode = st.checkbox("Strict mode (Review بدل تصحيح)", value=True)

        show_preview = st.checkbox("🖼️ Preview code ROI أثناء المعالجة", value=False)
        show_debug = st.checkbox("🧾 Debug reason/values", value=False)

        sheets = st.file_uploader("Upload Students Sheets PDF (يفضل ≤ 50 صفحة لكل ملف)", type=["pdf"])

        if sheets and st.button("🔍 Load File", type="primary"):
            with st.spinner("Loading pages..."):
                b = read_bytes(sheets)
                pages = load_pages(b, sheets.name, dpi=dpi)
                st.session_state.current_pages = pages
                st.session_state.current_idx = 0
                st.success(f"✅ Loaded {len(pages)} pages")

        if st.session_state.current_pages is None:
            st.info("ارفع ملف PDF ثم اضغط Load File.")
            return

        pages = st.session_state.current_pages
        idx = int(st.session_state.current_idx)
        total = len(pages)

        st.metric("File Progress", f"{idx}/{total} ({(idx/total*100 if total else 0):.0f}%)")

        if idx < total and st.button(f"🚀 Process next {min(batch_size, total-idx)}", type="primary"):
            end = min(idx + batch_size, total)
            prog = st.progress(0)
            status = st.empty()

            processed = 0

            for i in range(idx, end):
                prog.progress((i - idx + 1) / (end - idx))
                status.text(f"Page {i+1}/{total}")

                if i in st.session_state.processed_pages:
                    continue

                # ---- Load & align page
                bgr = pil_to_bgr(pages[i])
                aligned, ok_align = align_page(bgr)

                # ---- Read code (OMR bubbles only)
                code_read = read_code_from_bubbles(aligned, roi_cfg=roi_cfg, min_ink=min_ink, min_margin=min_margin)

                if show_preview:
                    roi_img = extract_code_roi(aligned, roi_cfg)
                    st.image(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB), caption=f"ROI Page {i+1}", use_container_width=True)

                if not code_read.ok:
                    st.session_state.review_pages.append({
                        "Page": i+1,
                        "Reason": code_read.reason,
                        "Code": "REVIEW"
                    })
                    st.session_state.processed_pages.add(i)
                    if show_debug:
                        st.warning(f"⚠️ Page {i+1}: {code_read.reason}")
                        st.write({"digits": code_read.row_digits, "scores": code_read.row_scores, "margins": code_read.row_margins})
                    if strict_mode:
                        continue
                    code = None
                else:
                    code = code_read.code

                # ---- Validate code in student list (NO guessing)
                student = find_student_by_code(st.session_state.students, code) if code else None
                if student is None:
                    st.session_state.review_pages.append({
                        "Page": i+1,
                        "Reason": "REVIEW: code not found in student list",
                        "Code": code if code else "REVIEW"
                    })
                    st.session_state.processed_pages.add(i)
                    if show_debug:
                        st.warning(f"⚠️ Page {i+1}: code '{code}' not in Excel list")
                    if strict_mode:
                        continue

                # ---- Read answers (AI optional)
                if not api_key:
                    st.session_state.review_pages.append({
                        "Page": i+1,
                        "Reason": "REVIEW: missing API key for answers",
                        "Code": code
                    })
                    st.session_state.processed_pages.add(i)
                    if strict_mode:
                        continue

                ai = analyze_answers_with_ai(bgr_to_png_bytes(aligned), api_key)
                if not ai.success:
                    st.session_state.review_pages.append({
                        "Page": i+1,
                        "Reason": "REVIEW: AI failed to read answers",
                        "Code": code
                    })
                    st.session_state.processed_pages.add(i)
                    if strict_mode:
                        continue

                score, total_q = grade_student(ai.answers, st.session_state.answer_key)

                st.session_state.results.append({
                    "Page": i+1,
                    "ID": student.student_id,
                    "Name": student.name,
                    "Code": code,
                    "Score": score,
                    "Total": total_q
                })

                st.session_state.processed_pages.add(i)
                processed += 1

                # Memory clean
                del bgr, aligned
                if (i - idx) % 7 == 0:
                    gc.collect()

            st.session_state.current_idx = end
            gc.collect()
            st.success(f"✅ Processed {processed} pages")

            if end >= total:
                st.success("🎉 File complete")
                st.session_state.current_pages = None
                st.session_state.current_idx = 0
                gc.collect()

    # -------------------------
    # TAB 4 — Results (Robust)
    # -------------------------
    with tab4:
        st.subheader("📊 Results")

        # REVIEW
        if st.session_state.get("review_pages"):
            review_df = pd.DataFrame(st.session_state.review_pages)
            st.error(f"⚠️ REVIEW pages: {len(review_df)}")
            with st.expander("عرض صفحات المراجعة"):
                st.dataframe(review_df, use_container_width=True)

        if not st.session_state.get("results"):
            st.info("No graded results yet.")
            return

        df = pd.DataFrame(st.session_state.results)

        # Ensure columns exist
        for c in ["Page", "ID", "Name", "Code", "Score", "Total"]:
            if c not in df.columns:
                df[c] = "" if c in ["Page", "ID", "Name", "Code"] else 0

        # Normalize numeric
        df["Score"] = pd.to_numeric(df["Score"], errors="coerce").fillna(0).astype(int)
        df["Total"] = pd.to_numeric(df["Total"], errors="coerce").fillna(0).astype(int)

        # Compute Percent safely (always)
        df["Percent"] = np.where(df["Total"] > 0, (df["Score"] / df["Total"]) * 100.0, 0.0).round(1)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Graded", int(len(df)))
        with col2:
            st.metric("Avg %", f"{float(df['Percent'].mean()):.1f}")
        with col3:
            st.metric("Max %", f"{float(df['Percent'].max()):.1f}")
        with col4:
            st.metric("Min %", f"{float(df['Percent'].min()):.1f}")

        show_cols = ["Page", "ID", "Name", "Code", "Score", "Total", "Percent"]
        show_cols = [c for c in show_cols if c in df.columns]
        st.dataframe(df[show_cols], use_container_width=True)

        st.markdown("---")
        if st.button("📥 Export Excel", type="primary"):
            out = export_results_df(df[show_cols])
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "⬇️ Download Excel",
                out,
                f"results_{ts}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


if __name__ == "__main__":
    main()
