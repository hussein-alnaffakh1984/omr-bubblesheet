"""
🤖 OMR (Code) + AI (Answers) — Production-Grade for Large Classes
✅ Code from bubbles ONLY (no OCR, no guessing)
✅ Perspective alignment (robust)
✅ Strict confidence rules: outputs REVIEW instead of wrong codes
✅ Scalable batch processing
"""

import io, base64, time, gc, re
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
# Basic IO Helpers
# =========================
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


def load_pages(file_bytes, filename, dpi=200):
    """Load PDF pages as PIL images. Keep dpi moderate for memory."""
    if filename.lower().endswith(".pdf"):
        pages = convert_from_bytes(file_bytes, dpi=dpi, fmt="jpeg",
                                  jpegopt={"quality": 88, "optimize": True})
        return [p.convert("RGB") for p in pages]
    return [Image.open(io.BytesIO(file_bytes)).convert("RGB")]


def pil_to_bgr(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def bgr_to_png_bytes(bgr):
    _, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    return buf.tobytes()


# =========================
# Student List
# =========================
@dataclass
class StudentRecord:
    student_id: str
    name: str
    code: str


def load_students_from_excel(file_bytes) -> List[StudentRecord]:
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

        students = []
        for _, row in df.iterrows():
            sid = str(row[id_col]).strip()
            nm = str(row[name_col]).strip()
            cd = str(row[code_col]).strip().replace(" ", "").replace("-", "")
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
# Perspective Alignment (Critical!)
# =========================
def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def align_page(bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Align scanned page using largest contour. Returns aligned page and success flag.
    If alignment fails, returns original.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge-based detection
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return bgr, False

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    page_cnt = None

    for c in cnts[:5]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.2 * (h * w):
            page_cnt = approx
            break

    if page_cnt is None:
        return bgr, False

    pts = order_points(page_cnt.reshape(4, 2))

    # Output size — keep consistent
    out_w = 1700
    out_h = 2400
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(bgr, M, (out_w, out_h))
    return warped, True


# =========================
# Code Grid OMR (No Guessing)
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
    Compute darkness/ink score robustly:
    - Normalize
    - Otsu threshold (ink vs paper)
    - Return %ink
    """
    # Normalize contrast
    g = cv2.GaussianBlur(cell_gray, (3, 3), 0)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)

    # Invert? We'll threshold ink as darker than paper
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Focus on inner circle area (avoid borders)
    h, w = th.shape
    mask = np.zeros_like(th)
    cy, cx = h // 2, w // 2
    r = int(min(h, w) * 0.33)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    ink = cv2.countNonZero(cv2.bitwise_and(th, mask))
    area = cv2.countNonZero(mask) + 1e-6
    return float(ink) / float(area)


def read_code_from_bubbles(aligned_bgr: np.ndarray) -> CodeRead:
    """
    Reads the 4-digit code from the bubble grid ONLY.
    Strict rules:
    - choose max ink per row
    - require clear margin between top1 and top2 (avoid multi-filled)
    - if uncertain -> REVIEW (no wrong codes)
    """
    H, W = aligned_bgr.shape[:2]

    # ROI for code grid (ADJUST ONCE for your form)
    # Based on your sheets: top-right code grid
    y1, y2 = int(0.12 * H), int(0.32 * H)
    x1, x2 = int(0.60 * W), int(0.88 * W)
    roi = aligned_bgr[y1:y2, x1:x2].copy()

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Contrast boost (helps in faint scans)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Grid dimensions (4 rows × 10 columns)
    rows = 4
    cols = 10

    rh = gray.shape[0] / rows
    cw = gray.shape[1] / cols

    chosen = []
    scores = []
    margins = []

    # Strict thresholds (tune if needed)
    MIN_INK = 0.06        # below this, likely empty row
    MIN_MARGIN = 0.018    # top1 - top2 must exceed this to be unambiguous

    for r in range(rows):
        row_scores = []
        for c in range(cols):
            yA = int(r * rh)
            yB = int((r + 1) * rh)
            xA = int(c * cw)
            xB = int((c + 1) * cw)

            cell = gray[yA:yB, xA:xB]
            sc = _ink_score(cell)
            row_scores.append(sc)

        row_scores = np.array(row_scores, dtype=np.float32)
        idx_sorted = np.argsort(-row_scores)  # descending
        best = int(idx_sorted[0])
        second = int(idx_sorted[1])

        best_sc = float(row_scores[best])
        second_sc = float(row_scores[second])
        margin = best_sc - second_sc

        chosen.append(best)
        scores.append(best_sc)
        margins.append(margin)

        # Hard rejection rules
        if best_sc < MIN_INK:
            return CodeRead(None, False, f"REVIEW: row {r+1} looks empty/too faint (best={best_sc:.3f})",
                            chosen, scores, margins)

        if margin < MIN_MARGIN:
            return CodeRead(None, False, f"REVIEW: row {r+1} ambiguous (margin={margin:.3f})",
                            chosen, scores, margins)

    code = "".join(str(d) for d in chosen)

    # OPTIONAL: range rule. If your codes always 1000-1999 or 1000-1057 set it strict.
    # Here we only enforce 4 digits.
    if not (code.isdigit() and len(code) == 4):
        return CodeRead(None, False, "REVIEW: invalid code format", chosen, scores, margins)

    return CodeRead(code, True, "OK", chosen, scores, margins)


# =========================
# AI Answers (Keep your AI approach)
# =========================
@dataclass
class AIResult:
    answers: Dict[int, str]
    success: bool
    notes: List[str]
    confidence: str = "medium"


def analyze_answers_with_ai(full_page_png_bytes: bytes, api_key: str) -> AIResult:
    """
    Uses AI to read answers. You can later replace this with full OMR answers if needed.
    """
    if not api_key or len(api_key) < 20:
        return AIResult({}, False, ["API Key required"], "no_api")

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        image_b64 = base64.b64encode(full_page_png_bytes).decode("utf-8")

        prompt = r"""
أنت نظام OMR خبير.
اقرأ إجابات الطالب (10 أسئلة) من الفقاعات.
قواعد:
1) إذا وُجد X فوق فقاعة => تُلغى (لا تُحسب حتى لو مظللة).
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
                    {"type": "text", "text": prompt}
                ]
            }]
        )

        txt = msg.content[0].text
        import json
        m = re.search(r"\{[\s\S]*\}", txt)
        data = json.loads(m.group()) if m else {}

        ans = data.get("answers", {})
        answers = {}
        for k, v in ans.items():
            try:
                q = int(k)
                answers[q] = str(v).strip().upper()
            except:
                pass

        if not answers:
            return AIResult({}, False, ["AI returned no answers"], "low")

        return AIResult(answers, True, [], data.get("confidence", "medium"))

    except Exception as e:
        return AIResult({}, False, [str(e)], "error")


def grade_student(student_answers: Dict[int, str], answer_key: Dict[int, str]) -> Tuple[int, int]:
    total = len(answer_key)
    score = 0
    for q, a in answer_key.items():
        if student_answers.get(q) == a:
            score += 1
    return score, total


def export_results(df: pd.DataFrame) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Results", index=False)
    return out.getvalue()


# =========================
# MAIN APP
# =========================
def main():
    st.set_page_config(page_title="OMR Pro (Code Bubbles) + AI Answers", layout="wide")
    st.title("✅ OMR احترافي — كود الطالب من الفقاعات فقط + إجابات بالـ AI")

    # Session State
    if "answer_key" not in st.session_state:
        st.session_state.answer_key = {}
    if "students" not in st.session_state:
        st.session_state.students = []
    if "results" not in st.session_state:
        st.session_state.results = []
    if "processed_pages" not in st.session_state:
        st.session_state.processed_pages = set()
    if "review_pages" not in st.session_state:
        st.session_state.review_pages = []  # pages that need manual check
    if "current_pages" not in st.session_state:
        st.session_state.current_pages = None
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        api_key = ""
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
            if api_key:
                st.success("✅ API Key loaded from secrets")
        except:
            pass
        if not api_key:
            api_key = st.text_input("🔑 Anthropic API Key (for answers)", type="password")

        st.markdown("---")
        st.metric("Answer Key", f"{len(st.session_state.answer_key)} Q")
        st.metric("Students", len(st.session_state.students))
        st.metric("Graded", len(st.session_state.results))
        st.metric("Review", len(st.session_state.review_pages))

        if st.button("🔄 Reset All", type="secondary"):
            st.session_state.answer_key = {}
            st.session_state.students = []
            st.session_state.results = []
            st.session_state.processed_pages = set()
            st.session_state.review_pages = []
            st.session_state.current_pages = None
            st.session_state.current_idx = 0
            st.rerun()

    tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ Answer Key", "2️⃣ Students", "3️⃣ Grade", "4️⃣ Results"])

    # TAB 1 — Answer Key (manual simple or AI)
    with tab1:
        st.subheader("📝 Answer Key")
        mode = st.radio("كيف تريد إدخال Answer Key؟", ["Manual", "AI from Answer Key sheet"], horizontal=True)

        if mode == "Manual":
            st.info("أدخل الإجابات (10 أسئلة) يدويًا مرة واحدة فقط.")
            cols = st.columns(10)
            tmp = {}
            for i in range(10):
                with cols[i]:
                    tmp[i+1] = st.selectbox(f"Q{i+1}", ["A", "B", "C", "D"], index=0, key=f"k{i+1}")
            if st.button("✅ Save Answer Key", type="primary"):
                st.session_state.answer_key = {q: a for q, a in tmp.items()}
                st.success("✅ Saved")
        else:
            key_file = st.file_uploader("Upload Answer Key sheet (PDF/IMG)", type=["pdf", "png", "jpg"])
            if key_file and st.button("🤖 Read Answer Key with AI", type="primary"):
                if not api_key:
                    st.error("❌ API Key required for AI")
                else:
                    b = read_bytes(key_file)
                    pages = load_pages(b, key_file.name, dpi=250)
                    bgr = pil_to_bgr(pages[0])
                    aligned, _ = align_page(bgr)
                    res = analyze_answers_with_ai(bgr_to_png_bytes(aligned), api_key)
                    if res.success:
                        st.session_state.answer_key = res.answers
                        st.success(f"✅ Loaded {len(res.answers)} answers")
                    else:
                        st.error("Failed: " + " | ".join(res.notes))

        if st.session_state.answer_key:
            st.markdown("**Current Key:** " + " | ".join([f"Q{q}:{a}" for q, a in sorted(st.session_state.answer_key.items())]))

    # TAB 2 — Students
    with tab2:
        st.subheader("👥 Students")
        excel = st.file_uploader("Upload Excel (ID, Name, Code)", type=["xlsx", "xls"])
        if excel and st.button("📊 Load Students", type="primary"):
            students = load_students_from_excel(read_bytes(excel))
            if students:
                st.session_state.students = students
                st.success(f"✅ Loaded {len(students)} students")
            else:
                st.error("❌ Could not detect columns (ID/Name/Code)")

        if st.session_state.students:
            st.info(f"Loaded: {len(st.session_state.students)} students")
            with st.expander("Preview (first 50)"):
                dfp = pd.DataFrame([{"ID": s.student_id, "Name": s.name, "Code": s.code} for s in st.session_state.students[:50]])
                st.dataframe(dfp, use_container_width=True)

    # TAB 3 — Grading
    with tab3:
        st.subheader("✅ Grade (Code = Bubble OMR only)")
        if not st.session_state.answer_key:
            st.warning("⚠️ Load Answer Key first")
            return
        if not st.session_state.students:
            st.warning("⚠️ Load Students first")
            return

        sheets = st.file_uploader("ارفع ملف PDF (يفضل ≤ 50 صفحة لكل ملف)", type=["pdf"])
        dpi = st.slider("DPI", 150, 300, 220, help="220 مناسب. ارفع لـ 260 إذا الورق ضعيف.")
        batch_size = st.slider("Batch size", 5, 30, 10)

        colA, colB = st.columns(2)
        with colA:
            strict_mode = st.checkbox("Strict Mode (لا يصحح إذا الكود REVIEW)", value=True)
        with colB:
            show_debug = st.checkbox("Debug (عرض سبب REVIEW)", value=False)

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
        idx = st.session_state.current_idx
        total = len(pages)

        st.metric("Progress", f"{idx}/{total} ({(idx/total*100 if total else 0):.0f}%)")

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

                bgr = pil_to_bgr(pages[i])
                aligned, ok_align = align_page(bgr)

                # 1) Read code from bubbles ONLY
                code_read = read_code_from_bubbles(aligned)

                if not code_read.ok:
                    st.session_state.review_pages.append({
                        "page": i+1,
                        "reason": code_read.reason,
                    })
                    st.session_state.processed_pages.add(i)
                    if show_debug:
                        st.warning(f"⚠️ Page {i+1}: {code_read.reason}")
                    if strict_mode:
                        continue
                    # if not strict, we still attempt AI answers but mark code as REVIEW
                    code = None
                else:
                    code = code_read.code

                # 2) Validate against student list (NO GUESSING)
                student = None
                if code is not None:
                    student = find_student_by_code(st.session_state.students, code)

                if code is None or student is None:
                    st.session_state.review_pages.append({
                        "page": i+1,
                        "reason": "REVIEW: code not found in student list (or missing)",
                        "code": code if code else "REVIEW"
                    })
                    st.session_state.processed_pages.add(i)
                    if show_debug:
                        st.warning(f"⚠️ Page {i+1}: code={code} not in list")
                    if strict_mode:
                        continue

                # 3) AI read answers
                if not api_key:
                    st.warning(f"⚠️ Page {i+1}: API key missing → cannot read answers")
                    st.session_state.processed_pages.add(i)
                    continue

                ai = analyze_answers_with_ai(bgr_to_png_bytes(aligned), api_key)
                if not ai.success:
                    st.session_state.review_pages.append({
                        "page": i+1,
                        "reason": "REVIEW: AI failed to read answers",
                        "code": code if code else "REVIEW"
                    })
                    st.session_state.processed_pages.add(i)
                    continue

                score, total_q = grade_student(ai.answers, st.session_state.answer_key)

                st.session_state.results.append({
                    "Page": i+1,
                    "ID": student.student_id if student else "",
                    "Name": student.name if student else "",
                    "Code": code if code else "REVIEW",
                    "Score": score,
                    "Total": total_q,
                    "Percent": round(score / total_q * 100, 1) if total_q else 0.0
                })

                st.session_state.processed_pages.add(i)
                processed += 1

                # Clean memory
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

    # TAB 4 — Results
    with tab4:
        st.subheader("📊 Results")

        if st.session_state.review_pages:
            st.error(f"⚠️ REVIEW pages: {len(st.session_state.review_pages)}")
            with st.expander("عرض صفحات المراجعة"):
                st.dataframe(pd.DataFrame(st.session_state.review_pages), use_container_width=True)

        if not st.session_state.results:
            st.info("No graded results yet.")
            return

        df = pd.DataFrame(st.session_state.results)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Graded", len(df))
        with col2:
            st.metric("Avg %", f"{df['Percent'].mean():.1f}")
        with col3:
            st.metric("Max %", f"{df['Percent'].max():.1f}")
        with col4:
            st.metric("Min %", f"{df['Percent'].min():.1f}")

        st.dataframe(df, use_container_width=True)

        st.markdown("---")
        if st.button("📥 Export Excel", type="primary"):
            x = export_results(df)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "⬇️ Download Excel",
                x,
                f"results_{ts}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


if __name__ == "__main__":
    main()
