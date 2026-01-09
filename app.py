"""
🤖 OMR PRO (Scanner PDF) — Full App
✅ Student Code: bubbles-only (Auto-detect 4x10 grid) NO OCR / NO ROI
✅ Answers: AI optional (Anthropic) OR you can plug your bubble-answers later
✅ Students from Excel
✅ Duplicate management
✅ Export Excel
✅ Fix KeyError Percent

Requirements:
pip install streamlit opencv-python numpy pandas pdf2image pillow openpyxl
Linux: install poppler for pdf2image
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
# Fixed constants (Scanner PDF)
# =========================
DPI = 260
TOP_REGION_RATIO = 0.50

MIN_CIRCULARITY = 0.55
MIN_AREA = 60
R_MIN = 6
R_MAX = 45

MIN_INK = 0.030
MIN_MARGIN = 0.010

CODE_MIN = 1000
CODE_MAX = 1999


# =========================
# Bytes / pages
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


def load_pages(file_bytes, filename, dpi=DPI):
    if filename.lower().endswith(".pdf"):
        pages = convert_from_bytes(
            file_bytes, dpi=dpi, fmt="jpeg",
            jpegopt={"quality": 90, "optimize": True}
        )
        return [p.convert("RGB") for p in pages]
    return [Image.open(io.BytesIO(file_bytes)).convert("RGB")]


def pil_to_bgr(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def bgr_to_png_bytes(bgr):
    _, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    return buf.tobytes()


# =========================
# Student list
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
            students.append(
                StudentRecord(
                    str(row[id_col]).strip(),
                    str(row[name_col]).strip(),
                    str(row[code_col]).strip()
                )
            )
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
# Code reading (BUBBLES ONLY, Auto 4x10)
# =========================
def kmeans_1d(vals: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    vals = vals.astype(np.float32).reshape(-1, 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.01)
    _, labels, centers = cv2.kmeans(vals, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers = centers.flatten()
    order = np.argsort(centers)

    remap = np.zeros_like(order)
    for new, old in enumerate(order):
        remap[old] = new

    labels_sorted = np.array([remap[int(l)] for l in labels.flatten()], dtype=int)
    centers_sorted = centers[order]
    return labels_sorted, centers_sorted


def ink_score_in_circle(gray: np.ndarray, cx: int, cy: int, r: int) -> float:
    r2 = max(8, int(r * 0.85))
    y1, y2 = max(0, cy - r2), min(gray.shape[0], cy + r2)
    x1, x2 = max(0, cx - r2), min(gray.shape[1], cx + r2)
    patch = gray[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0

    patch = cv2.GaussianBlur(patch, (3, 3), 0)
    patch = cv2.normalize(patch, None, 0, 255, cv2.NORM_MINMAX)

    th = cv2.adaptiveThreshold(
        patch, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 3
    )

    h, w = th.shape
    mask = np.zeros_like(th)
    cv2.circle(mask, (w // 2, h // 2), int(min(h, w) * 0.38), 255, -1)

    ink = cv2.countNonZero(cv2.bitwise_and(th, mask))
    area = cv2.countNonZero(mask) + 1e-6
    return float(ink) / float(area)


@dataclass
class CodeResult:
    code: Optional[str]
    ok: bool
    reason: str


def read_code_auto(page_bgr: np.ndarray) -> CodeResult:
    H, W = page_bgr.shape[:2]
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray_blur, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circles = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue
        peri = cv2.arcLength(c, True) + 1e-6
        circ = 4 * np.pi * area / (peri * peri)
        if circ < MIN_CIRCULARITY:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if r < R_MIN or r > R_MAX:
            continue
        circles.append((float(x), float(y), float(r)))

    if len(circles) < 60:
        return CodeResult(None, False, "REVIEW: not enough bubbles detected")

    circles = np.array(circles, dtype=np.float32)
    rs = circles[:, 2]
    r_med = float(np.median(rs))
    keep = (rs > r_med * 0.70) & (rs < r_med * 1.40)
    circles = circles[keep]
    if len(circles) < 50:
        return CodeResult(None, False, "REVIEW: bubble size filtering failed")

    top = circles[circles[:, 1] < TOP_REGION_RATIO * H]
    if len(top) < 40:
        top = circles

    pts = top
    x = pts[:, 0]
    y = pts[:, 1]

    try:
        row_labels, row_centers = kmeans_1d(y, 4)
    except Exception:
        return CodeResult(None, False, "REVIEW: row clustering failed")

    ydist = np.abs(y - row_centers[row_labels])
    thr = np.percentile(ydist, 70) * 1.8 + 1e-6
    good = ydist < thr
    pts2 = pts[good]
    row_labels2 = row_labels[good]

    if len(pts2) < 35:
        return CodeResult(None, False, "REVIEW: row grid unstable")

    try:
        col_labels, col_centers = kmeans_1d(pts2[:, 0], 10)
    except Exception:
        return CodeResult(None, False, "REVIEW: column clustering failed")

    grid = [[None for _ in range(10)] for _ in range(4)]
    for (cx, cy, r), rr, cc in zip(pts2, row_labels2, col_labels):
        dx = abs(cx - col_centers[cc])
        dy = abs(cy - row_centers[rr])
        d = dx + dy
        if grid[rr][cc] is None or d < grid[rr][cc][0]:
            grid[rr][cc] = (d, int(cx), int(cy), int(r))

    missing = sum(1 for rr in range(4) for cc in range(10) if grid[rr][cc] is None)
    if missing > 8:
        return CodeResult(None, False, "REVIEW: code grid not found")

    digits = []
    for rr in range(4):
        scores = np.zeros((10,), dtype=np.float32)
        for cc in range(10):
            cell = grid[rr][cc]
            if cell is None:
                scores[cc] = 0.0
                continue
            _, cx, cy, r = cell
            scores[cc] = ink_score_in_circle(gray, cx, cy, r)

        best = int(np.argmax(scores))
        best_sc = float(scores[best])
        second_sc = float(np.partition(scores, -2)[-2])
        margin = best_sc - second_sc

        if best_sc < MIN_INK:
            return CodeResult(None, False, f"REVIEW: row {rr+1} too faint")
        if margin < MIN_MARGIN:
            return CodeResult(None, False, f"REVIEW: row {rr+1} ambiguous")

        digits.append(best)

    code = "".join(map(str, digits))
    if not code.isdigit():
        return CodeResult(None, False, "REVIEW: invalid code")

    code_int = int(code)
    if not (CODE_MIN <= code_int <= CODE_MAX):
        return CodeResult(None, False, f"REVIEW: code out of range ({code})")

    return CodeResult(code, True, "OK")


# =========================
# AI answers (optional)
# =========================
@dataclass
class AIResult:
    answers: Dict[int, str]
    success: bool
    notes: List[str]


def analyze_answers_with_ai(image_png_bytes: bytes, api_key: str, is_answer_key: bool):
    if not api_key or len(api_key) < 20:
        return AIResult({}, False, ["API key missing"])

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        img_b64 = base64.b64encode(image_png_bytes).decode("utf-8")

        if is_answer_key:
            prompt = "اقرأ Answer Key. JSON فقط: {\"answers\": {\"1\": \"A\", ...}}"
        else:
            prompt = "اقرأ إجابات الطالب فقط. JSON فقط: {\"answers\": {\"1\": \"A\", ...}}"

        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1200,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                    {"type": "text", "text": prompt}
                ]
            }]
        )

        text = msg.content[0].text
        import json

        json_text = text
        if "```json" in text:
            json_text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            json_text = text.split("```")[1].split("```")[0].strip()

        try:
            obj = json.loads(json_text)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", text)
            if not m:
                raise ValueError("No JSON found")
            obj = json.loads(m.group())

        answers = {int(k): str(v).strip().upper() for k, v in obj.get("answers", {}).items()}
        return AIResult(answers, True, [])
    except Exception as e:
        return AIResult({}, False, [str(e)])


def grade_student(student_answers: Dict[int, str], answer_key: Dict[int, str]) -> Tuple[int, int]:
    total = len(answer_key)
    score = 0
    for q, a in answer_key.items():
        if student_answers.get(q) == a:
            score += 1
    return score, total


def export_results(results_rows: List[dict], review_rows: List[dict], dup_rows: List[dict]) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        pd.DataFrame(results_rows).to_excel(w, index=False, sheet_name="Results")
        if review_rows:
            pd.DataFrame(review_rows).to_excel(w, index=False, sheet_name="Review")
        if dup_rows:
            pd.DataFrame(dup_rows).to_excel(w, index=False, sheet_name="Duplicates")
    return out.getvalue()


# =========================
# MAIN APP
# =========================
def main():
    st.set_page_config(page_title="🤖 OMR PRO", layout="wide")
    st.title("🤖 OMR PRO — كود من الفقاعات (بدون إعدادات) + تصحيح")

    # session
    if "answer_key" not in st.session_state: st.session_state.answer_key = {}
    if "students" not in st.session_state: st.session_state.students = []
    if "results" not in st.session_state: st.session_state.results = []
    if "review" not in st.session_state: st.session_state.review = []
    if "duplicate_warnings" not in st.session_state: st.session_state.duplicate_warnings = []

    # sidebar
    with st.sidebar:
        st.header("⚙️ (ثابت - لا إعدادات)")
        api_key = ""
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:
            pass
        if not api_key:
            api_key = st.text_input("🔑 API Key (للإجابات فقط)", type="password")

        st.markdown("---")
        st.metric("Answer Key", len(st.session_state.answer_key))
        st.metric("Students", len(st.session_state.students))
        st.metric("Graded", len(st.session_state.results))

        if st.button("🔄 Reset All"):
            st.session_state.answer_key = {}
            st.session_state.students = []
            st.session_state.results = []
            st.session_state.review = []
            st.session_state.duplicate_warnings = []
            st.rerun()

    tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ Answer Key", "2️⃣ Students", "3️⃣ Grade", "4️⃣ Results"])

    # TAB 1
    with tab1:
        st.subheader("📝 Answer Key (AI)")
        key_file = st.file_uploader("Upload Answer Key PDF/PNG", type=["pdf", "png", "jpg"], key="key")
        if key_file and st.button("🤖 Extract Answer Key"):
            pages = load_pages(read_bytes(key_file), key_file.name, dpi=DPI)
            bgr = pil_to_bgr(pages[0])
            res = analyze_answers_with_ai(bgr_to_png_bytes(bgr), api_key, True)
            if res.success and res.answers:
                st.session_state.answer_key = res.answers
                st.success(f"✅ Loaded {len(res.answers)} answers")
            else:
                st.error("❌ Failed to read Answer Key")
                st.write(res.notes)

        if st.session_state.answer_key:
            st.info(" | ".join([f"Q{q}:{a}" for q, a in sorted(st.session_state.answer_key.items())]))

    # TAB 2
    with tab2:
        st.subheader("👥 Students (Excel)")
        excel = st.file_uploader("Upload Excel (ID, Name, Code)", type=["xlsx", "xls"], key="excel")
        if excel and st.button("📊 Load Students"):
            students = load_students_from_excel(read_bytes(excel))
            if students:
                st.session_state.students = students
                st.success(f"✅ Loaded {len(students)} students")
            else:
                st.error("❌ Could not detect columns (ID/Name/Code)")

        if st.session_state.students:
            st.dataframe(pd.DataFrame([s.__dict__ for s in st.session_state.students[:50]]), use_container_width=True)

    # TAB 3
    with tab3:
        st.subheader("✅ Grading")

        if not st.session_state.answer_key:
            st.warning("Load Answer Key first.")
            st.stop()
        if not st.session_state.students:
            st.warning("Load Students first.")
            st.stop()

        sheets = st.file_uploader("Upload Student Sheets PDF (Scanner)", type=["pdf"], key="sheets")
        dup_mode = st.radio(
            "Duplicate handling",
            ["⚠️ Warn only", "🚫 Skip duplicates", "✅ Allow duplicates"],
            index=0
        )
        batch_size = 10

        if sheets and st.button("🚀 Process PDF"):
            st.session_state.results = []
            st.session_state.review = []
            st.session_state.duplicate_warnings = []

            pages = load_pages(read_bytes(sheets), sheets.name, dpi=DPI)

            prog = st.progress(0)
            status = st.empty()

            seen_codes = {}  # code -> first page

            for i, p in enumerate(pages):
                prog.progress((i + 1) / len(pages))
                status.text(f"Page {i+1}/{len(pages)}")

                bgr = pil_to_bgr(p)

                # 1) code from bubbles
                cr = read_code_auto(bgr)
                if not cr.ok or not cr.code:
                    st.session_state.review.append({"Page": i+1, "Reason": cr.reason, "Code": "REVIEW"})
                    del bgr
                    continue

                code = cr.code
                student = find_student_by_code(st.session_state.students, code)
                if not student:
                    st.session_state.review.append({"Page": i+1, "Reason": f"Code {code} not in Excel", "Code": "REVIEW"})
                    del bgr
                    continue

                # 2) duplicates
                if code in seen_codes:
                    if "Skip" in dup_mode:
                        st.session_state.duplicate_warnings.append({"Code": code, "Name": student.name, "Pages": f"{seen_codes[code]},{i+1}"})
                        del bgr
                        continue
                    if "Warn" in dup_mode:
                        st.session_state.duplicate_warnings.append({"Code": code, "Name": student.name, "Pages": f"{seen_codes[code]},{i+1}"})
                else:
                    seen_codes[code] = i + 1

                # 3) answers with AI
                ai_res = analyze_answers_with_ai(bgr_to_png_bytes(bgr), api_key, False)
                if not ai_res.success:
                    st.session_state.review.append({"Page": i+1, "Reason": "AI failed reading answers", "Code": code})
                    del bgr
                    continue

                score, total = grade_student(ai_res.answers, st.session_state.answer_key)

                st.session_state.results.append({
                    "Page": i+1,
                    "ID": student.student_id,
                    "Name": student.name,
                    "Code": code,
                    "Score": score,
                    "Total": total,
                    "Percent": (score / total * 100.0) if total else 0.0
                })

                del bgr
                if (i % batch_size) == 0:
                    gc.collect()

            gc.collect()
            st.success("✅ Processing complete")

    # TAB 4
    with tab4:
        st.subheader("📊 Results")

        if not st.session_state.results:
            st.info("No results yet.")
            st.stop()

        df = pd.DataFrame(st.session_state.results)

        # Fix Percent safely
        if "Percent" not in df.columns:
            df["Percent"] = 0.0

        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Graded", len(df))
        with col2: st.metric("Avg %", f"{df['Percent'].mean():.1f}")
        with col3: st.metric("Max %", f"{df['Percent'].max():.1f}")
        with col4: st.metric("Min %", f"{df['Percent'].min():.1f}")

        st.dataframe(df, use_container_width=True)

        # duplicates table
        dup_rows = []
        if st.session_state.duplicate_warnings:
            st.error(f"⚠️ Duplicates found: {len(st.session_state.duplicate_warnings)}")
            dup_df = pd.DataFrame(st.session_state.duplicate_warnings)
            st.dataframe(dup_df, use_container_width=True)
            dup_rows = st.session_state.duplicate_warnings

        # review table
        if st.session_state.review:
            st.warning(f"🟨 Review pages: {len(st.session_state.review)}")
            review_df = pd.DataFrame(st.session_state.review)
            st.dataframe(review_df, use_container_width=True)

        # export
        if st.button("📥 Export Excel", type="primary"):
            payload = export_results(
                results_rows=df.to_dict(orient="records"),
                review_rows=st.session_state.review,
                dup_rows=dup_rows
            )
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "⬇️ Download Excel",
                payload,
                f"results_{ts}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


if __name__ == "__main__":
    main()
