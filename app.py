"""
OMR PRO — Code from bubbles + AI answers (No settings)
- Reads student code from bubbles (4x10) with robust adaptive thresholds
- Reads Answer Key and student answers via AI (Anthropic)
- Shows REVIEW table even if no graded results
- Exports Results + Review + Duplicates to Excel
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
# Fixed defaults (no user tuning)
# =========================
DPI = 200  # fixed; good for scanner PDFs
TOP_REGION_RATIO = 0.40  # focus top area where code exists

# Bubble detection / filtering
MIN_AREA = 60
MIN_CIRCULARITY = 0.58
R_MIN = 8
R_MAX = 35

# Code range (adjust if your college uses different range)
CODE_MIN = 1000
CODE_MAX = 1999   # إذا عندك دائمًا 1000-1057 غيّرها إلى 1057


# =========================
# Data classes
# =========================
@dataclass
class AIResult:
    answers: Dict[int, str]
    confidence: str
    notes: List[str]
    success: bool
    student_code: Optional[str] = None


@dataclass
class StudentRecord:
    student_id: str
    name: str
    code: str


@dataclass
class CodeResult:
    code: Optional[str]
    ok: bool
    reason: str


# =========================
# Basic helpers
# =========================
def read_bytes(f) -> bytes:
    if not f:
        return b""
    try:
        return f.getbuffer().tobytes()
    except Exception:
        try:
            return f.read()
        except Exception:
            return b""


def load_pages(file_bytes: bytes, filename: str, dpi: int = DPI) -> List[Image.Image]:
    if filename.lower().endswith(".pdf"):
        pages = convert_from_bytes(file_bytes, dpi=dpi, fmt="jpeg",
                                  jpegopt={"quality": 85, "optimize": True})
        return [p.convert("RGB") for p in pages]
    return [Image.open(io.BytesIO(file_bytes)).convert("RGB")]


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def bgr_to_png_bytes(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    return buf.tobytes() if ok else b""


# =========================
# Students Excel
# =========================
def load_students_from_excel(file_bytes: bytes) -> List[StudentRecord]:
    try:
        df = pd.read_excel(io.BytesIO(file_bytes))
    except Exception as e:
        st.error(f"Excel error: {e}")
        return []

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
        st.error("Excel must contain ID/Name/Code columns (or Arabic equivalents).")
        return []

    students: List[StudentRecord] = []
    for _, row in df.iterrows():
        sid = str(row[id_col]).strip()
        nm = str(row[name_col]).strip()
        cd = str(row[code_col]).strip()
        if cd.lower() in ["nan", "none"]:
            cd = ""
        students.append(StudentRecord(sid, nm, cd))
    return students


def normalize_code(code: str) -> str:
    return str(code).strip().replace(" ", "").replace("-", "").replace("_", "")


def find_student_by_code(students: List[StudentRecord], code: str) -> Optional[StudentRecord]:
    code_norm = normalize_code(code)
    for s in students:
        if normalize_code(s.code) == code_norm:
            return s
    return None


# =========================
# Grading
# =========================
def grade_student(student_answers: Dict[int, str], answer_key: Dict[int, str]) -> Tuple[int, int]:
    total = len(answer_key)
    score = 0
    for q, ans in answer_key.items():
        if student_answers.get(q) == ans:
            score += 1
    return score, total


# =========================
# Export Excel (3 sheets)
# =========================
def export_results(results_rows: List[dict], review_rows: List[dict], dup_rows: List[dict]) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        pd.DataFrame(results_rows).to_excel(writer, sheet_name="Results", index=False)
        pd.DataFrame(review_rows).to_excel(writer, sheet_name="Review", index=False)
        pd.DataFrame(dup_rows).to_excel(writer, sheet_name="Duplicates", index=False)
    return out.getvalue()


# =========================
# AI (Anthropic)
# =========================
def analyze_with_ai(image_bytes: bytes, api_key: str, is_answer_key: bool) -> AIResult:
    if not api_key or len(api_key) < 20:
        return AIResult({}, "no_api", ["API key required"], False)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        if is_answer_key:
            prompt = (
                "اقرأ ورقة Answer Key بدقة.\n"
                "أرجع JSON فقط بالشكل:\n"
                "{\"answers\": {\"1\":\"A\", \"2\":\"B\", ...}}\n"
                "بدون أي نص إضافي."
            )
        else:
            prompt = (
                "أنت نظام OMR خبير.\n"
                "اقرأ إجابات الطالب (10 أسئلة).\n"
                "إذا أكثر من فقاعة مظللة اختر الأقتم.\n"
                "إذا لا يوجد خيار مظلل: \"?\".\n"
                "أرجع JSON فقط:\n"
                "{\"answers\": {\"1\":\"A\",...\"10\":\"D\"}}\n"
                "بدون أي نص إضافي."
            )

        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1200,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}},
                    {"type": "text", "text": prompt}
                ]
            }]
        )

        response_text = msg.content[0].text

        import json
        json_text = response_text.strip()
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0].strip()
        elif "```" in json_text:
            json_text = json_text.split("```")[1].split("```")[0].strip()
        else:
            m = re.search(r"\{[\s\S]*\}", json_text)
            if m:
                json_text = m.group(0)

        data = json.loads(json_text)
        answers_raw = data.get("answers", {}) if isinstance(data, dict) else {}
        answers = {}
        for k, v in answers_raw.items():
            try:
                answers[int(k)] = str(v).strip().upper()
            except Exception:
                pass

        return AIResult(answers, data.get("confidence", "medium"), data.get("notes", []), True)

    except Exception as e:
        return AIResult({}, "error", [str(e)], False)


# =========================
# Geometry / clustering
# =========================
def kmeans_1d(values: np.ndarray, k: int, iters: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    """Simple deterministic 1D kmeans."""
    v = values.astype(np.float32)
    vmin, vmax = float(v.min()), float(v.max())
    centers = np.linspace(vmin, vmax, k).astype(np.float32)

    labels = np.zeros_like(v, dtype=np.int32)
    for _ in range(iters):
        dists = np.abs(v[:, None] - centers[None, :])
        new_labels = np.argmin(dists, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                centers[i] = float(np.mean(v[mask]))
    order = np.argsort(centers)
    inv = np.zeros_like(order)
    inv[order] = np.arange(k)
    labels = inv[labels]
    centers = centers[order]
    return labels, centers


def ink_score_in_circle(gray: np.ndarray, cx: int, cy: int, r: int) -> float:
    """Higher = darker ink inside circle."""
    h, w = gray.shape[:2]
    r1 = max(6, int(r * 0.70))
    x1, x2 = max(0, cx - r1), min(w, cx + r1)
    y1, y2 = max(0, cy - r1), min(h, cy + r1)
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0

    # create circular mask
    mh, mw = crop.shape[:2]
    yy, xx = np.ogrid[:mh, :mw]
    mask = (xx - (cx - x1)) ** 2 + (yy - (cy - y1)) ** 2 <= r1 ** 2

    # ink score: inverted mean intensity inside mask (darker => higher)
    vals = crop[mask]
    if vals.size == 0:
        return 0.0
    return float(255.0 - np.mean(vals))


# =========================
# Code reading (BUBBLES ONLY) — robust + adaptive
# =========================
def read_code_auto(page_bgr: np.ndarray) -> CodeResult:
    H, W = page_bgr.shape[:2]
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    def detect_circles(gray_img: np.ndarray) -> List[Tuple[float, float, float]]:
        edges = cv2.Canny(gray_img, 40, 120)
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

        # Fallback: HoughCircles
        if len(circles) < 60:
            g = cv2.medianBlur(gray_img, 5)
            hc = cv2.HoughCircles(
                g, cv2.HOUGH_GRADIENT,
                dp=1.2, minDist=18,
                param1=120, param2=18,
                minRadius=R_MIN, maxRadius=R_MAX
            )
            if hc is not None:
                hc = np.squeeze(hc, axis=0)
                circles = [(float(x), float(y), float(r)) for x, y, r in hc]

        return circles

    circles = detect_circles(gray)
    if len(circles) < 50:
        return CodeResult(None, False, "REVIEW: not enough bubbles detected")

    circles = np.array(circles, dtype=np.float32)
    rs = circles[:, 2]
    r_med = float(np.median(rs))
    keep = (rs > r_med * 0.65) & (rs < r_med * 1.55)
    circles = circles[keep]
    if len(circles) < 45:
        return CodeResult(None, False, "REVIEW: bubble size filtering failed")

    # Focus on top region (where code is)
    top = circles[circles[:, 1] < TOP_REGION_RATIO * H]
    pts = top if len(top) >= 40 else circles

    x = pts[:, 0]
    y = pts[:, 1]

    try:
        row_labels, row_centers = kmeans_1d(y, 4)
    except Exception:
        return CodeResult(None, False, "REVIEW: row clustering failed")

    ydist = np.abs(y - row_centers[row_labels])
    thr = np.percentile(ydist, 70) * 2.0 + 1e-6
    good = ydist < thr

    pts2 = pts[good]
    row_labels2 = row_labels[good]
    if len(pts2) < 35:
        return CodeResult(None, False, "REVIEW: row grid unstable")

    try:
        col_labels, col_centers = kmeans_1d(pts2[:, 0], 10)
    except Exception:
        return CodeResult(None, False, "REVIEW: column clustering failed")

    # Build 4x10 grid by nearest point per cell
    grid = [[None for _ in range(10)] for _ in range(4)]
    for (cx, cy, r), rr, cc in zip(pts2, row_labels2, col_labels):
        dx = abs(cx - col_centers[cc])
        dy = abs(cy - row_centers[rr])
        d = dx + dy
        if grid[rr][cc] is None or d < grid[rr][cc][0]:
            grid[rr][cc] = (d, int(cx), int(cy), int(r))

    missing = sum(1 for rr in range(4) for cc in range(10) if grid[rr][cc] is None)
    if missing > 10:
        return CodeResult(None, False, "REVIEW: code grid not found")

    digits: List[int] = []
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

        med = float(np.median(scores))
        mad = float(np.median(np.abs(scores - med)) + 1e-6)

        # must stand out
        if best_sc < med + 3.0 * mad:
            return CodeResult(None, False, f"REVIEW: row {rr+1} faint/unclear")

        # must beat 2nd place by relative margin
        sorted_scores = np.sort(scores)
        second_sc = float(sorted_scores[-2])
        margin = best_sc - second_sc
        if margin < max(0.12 * best_sc, 2.0 * mad):
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
# Streamlit App
# =========================
def main():
    st.set_page_config(page_title="OMR PRO — Code from bubbles + AI", layout="wide")
    st.title("😃 OMR PRO — كود من الفقاعات + تصحيح بالـ AI (بدون إعدادات)")

    # Session state
    if "answer_key" not in st.session_state:
        st.session_state.answer_key = {}
    if "students" not in st.session_state:
        st.session_state.students = []
    if "results" not in st.session_state:
        st.session_state.results = []
    if "review" not in st.session_state:
        st.session_state.review = []
    if "duplicates" not in st.session_state:
        st.session_state.duplicates = []

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        api_key = ""
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:
            pass

        if not api_key:
            api_key = st.text_input("🔑 Anthropic API Key", type="password")

        st.markdown("---")
        st.metric("Answer Key", f"{len(st.session_state.answer_key)} Q")
        st.metric("Students", len(st.session_state.students))
        st.metric("Results", len(st.session_state.results))
        st.metric("Review", len(st.session_state.review))

        if st.button("🔄 Reset", type="secondary"):
            st.session_state.answer_key = {}
            st.session_state.students = []
            st.session_state.results = []
            st.session_state.review = []
            st.session_state.duplicates = []
            st.rerun()

    tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ Answer Key", "2️⃣ Students", "3️⃣ Grade", "4️⃣ Results"])

    # TAB 1: Answer Key
    with tab1:
        st.subheader("📝 Answer Key")
        key_file = st.file_uploader("Upload Answer Key (PDF/PNG/JPG)", type=["pdf", "png", "jpg"], key="key_file")
        if key_file and st.button("🤖 Analyze Answer Key", type="primary"):
            if not api_key or len(api_key) < 20:
                st.error("❌ API Key required.")
            else:
                pages = load_pages(read_bytes(key_file), key_file.name, dpi=DPI)
                if not pages:
                    st.error("No pages found.")
                else:
                    img_bgr = pil_to_bgr(pages[0])
                    res = analyze_with_ai(bgr_to_png_bytes(img_bgr), api_key, True)
                    if res.success and res.answers:
                        st.session_state.answer_key = res.answers
                        st.success(f"✅ Loaded {len(res.answers)} answers")
                    else:
                        st.error("Failed to read Answer Key with AI.")
        if st.session_state.answer_key:
            show = " | ".join([f"Q{q}: {a}" for q, a in sorted(st.session_state.answer_key.items())])
            st.info(show)

    # TAB 2: Students
    with tab2:
        st.subheader("👥 Students List")
        excel = st.file_uploader("Upload Excel (ID / Name / Code)", type=["xlsx", "xls"], key="students_excel")
        if excel and st.button("📊 Load Students"):
            students = load_students_from_excel(read_bytes(excel))
            if students:
                st.session_state.students = students
                st.success(f"✅ Loaded {len(students)} students")
        if st.session_state.students:
            with st.expander("Preview (first 50)"):
                dfp = pd.DataFrame([{"ID": s.student_id, "Name": s.name, "Code": s.code}
                                    for s in st.session_state.students[:50]])
                st.dataframe(dfp, use_container_width=True)

    # TAB 3: Grade
    with tab3:
        st.subheader("✅ Grading")
        if not st.session_state.answer_key:
            st.warning("⚠️ Load Answer Key first.")
            st.stop()
        if not st.session_state.students:
            st.warning("⚠️ Load Students first.")
            st.stop()
        if not api_key or len(api_key) < 20:
            st.error("❌ API Key required to read student answers with AI.")
            st.stop()

        sheets = st.file_uploader("Upload Student Sheets PDF (Scanner)", type=["pdf"], key="sheets_pdf")

        dup_mode = st.radio(
            "Duplicate handling",
            ["⚠️ Warn only", "🚫 Skip duplicates", "✅ Allow duplicates"],
            index=0
        )

        if sheets and st.button("🚀 Process PDF", type="primary"):
            st.session_state.results = []
            st.session_state.review = []
            st.session_state.duplicates = []

            pages = load_pages(read_bytes(sheets), sheets.name, dpi=DPI)
            if not pages:
                st.error("No pages found in PDF.")
                st.stop()

            prog = st.progress(0)
            status = st.empty()

            seen_codes = {}  # code -> first page number
            corrected = 0
            reviewed = 0

            for i, p in enumerate(pages):
                prog.progress((i + 1) / len(pages))
                status.text(f"Page {i+1}/{len(pages)}")

                bgr = pil_to_bgr(p)

                # 1) code from bubbles
                cr = read_code_auto(bgr)
                if not cr.ok or not cr.code:
                    reviewed += 1
                    st.session_state.review.append({"Page": i + 1, "Reason": cr.reason, "Code": "REVIEW"})
                    del bgr
                    if i % 10 == 0:
                        gc.collect()
                    continue

                code = cr.code
                student = find_student_by_code(st.session_state.students, code)
                if not student:
                    reviewed += 1
                    st.session_state.review.append({"Page": i + 1, "Reason": f"Code {code} not in Excel", "Code": code})
                    del bgr
                    if i % 10 == 0:
                        gc.collect()
                    continue

                # 2) duplicates (true duplicates only)
                if code in seen_codes:
                    st.session_state.duplicates.append({
                        "Code": code,
                        "Name": student.name,
                        "Pages": f"{seen_codes[code]},{i+1}"
                    })
                    if "Skip" in dup_mode:
                        del bgr
                        continue
                else:
                    seen_codes[code] = i + 1

                # 3) answers by AI
                ai_res = analyze_with_ai(bgr_to_png_bytes(bgr), api_key, False)
                if not ai_res.success or not ai_res.answers:
                    reviewed += 1
                    st.session_state.review.append({"Page": i + 1, "Reason": "AI failed reading answers", "Code": code})
                    del bgr
                    if i % 10 == 0:
                        gc.collect()
                    continue

                score, total = grade_student(ai_res.answers, st.session_state.answer_key)
                percent = (score / total * 100.0) if total else 0.0

                st.session_state.results.append({
                    "Page": i + 1,
                    "ID": student.student_id,
                    "Name": student.name,
                    "Code": code,
                    "Score": score,
                    "Total": total,
                    "Percent": percent
                })

                corrected += 1
                del bgr
                if i % 10 == 0:
                    gc.collect()

            gc.collect()
            st.success("✅ Processing complete")
            st.info(f"✅ Corrected: {corrected} | 🟨 Review: {reviewed} | 📄 Total: {len(pages)}")

            # Show review immediately if results are empty
            if corrected == 0 and st.session_state.review:
                st.warning("لم يتم تصحيح أي ورقة. هذه صفحات Review (مع السبب):")
                st.dataframe(pd.DataFrame(st.session_state.review), use_container_width=True)

    # TAB 4: Results
    with tab4:
        st.subheader("📊 Results")

        # Always show Review if exists
        if st.session_state.review:
            st.warning(f"🟨 Review pages: {len(st.session_state.review)}")
            st.dataframe(pd.DataFrame(st.session_state.review), use_container_width=True)

        if st.session_state.duplicates:
            st.error(f"⚠️ Duplicates found: {len(st.session_state.duplicates)}")
            st.dataframe(pd.DataFrame(st.session_state.duplicates), use_container_width=True)

        if not st.session_state.results:
            st.info("No graded results yet.")
            st.stop()

        df = pd.DataFrame(st.session_state.results)
        if "Percent" not in df.columns:
            df["Percent"] = 0.0

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Graded", len(df))
        with c2:
            st.metric("Avg %", f"{df['Percent'].mean():.1f}")
        with c3:
            st.metric("Max %", f"{df['Percent'].max():.1f}")
        with c4:
            st.metric("Min %", f"{df['Percent'].min():.1f}")

        st.dataframe(df, use_container_width=True)

        st.markdown("---")
        if st.button("📥 Export Excel", type="primary"):
            payload = export_results(
                results_rows=df.to_dict(orient="records"),
                review_rows=st.session_state.review,
                dup_rows=st.session_state.duplicates
            )
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "⬇️ Download Excel (Results + Review + Duplicates)",
                payload,
                f"results_{ts}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


if __name__ == "__main__":
    main()
