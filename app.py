# =============================================================================
# Hybrid OMR Bubble Sheet Scanner (Deterministic + Full Debug)
# Works for: 10 questions, 4 choices (A-D), ID grid: 4 digits x 10 rows (0-9)
# =============================================================================

import io
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image

# --------------------------- CONFIG (for your exact form) ---------------------------
# Reference size taken from your answer key rendering (dpi ~250)
REF_W = 2065
REF_H = 2943

# ROIs tuned from your provided answer key layout
# ID bubbles at top-right
ID_ROI = (1310, 260, 450, 808)   # x, y, w, h

# Questions block at left-bottom
Q_ROI  = (170, 1190, 550, 1050)  # x, y, w, h

NUM_Q = 10
NUM_CHOICES = 4
CHOICES = ["A", "B", "C", "D"]

ID_DIGITS = 4
ID_ROWS = 10

# Detection thresholds
MIN_FILL = 0.18          # minimum normalized fill to accept
DOUBLE_RATIO = 1.35      # if top/second < this => DOUBLE
# -------------------------------------------------------------------------------


@dataclass
class GradeResult:
    page_index: int
    student_code: str
    student_name: str
    score: int
    total: int
    percentage: float
    status: str


# ============================ IMAGE LOADING ============================

def load_first_page_image(file_bytes: bytes, filename: str, dpi: int = 250) -> Image.Image:
    """Load first page from PDF or image file into PIL RGB."""
    if filename.lower().endswith(".pdf"):
        pages = convert_from_bytes(file_bytes, dpi=dpi)
        if not pages:
            raise ValueError("PDF فارغ أو لم يتم تحويله")
        return pages[0].convert("RGB")
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def load_all_pages_images(file_bytes: bytes, filename: str, dpi: int = 250) -> List[Image.Image]:
    """Load all pages from PDF or single image as a list."""
    if filename.lower().endswith(".pdf"):
        pages = convert_from_bytes(file_bytes, dpi=dpi)
        return [p.convert("RGB") for p in pages]
    return [Image.open(io.BytesIO(file_bytes)).convert("RGB")]


# ============================ PREPROCESS + WARP ============================

def to_bgr(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points: TL, TR, BR, BL."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def find_document_quad(gray: np.ndarray) -> Optional[np.ndarray]:
    """Find the biggest 4-point contour (paper border)."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts[:8]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.2 * gray.size:
            return approx.reshape(4, 2).astype("float32")
    return None

def warp_to_reference(bgr: np.ndarray, debug: bool = False) -> Tuple[np.ndarray, Dict]:
    """Warp page to REF_W x REF_H using detected document quad."""
    dbg = {}
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    quad = find_document_quad(gray)
    if quad is None:
        # fallback: simple resize (still works sometimes)
        warped = cv2.resize(bgr, (REF_W, REF_H), interpolation=cv2.INTER_AREA)
        dbg["warp_mode"] = "FALLBACK_RESIZE"
        return warped, dbg

    rect = order_points(quad)
    dst = np.array([[0, 0], [REF_W - 1, 0], [REF_W - 1, REF_H - 1], [0, REF_H - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(bgr, M, (REF_W, REF_H))
    dbg["warp_mode"] = "PERSPECTIVE"
    dbg["quad"] = rect
    return warped, dbg

def preprocess_binary(warped_bgr: np.ndarray) -> np.ndarray:
    """Binary image for fill detection (bubbles become white in binary)."""
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    # stabilize contrast
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # Adaptive threshold then invert: filled marks -> 255
    bin_img = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 7
    )
    # small cleanup
    bin_img = cv2.medianBlur(bin_img, 3)
    return bin_img


# ============================ FILL DETECTION ============================

def inner_fill_ratio(cell_bin: np.ndarray) -> float:
    """Compute fill ratio inside inner region (avoid bubble ring)."""
    if cell_bin.size == 0:
        return 0.0
    h, w = cell_bin.shape[:2]
    mh = int(h * 0.25)
    mw = int(w * 0.25)
    inner = cell_bin[mh:h-mh, mw:w-mw]
    if inner.size == 0:
        return 0.0
    return float(np.sum(inner > 0)) / float(inner.size)

def pick_mark(fills: List[float]) -> Tuple[str, str, float, float]:
    """Return (answer, status, top_fill, second_fill)."""
    idx = np.argsort(fills)[::-1]
    top = int(idx[0])
    top_fill = fills[top]
    second_fill = fills[int(idx[1])] if len(fills) > 1 else 0.0

    if top_fill < MIN_FILL:
        return "?", "BLANK", top_fill, second_fill
    if second_fill >= MIN_FILL and (top_fill / (second_fill + 1e-9)) < DOUBLE_RATIO:
        return "!", "DOUBLE", top_fill, second_fill
    return str(top), "OK", top_fill, second_fill


# ============================ ROI GRID CUTTERS ============================

def cut_grid(bin_img: np.ndarray, roi: Tuple[int, int, int, int], rows: int, cols: int) -> List[List[np.ndarray]]:
    x, y, w, h = roi
    roi_bin = bin_img[y:y+h, x:x+w]
    cell_h = h // rows
    cell_w = w // cols
    grid = []
    for r in range(rows):
        row_cells = []
        for c in range(cols):
            cell = roi_bin[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            row_cells.append(cell)
        grid.append(row_cells)
    return grid

def read_student_code(bin_img: np.ndarray, debug: bool = False) -> Tuple[str, Dict]:
    """Read 4-digit code from ID bubble grid (10 rows x 4 cols)."""
    dbg = {}
    grid = cut_grid(bin_img, ID_ROI, ID_ROWS, ID_DIGITS)

    # For each digit column: we choose the row (0..9) with max fill
    digits = []
    fills_table = []
    for c in range(ID_DIGITS):
        col_fills = []
        col_cells = [grid[r][c] for r in range(ID_ROWS)]
        for r in range(ID_ROWS):
            col_fills.append(inner_fill_ratio(col_cells[r]))
        fills_table.append(col_fills)

        best_r = int(np.argmax(col_fills))
        best_fill = col_fills[best_r]

        # detect blank/double similarly (use top-2)
        sorted_idx = np.argsort(col_fills)[::-1]
        top_fill = col_fills[int(sorted_idx[0])]
        second_fill = col_fills[int(sorted_idx[1])] if len(sorted_idx) > 1 else 0.0

        if top_fill < MIN_FILL:
            digits.append("X")
        elif second_fill >= MIN_FILL and (top_fill / (second_fill + 1e-9)) < DOUBLE_RATIO:
            digits.append("X")
        else:
            digits.append(str(best_r))

    code = "".join(digits)
    dbg["id_fills"] = fills_table
    dbg["id_code"] = code
    return code, dbg

def read_answers(bin_img: np.ndarray, debug: bool = False) -> Tuple[Dict[int, Dict], Dict]:
    """Read answers for Q1..Q10 from question ROI (10 rows x 4 cols)."""
    dbg = {}
    grid = cut_grid(bin_img, Q_ROI, NUM_Q, NUM_CHOICES)

    answers = {}
    fills_all = []
    for q in range(NUM_Q):
        fills = [inner_fill_ratio(grid[q][c]) for c in range(NUM_CHOICES)]
        fills_all.append(fills)

        idx = np.argsort(fills)[::-1]
        top = int(idx[0])
        top_fill = fills[top]
        second_fill = fills[int(idx[1])] if len(idx) > 1 else 0.0

        if top_fill < MIN_FILL:
            answers[q+1] = {"answer": "?", "status": "BLANK", "fills": fills}
        elif second_fill >= MIN_FILL and (top_fill / (second_fill + 1e-9)) < DOUBLE_RATIO:
            answers[q+1] = {"answer": "!", "status": "DOUBLE", "fills": fills}
        else:
            answers[q+1] = {"answer": CHOICES[top], "status": "OK", "fills": fills}

    dbg["q_fills"] = fills_all
    return answers, dbg


# ============================ GRADING ============================

def build_roster_dict(df: pd.DataFrame) -> Dict[str, str]:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "student_code" not in df.columns or "student_name" not in df.columns:
        raise ValueError("ملف الطلاب لازم يحتوي أعمدة: student_code و student_name")

    codes = df["student_code"].astype(str).str.strip().str.replace(".0", "", regex=False)
    # ensure 4 digits for this form
    codes = codes.apply(lambda x: x.zfill(4) if x.isdigit() and len(x) < 4 else x)
    names = df["student_name"].astype(str).str.strip()
    return dict(zip(codes, names))

def extract_answer_key(key_img_bgr: np.ndarray, debug: bool = False) -> Tuple[Dict[int, str], Dict]:
    warped, wdbg = warp_to_reference(key_img_bgr, debug=debug)
    bin_img = preprocess_binary(warped)
    ans, adbg = read_answers(bin_img, debug=debug)

    key = {}
    for q, d in ans.items():
        if d["status"] == "OK":
            key[q] = d["answer"]

    dbg = {"warp": wdbg, "answers_dbg": adbg, "key": key, "warped": warped, "binary": bin_img}
    return key, dbg

def grade_page(page_bgr: np.ndarray, answer_key: Dict[int, str], roster: Dict[str, str],
               strict: bool = True, debug: bool = False) -> Tuple[GradeResult, Dict]:
    warped, wdbg = warp_to_reference(page_bgr, debug=debug)
    bin_img = preprocess_binary(warped)

    code, cdbg = read_student_code(bin_img, debug=debug)
    name = roster.get(code, "غير موجود")

    answers, adbg = read_answers(bin_img, debug=debug)

    correct = 0
    total = len(answer_key)
    for q, right in answer_key.items():
        if q not in answers:
            continue
        st_res = answers[q]
        if strict and st_res["status"] != "OK":
            continue
        if st_res["answer"] == right:
            correct += 1

    pct = (correct / total * 100.0) if total else 0.0
    status = "ناجح ✓" if pct >= 50 else "راسب ✗"

    dbg = {
        "warp": wdbg,
        "code_dbg": cdbg,
        "answers_dbg": adbg,
        "warped": warped,
        "binary": bin_img,
        "answers": answers
    }

    return GradeResult(
        page_index=0,
        student_code=code,
        student_name=name,
        score=correct,
        total=total,
        percentage=pct,
        status=status
    ), dbg


# ============================ STREAMLIT UI ============================

def show_debug_images(warped: np.ndarray, binary: np.ndarray):
    st.subheader("Debug: Warp & Binary")
    st.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), caption="Warped to Reference", use_container_width=True)
    st.image(binary, caption="Binary (filled marks = white)", use_container_width=True)

def draw_rois_on_image(warped: np.ndarray) -> np.ndarray:
    img = warped.copy()
    def rect_draw(roi, color, label):
        x,y,w,h = roi
        cv2.rectangle(img, (x,y), (x+w,y+h), color, 4)
        cv2.putText(img, label, (x+10, y+35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)
    rect_draw(ID_ROI, (0,0,255), "ID ROI")
    rect_draw(Q_ROI,  (0,255,0), "Q ROI")
    return img

def main():
    st.set_page_config(page_title="Hybrid OMR + Debug", layout="wide")

    st.title("✅ Hybrid OMR Bubble Sheet Scanner (مع Debug كامل)")
    st.caption("Warp تلقائي + استخراج كود الطالب + استخراج نموذج الإجابات + تصحيح كل صفحات PDF")

    with st.expander("معلومة مهمة عن نموذجكم"):
        st.write("النموذج يحتوي كود فقاعات (4 خانات × 10 صفوف) + 10 أسئلة (A-D).")
        st.write("تمت معايرة ROIs على نموذج الإجابة المرفوع.")

    colA, colB, colC = st.columns(3)
    with colA:
        roster_file = st.file_uploader("📋 ملف الطلاب (Excel/CSV)", type=["xlsx", "csv"])
    with colB:
        key_file = st.file_uploader("🔑 ملف الإجابة النموذجية (PDF/صورة)", type=["pdf", "png", "jpg", "jpeg"])
    with colC:
        sheets_file = st.file_uploader("📚 ملف أوراق الطلاب (PDF/صورة)", type=["pdf", "png", "jpg", "jpeg"])

    strict = st.checkbox("وضع صارم: BLANK/DOUBLE تُحسب خطأ", value=True)
    debug_mode = st.checkbox("إظهار Debug خطوة بخطوة", value=True)
    dpi = st.slider("DPI للتحويل من PDF", 150, 350, 250, 10)

    if not (roster_file and key_file and sheets_file):
        st.info("ارفع الملفات الثلاثة حتى نبدأ.")
        return

    # Load roster
    try:
        if roster_file.name.lower().endswith((".xlsx", ".xls")):
            df_roster = pd.read_excel(roster_file)
        else:
            df_roster = pd.read_csv(roster_file)
        roster = build_roster_dict(df_roster)
        st.success(f"تم تحميل قائمة الطلاب: {len(roster)} طالب")
    except Exception as e:
        st.error(f"خطأ في ملف الطلاب: {e}")
        return

    # Extract key
    try:
        key_img = load_first_page_image(key_file.getvalue(), key_file.name, dpi=dpi)
        key_bgr = to_bgr(key_img)
        answer_key, key_dbg = extract_answer_key(key_bgr, debug=debug_mode)

        st.success(f"تم استخراج Answer Key: {len(answer_key)} إجابة")
        st.write("Answer Key:", answer_key)

        if debug_mode:
            st.subheader("Debug: Answer Key Page")
            rois_img = draw_rois_on_image(key_dbg["warped"])
            st.image(cv2.cvtColor(rois_img, cv2.COLOR_BGR2RGB), caption="ROIs on Warped Key", use_container_width=True)
            show_debug_images(key_dbg["warped"], key_dbg["binary"])

    except Exception as e:
        st.error(f"فشل استخراج نموذج الإجابة: {e}")
        return

    # Grade sheets
    if st.button("🚀 ابدأ التصحيح الآن", type="primary", use_container_width=True):
        try:
            pages = load_all_pages_images(sheets_file.getvalue(), sheets_file.name, dpi=dpi)

            results: List[GradeResult] = []
            debug_samples = []

            prog = st.progress(0)
            for i, pil_page in enumerate(pages):
                page_bgr = to_bgr(pil_page)
                res, dbg = grade_page(page_bgr, answer_key, roster, strict=strict, debug=debug_mode)
                res.page_index = i + 1
                results.append(res)

                # keep few debug samples only
                if debug_mode and len(debug_samples) < 3:
                    debug_samples.append((i+1, dbg))

                prog.progress(int((i+1) / len(pages) * 100))

            df_out = pd.DataFrame([{
                "page_index": r.page_index,
                "student_code": r.student_code,
                "student_name": r.student_name,
                "score": r.score,
                "total": r.total,
                "percentage": round(r.percentage, 2),
                "status": r.status
            } for r in results])

            st.success("✅ اكتمل التصحيح")
            st.dataframe(df_out, use_container_width=True)

            # Download
            buf = io.BytesIO()
            df_out.to_excel(buf, index=False, engine="openpyxl")
            st.download_button(
                "⬇️ تحميل النتائج Excel",
                data=buf.getvalue(),
                file_name="results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

            # Debug samples
            if debug_mode and debug_samples:
                st.markdown("---")
                st.header("Debug Samples (أول 3 صفحات)")
                for page_no, dbg in debug_samples:
                    st.subheader(f"صفحة رقم {page_no}")
                    rois_img = draw_rois_on_image(dbg["warped"])
                    st.image(cv2.cvtColor(rois_img, cv2.COLOR_BGR2RGB), caption="ROIs on Warped Page", use_container_width=True)
                    show_debug_images(dbg["warped"], dbg["binary"])

                    st.write("📌 كود الطالب المستخرج:", dbg["code_dbg"]["id_code"])
                    st.write("📊 ID fills (لكل خانة 10 قيم):")
                    st.dataframe(pd.DataFrame(dbg["code_dbg"]["id_fills"]).T, use_container_width=True)

                    st.write("📊 Q fills (10 أسئلة × 4 خيارات):")
                    st.dataframe(pd.DataFrame(dbg["answers_dbg"]["q_fills"], columns=CHOICES), use_container_width=True)

        except Exception as e:
            st.error(f"❌ حدث خطأ أثناء التصحيح: {e}")
            import traceback
            with st.expander("تفاصيل الخطأ"):
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
