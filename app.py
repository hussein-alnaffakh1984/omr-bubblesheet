import io
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image


# -----------------------------
# Data models
# -----------------------------
@dataclass
class QBlock:
    x: int
    y: int
    w: int
    h: int
    start_q: int
    end_q: int
    rows: int


@dataclass
class TemplateConfig:
    # ID area (x, y, w, h) in TEMPLATE coordinates
    id_roi: Tuple[int, int, int, int] = (0, 0, 0, 0)
    id_digits: int = 4
    id_rows: int = 10  # 0..9 rows

    # question blocks
    q_blocks: List[QBlock] = None

    # detection params
    min_ratio: float = 1.25
    # cell fill thresholds (normalized 0..1)
    id_min_fill: float = 0.12
    q_min_fill: float = 0.10


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def load_pages(file_bytes: bytes, name: str) -> List[Image.Image]:
    name = name.lower()
    if name.endswith(".pdf"):
        return convert_from_bytes(file_bytes)
    return [Image.open(io.BytesIO(file_bytes))]


# -----------------------------
# Preprocess / binarize
# -----------------------------
def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 10
    )
    return thr


# -----------------------------
# Alignment (ORB Homography)
# -----------------------------
def orb_homography_align(page_bgr: np.ndarray, template_bgr: np.ndarray) -> Optional[np.ndarray]:
    page_gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    tpl_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=3000, fastThreshold=10)
    kp1, des1 = orb.detectAndCompute(page_gray, None)
    kp2, des2 = orb.detectAndCompute(tpl_gray, None)

    if des1 is None or des2 is None or len(kp1) < 30 or len(kp2) < 30:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 25:
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, _mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H


def warp_to_template(page_bgr: np.ndarray, template_bgr: np.ndarray, tw: int, th: int) -> np.ndarray:
    H = orb_homography_align(page_bgr, template_bgr)
    if H is not None:
        return cv2.warpPerspective(page_bgr, H, (tw, th))
    # fallback: resize فقط
    return cv2.resize(page_bgr, (tw, th), interpolation=cv2.INTER_AREA)


# -----------------------------
# Bubble scoring (ignore circle borders)
# -----------------------------
def score_cell_norm(bin_cell: np.ndarray) -> float:
    h, w = bin_cell.shape[:2]
    mx = int(w * 0.28)
    my = int(h * 0.28)
    if mx * 2 >= w or my * 2 >= h:
        c = bin_cell
    else:
        c = bin_cell[my:h - my, mx:w - mx]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    c = cv2.morphologyEx(c, cv2.MORPH_CLOSE, kernel, iterations=2)

    return float(np.sum(c > 0)) / (c.shape[0] * c.shape[1] + 1e-9)


def pick_one(scores: List[Tuple[str, float]], min_fill: float, min_ratio: float) -> Tuple[str, str]:
    # scores = [(choice, value_norm), ...]
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_c, top_s = scores[0]
    second_s = scores[1][1] if len(scores) > 1 else 0.0

    if top_s < min_fill:
        return "?", "BLANK"
    if second_s > 0 and (top_s / (second_s + 1e-9)) < min_ratio:
        return "!", "DOUBLE"
    return top_c, "OK"


# -----------------------------
# Read ID
# -----------------------------
def read_student_code(thr: np.ndarray, cfg: TemplateConfig) -> Tuple[str, Dict]:
    x, y, w, h = cfg.id_roi
    roi = thr[y:y + h, x:x + w]
    if roi.size == 0:
        return "UNKNOWN", {"error": "ID ROI empty"}

    rows = cfg.id_rows
    cols = cfg.id_digits
    ch = h // rows
    cw = w // cols
    digits = []
    states = []

    for c in range(cols):
        scores = []
        for r in range(rows):
            cell = roi[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
            scores.append((str(r), score_cell_norm(cell)))
        d, stt = pick_one(scores, cfg.id_min_fill, cfg.min_ratio)
        states.append(stt)
        digits.append("" if d in ("?", "!") else d)

    code = "".join(digits).strip()

    if len(code) != cfg.id_digits:
        return "UNKNOWN", {"raw_digits": digits, "states": states, "error": "incomplete ID"}
    return code.zfill(cfg.id_digits), {"raw_digits": digits, "states": states}


# -----------------------------
# Read Answers
# -----------------------------
def read_answers(thr: np.ndarray, cfg: TemplateConfig, choices: int) -> Dict[int, Tuple[str, str]]:
    letters = "ABCDE"[:choices]
    out: Dict[int, Tuple[str, str]] = {}

    for blk in cfg.q_blocks:
        x, y, w, h = blk.x, blk.y, blk.w, blk.h
        roi = thr[y:y + h, x:x + w]
        if roi.size == 0:
            continue

        rows = blk.rows
        rh = h // rows
        cw = w // choices

        q = blk.start_q
        for r in range(rows):
            if q > blk.end_q:
                break
            scores = []
            for c in range(choices):
                cell = roi[r * rh:(r + 1) * rh, c * cw:(c + 1) * cw]
                scores.append((letters[c], score_cell_norm(cell)))
            a, stt = pick_one(scores, cfg.q_min_fill, cfg.min_ratio)
            out[q] = (a, stt)
            q += 1

    return out


# -----------------------------
# Helpers
# -----------------------------
def parse_ranges(txt: str) -> List[Tuple[int, int]]:
    if not txt.strip():
        return []
    out = []
    for part in txt.split(","):
        p = part.strip()
        m = re.match(r"^(\d+)\s*-\s*(\d+)$", p)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            out.append((min(a, b), max(a, b)))
        elif p.isdigit():
            x = int(p)
            out.append((x, x))
    return out


def in_ranges(q: int, ranges: List[Tuple[int, int]]) -> bool:
    return any(a <= q <= b for a, b in ranges)


def overlay_rects(img_bgr: np.ndarray, cfg: TemplateConfig, choices: int) -> np.ndarray:
    vis = img_bgr.copy()

    # ID
    x, y, w, h = cfg.id_roi
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 3)
    cv2.putText(vis, "ID ROI", (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Q blocks
    for i, blk in enumerate(cfg.q_blocks, 1):
        cv2.rectangle(vis, (blk.x, blk.y), (blk.x + blk.w, blk.y + blk.h), (0, 0, 255), 3)
        cv2.putText(
            vis, f"Q{i}: {blk.start_q}-{blk.end_q}",
            (blk.x, max(20, blk.y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
        )

    return vis


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="OMR Bubble Sheet (Stable)", layout="wide")
st.title("✅ تصحيح ببل شيت (مستقر على Streamlit Cloud) — بدون Canvas إضافي")


# ---------- Uploads ----------
c1, c2 = st.columns([1, 1])

with c1:
    st.subheader("1) Template (نموذج الورقة)")
    tpl_file = st.file_uploader("PDF/PNG/JPG (صفحة النموذج)", type=["pdf", "png", "jpg", "jpeg"], key="tpl")

with c2:
    st.subheader("2) Answer Key (مفتاح الإجابة)")
    key_file = st.file_uploader("PDF/PNG/JPG (يفضل نفس النموذج مع إجابات مظللة)", type=["pdf", "png", "jpg", "jpeg"], key="key")


st.subheader("3) Roster (قائمة الطلاب)")
roster_file = st.file_uploader("Excel/CSV يحتوي: student_code, student_name", type=["xlsx", "xls", "csv"], key="roster")

st.subheader("4) Student Sheets (أوراق الطلاب)")
sheets_file = st.file_uploader("PDF متعدد الصفحات أو صور", type=["pdf", "png", "jpg", "jpeg"], key="sheets")


# ---------- Exam settings ----------
st.subheader("⚙️ إعدادات الامتحان")
s1, s2, s3, s4 = st.columns([1, 1, 1, 1])
with s1:
    choices = st.radio("عدد الخيارات", [4, 5], horizontal=True)
with s2:
    strict = st.checkbox("وضع صارم (BLANK/DOUBLE = خطأ)", True)
with s3:
    theory_txt = st.text_input("نطاق النظري (مثال: 1-40)", value="1-60")
with s4:
    practical_txt = st.text_input("نطاق العملي (اختياري)", value="")

theory_ranges = parse_ranges(theory_txt)
practical_ranges = parse_ranges(practical_txt)


# ---------- Config editing ----------
st.subheader("🧩 إعداد مناطق القراءة (إحداثيات على TEMPLATE)")

if "cfg" not in st.session_state:
    st.session_state.cfg = TemplateConfig(
        id_roi=(0, 0, 0, 0),
        id_digits=4,
        id_rows=10,
        q_blocks=[],
        min_ratio=1.25,
        id_min_fill=0.12,
        q_min_fill=0.10,
    )

cfg: TemplateConfig = st.session_state.cfg

left, right = st.columns([1.2, 1])

with right:
    st.markdown("### إعداد الكود")
    cfg.id_digits = st.number_input("عدد خانات الكود", min_value=1, max_value=12, value=int(cfg.id_digits), step=1)
    cfg.id_rows = st.number_input("عدد صفوف أرقام الكود (عادة 10)", min_value=5, max_value=15, value=int(cfg.id_rows), step=1)

    st.markdown("### حساسية القراءة")
    cfg.min_ratio = st.slider("min_ratio (تمييز المظلّل عن الثاني)", 1.05, 2.00, float(cfg.min_ratio), 0.01)
    cfg.id_min_fill = st.slider("ID min_fill", 0.02, 0.40, float(cfg.id_min_fill), 0.01)
    cfg.q_min_fill = st.slider("Q min_fill", 0.02, 0.40, float(cfg.q_min_fill), 0.01)

    st.markdown("---")
    st.markdown("### ID ROI (x,y,w,h)")
    ix, iy, iw, ih = cfg.id_roi
    ix = st.number_input("ID x", min_value=0, value=int(ix), step=1)
    iy = st.number_input("ID y", min_value=0, value=int(iy), step=1)
    iw = st.number_input("ID w", min_value=0, value=int(iw), step=1)
    ih = st.number_input("ID h", min_value=0, value=int(ih), step=1)
    cfg.id_roi = (int(ix), int(iy), int(iw), int(ih))

    st.markdown("---")
    st.markdown("### Q Blocks")
    # editable table for blocks
    if len(cfg.q_blocks) == 0:
        df_blocks = pd.DataFrame(columns=["x", "y", "w", "h", "start_q", "end_q", "rows"])
    else:
        df_blocks = pd.DataFrame([vars(b) for b in cfg.q_blocks])

    edited = st.data_editor(
        df_blocks,
        num_rows="dynamic",
        use_container_width=True,
        key="blocks_editor"
    )

    # apply edited blocks
    new_blocks = []
    for _, r in edited.iterrows():
        try:
            new_blocks.append(QBlock(
                x=int(r["x"]), y=int(r["y"]), w=int(r["w"]), h=int(r["h"]),
                start_q=int(r["start_q"]), end_q=int(r["end_q"]), rows=int(r["rows"])
            ))
        except Exception:
            continue
    cfg.q_blocks = new_blocks

    st.markdown("✅ بعد تعديل البلوكات، ارجع لليسار وشوف المعاينة بالمربعات الحمراء.")


with left:
    st.markdown("### معاينة Template مع المناطق (مستطيلات حمراء)")
    debug_view = st.checkbox("عرض المعاينة (Overlay) + قص مناطق القراءة", True)
    canvas_width = st.slider("عرض المعاينة (كلما أقل أسرع)", 600, 1400, 900, 50)

    if tpl_file:
        tpl_pages = load_pages(tpl_file.getvalue(), tpl_file.name)
        tpl_img = tpl_pages[0]
        tpl_bgr = pil_to_bgr(tpl_img)
        th, tw = tpl_bgr.shape[:2]

        st.caption(f"Template size: {tw} x {th}")

        if debug_view:
            vis = overlay_rects(tpl_bgr, cfg, choices)
            st.image(bgr_to_pil(vis), use_container_width=False, width=canvas_width)

            # show cropped ROI previews
            ix, iy, iw, ih = cfg.id_roi
            if iw > 0 and ih > 0:
                crop_id = tpl_bgr[iy:iy+ih, ix:ix+iw]
                if crop_id.size > 0:
                    st.image(bgr_to_pil(crop_id), caption="ID ROI crop", width=min(canvas_width, 500))

            for i, blk in enumerate(cfg.q_blocks, 1):
                crop_q = tpl_bgr[blk.y:blk.y+blk.h, blk.x:blk.x+blk.w]
                if crop_q.size > 0:
                    st.image(bgr_to_pil(crop_q), caption=f"QBlock {i} crop", width=min(canvas_width, 500))
    else:
        st.info("ارفع ملف Template أولاً حتى تظهر المعاينة.")


# ---------- Load roster ----------
roster: Dict[str, str] = {}
if roster_file:
    if roster_file.name.lower().endswith(("xlsx", "xls")):
        df_r = pd.read_excel(roster_file)
    else:
        df_r = pd.read_csv(roster_file)

    df_r.columns = [c.strip().lower() for c in df_r.columns]
    if "student_code" not in df_r.columns or "student_name" not in df_r.columns:
        st.error("Roster لازم يحتوي أعمدة: student_code و student_name")
    else:
        roster = dict(zip(df_r["student_code"].astype(str), df_r["student_name"].astype(str)))
        st.success(f"تم تحميل {len(roster)} طالب")


# ---------- Run grading ----------
st.markdown("---")
st.subheader("🚀 التصحيح")

debug_first = st.checkbox("🔎 Debug: اعرض أول ورقة طالب بعد المحاذاة + Threshold + المناطق", False)

if st.button("ابدأ التصحيح الآن"):
    if not tpl_file or not sheets_file:
        st.error("لازم ترفع Template و Student Sheets على الأقل.")
        st.stop()

    if not key_file:
        st.warning("لم ترفع Answer Key — سيتم استخدام Template كمفتاح (إذا كان مظلل عليه الإجابات).")

    # load template
    tpl_pages = load_pages(tpl_file.getvalue(), tpl_file.name)
    tpl_bgr = pil_to_bgr(tpl_pages[0])
    tpl_h, tpl_w = tpl_bgr.shape[:2]

    # load key
    key_pages = load_pages((key_file.getvalue() if key_file else tpl_file.getvalue()),
                           (key_file.name if key_file else tpl_file.name))
    key_bgr_raw = pil_to_bgr(key_pages[0])
    key_bgr = warp_to_template(key_bgr_raw, tpl_bgr, tpl_w, tpl_h)
    key_thr = preprocess(key_bgr)
    key_answers = read_answers(key_thr, cfg, choices)

    if len(key_answers) == 0:
        st.error("لم يتم استخراج أي إجابات من Answer Key. تأكد من Q Blocks صحيحة.")
        st.stop()

    # load student sheets
    pages = load_pages(sheets_file.getvalue(), sheets_file.name)
    total_pages = len(pages)

    results = []
    prog = st.progress(0)

    total_questions = len(key_answers)

    for idx, pg in enumerate(pages, 1):
        page_bgr_raw = pil_to_bgr(pg)
        page_bgr = warp_to_template(page_bgr_raw, tpl_bgr, tpl_w, tpl_h)
        thr = preprocess(page_bgr)

        # read code
        code, code_dbg = read_student_code(thr, cfg)
        name = roster.get(code, "") if code != "UNKNOWN" else ""

        # read answers
        stu_ans = read_answers(thr, cfg, choices)

        score = 0
        blanks = 0
        doubles = 0
        correct = 0

        for q, (ka, _) in key_answers.items():
            sa, stt = stu_ans.get(q, ("?", "BLANK"))
            if stt == "BLANK":
                blanks += 1
            if stt == "DOUBLE":
                doubles += 1

            if strict and (stt in ("BLANK", "DOUBLE")):
                pass
            else:
                if sa == ka:
                    correct += 1

        # apply theory/practical ranges (optional)
        if theory_ranges or practical_ranges:
            ranged_correct = 0
            ranged_total = 0
            for q, (ka, _) in key_answers.items():
                if (theory_ranges and in_ranges(q, theory_ranges)) or (practical_ranges and in_ranges(q, practical_ranges)):
                    sa, stt = stu_ans.get(q, ("?", "BLANK"))
                    if strict and (stt in ("BLANK", "DOUBLE")):
                        pass
                    else:
                        if sa == ka:
                            ranged_correct += 1
                    ranged_total += 1
            score = ranged_correct
            total_used = ranged_total
        else:
            score = correct
            total_used = total_questions

        # debug first page
        if debug_first and idx == 1:
            st.markdown("### 🔎 Debug (First student page)")
            vis = overlay_rects(page_bgr, cfg, choices)
            st.image(bgr_to_pil(vis), caption="Aligned page + ROIs", use_container_width=True)

            st.image(thr, caption="Threshold image", clamp=True)

            st.write("Student code:", code)
            st.write("Code debug:", code_dbg)
            st.write("First 10 answers:", {k: stu_ans.get(k) for k in sorted(list(key_answers.keys()))[:10]})

        results.append({
            "sheet_index": idx,
            "student_code": code,
            "student_name": name,
            "score": int(score),
            "total_questions": int(total_used),
            "blanks": int(blanks),
            "doubles": int(doubles),
        })

        prog.progress(int(idx / total_pages * 100))

    out = pd.DataFrame(results)

    # export excel
    buf = io.BytesIO()
    out.to_excel(buf, index=False)
    st.success("✅ تم إنشاء ملف Excel للنتائج")
    st.download_button("تحميل Excel", buf.getvalue(), "results.xlsx")
    st.dataframe(out, use_container_width=True)
