# app.py
# -*- coding: utf-8 -*-

import io
import os
import json
import math
import base64
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, ImageDraw

import cv2
from pdf2image import convert_from_bytes, convert_from_path

# ----------------------------
# Patch: Fix streamlit_drawable_canvas background_image on newer Streamlit
# ----------------------------
def _patch_streamlit_image_to_url():
    """
    streamlit-drawable-canvas calls: streamlit.elements.image.image_to_url()
    In some Streamlit versions this is missing or breaks.
    We override it with a safe Data-URL generator (PNG).
    """
    try:
        try:
            import streamlit.elements.image as st_image_mod
        except Exception:
            from streamlit.elements import image as st_image_mod  # type: ignore

        def image_to_url_safe(image, *args, **kwargs):
            # Accept PIL / np.ndarray / bytes
            if image is None:
                return None
            if isinstance(image, np.ndarray):
                # Convert BGR/RGB arrays to PIL
                if image.ndim == 3 and image.shape[2] == 3:
                    # assume RGB
                    pil = Image.fromarray(image.astype(np.uint8), mode="RGB")
                elif image.ndim == 2:
                    pil = Image.fromarray(image.astype(np.uint8), mode="L").convert("RGB")
                else:
                    pil = Image.fromarray(image.astype(np.uint8))
            elif isinstance(image, Image.Image):
                pil = image
            elif isinstance(image, (bytes, bytearray)):
                pil = Image.open(io.BytesIO(image)).convert("RGB")
            else:
                # Try to coerce
                pil = Image.fromarray(np.array(image)).convert("RGB")

            buff = io.BytesIO()
            pil.convert("RGB").save(buff, format="PNG", optimize=True)
            b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{b64}"

        # Always override (more robust)
        st_image_mod.image_to_url = image_to_url_safe  # type: ignore

    except Exception:
        # If patch fails, app can still run (but canvas background may fail)
        pass


_patch_streamlit_image_to_url()
from streamlit_drawable_canvas import st_canvas  # noqa: E402


# ----------------------------
# UI config
# ----------------------------
st.set_page_config(
    page_title="OMR BubbleSheet - Professional",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
/* RTL for Arabic UI */
html, body, [class*="css"]  { direction: rtl; }
.block-container { padding-top: 1.2rem; }

/* nicer headers */
h1, h2, h3 { letter-spacing: 0.2px; }

/* card-like containers */
.omr-card {
    background: rgba(255,255,255,0.75);
    border: 1px solid rgba(0,0,0,0.06);
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.04);
}

/* smaller expander padding */
div[data-testid="stExpander"] details summary { font-size: 0.95rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("✅ OMR BubbleSheet — واجهة احترافية + تصحيح نهائي")
st.caption("رفع القالب → تحديد مناطق (كود + بلوكات الأسئلة) → رفع مفتاح الإجابة + أوراق الطلبة + ملف الأسماء → استخراج النتائج Excel")


# ----------------------------
# Data models
# ----------------------------
@dataclass
class ROI:
    x: int
    y: int
    w: int
    h: int

@dataclass
class QBlock:
    name: str
    start_q: int
    end_q: int
    choices: int
    roi: ROI

@dataclass
class OMRConfig:
    dpi: int = 200
    id_digits: int = 4            # عدد خانات الكود
    id_rows: int = 10             # 0..9
    id_roi: Optional[ROI] = None
    q_blocks: List[QBlock] = None
    # detection tuning
    fill_thresh: float = 0.12     # نسبة تعبئة الخانة (0..1)
    winner_ratio: float = 1.18    # الفائز أكبر من الثاني بهذه النسبة
    inner_pad: float = 0.15       # قص داخلي للخانة لتجنب الحواف

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @staticmethod
    def from_json(s: str) -> "OMRConfig":
        d = json.loads(s)
        cfg = OMRConfig(
            dpi=int(d.get("dpi", 200)),
            id_digits=int(d.get("id_digits", 4)),
            id_rows=int(d.get("id_rows", 10)),
            fill_thresh=float(d.get("fill_thresh", 0.12)),
            winner_ratio=float(d.get("winner_ratio", 1.18)),
            inner_pad=float(d.get("inner_pad", 0.15)),
            id_roi=ROI(**d["id_roi"]) if d.get("id_roi") else None,
            q_blocks=[],
        )
        for qb in d.get("q_blocks", []) or []:
            cfg.q_blocks.append(
                QBlock(
                    name=qb["name"],
                    start_q=int(qb["start_q"]),
                    end_q=int(qb["end_q"]),
                    choices=int(qb.get("choices", 4)),
                    roi=ROI(**qb["roi"]),
                )
            )
        return cfg


# ----------------------------
# Helpers: file/image
# ----------------------------
def load_first_page_as_pil(uploaded_file, dpi: int) -> Image.Image:
    """
    Supports PDF/PNG/JPG/JPEG.
    Returns PIL RGB image.
    """
    if uploaded_file is None:
        raise ValueError("لا يوجد ملف.")
    name = uploaded_file.name.lower()
    data = uploaded_file.read()

    if name.endswith(".pdf"):
        pages = convert_from_bytes(data, dpi=dpi, first_page=1, last_page=1)
        img = pages[0].convert("RGB")
        return img
    else:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return img


def pil_resize_keep_aspect(img: Image.Image, target_w: int) -> Tuple[Image.Image, float]:
    w, h = img.size
    if target_w <= 0:
        return img, 1.0
    scale = target_w / float(w)
    target_h = max(1, int(round(h * scale)))
    out = img.resize((target_w, target_h), Image.LANCZOS)
    return out, scale


def roi_from_canvas_obj(obj: dict) -> ROI:
    # Fabric.js rect: left, top, width, height
    x = int(round(obj.get("left", 0)))
    y = int(round(obj.get("top", 0)))
    w = int(round(obj.get("width", 0)))
    h = int(round(obj.get("height", 0)))
    # Sometimes scaleX/scaleY used
    sx = float(obj.get("scaleX", 1.0))
    sy = float(obj.get("scaleY", 1.0))
    w = int(round(w * sx))
    h = int(round(h * sy))
    return ROI(x=x, y=y, w=w, h=h)


def draw_rois_preview(img: Image.Image, cfg: OMRConfig) -> Image.Image:
    preview = img.copy()
    dr = ImageDraw.Draw(preview)
    if cfg.id_roi:
        r = cfg.id_roi
        dr.rectangle([r.x, r.y, r.x + r.w, r.y + r.h], outline=(255, 0, 0), width=5)
        dr.text((r.x + 6, r.y + 6), "ID ROI", fill=(255, 0, 0))
    if cfg.q_blocks:
        for i, qb in enumerate(cfg.q_blocks, start=1):
            r = qb.roi
            dr.rectangle([r.x, r.y, r.x + r.w, r.y + r.h], outline=(0, 128, 255), width=5)
            dr.text((r.x + 6, r.y + 6), f"{i}) {qb.name} Q{qb.start_q}-{qb.end_q}", fill=(0, 90, 180))
    return preview


# ----------------------------
# Image normalization & OMR
# ----------------------------
def pil_to_bgr(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def normalize_to_template(page_bgr: np.ndarray, template_wh: Tuple[int, int]) -> np.ndarray:
    """
    Try to find the biggest rectangle (paper border) and warp to template size.
    If fails, fallback to resize.
    """
    tw, th = template_wh
    img = page_bgr.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    best = None
    for c in cnts[:10]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.2 * (img.shape[0] * img.shape[1]):
            best = approx
            break

    if best is None:
        return cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)

    pts = best.reshape(4, 2).astype(np.float32)

    # order points: tl, tr, br, bl
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    src = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (tw, th))
    return warped


def binarize_for_bubbles(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # adaptive threshold to handle scan lighting
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 7
    )
    # small morph clean
    k = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    return th


def crop_roi(bin_img: np.ndarray, roi: ROI) -> np.ndarray:
    h, w = bin_img.shape[:2]
    x1 = max(0, roi.x)
    y1 = max(0, roi.y)
    x2 = min(w, roi.x + roi.w)
    y2 = min(h, roi.y + roi.h)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1), dtype=np.uint8)
    return bin_img[y1:y2, x1:x2]


def split_grid(img: np.ndarray, rows: int, cols: int, inner_pad: float) -> List[List[np.ndarray]]:
    """
    split ROI into rows x cols cells with inner padding.
    """
    h, w = img.shape[:2]
    ch = h / float(rows)
    cw = w / float(cols)

    out = []
    for r in range(rows):
        row_cells = []
        for c in range(cols):
            y1 = int(round(r * ch))
            y2 = int(round((r + 1) * ch))
            x1 = int(round(c * cw))
            x2 = int(round((c + 1) * cw))

            cell = img[y1:y2, x1:x2]

            # inner pad (remove borders)
            ph = int(round((y2 - y1) * inner_pad))
            pw = int(round((x2 - x1) * inner_pad))
            cell = cell[ph:(y2 - y1 - ph), pw:(x2 - x1 - pw)]
            if cell.size == 0:
                cell = img[y1:y2, x1:x2]
            row_cells.append(cell)
        out.append(row_cells)
    return out


def cell_fill_fraction(cell: np.ndarray) -> float:
    # binary_inv: filled = 255
    if cell.size == 0:
        return 0.0
    return float(np.mean(cell > 0))


def pick_one(scores: List[float], fill_thresh: float, winner_ratio: float) -> Tuple[Optional[int], str]:
    """
    returns chosen index or None.
    status: OK / BLANK / DOUBLE / WEAK
    """
    if not scores:
        return None, "BLANK"
    idx_sorted = np.argsort(scores)[::-1]
    best = scores[idx_sorted[0]]
    second = scores[idx_sorted[1]] if len(scores) > 1 else 0.0

    if best < fill_thresh:
        return None, "BLANK"
    if second > 0 and (best / (second + 1e-9)) < winner_ratio:
        return None, "DOUBLE"
    return int(idx_sorted[0]), "OK"


def read_student_code(bin_img: np.ndarray, cfg: OMRConfig) -> Tuple[str, Dict]:
    if cfg.id_roi is None:
        return "????", {"status": "NO_ID_ROI"}

    roi_img = crop_roi(bin_img, cfg.id_roi)
    grid = split_grid(roi_img, rows=cfg.id_rows, cols=cfg.id_digits, inner_pad=cfg.inner_pad)

    digits = []
    debug = {"digit_cells": [], "digit_scores": [], "digit_status": []}

    for c in range(cfg.id_digits):
        # for a digit column, rows represent 0..9
        scores = []
        for r in range(cfg.id_rows):
            cell = grid[r][c]
            scores.append(cell_fill_fraction(cell))
        choice, status = pick_one(scores, cfg.fill_thresh, cfg.winner_ratio)
        debug["digit_scores"].append(scores)
        debug["digit_status"].append(status)

        if choice is None:
            digits.append("0")  # fallback
        else:
            digits.append(str(choice))

    code = "".join(digits)
    return code, debug


def read_qblock_answers(bin_img: np.ndarray, qb: QBlock, cfg: OMRConfig) -> Tuple[Dict[int, str], Dict]:
    """
    Returns dict: {q: 'A'/'B'/'C'/'D'/... or '?' } and status debug
    """
    roi_img = crop_roi(bin_img, qb.roi)
    n_q = qb.end_q - qb.start_q + 1
    rows = n_q
    cols = qb.choices

    grid = split_grid(roi_img, rows=rows, cols=cols, inner_pad=cfg.inner_pad)

    answers = {}
    debug = {"q_scores": {}, "q_status": {}}

    for i in range(n_q):
        qnum = qb.start_q + i
        scores = [cell_fill_fraction(grid[i][c]) for c in range(cols)]
        choice, status = pick_one(scores, cfg.fill_thresh, cfg.winner_ratio)
        debug["q_scores"][qnum] = scores
        debug["q_status"][qnum] = status

        if choice is None:
            answers[qnum] = "?"
        else:
            answers[qnum] = chr(ord("A") + choice)

    return answers, debug


def merge_blocks_answers(blocks: List[QBlock], bin_img: np.ndarray, cfg: OMRConfig) -> Tuple[Dict[int, str], Dict]:
    all_ans = {}
    all_dbg = {"blocks": []}
    for qb in blocks:
        ans, dbg = read_qblock_answers(bin_img, qb, cfg)
        all_ans.update(ans)
        all_dbg["blocks"].append({"name": qb.name, "debug": dbg})
    return all_ans, all_dbg


def score_answers(student: Dict[int, str], key: Dict[int, str]) -> Tuple[int, int, Dict]:
    """
    Only count questions where key is a valid option (A/B/C/D/..).
    """
    valid = [q for q, a in key.items() if isinstance(a, str) and len(a) == 1 and a.isalpha() and a != "?"]
    total = len(valid)
    score = 0
    detail = {"wrong": [], "blank": [], "double": []}

    for q in valid:
        sa = student.get(q, "?")
        ka = key.get(q, "?")
        if sa == "?":
            detail["blank"].append(q)
        if sa == ka:
            score += 1
        else:
            detail["wrong"].append(q)

    return score, total, detail


# ----------------------------
# Session state
# ----------------------------
if "cfg" not in st.session_state:
    st.session_state.cfg = OMRConfig(q_blocks=[])

if "template_img" not in st.session_state:
    st.session_state.template_img = None  # original PIL
if "template_disp" not in st.session_state:
    st.session_state.template_disp = None  # resized PIL
if "template_scale" not in st.session_state:
    st.session_state.template_scale = 1.0
if "last_rect" not in st.session_state:
    st.session_state.last_rect = None


# ----------------------------
# Sidebar: config & tuning
# ----------------------------
with st.sidebar:
    st.markdown('<div class="omr-card">', unsafe_allow_html=True)
    st.subheader("⚙️ إعدادات عامة")

    cfg: OMRConfig = st.session_state.cfg

    cfg.dpi = st.slider("DPI عند تحويل PDF", 120, 350, int(cfg.dpi), 10)
    cfg.id_digits = st.slider("عدد خانات كود الطالب", 2, 10, int(cfg.id_digits), 1)
    cfg.id_rows = st.slider("عدد صفوف أرقام الكود (عادة 10)", 10, 10, 10, 1)

    st.divider()
    st.subheader("🎯 حساسية القراءة")
    cfg.fill_thresh = st.slider("Fill Threshold (نسبة تعبئة الخانة)", 0.05, 0.30, float(cfg.fill_thresh), 0.01)
    cfg.winner_ratio = st.slider("Winner Ratio (تمييز أفضل خيار)", 1.05, 1.60, float(cfg.winner_ratio), 0.01)
    cfg.inner_pad = st.slider("قص داخلي للخانة (لتجنب الحواف)", 0.05, 0.35, float(cfg.inner_pad), 0.01)

    st.divider()
    st.subheader("💾 حفظ/تحميل الإعدادات")
    cfg_json = cfg.to_json()
    st.download_button("⬇️ تنزيل config.json", data=cfg_json.encode("utf-8"), file_name="config.json", mime="application/json")

    up_cfg = st.file_uploader("رفع config.json", type=["json"], key="cfg_json_up")
    if up_cfg is not None:
        try:
            loaded = up_cfg.read().decode("utf-8")
            st.session_state.cfg = OMRConfig.from_json(loaded)
            st.success("تم تحميل الإعدادات ✅")
        except Exception as e:
            st.error(f"فشل تحميل config: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs(["1) القالب + التحديد", "2) التصحيح واستخراج النتائج", "3) Debug / معاينة"])

# ----------------------------
# TAB 1: Template & ROI
# ----------------------------
with tab1:
    st.markdown('<div class="omr-card">', unsafe_allow_html=True)
    st.subheader("📄 1) رفع قالب الورقة (Template)")
    colA, colB = st.columns([1.35, 1])

    with colB:
        tpl_file = st.file_uploader("رفع PDF/PNG/JPG (صفحة القالب)", type=["pdf", "png", "jpg", "jpeg"], key="tpl")
        disp_w = st.slider("عرض القالب على الشاشة (تصغير/تكبير)", 650, 1400, 950, 10)
        st.caption("كلما قللت العرض → يصير البابل أصغر وما تحتاج تصغّر المتصفح.")

        if st.button("🧹 Reset التحديد بالكامل", use_container_width=True):
            st.session_state.cfg.id_roi = None
            st.session_state.cfg.q_blocks = []
            st.session_state.last_rect = None
            st.success("تمت إعادة التعيين ✅")

        st.divider()
        st.subheader("➕ إضافة بلوك أسئلة بسهولة")
        bname = st.text_input("اسم البلوك", value="Block 1")
        bchoices = st.number_input("عدد الخيارات لكل سؤال", min_value=2, max_value=6, value=4, step=1)
        bstart = st.number_input("Start Q", min_value=1, max_value=500, value=1, step=1)
        bend = st.number_input("End Q", min_value=1, max_value=500, value=20, step=1)

        st.caption("بعد ما ترسم مستطيل على الورقة (يسار) اضغط زر حفظ ID أو حفظ Block.")

        btn1 = st.button("✅ حفظ آخر مستطيل كـ ID ROI", use_container_width=True)
        btn2 = st.button("✅ حفظ آخر مستطيل كـ Q Block", use_container_width=True)

    with colA:
        st.subheader("🖊️ واجهة التحديد (ارسم مستطيل واحد)")
        st.caption("اختَر وضع الرسم Rect وارسم مستطيل. إذا تريد تعدل: ارسم مستطيل جديد (نأخذ الأخير).")

        if tpl_file is not None:
            try:
                tpl_img = load_first_page_as_pil(tpl_file, dpi=st.session_state.cfg.dpi)
                st.session_state.template_img = tpl_img

                disp_img, scale = pil_resize_keep_aspect(tpl_img, disp_w)
                st.session_state.template_disp = disp_img
                st.session_state.template_scale = scale

                # Canvas with background image (patched)
                canvas_result = st_canvas(
                    fill_color="rgba(255,0,0,0.06)",
                    stroke_width=3,
                    stroke_color="rgba(255,0,0,0.90)",
                    background_image=disp_img,   # <-- works due to patch
                    update_streamlit=True,
                    height=disp_img.size[1],
                    width=disp_img.size[0],
                    drawing_mode="rect",
                    key="canvas_tpl",
                    display_toolbar=True
                )

                # Extract last rect
                if canvas_result and canvas_result.json_data:
                    objs = canvas_result.json_data.get("objects", [])
                    rects = [o for o in objs if o.get("type") == "rect"]
                    if rects:
                        last = rects[-1]
                        r_disp = roi_from_canvas_obj(last)
                        st.session_state.last_rect = r_disp

                        st.info(f"آخر مستطيل (عرض الشاشة): x={r_disp.x}, y={r_disp.y}, w={r_disp.w}, h={r_disp.h}")

                        # show in original coordinates
                        sc = st.session_state.template_scale
                        r_org = ROI(
                            x=int(round(r_disp.x / sc)),
                            y=int(round(r_disp.y / sc)),
                            w=int(round(r_disp.w / sc)),
                            h=int(round(r_disp.h / sc)),
                        )
                        st.write("✅ نفس المستطيل بإحداثيات القالب الأصلية:", r_org)

            except Exception as e:
                st.error(f"فشل عرض القالب: {e}")
        else:
            st.warning("ارفع القالب أولاً.")

    # handle save buttons
    if btn1 or btn2:
        if st.session_state.template_img is None or st.session_state.last_rect is None:
            st.warning("لا يوجد مستطيل مرسوم أو لا يوجد قالب.")
        else:
            sc = float(st.session_state.template_scale)
            r_disp: ROI = st.session_state.last_rect
            r_org = ROI(
                x=int(round(r_disp.x / sc)),
                y=int(round(r_disp.y / sc)),
                w=int(round(r_disp.w / sc)),
                h=int(round(r_disp.h / sc)),
            )

            if btn1:
                st.session_state.cfg.id_roi = r_org
                st.success("تم حفظ ID ROI ✅")

            if btn2:
                if bend < bstart:
                    st.error("End Q لازم أكبر أو يساوي Start Q")
                else:
                    qb = QBlock(
                        name=bname.strip() if bname.strip() else f"Block {len(st.session_state.cfg.q_blocks) + 1}",
                        start_q=int(bstart),
                        end_q=int(bend),
                        choices=int(bchoices),
                        roi=r_org,
                    )
                    st.session_state.cfg.q_blocks.append(qb)
                    st.success("تم حفظ Q Block ✅")

    # Preview
    st.divider()
    st.subheader("👁️ معاينة التحديد الحالي")
    if st.session_state.template_img is not None:
        prev = draw_rois_preview(st.session_state.template_img, st.session_state.cfg)
        prev_disp, _ = pil_resize_keep_aspect(prev, min(1100, prev.size[0]))
        st.image(prev_disp, caption="المعاينة (ID باللون الأحمر - البلوكات بالأزرق)", use_container_width=True)
    else:
        st.info("لا توجد معاينة لأن القالب غير مرفوع.")

    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# TAB 2: Grade & Export
# ----------------------------
with tab2:
    st.markdown('<div class="omr-card">', unsafe_allow_html=True)
    st.subheader("🧾 2) التصحيح واستخراج النتائج")

    cfg: OMRConfig = st.session_state.cfg
    if cfg.id_roi is None or not cfg.q_blocks:
        st.warning("قبل التصحيح: لازم تحدد ID ROI وتضيف على الأقل Q Block واحد في تبويب (القالب + التحديد).")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📌 الملفات المطلوبة")
        key_file = st.file_uploader("مفتاح الإجابة (PDF/PNG/JPG)", type=["pdf", "png", "jpg", "jpeg"], key="key_file")
        student_file = st.file_uploader("أوراق الطلبة (PDF واحد متعدد الصفحات)", type=["pdf"], key="stu_pdf")
        roster_file = st.file_uploader("ملف الأسماء/الأكواد (Excel)", type=["xlsx"], key="roster")

        st.markdown("### ⚙️ إعدادات الإخراج")
        sheet_name = st.text_input("اسم صفحة Excel", value="Results")
        max_pages = st.number_input("حد أقصى صفحات للمعالجة (0 = الكل)", min_value=0, max_value=2000, value=0, step=10)

    with col2:
        st.markdown("### ✅ ملخص الإعدادات الحالية")
        st.write("DPI:", cfg.dpi)
        st.write("ID digits:", cfg.id_digits)
        st.write("Blocks:", len(cfg.q_blocks))
        if cfg.id_roi:
            st.write("ID ROI:", cfg.id_roi)
        if cfg.q_blocks:
            for i, qb in enumerate(cfg.q_blocks, start=1):
                st.write(f"{i}) {qb.name}: Q{qb.start_q}-{qb.end_q} choices={qb.choices} roi={qb.roi}")

        st.divider()
        st.markdown("### ▶️ تشغيل التصحيح")
        run = st.button("🚀 ابدأ التصحيح", use_container_width=True)

    def _read_pages_any(file_up, dpi: int) -> List[Image.Image]:
        name = file_up.name.lower()
        data = file_up.read()
        if name.endswith(".pdf"):
            # Convert all pages
            pages = convert_from_bytes(data, dpi=dpi)
            return [p.convert("RGB") for p in pages]
        else:
            return [Image.open(io.BytesIO(data)).convert("RGB")]

    def _read_student_pdf_pages(file_up, dpi: int, limit: int) -> List[Image.Image]:
        data = file_up.read()
        pages = convert_from_bytes(data, dpi=dpi)
        if limit and limit > 0:
            pages = pages[:limit]
        return [p.convert("RGB") for p in pages]

    def _load_roster(file_up) -> Dict[str, str]:
        df = pd.read_excel(file_up)
        # try find columns
        cols = [c.lower().strip() for c in df.columns]
        code_col = None
        name_col = None
        for i, c in enumerate(cols):
            if "code" in c or "كود" in c or "id" in c:
                code_col = df.columns[i]
            if "name" in c or "اسم" in c:
                name_col = df.columns[i]
        if code_col is None:
            code_col = df.columns[0]
        if name_col is None and len(df.columns) > 1:
            name_col = df.columns[1]

        out = {}
        for _, row in df.iterrows():
            code = str(row.get(code_col, "")).strip()
            if code in ("", "nan", "None"):
                continue
            # keep digits only
            code_digits = "".join([ch for ch in code if ch.isdigit()])
            if code_digits == "":
                continue
            code_digits = code_digits.zfill(cfg.id_digits)[-cfg.id_digits:]
            nm = ""
            if name_col is not None:
                nm = str(row.get(name_col, "")).strip()
                if nm.lower() in ("nan", "none"):
                    nm = ""
            out[code_digits] = nm
        return out

    if run:
        if st.session_state.template_img is None:
            st.error("ارفع القالب في التبويب الأول أولاً.")
        elif key_file is None or student_file is None:
            st.error("ارفع مفتاح الإجابة + ملف أوراق الطلبة.")
        elif cfg.id_roi is None or not cfg.q_blocks:
            st.error("لا يمكن التصحيح بدون ID ROI و Q Blocks.")
        else:
            try:
                roster_map = {}
                if roster_file is not None:
                    roster_map = _load_roster(roster_file)

                # Template base size (original)
                tw, th = st.session_state.template_img.size

                # Read key page(s) - use first page only
                key_pages = _read_pages_any(key_file, cfg.dpi)
                key_pil = key_pages[0]
                key_bgr = pil_to_bgr(key_pil)
                key_bgr = normalize_to_template(key_bgr, (tw, th))
                key_bin = binarize_for_bubbles(key_bgr)
                key_ans, key_dbg = merge_blocks_answers(cfg.q_blocks, key_bin, cfg)

                # Quick sanity check: if key is mostly '?', warn
                key_valid = [a for a in key_ans.values() if a != "?"]
                if len(key_valid) < max(3, int(0.3 * len(key_ans))):
                    st.warning("⚠️ مفتاح الإجابة يبدو غير مقروء (الكثير من '?'). عدّل ROI أو حساسية القراءة (fill_thresh).")

                # Read student pages
                pages = _read_student_pdf_pages(student_file, cfg.dpi, int(max_pages))
                total_pages = len(pages)

                prog = st.progress(0)
                results = []
                debug_samples = []

                for idx, pil_page in enumerate(pages, start=1):
                    bgr = pil_to_bgr(pil_page)
                    bgr = normalize_to_template(bgr, (tw, th))
                    bin_img = binarize_for_bubbles(bgr)

                    code, code_dbg = read_student_code(bin_img, cfg)
                    stu_ans, stu_dbg = merge_blocks_answers(cfg.q_blocks, bin_img, cfg)
                    score, total_q, detail = score_answers(stu_ans, key_ans)

                    name = roster_map.get(code, "")

                    results.append({
                        "sheet_index": idx,
                        "student_code": code,
                        "student_name": name,
                        "score": score,
                        "total_questions": total_q,
                        "percent": round((score / total_q * 100.0), 2) if total_q > 0 else 0.0,
                    })

                    # keep few debug samples
                    if idx <= 3:
                        debug_samples.append({
                            "sheet_index": idx,
                            "code": code,
                            "code_status": code_dbg.get("digit_status"),
                            "score": score,
                            "total": total_q,
                            "wrong_q": detail["wrong"][:10],
                        })

                    prog.progress(int(idx / total_pages * 100))

                df_out = pd.DataFrame(results)

                # export excel
                out_buf = io.BytesIO()
                with pd.ExcelWriter(out_buf, engine="openpyxl") as writer:
                    df_out.to_excel(writer, index=False, sheet_name=sheet_name[:31])

                    # add key sheet
                    key_df = pd.DataFrame(sorted(key_ans.items(), key=lambda x: x[0]), columns=["question", "key_answer"])
                    key_df.to_excel(writer, index=False, sheet_name="AnswerKey")

                    # add debug sample sheet
                    if debug_samples:
                        dbg_df = pd.DataFrame(debug_samples)
                        dbg_df.to_excel(writer, index=False, sheet_name="DebugSample")

                st.success("✅ تم استخراج النتائج بنجاح")
                st.dataframe(df_out, use_container_width=True)

                st.download_button(
                    "⬇️ تنزيل ملف النتائج Excel",
                    data=out_buf.getvalue(),
                    file_name="OMR_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"حصل خطأ أثناء التصحيح: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# TAB 3: Debug / Preview
# ----------------------------
with tab3:
    st.markdown('<div class="omr-card">', unsafe_allow_html=True)
    st.subheader("🧪 Debug / معاينة القراءة (للتأكد قبل التصحيح الكامل)")

    cfg: OMRConfig = st.session_state.cfg
    if st.session_state.template_img is None:
        st.info("ارفع القالب أولاً من تبويب (القالب + التحديد).")
    else:
        st.markdown("### 1) معاينة binarization للقالب")
        tw, th = st.session_state.template_img.size
        bgr = pil_to_bgr(st.session_state.template_img)
        bin_img = binarize_for_bubbles(bgr)
        # show a smaller preview
        pil_bin = Image.fromarray(bin_img).convert("RGB")
        pil_bin_disp, _ = pil_resize_keep_aspect(pil_bin, 900)
        st.image(pil_bin_disp, caption="Binarized template preview", use_container_width=True)

        st.divider()
        st.markdown("### 2) اختبار صفحة واحدة (طالب أو مفتاح)")

        test_file = st.file_uploader("ارفع صفحة واحدة للتجربة (PDF/PNG/JPG)", type=["pdf", "png", "jpg", "jpeg"], key="test_one")
        if test_file is not None and cfg.id_roi is not None and cfg.q_blocks:
            try:
                test_img = load_first_page_as_pil(test_file, dpi=cfg.dpi)
                tbgr = pil_to_bgr(test_img)
                tbgr = normalize_to_template(tbgr, (tw, th))
                tbin = binarize_for_bubbles(tbgr)

                code, code_dbg = read_student_code(tbin, cfg)
                ans, dbg = merge_blocks_answers(cfg.q_blocks, tbin, cfg)

                st.success(f"✅ الكود المقروء: {code}")
                st.write("حالة كل خانة:", code_dbg.get("digit_status"))

                # show first 15 answers
                items = sorted(ans.items(), key=lambda x: x[0])[:15]
                st.write("أول 15 سؤال مقروء:")
                st.table(pd.DataFrame(items, columns=["Q", "Answer"]))

                # ROI previews
                st.divider()
                st.markdown("### 3) معاينة قصّ الـ ROI")
                # ID ROI crop
                id_crop = crop_roi(tbin, cfg.id_roi)
                st.image(Image.fromarray(id_crop), caption="ID ROI (binary)", use_container_width=True)

                # first block crop
                qb0 = cfg.q_blocks[0]
                qb_crop = crop_roi(tbin, qb0.roi)
                st.image(Image.fromarray(qb_crop), caption=f"QBlock ROI (binary) - {qb0.name}", use_container_width=True)

            except Exception as e:
                st.error(f"فشل اختبار الصفحة: {e}")
        else:
            st.info("ارفع ملف تجربة + تأكد محدد ID ROI ومضيف Q Blocks.")

    st.markdown("</div>", unsafe_allow_html=True)
