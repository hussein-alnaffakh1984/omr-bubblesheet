# ============================================================
# OMR BUBBLE SHEET SCANNER (SCANNER) — DEBUG / VERIFY STEP-BY-STEP
# ============================================================
# ✅ يعرض ناتج كل إجراء: Aligned → Binary → ROI → Grid → Fill Tables
# ✅ يدعم PDF/صور + Multiple student files
# ✅ إصلاح مشكلة Streamlit getvalue(): نستخدم getbuffer/read بطريقة آمنة
# ============================================================

import io
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image, ImageDraw


# ============================================================
# HELPERS (SAFE FILE READ)
# ============================================================

def read_uploaded_file_bytes(uploaded_file) -> bytes:
    """Most compatible way to read bytes from Streamlit UploadedFile."""
    if uploaded_file is None:
        return b""
    try:
        return uploaded_file.getbuffer().tobytes()
    except Exception:
        try:
            # WARNING: read() can consume the stream; but acceptable if used once per file
            return uploaded_file.read()
        except Exception:
            return b""


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class Rectangle:
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height


@dataclass
class QuestionBlock:
    rect: Rectangle
    start_q: int
    end_q: int
    num_rows: int


@dataclass
class Template:
    width: int
    height: int
    id_block: Optional[Rectangle] = None
    q_blocks: List[QuestionBlock] = None
    num_choices: int = 4
    id_digits: int = 4
    id_rows: int = 10

    def __post_init__(self):
        if self.q_blocks is None:
            self.q_blocks = []


# ============================================================
# IMAGE PROCESSING
# ============================================================

class ImageProcessor:

    @staticmethod
    def load_first_page(file_bytes: bytes, filename: str, dpi: int = 250) -> Optional[Image.Image]:
        try:
            if filename.lower().endswith(".pdf"):
                pages = convert_from_bytes(file_bytes, dpi=dpi)
                return pages[0].convert("RGB") if pages else None
            return Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception as e:
            st.error(f"خطأ تحميل الصورة: {e}")
            return None

    @staticmethod
    def load_all_pages(file_bytes: bytes, filename: str, dpi: int = 250) -> List[Image.Image]:
        try:
            if filename.lower().endswith(".pdf"):
                pages = convert_from_bytes(file_bytes, dpi=dpi)
                return [p.convert("RGB") for p in pages]
            return [Image.open(io.BytesIO(file_bytes)).convert("RGB")]
        except Exception as e:
            st.error(f"خطأ تحميل الصفحات: {e}")
            return []

    @staticmethod
    def pil_to_bgr(img: Image.Image) -> np.ndarray:
        arr = np.array(img)  # RGB
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    @staticmethod
    def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    @staticmethod
    def preprocess_binary(img_bgr: np.ndarray, blur_ksize: int, block_size: int, C: int) -> np.ndarray:
        """
        Output binary (white=ink) using adaptive threshold, inverted.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        if blur_ksize and blur_ksize > 0:
            gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

        block_size = int(block_size)
        if block_size % 2 == 0:
            block_size += 1

        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size, int(C)
        )
        return binary

    @staticmethod
    def resize_to_template(img_bgr: np.ndarray, w: int, h: int) -> np.ndarray:
        return cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def align_to_template_warp(img_bgr: np.ndarray, target_w: int, target_h: int) -> Tuple[np.ndarray, bool]:
        """
        Detect largest 4-point contour (paper) then warp to template size.
        If not found, fall back to resize.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return ImageProcessor.resize_to_template(img_bgr, target_w, target_h), False

        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        sheet = None
        for c in cnts[:10]:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                sheet = approx
                break

        if sheet is None:
            return ImageProcessor.resize_to_template(img_bgr, target_w, target_h), False

        pts = sheet.reshape(4, 2).astype(np.float32)

        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]

        src = np.array([tl, tr, br, bl], dtype=np.float32)
        dst = np.array([[0, 0],
                        [target_w - 1, 0],
                        [target_w - 1, target_h - 1],
                        [0, target_h - 1]], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img_bgr, M, (target_w, target_h))
        return warped, True


# ============================================================
# BUBBLE DETECTION
# ============================================================

class BubbleDetector:
    def __init__(self, min_fill: float = 0.10, margin: float = 0.15, double_ratio: float = 1.35):
        self.min_fill = float(min_fill)
        self.margin = float(margin)
        self.double_ratio = float(double_ratio)

    def calculate_fill(self, cell: np.ndarray) -> float:
        if cell is None or cell.size == 0:
            return 0.0

        h, w = cell.shape[:2]
        mh = int(h * self.margin)
        mw = int(w * self.margin)

        y1, y2 = mh, h - mh
        x1, x2 = mw, w - mw

        if y2 <= y1 or x2 <= x1:
            return 0.0

        inner = cell[y1:y2, x1:x2]
        if inner.size == 0:
            return 0.0

        return float(np.sum(inner > 0) / inner.size)

    def detect_answer(self, cells: List[np.ndarray], choices: List[str]) -> Dict:
        fills = [self.calculate_fill(c) for c in cells]
        order = sorted(range(len(fills)), key=lambda i: fills[i], reverse=True)

        top = order[0]
        top_fill = fills[top]
        second_fill = fills[order[1]] if len(order) > 1 else 0.0

        if top_fill < self.min_fill:
            return {"answer": "?", "status": "BLANK", "fills": fills}

        if second_fill >= self.min_fill and (top_fill / (second_fill + 1e-9)) < self.double_ratio:
            return {"answer": "!", "status": "DOUBLE", "fills": fills}

        return {"answer": choices[top], "status": "OK", "fills": fills}


# ============================================================
# GRADING ENGINE
# ============================================================

class GradingEngine:
    def __init__(self, template: Template, detector: BubbleDetector):
        self.template = template
        self.detector = detector

    def _safe_roi(self, binary: np.ndarray, rect: Rectangle) -> Optional[np.ndarray]:
        h, w = binary.shape[:2]
        if rect.x < 0 or rect.y < 0 or rect.x2 > w or rect.y2 > h:
            return None
        return binary[rect.y:rect.y2, rect.x:rect.x2]

    def extract_id(self, binary: np.ndarray) -> Tuple[str, pd.DataFrame]:
        """
        Returns (id_string, debug_table).
        debug_table shows status and fills per row (0..9) for each digit column.
        """
        if not self.template.id_block:
            return "", pd.DataFrame()

        roi = self._safe_roi(binary, self.template.id_block)
        if roi is None:
            return "OUT_OF_BOUNDS", pd.DataFrame()

        rows = int(self.template.id_rows)
        cols = int(self.template.id_digits)

        cell_h = max(1, self.template.id_block.height // rows)
        cell_w = max(1, self.template.id_block.width // cols)

        digits = []
        dbg_rows = []

        for col in range(cols):
            col_cells = []
            for row in range(rows):
                y1, y2 = row * cell_h, (row + 1) * cell_h
                x1, x2 = col * cell_w, (col + 1) * cell_w
                col_cells.append(roi[y1:y2, x1:x2])

            res = self.detector.detect_answer(col_cells, [str(i) for i in range(10)])
            digit = res["answer"] if res["status"] == "OK" else "X"
            digits.append(digit)

            row_obj = {"digit_col": col + 1, "status": res["status"], "picked": res["answer"]}
            for r in range(min(10, len(res["fills"]))):
                row_obj[f"r{r}"] = round(res["fills"][r], 3)
            dbg_rows.append(row_obj)

        return "".join(digits), pd.DataFrame(dbg_rows)

    def extract_answers_block(self, binary: np.ndarray, block: QuestionBlock) -> Tuple[Dict[int, Dict], pd.DataFrame]:
        roi = self._safe_roi(binary, block.rect)
        if roi is None:
            return {}, pd.DataFrame()

        rows = int(block.num_rows)
        cols = int(self.template.num_choices)
        cell_h = max(1, block.rect.height // rows)
        cell_w = max(1, block.rect.width // cols)

        choices = list("ABCDEFGH"[:cols])
        answers: Dict[int, Dict] = {}
        dbg = []

        q = block.start_q
        for r in range(rows):
            if q > block.end_q:
                break

            row_cells = []
            for c in range(cols):
                y1, y2 = r * cell_h, (r + 1) * cell_h
                x1, x2 = c * cell_w, (c + 1) * cell_w
                row_cells.append(roi[y1:y2, x1:x2])

            res = self.detector.detect_answer(row_cells, choices)
            answers[q] = res

            dbg_row = {"q": q, "status": res["status"], "answer": res["answer"]}
            for i, ch in enumerate(choices):
                dbg_row[ch] = round(res["fills"][i], 3)
            dbg.append(dbg_row)

            q += 1

        return answers, pd.DataFrame(dbg)

    def build_answer_key_from_key_binary(self, key_binary: np.ndarray) -> Tuple[Dict[int, str], List[pd.DataFrame]]:
        answer_key: Dict[int, str] = {}
        debug_tables: List[pd.DataFrame] = []

        for block in self.template.q_blocks:
            ans, dbg = self.extract_answers_block(key_binary, block)
            debug_tables.append(dbg)
            for q, res in ans.items():
                if res["status"] == "OK":
                    answer_key[q] = res["answer"]

        return answer_key, debug_tables

    def grade_one(self, binary: np.ndarray, answer_key: Dict[int, str], strict: bool) -> Tuple[int, int, float, pd.DataFrame]:
        # extract all answers
        all_answers: Dict[int, Dict] = {}
        for block in self.template.q_blocks:
            ans, _ = self.extract_answers_block(binary, block)
            all_answers.update(ans)

        correct = 0
        total = len(answer_key)
        per_q = []

        for q, k in answer_key.items():
            if q not in all_answers:
                per_q.append({"q": q, "key": k, "student": "-", "status": "MISSING", "is_correct": False})
                continue

            res = all_answers[q]
            if strict and res["status"] != "OK":
                per_q.append({"q": q, "key": k, "student": res["answer"], "status": res["status"], "is_correct": False})
                continue

            is_ok = (res["answer"] == k)
            correct += int(is_ok)
            per_q.append({"q": q, "key": k, "student": res["answer"], "status": res["status"], "is_correct": is_ok})

        pct = (correct / total * 100) if total else 0.0
        return correct, total, pct, pd.DataFrame(per_q)


# ============================================================
# UI HELPERS
# ============================================================

def draw_preview(img: Image.Image, template: Template) -> Image.Image:
    preview = img.copy()
    draw = ImageDraw.Draw(preview)

    if template.id_block:
        r = template.id_block
        draw.rectangle([r.x, r.y, r.x2, r.y2], outline="red", width=4)
        draw.text((r.x + 8, r.y + 8), "ID", fill="red")

    for i, block in enumerate(template.q_blocks, 1):
        r = block.rect
        draw.rectangle([r.x, r.y, r.x2, r.y2], outline="green", width=4)
        draw.text((r.x + 8, r.y + 8), f"B{i}:Q{block.start_q}-{block.end_q}", fill="green")

    return preview


def make_rect_from_points(x1, y1, x2, y2) -> Optional[Rectangle]:
    x = int(min(x1, x2))
    y = int(min(y1, y2))
    w = int(abs(x2 - x1))
    h = int(abs(y2 - y1))
    if w < 10 or h < 10:
        return None
    return Rectangle(x, y, w, h)


# ============================================================
# MAIN
# ============================================================

def main():
    st.set_page_config(page_title="OMR Debug (Scanner)", layout="wide")
    st.title("✅ OMR Bubble Sheet — Debug خطوة بخطوة (Scanner)")

    # Session state
    if "template" not in st.session_state:
        st.session_state.template = None
    if "template_img" not in st.session_state:
        st.session_state.template_img = None
    if "answer_key" not in st.session_state:
        st.session_state.answer_key = None

    st.markdown("---")
    st.subheader("1) رفع نموذج البابل شيت (Template)")
    template_file = st.file_uploader("Template (PDF/PNG/JPG)", type=["pdf", "png", "jpg", "jpeg"], key="tpl")

    if template_file:
        tpl_bytes = read_uploaded_file_bytes(template_file)
        tpl_img = ImageProcessor.load_first_page(tpl_bytes, template_file.name, dpi=250)
        if tpl_img:
            st.session_state.template_img = tpl_img
            w, h = tpl_img.size
            if st.session_state.template is None:
                st.session_state.template = Template(w, h)
            else:
                st.session_state.template.width = w
                st.session_state.template.height = h

            st.success(f"✅ Template جاهز: {w}×{h}")
            st.image(draw_preview(tpl_img, st.session_state.template), use_container_width=True)

    if not st.session_state.template_img:
        st.info("ارفع Template للبدء.")
        st.stop()

    st.markdown("---")
    st.subheader("2) إعدادات الكشف (Detector + Binary)")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        num_choices = st.selectbox("عدد الخيارات", [4, 5, 6], index=0)
    with c2:
        id_digits = st.number_input("خانات الكود", 1, 12, 4, 1)
    with c3:
        id_rows = st.number_input("صفوف الكود", 5, 15, 10, 1)
    with c4:
        min_fill = st.slider("min_fill", 0.03, 0.30, 0.10, 0.01)
    with c5:
        margin = st.slider("margin", 0.05, 0.35, 0.15, 0.01)

    p1, p2, p3 = st.columns(3)
    with p1:
        blur_k = st.selectbox("Gaussian blur", [0, 3, 5], index=1)
    with p2:
        block_size = st.selectbox("Adaptive block size", [15, 21, 25, 31], index=1)
    with p3:
        C = st.selectbox("Adaptive C", [2, 4, 6, 8, 10], index=2)

    use_warp = st.checkbox("استخدم Warp قبل القصّ (مفيد إذا يوجد قص/إزاحة بالسكنر)", value=True)
    strict = st.checkbox("وضع صارم: BLANK/DOUBLE لا تُحسب", value=False)

    st.session_state.template.num_choices = int(num_choices)
    st.session_state.template.id_digits = int(id_digits)
    st.session_state.template.id_rows = int(id_rows)

    detector = BubbleDetector(min_fill=min_fill, margin=margin)
    engine = GradingEngine(st.session_state.template, detector)

    def align(img_bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
        if use_warp:
            return ImageProcessor.align_to_template_warp(img_bgr, st.session_state.template.width, st.session_state.template.height)
        return ImageProcessor.resize_to_template(img_bgr, st.session_state.template.width, st.session_state.template.height), True

    st.markdown("---")
    st.subheader("3) تحديد المناطق (ID + Blocks)")
    st.caption("استخدم Paint للحصول على الإحداثيات (x1,y1) و(x2,y2) لنفس أبعاد الـTemplate.")

    st.image(draw_preview(st.session_state.template_img, st.session_state.template), caption="Preview", use_container_width=True)

    mode = st.radio("نوع المنطقة", ["ID", "Q_BLOCK"], horizontal=True)

    if mode == "Q_BLOCK":
        a, b, c = st.columns(3)
        with a:
            start_q = st.number_input("من سؤال", 1, 500, 1)
        with b:
            end_q = st.number_input("إلى سؤال", 1, 500, 20)
        with c:
            num_rows_block = st.number_input("عدد الصفوف", 1, 200, 20)
    else:
        start_q, end_q, num_rows_block = 0, 0, 0

    colL, colR = st.columns(2)
    with colL:
        x1 = st.number_input("x1", 0, st.session_state.template.width, 0, 10)
        y1 = st.number_input("y1", 0, st.session_state.template.height, 0, 10)
    with colR:
        x2 = st.number_input("x2", 0, st.session_state.template.width, 200, 10)
        y2 = st.number_input("y2", 0, st.session_state.template.height, 200, 10)

    rect = make_rect_from_points(x1, y1, x2, y2)
    if rect:
        st.info(f"المستطيل: ({rect.x},{rect.y}) → ({rect.x2},{rect.y2}) | size={rect.width}×{rect.height}")
    else:
        st.warning("المستطيل صغير جدًا (لازم ≥ 10×10).")

    if st.button("💾 حفظ المنطقة", type="primary", use_container_width=True):
        if rect is None:
            st.error("❌ لا يمكن حفظ مستطيل صغير.")
        else:
            if mode == "ID":
                st.session_state.template.id_block = rect
                st.success("✅ تم حفظ منطقة ID")
            else:
                qb = QuestionBlock(rect=rect, start_q=int(start_q), end_q=int(end_q), num_rows=int(num_rows_block))
                st.session_state.template.q_blocks.append(qb)
                st.success(f"✅ تم حفظ Block: Q{start_q}-{end_q}")
            st.rerun()

    if st.session_state.template.id_block or st.session_state.template.q_blocks:
        st.markdown("#### المناطق المحفوظة")
        if st.session_state.template.id_block:
            r = st.session_state.template.id_block
            st.success(f"ID: ({r.x},{r.y}) → ({r.x2},{r.y2})")
        for i, b in enumerate(st.session_state.template.q_blocks, 1):
            r = b.rect
            colA, colB = st.columns([4, 1])
            with colA:
                st.success(f"Block {i}: Q{b.start_q}-{b.end_q} | ({r.x},{r.y}) → ({r.x2},{r.y2}) | rows={b.num_rows}")
            with colB:
                if st.button("🗑️ حذف", key=f"del_{i}"):
                    st.session_state.template.q_blocks.pop(i - 1)
                    st.rerun()

    if not st.session_state.template.id_block or not st.session_state.template.q_blocks:
        st.warning("لازم تحدد ID وBlock واحد على الأقل قبل رفع الملفات.")
        st.stop()

    st.markdown("---")
    st.subheader("4) رفع الملفات (Roster + Key + Sheets)")
    c1, c2, c3 = st.columns(3)
    with c1:
        roster_file = st.file_uploader("📋 Roster (xlsx/csv)", type=["xlsx", "xls", "csv"], key="roster")
    with c2:
        key_file = st.file_uploader("🔑 Answer Key (pdf/jpg/png)", type=["pdf", "png", "jpg", "jpeg"], key="key")
    with c3:
        sheets_files = st.file_uploader("📚 Student Sheets (multiple)", type=["pdf", "png", "jpg", "jpeg"],
                                        accept_multiple_files=True, key="sheets")

    # -------- ROSTER LOAD (WITH ZFILL) --------
    roster_dict: Dict[str, str] = {}
    if roster_file:
        try:
            if roster_file.name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(roster_file)
            else:
                df = pd.read_csv(roster_file)

            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            if "student_code" not in df.columns or "student_name" not in df.columns:
                st.error("❌ لازم الأعمدة تكون: student_code و student_name")
            else:
                digits = st.session_state.template.id_digits
                df["student_code"] = df["student_code"].astype(str).str.strip().str.zfill(digits)
                df["student_name"] = df["student_name"].astype(str).str.strip()
                roster_dict = dict(zip(df["student_code"], df["student_name"]))
                st.success(f"✅ roster جاهز: {len(roster_dict)} طالب")
                with st.expander("عرض أول 10 صفوف من roster"):
                    st.dataframe(df.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"❌ خطأ roster: {e}")

    # -------- KEY VERIFY + BUILD ANSWER KEY --------
    if key_file:
        st.markdown("---")
        st.subheader("5) فحص نموذج الإجابة (Key) — خطوة بخطوة")

        key_bytes = read_uploaded_file_bytes(key_file)
        key_img = ImageProcessor.load_first_page(key_bytes, key_file.name, dpi=250)
        if key_img:
            key_bgr = ImageProcessor.pil_to_bgr(key_img)
            key_aligned, ok_warp = align(key_bgr)
            key_binary = ImageProcessor.preprocess_binary(key_aligned, blur_ksize=int(blur_k), block_size=int(block_size), C=int(C))

            colA, colB = st.columns(2)
            with colA:
                st.image(ImageProcessor.bgr_to_pil(key_aligned), caption=f"Aligned (warp_ok={ok_warp})", use_container_width=True)
            with colB:
                st.image(key_binary, caption="Binary", clamp=True, use_container_width=True)

            # ROIs
            r = st.session_state.template.id_block
            st.image(key_binary[r.y:r.y2, r.x:r.x2], caption="ID ROI (Key)", clamp=True, use_container_width=True)

            b0 = st.session_state.template.q_blocks[0].rect
            st.image(key_binary[b0.y:b0.y2, b0.x:b0.x2], caption="Q Block ROI (Key)", clamp=True, use_container_width=True)

            answer_key, key_debug_tables = engine.build_answer_key_from_key_binary(key_binary)
            st.session_state.answer_key = answer_key

            st.success(f"✅ تم استخراج {len(answer_key)} إجابة صحيحة من Key")
            with st.expander("تفاصيل fills للـKey (لكل Block)"):
                for i, dbg in enumerate(key_debug_tables, 1):
                    st.write(f"Block {i}")
                    st.dataframe(dbg, use_container_width=True)
        else:
            st.error("❌ فشل تحميل Key")

    # -------- SAMPLE SHEET VERIFY --------
    if sheets_files:
        st.markdown("---")
        st.subheader("6) فحص ورقة طالب (Sample) — خطوة بخطوة")

        sample = sheets_files[0]
        sample_bytes = read_uploaded_file_bytes(sample)
        pages = ImageProcessor.load_all_pages(sample_bytes, sample.name, dpi=250)

        if pages:
            st.write(f"الملف: {sample.name} | الصفحات: {len(pages)} (نستخدم أول صفحة)")
            stud_bgr = ImageProcessor.pil_to_bgr(pages[0])
            stud_aligned, ok_warp = align(stud_bgr)
            stud_binary = ImageProcessor.preprocess_binary(stud_aligned, blur_ksize=int(blur_k), block_size=int(block_size), C=int(C))

            colA, colB = st.columns(2)
            with colA:
                st.image(ImageProcessor.bgr_to_pil(stud_aligned), caption=f"Aligned (warp_ok={ok_warp})", use_container_width=True)
            with colB:
                st.image(stud_binary, caption="Binary", clamp=True, use_container_width=True)

            r = st.session_state.template.id_block
            roi_id = stud_binary[r.y:r.y2, r.x:r.x2]
            st.image(roi_id, caption="ID ROI (Student)", clamp=True, use_container_width=True)

            b0 = st.session_state.template.q_blocks[0].rect
            roi_q = stud_binary[b0.y:b0.y2, b0.x:b0.x2]
            st.image(roi_q, caption="Q Block ROI (Student)", clamp=True, use_container_width=True)

            sid, id_dbg = engine.extract_id(stud_binary)
            sid_z = sid.zfill(st.session_state.template.id_digits) if sid.isdigit() else sid

            st.success(f"🆔 ID المستخرج: {sid_z}")
            with st.expander("تفاصيل ID (fills لكل عمود)"):
                if id_dbg is not None and not id_dbg.empty:
                    st.dataframe(id_dbg, use_container_width=True)
                else:
                    st.write("لا يوجد جدول (قد يكون OUT_OF_BOUNDS أو مشكلة بالـID ROI).")

            # Answer blocks debug
            for i, block in enumerate(st.session_state.template.q_blocks, 1):
                _, dbg = engine.extract_answers_block(stud_binary, block)
                with st.expander(f"تفاصيل Block {i} (Q{block.start_q}-{block.end_q})"):
                    st.dataframe(dbg, use_container_width=True)

            # Roster match
            if roster_dict:
                if sid_z in roster_dict:
                    st.success(f"✅ الاسم: {roster_dict[sid_z]}")
                else:
                    st.warning("⚠️ لم يتم العثور على الاسم في roster — تأكد من عدد خانات الكود (zfill) وتحديد منطقة ID.")
        else:
            st.error("❌ لا يمكن قراءة صفحات Sample")

    # -------- FINAL GRADING --------
    st.markdown("---")
    st.subheader("7) التصحيح النهائي لكل الأوراق")

    if not st.session_state.answer_key:
        st.warning("ارفع Key أولاً وتأكد أنه استخرج الإجابات.")
        st.stop()
    if not roster_dict:
        st.warning("ارفع Roster صحيحاً (student_code, student_name).")
        st.stop()
    if not sheets_files:
        st.warning("ارفع أوراق الطلاب.")
        st.stop()

    if st.button("🚀 ابدأ التصحيح", type="primary", use_container_width=True):
        answer_key = st.session_state.answer_key
        results = []

        for f in sheets_files:
            f_bytes = read_uploaded_file_bytes(f)
            pages = ImageProcessor.load_all_pages(f_bytes, f.name, dpi=250)

            for page_idx, pil_page in enumerate(pages, 1):
                img_bgr = ImageProcessor.pil_to_bgr(pil_page)
                aligned, ok_warp = align(img_bgr)
                binary = ImageProcessor.preprocess_binary(aligned, blur_ksize=int(blur_k), block_size=int(block_size), C=int(C))

                sid, _ = engine.extract_id(binary)
                sid_z = sid.zfill(st.session_state.template.id_digits) if sid.isdigit() else sid
                name = roster_dict.get(sid_z, "غير موجود")

                score, total, pct, _ = engine.grade_one(binary, answer_key, strict=strict)

                results.append({
                    "file": f.name,
                    "page": page_idx,
                    "student_code": sid_z,
                    "student_name": name,
                    "score": score,
                    "total": total,
                    "percentage": round(pct, 2),
                    "passed": "ناجح" if pct >= 50 else "راسب"
                })

        df_res = pd.DataFrame(results)
        st.success("✅ اكتمل التصحيح")
        st.dataframe(df_res, use_container_width=True, height=420)

        buf = io.BytesIO()
        df_res.to_excel(buf, index=False, engine="openpyxl")
        st.download_button(
            "⬇️ تحميل النتائج Excel",
            data=buf.getvalue(),
            file_name="omr_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )


if __name__ == "__main__":
    main()
