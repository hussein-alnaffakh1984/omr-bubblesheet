# ============================================================
# OMR BUBBLE SHEET SCANNER - DEBUG / VERIFY EVERY STEP
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
    def preprocess_binary(img_bgr: np.ndarray,
                          blur_ksize: int = 3,
                          block_size: int = 21,
                          C: int = 6) -> np.ndarray:
        """
        returns binary image (white=ink) using THRESH_BINARY_INV
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if blur_ksize and blur_ksize > 0:
            gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

        # adaptive threshold
        block_size = block_size if block_size % 2 == 1 else block_size + 1
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size, C
        )
        return binary

    @staticmethod
    def resize_to_template(img_bgr: np.ndarray, w: int, h: int) -> np.ndarray:
        return cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def align_to_template_warp(img_bgr: np.ndarray, target_w: int, target_h: int) -> Tuple[np.ndarray, bool]:
        """
        Detect paper boundary as quadrilateral then warp to template size.
        Returns (warped, ok).
        For scanner: often helps if there is slight shift/crop.
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

        # binary is 0/255: count "ink" as >0
        return float(np.sum(inner > 0) / inner.size)

    def detect_answer(self, cells: List[np.ndarray], choices: List[str]) -> Dict:
        fills = [self.calculate_fill(c) for c in cells]
        order = sorted(range(len(fills)), key=lambda i: fills[i], reverse=True)

        top = order[0]
        top_fill = fills[top]
        second_fill = fills[order[1]] if len(order) > 1 else 0.0

        if top_fill < self.min_fill:
            return {"answer": "?", "status": "BLANK", "fills": fills}

        # double-mark condition
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

    def extract_id(self, binary: np.ndarray) -> Tuple[str, Dict]:
        """
        Returns (id_string, debug_info)
        """
        dbg = {"ok": False, "reason": "", "digits": [], "fills_table": None}
        if not self.template.id_block:
            dbg["reason"] = "NO_ID_BLOCK"
            return "", dbg

        roi = self._safe_roi(binary, self.template.id_block)
        if roi is None:
            dbg["reason"] = "ID_OUT_OF_BOUNDS"
            return "OUT_OF_BOUNDS", dbg

        rows = int(self.template.id_rows)
        cols = int(self.template.id_digits)

        cell_h = max(1, self.template.id_block.height // rows)
        cell_w = max(1, self.template.id_block.width // cols)

        digits = []
        fills_rows = []  # for dataframe

        for col in range(cols):
            col_cells = []
            for row in range(rows):
                y1, y2 = row * cell_h, (row + 1) * cell_h
                x1, x2 = col * cell_w, (col + 1) * cell_w
                col_cells.append(roi[y1:y2, x1:x2])

            res = self.detector.detect_answer(col_cells, [str(i) for i in range(10)])
            # For ID: if not OK, write X (strict)
            digit = res["answer"] if res["status"] == "OK" else "X"
            digits.append(digit)

            fills_rows.append({
                "digit_col": col + 1,
                "status": res["status"],
                "picked": res["answer"],
                **{f"r{r}": round(res["fills"][r], 3) for r in range(min(10, len(res["fills"])))}
            })

        out = "".join(digits)
        dbg["ok"] = True
        dbg["digits"] = digits
        dbg["fills_table"] = pd.DataFrame(fills_rows)
        return out, dbg

    def extract_answers_block(self, binary: np.ndarray, block: QuestionBlock) -> Tuple[Dict[int, Dict], pd.DataFrame]:
        rect = block.rect
        roi = self._safe_roi(binary, rect)
        if roi is None:
            return {}, pd.DataFrame()

        rows = int(block.num_rows)
        cols = int(self.template.num_choices)

        cell_h = max(1, rect.height // rows)
        cell_w = max(1, rect.width // cols)

        choices = list("ABCDEFGH"[:cols])
        answers = {}
        debug_rows = []

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

            debug_rows.append({
                "q": q,
                "status": res["status"],
                "answer": res["answer"],
                **{choices[i]: round(res["fills"][i], 3) for i in range(len(choices))}
            })
            q += 1

        return answers, pd.DataFrame(debug_rows)

    def grade_one(self, binary: np.ndarray, answer_key: Dict[int, str], strict: bool) -> Dict:
        # Extract all answers
        all_answers = {}
        debug_tables = []
        for b in self.template.q_blocks:
            ans, dbg = self.extract_answers_block(binary, b)
            all_answers.update(ans)
            if not dbg.empty:
                dbg.insert(0, "block", f"{b.start_q}-{b.end_q}")
                debug_tables.append(dbg)

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

        return {
            "score": correct,
            "total": total,
            "percentage": pct,
            "per_question": pd.DataFrame(per_q),
            "debug_answers_tables": debug_tables
        }


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


def show_step(title: str, body: str):
    st.markdown(f"### {title}")
    st.info(body)


def make_rect_from_points(x1, y1, x2, y2) -> Optional[Rectangle]:
    x = int(min(x1, x2))
    y = int(min(y1, y2))
    w = int(abs(x2 - x1))
    h = int(abs(y2 - y1))
    if w < 10 or h < 10:
        return None
    return Rectangle(x, y, w, h)


# ============================================================
# MAIN APP
# ============================================================

def main():
    st.set_page_config(page_title="OMR Debug Scanner", layout="wide")

    st.title("✅ OMR Scanner — Debug خطوة بخطوة")
    st.caption("نسخة تحقق/تشخيص: نعرض ناتج كل إجراء قبل المتابعة.")

    # Session state
    if "template" not in st.session_state:
        st.session_state.template = None
    if "template_img" not in st.session_state:
        st.session_state.template_img = None

    # ------------------------------------------------------------
    # STEP 1: Upload Template
    # ------------------------------------------------------------
    st.markdown("---")
    show_step("1) رفع نموذج البابل شيت (Template)",
              "ارفع ملف النموذج (PDF أو صورة). سنستخدمه لتحديد مناطق ID وBlocks بدقة.")

    template_file = st.file_uploader("Template", type=["pdf", "png", "jpg", "jpeg"], key="template")
    if template_file:
        img = ImageProcessor.load_first_page(template_file.getvalue(), template_file.name, dpi=250)
        if img:
            st.session_state.template_img = img
            w, h = img.size
            if st.session_state.template is None:
                st.session_state.template = Template(w, h)
            else:
                st.session_state.template.width = w
                st.session_state.template.height = h

            st.success(f"✅ Template جاهز: {w}×{h}")

    if not st.session_state.template_img:
        st.stop()

    # ------------------------------------------------------------
    # STEP 2: Settings + Detector
    # ------------------------------------------------------------
    st.markdown("---")
    show_step("2) إعدادات الكشف (Detector)",
              "هذه الإعدادات تؤثر على اكتشاف التظليل. للسكنر عادة min_fill = 0.08–0.12 ممتاز.")

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

    st.session_state.template.num_choices = int(num_choices)
    st.session_state.template.id_digits = int(id_digits)
    st.session_state.template.id_rows = int(id_rows)

    detector = BubbleDetector(min_fill=min_fill, margin=margin)

    # Preprocess tunings
    st.markdown("#### إعدادات التحويل إلى Binary (Threshold)")
    p1, p2, p3 = st.columns(3)
    with p1:
        blur_k = st.selectbox("Gaussian blur", [0, 3, 5], index=1)
    with p2:
        block_size = st.selectbox("Adaptive block size", [15, 21, 25, 31], index=1)
    with p3:
        C = st.selectbox("Adaptive C", [2, 4, 6, 8, 10], index=2)

    # ------------------------------------------------------------
    # STEP 3: Define Regions (Coordinates)
    # ------------------------------------------------------------
    st.markdown("---")
    show_step("3) تحديد المناطق (ID + Blocks)",
              "حدد المناطق بإحداثيات (x1,y1) و (x2,y2). بعد الحفظ سنعرض Preview للتأكد.")

    preview = draw_preview(st.session_state.template_img, st.session_state.template)
    st.image(preview, caption="Template Preview", use_container_width=True)

    mode = st.radio("نوع المنطقة", ["ID", "Q_BLOCK"], horizontal=True)

    if mode == "Q_BLOCK":
        a, b, c = st.columns(3)
        with a:
            start_q = st.number_input("من سؤال", 1, 500, 1)
        with b:
            end_q = st.number_input("إلى سؤال", 1, 500, 20)
        with c:
            num_rows = st.number_input("عدد الصفوف", 1, 200, 20)
    else:
        start_q = end_q = num_rows = 0

    st.markdown("**ادخل الإحداثيات:**")
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
                qb = QuestionBlock(rect=rect, start_q=int(start_q), end_q=int(end_q), num_rows=int(num_rows))
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
                st.success(f"Block {i}: Q{b.start_q}-{b.end_q} | ({r.x},{r.y}) → ({r.x2},{r.y2})")
            with colB:
                if st.button("🗑️ حذف", key=f"del_{i}"):
                    st.session_state.template.q_blocks.pop(i - 1)
                    st.rerun()

    # ------------------------------------------------------------
    # STEP 4: Upload roster + key + student sheets
    # ------------------------------------------------------------
    st.markdown("---")
    show_step("4) رفع الملفات (Roster + Key + Sheets)",
              "رفع قائمة الطلاب + نموذج الإجابات + أوراق الطلاب. (PDF أو صور).")

    c1, c2, c3 = st.columns(3)
    with c1:
        roster_file = st.file_uploader("📋 Roster (xlsx/csv)", type=["xlsx", "xls", "csv"], key="roster")
    with c2:
        key_file = st.file_uploader("🔑 Answer Key (pdf/jpg/png)", type=["pdf", "png", "jpg", "jpeg"], key="key")
    with c3:
        sheets_files = st.file_uploader("📚 Student Sheets (pdf/images) - multiple",
                                        type=["pdf", "png", "jpg", "jpeg"],
                                        accept_multiple_files=True,
                                        key="sheets")

    strict = st.checkbox("وضع صارم: BLANK/DOUBLE لا تُحسب", value=False)

    # Alignment choice
    st.markdown("#### محاذاة السكنر")
    use_warp = st.checkbox("استخدم Warp (Perspective) قبل القصّ", value=True)
    st.caption("لو الـROI طالع بمكان غلط، فعل Warp. للسكنر غالبًا يفيد إذا يوجد قص/إزاحة بسيطة.")

    # ------------------------------------------------------------
    # STEP 5: VERIFY PIPELINE ON KEY & ONE SHEET
    # ------------------------------------------------------------
    st.markdown("---")
    show_step("5) تحقق خطوة بخطوة قبل التصحيح النهائي",
              "سنطبق المعالجة ونعرض (Aligned → Binary → ID ROI → Q ROI → fills) حتى تتأكد أنها صحيحة.")

    # Validate minimum
    if not (st.session_state.template.id_block and st.session_state.template.q_blocks):
        st.warning("لازم تحدد ID وBlock واحد على الأقل قبل الفحص.")
        st.stop()

    engine = GradingEngine(st.session_state.template, detector)

    def align(img_bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
        if use_warp:
            warped, ok = ImageProcessor.align_to_template_warp(img_bgr, st.session_state.template.width, st.session_state.template.height)
            return warped, ok
        return ImageProcessor.resize_to_template(img_bgr, st.session_state.template.width, st.session_state.template.height), True

    # --- Verify KEY
    if key_file:
        st.markdown("### 🔎 فحص نموذج الإجابة (Key)")
        key_img = ImageProcessor.load_first_page(key_file.getvalue(), key_file.name, dpi=250)
        if key_img:
            key_bgr = ImageProcessor.pil_to_bgr(key_img)
            key_aligned, ok_warp = align(key_bgr)
            key_binary = ImageProcessor.preprocess_binary(key_aligned, blur_ksize=blur_k, block_size=block_size, C=C)

            colA, colB = st.columns(2)
            with colA:
                st.image(ImageProcessor.bgr_to_pil(key_aligned), caption=f"Aligned (warp_ok={ok_warp})", use_container_width=True)
            with colB:
                st.image(key_binary, caption="Binary", clamp=True, use_container_width=True)

            # Show ROIs
            r = st.session_state.template.id_block
            roi_id = key_binary[r.y:r.y2, r.x:r.x2]
            st.image(roi_id, caption="ID ROI (Key)", clamp=True, use_container_width=True)

            b0 = st.session_state.template.q_blocks[0].rect
            roi_q = key_binary[b0.y:b0.y2, b0.x:b0.x2]
            st.image(roi_q, caption="Q Block ROI (Key)", clamp=True, use_container_width=True)

            # Extract key answers (debug table)
            answer_key = {}
            key_debug_tables = []
            for b in st.session_state.template.q_blocks:
                ans, dbg = engine.extract_answers_block(key_binary, b)
                key_debug_tables.append(dbg)
                for q, res in ans.items():
                    if res["status"] == "OK":
                        answer_key[q] = res["answer"]

            st.success(f"✅ استخرجت {len(answer_key)} إجابة صحيحة من Key")
            with st.expander("عرض جدول fills لنموذج الإجابة (Key)"):
                for i, dbg in enumerate(key_debug_tables, 1):
                    st.write(f"Block {i}")
                    st.dataframe(dbg, use_container_width=True)

            st.session_state["answer_key"] = answer_key
        else:
            st.error("فشل تحميل Key.")

    # --- Load roster (verify)
    roster_dict = {}
    if roster_file:
        st.markdown("### 🔎 فحص قائمة الطلاب (Roster)")
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
                st.dataframe(df.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"❌ خطأ roster: {e}")

    # --- Verify on first student sheet (one sample)
    if sheets_files:
        st.markdown("### 🔎 فحص ورقة طالب (Sample) قبل التصحيح")
        sample = sheets_files[0]
        pages = ImageProcessor.load_all_pages(sample.getvalue(), sample.name, dpi=250)
        if pages:
            st.write(f"الملف: {sample.name} | عدد الصفحات: {len(pages)} (نستخدم أول صفحة للفحص)")

            stud_bgr = ImageProcessor.pil_to_bgr(pages[0])
            stud_aligned, ok_warp = align(stud_bgr)
            stud_binary = ImageProcessor.preprocess_binary(stud_aligned, blur_ksize=blur_k, block_size=block_size, C=C)

            colA, colB = st.columns(2)
            with colA:
                st.image(ImageProcessor.bgr_to_pil(stud_aligned), caption=f"Aligned (warp_ok={ok_warp})", use_container_width=True)
            with colB:
                st.image(stud_binary, caption="Binary", clamp=True, use_container_width=True)

            # ROIs
            r = st.session_state.template.id_block
            roi_id = stud_binary[r.y:r.y2, r.x:r.x2]
            st.image(roi_id, caption="ID ROI (Student)", clamp=True, use_container_width=True)

            b0 = st.session_state.template.q_blocks[0].rect
            roi_q = stud_binary[b0.y:b0.y2, b0.x:b0.x2]
            st.image(roi_q, caption="Q Block ROI (Student)", clamp=True, use_container_width=True)

            # Extract ID with fills table
            sid, id_dbg = engine.extract_id(stud_binary)
            sid_z = sid.zfill(st.session_state.template.id_digits) if sid.isdigit() else sid
            st.success(f"🆔 ID المستخرج: {sid_z}")

            if isinstance(id_dbg.get("fills_table"), pd.DataFrame) and not id_dbg["fills_table"].empty:
                with st.expander("تفاصيل ID (fills لكل عمود)"):
                    st.dataframe(id_dbg["fills_table"], use_container_width=True)

            # Extract answers debug
            for i, b in enumerate(st.session_state.template.q_blocks, 1):
                _, dbg = engine.extract_answers_block(stud_binary, b)
                with st.expander(f"تفاصيل Block {i} (Q{b.start_q}-{b.end_q})"):
                    st.dataframe(dbg, use_container_width=True)

            # Name matching
            if roster_dict and sid_z in roster_dict:
                st.success(f"✅ الاسم مطابق في roster: {roster_dict[sid_z]}")
            elif roster_dict:
                st.warning("⚠️ لم يتم إيجاد الاسم في roster (تأكد من zfill وعدد خانات الكود)")
        else:
            st.error("لا يمكن قراءة ورقة الطالب.")

    # ------------------------------------------------------------
    # STEP 6: FINAL GRADING (after user confirms)
    # ------------------------------------------------------------
    st.markdown("---")
    show_step("6) التصحيح النهائي (بعد ما تتأكد أن كل شيء صحيح)",
              "إذا نتائج الفحص أعلاه صحيحة (ROI صحيح + ID صحيح + Key صحيح) اضغط تصحيح.")

    can_grade = bool(st.session_state.get("answer_key")) and bool(roster_dict) and bool(sheets_files)
    if not can_grade:
        st.warning("لا يمكن التصحيح: تأكد من رفع Key + Roster + Sheets، وأن Key استُخرج بنجاح.")
        st.stop()

    if st.button("🚀 ابدأ التصحيح لكل الأوراق", type="primary", use_container_width=True):
        answer_key = st.session_state["answer_key"]
        results = []

        for f in sheets_files:
            pages = ImageProcessor.load_all_pages(f.getvalue(), f.name, dpi=250)
            for page_idx, pil_page in enumerate(pages, 1):
                img_bgr = ImageProcessor.pil_to_bgr(pil_page)
                aligned, ok_warp = align(img_bgr)
                binary = ImageProcessor.preprocess_binary(aligned, blur_ksize=blur_k, block_size=block_size, C=C)

                sid, _ = engine.extract_id(binary)
                sid_z = sid.zfill(st.session_state.template.id_digits) if sid.isdigit() else sid
                name = roster_dict.get(sid_z, "غير موجود")

                g = engine.grade_one(binary, answer_key, strict=strict)

                results.append({
                    "file": f.name,
                    "page": page_idx,
                    "student_code": sid_z,
                    "student_name": name,
                    "score": g["score"],
                    "total": g["total"],
                    "percentage": round(g["percentage"], 2),
                    "passed": "ناجح" if g["percentage"] >= 50 else "راسب"
                })

        df_res = pd.DataFrame(results)
        st.success("✅ اكتمل التصحيح")
        st.dataframe(df_res, use_container_width=True, height=350)

        # Export
        buf = io.BytesIO()
        df_res.to_excel(buf, index=False, engine="openpyxl")
        st.download_button(
            "⬇️ تحميل النتائج Excel",
            buf.getvalue(),
            file_name="omr_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )


if __name__ == "__main__":
    main()
