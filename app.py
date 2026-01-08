# ============================================================
# ✅ OMR Bubble Sheet (Streamlit)
# Hybrid rule-based + Smart Auto-Template (No manual ROI)
# FIXES:
# 1) Stable ID grid detection (4 cols x 10 rows) from FAR RIGHT
# 2) Prevent mixing ID cols with question cols
# 3) Disable X detection on Answer Key (letters A/B/C/D false-X fixed)
# 4) For students: X cancels only when bubble is NOT filled (X on empty bubble)
# 5) Always show Answer Key table with Status (no empty confusion)
# ============================================================

import io
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes


# ----------------------------
# Utilities
# ----------------------------

def read_uploaded_bytes(up):
    if up is None:
        return None
    if hasattr(up, "getvalue"):
        return up.getvalue()
    return up.read()

def pil_from_bytes(file_bytes: bytes, filename: str, dpi: int = 300) -> List[Image.Image]:
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        pages = convert_from_bytes(file_bytes, dpi=dpi)
        return [p.convert("RGB") for p in pages] if pages else []
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return [img]

def to_bgr(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def resize_max(img: np.ndarray, max_side: int = 2400) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / m
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def gray_blur(img_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    return g

def adaptive_bin_inv(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 7
    )


# ----------------------------
# Data classes
# ----------------------------

@dataclass
class Bubble:
    x: float
    y: float
    r: float
    fill: float = 0.0
    x_score: float = 0.0
    cancelled: bool = False

@dataclass
class AutoTemplate:
    id_cols_x: List[float]      # 4
    id_rows_y: List[float]      # 10
    q_cols_x: List[float]       # 2/4/5
    q_rows_y: List[float]       # N
    choice_count: int
    q_count: int
    debug: Dict


# ----------------------------
# Engine
# ----------------------------

class OMRSmartEngine:

    def __init__(
        self,
        blank_fill_thr: float = 0.14,
        double_gap_thr: float = 0.03,
        x_hough_min_votes: int = 12,
        x_score_thr_student: float = 2.5,
        bubble_margin_ratio: float = 0.22
    ):
        self.blank_fill_thr = float(blank_fill_thr)
        self.double_gap_thr = float(double_gap_thr)
        self.x_hough_min_votes = int(x_hough_min_votes)
        self.x_score_thr_student = float(x_score_thr_student)
        self.bubble_margin_ratio = float(bubble_margin_ratio)

    # ---------- bubble detection ----------

    def detect_bubbles(self, img_bgr: np.ndarray) -> Tuple[List[Bubble], Dict]:
        dbg = {}
        img = resize_max(img_bgr, 2400)
        g = gray_blur(img)
        b = adaptive_bin_inv(g)

        # open to remove small noise
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, k, iterations=1)

        contours, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = b.shape[:2]

        min_area = max(80, int(0.00002 * (h * w)))
        max_area = int(0.0035 * (h * w))

        tmp = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue
            per = cv2.arcLength(c, True)
            if per <= 0:
                continue
            circ = 4 * math.pi * (area / (per * per + 1e-9))
            # stricter circularity
            if circ < 0.45:
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            if r < 7:
                continue
            tmp.append((float(x), float(y), float(r), float(circ)))

        if not tmp:
            dbg["bubbles_found_filtered"] = 0
            dbg["img_shape"] = (h, w)
            return [], dbg

        rs = np.array([t[2] for t in tmp], dtype=float)
        r_med = float(np.median(rs))
        r_lo = 0.70 * r_med
        r_hi = 1.45 * r_med

        bubbles = [Bubble(x=x, y=y, r=r) for (x, y, r, _) in tmp if r_lo <= r <= r_hi]

        dbg.update({
            "img_shape": (h, w),
            "min_area": min_area,
            "max_area": max_area,
            "r_med": r_med,
            "r_band": (r_lo, r_hi),
            "bubbles_found_raw": len(tmp),
            "bubbles_found_filtered": len(bubbles),
        })
        return bubbles, dbg

    # ---------- clustering ----------

    @staticmethod
    def _cluster_1d(values: np.ndarray, eps: float) -> List[float]:
        if len(values) == 0:
            return []
        v = np.sort(values.astype(float))
        groups = [[v[0]]]
        for x in v[1:]:
            if abs(x - groups[-1][-1]) <= eps:
                groups[-1].append(x)
            else:
                groups.append([x])
        return [float(np.mean(g)) for g in groups]

    # ---------- template inference ----------

    def infer_template(self, bubbles: List[Bubble], img_shape: Tuple[int, int]) -> AutoTemplate:
        h, w = img_shape
        if len(bubbles) < 40:
            raise ValueError("عدد الفقاعات المكتشفة قليل جداً. ارفع DPI أو تأكد وضوح الصورة.")

        xs = np.array([b.x for b in bubbles], dtype=float)
        ys = np.array([b.y for b in bubbles], dtype=float)
        rs = np.array([b.r for b in bubbles], dtype=float)

        r_med = float(np.median(rs))
        eps_x = max(10.0, r_med * 2.2)
        eps_y = max(10.0, r_med * 2.2)

        col_centers = sorted(self._cluster_1d(xs, eps=eps_x))
        if len(col_centers) < 6:
            raise ValueError("لم أستطع تكوين أعمدة كافية. قد تكون الصفحة مقصوصة أو DPI منخفض.")

        debug = {
            "r_med": r_med, "eps_x": eps_x, "eps_y": eps_y,
            "col_centers_total": len(col_centers),
            "bubbles_total": len(bubbles),
        }

        # ---------------------------
        # 1) Detect ID columns (FAR RIGHT)
        # ---------------------------
        right_bands = [0.72, 0.66, 0.60, 0.55, 0.50]
        right_cols = []
        used_rb = None
        for rb in right_bands:
            cand = [c for c in col_centers if c >= w * rb]
            if len(cand) >= 4:
                right_cols = cand[-10:]
                used_rb = rb
                break
        if len(right_cols) < 4:
            raise ValueError("فشل تحديد منطقة كود الطالب يمين الصفحة. تأكد الصفحة كاملة.")

        debug["right_band_used"] = used_rb
        debug["right_cols"] = right_cols

        from itertools import combinations

        def score_id(cols4: List[float]) -> Tuple[float, List[float]]:
            # bubbles near these columns
            idx = []
            for i, b in enumerate(bubbles):
                if min(abs(b.x - c) for c in cols4) <= eps_x:
                    idx.append(i)
            if len(idx) < 25:
                return -1e9, []

            y_local = np.array([bubbles[i].y for i in idx], dtype=float)
            y_centers = sorted(self._cluster_1d(y_local, eps=eps_y))

            # around 10 rows
            rc = len(y_centers)
            if rc < 8 or rc > 12:
                return -1e9, y_centers

            diffs = np.diff(y_centers) if len(y_centers) >= 2 else np.array([9999.0])
            cv = float(np.std(diffs) / (np.mean(diffs) + 1e-9))

            # prefer tight width (4 columns close)
            width = max(cols4) - min(cols4)
            score = -abs(rc - 10) * 6.0 - cv * 10.0 - width * 0.02 + len(idx) * 0.05
            return score, y_centers

        best_id = None
        best_score = -1e9
        for comb in combinations(right_cols, 4):
            cols4 = sorted(list(comb))
            s, y_centers = score_id(cols4)
            if s > best_score:
                best_score = s
                best_id = (cols4, y_centers)

        if best_id is None:
            raise ValueError("فشل اكتشاف كود الطالب (4×10).")

        id_cols_x, id_rows_y = best_id
        if len(id_rows_y) > 10:
            id_rows_y = id_rows_y[:10]

        debug["id_cols_x"] = id_cols_x
        debug["id_rows_detected"] = len(id_rows_y)

        # ---------------------------
        # 2) Build question candidates by removing bubbles near ID columns (X only)
        # ---------------------------
        def near_any(val, centers, eps):
            return (min(abs(val - c) for c in centers) <= eps) if centers else False

        q_candidate = [b for b in bubbles if not near_any(b.x, id_cols_x, eps_x)]
        debug["bubbles_q_candidate"] = len(q_candidate)

        if len(q_candidate) < 40:
            raise ValueError("لم يتبقّ فقاعات كافية للأسئلة بعد فصل الكود. الكود مكتشف خطأ.")

        # ---------------------------
        # 3) Detect question columns (2/4/5) LEFT of ID group
        # ---------------------------
        q_xs = np.array([b.x for b in q_candidate], dtype=float)
        q_cols_all = sorted(self._cluster_1d(q_xs, eps=eps_x))

        id_left_edge = min(id_cols_x)
        q_cols_filtered = [c for c in q_cols_all if c < (id_left_edge - eps_x)]
        if len(q_cols_filtered) < 2:
            q_cols_filtered = [c for c in q_cols_all if c < w * 0.70]
        if len(q_cols_filtered) < 2:
            q_cols_filtered = q_cols_all

        q_cols_filtered = q_cols_filtered[:14]
        debug["q_cols_filtered_count"] = len(q_cols_filtered)

        def score_questions(cols_k: List[float]) -> Tuple[float, List[float]]:
            idx = []
            for i, b in enumerate(q_candidate):
                if min(abs(b.x - c) for c in cols_k) <= eps_x:
                    idx.append(i)
            if len(idx) < 10 * len(cols_k):
                return -1e9, []

            y_local = np.array([q_candidate[i].y for i in idx], dtype=float)
            y_centers = sorted(self._cluster_1d(y_local, eps=eps_y))
            if len(y_centers) < 5:
                return -1e9, y_centers

            diffs = np.diff(y_centers) if len(y_centers) >= 2 else np.array([9999.0])
            cv = float(np.std(diffs) / (np.mean(diffs) + 1e-9))
            density = len(idx) / (len(cols_k) * max(len(y_centers), 1))

            score = -cv * 10.0 + density * 4.0 + len(y_centers) * 0.12 + len(idx) * 0.01
            return score, y_centers

        best_q = None
        best_q_score = -1e9
        for k in [2, 4, 5]:
            if len(q_cols_filtered) < k:
                continue
            for comb in combinations(q_cols_filtered, k):
                cols_k = sorted(list(comb))
                s, y_centers = score_questions(cols_k)
                if s > best_q_score:
                    best_q_score = s
                    best_q = (cols_k, y_centers, k)

        if best_q is None:
            raise ValueError("فشل اكتشاف شبكة الأسئلة. جرّب DPI أعلى أو ملف أوضح.")

        q_cols_x, q_rows_y, choice_count = best_q
        q_count = len(q_rows_y)

        debug["q_cols_x"] = q_cols_x
        debug["q_rows_detected"] = q_count
        debug["choice_count"] = choice_count

        return AutoTemplate(
            id_cols_x=id_cols_x,
            id_rows_y=id_rows_y,
            q_cols_x=q_cols_x,
            q_rows_y=q_rows_y,
            choice_count=choice_count,
            q_count=q_count,
            debug=debug
        )

    # ---------- scoring ----------

    def _extract_roi(self, img: np.ndarray, cx: float, cy: float, r: float) -> np.ndarray:
        h, w = img.shape[:2]
        rr = max(8, int(r * 1.15))
        x1 = max(0, int(cx - rr))
        y1 = max(0, int(cy - rr))
        x2 = min(w, int(cx + rr))
        y2 = min(h, int(cy + rr))
        return img[y1:y2, x1:x2]

    def fill_ratio(self, cell_bin: np.ndarray) -> float:
        if cell_bin.size == 0:
            return 0.0
        h, w = cell_bin.shape[:2]
        mh = int(h * self.bubble_margin_ratio)
        mw = int(w * self.bubble_margin_ratio)
        inner = cell_bin if (h - 2 * mh <= 0 or w - 2 * mw <= 0) else cell_bin[mh:h-mh, mw:w-mw]
        return float(np.mean(inner > 0))

    def x_cancellation_score(self, cell_gray: np.ndarray) -> float:
        if cell_gray.size == 0:
            return 0.0
        edges = cv2.Canny(cell_gray, 60, 140)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=self.x_hough_min_votes,
            minLineLength=max(8, int(min(cell_gray.shape[:2]) * 0.28)),
            maxLineGap=6
        )
        if lines is None:
            return 0.0

        pos = 0
        neg = 0
        for l in lines[:, 0, :]:
            x1, y1, x2, y2 = map(int, l)
            dx = x2 - x1
            dy = y2 - y1
            ang = math.degrees(math.atan2(dy, dx + 1e-9))
            ang = (ang + 180) % 180
            if 25 <= ang <= 65:
                pos += 1
            if 115 <= ang <= 155:
                neg += 1

        if pos > 0 and neg > 0:
            return float(min(pos, 3) + min(neg, 3))  # 2..6
        return 0.0

    def evaluate_cells(self, img_bgr: np.ndarray, template: AutoTemplate, for_key: bool) -> Dict[Tuple[int, int], Bubble]:
        img = resize_max(img_bgr, 2400)
        g = gray_blur(img)
        b = adaptive_bin_inv(g)

        out: Dict[Tuple[int, int], Bubble] = {}

        # radius guess from row spacing
        id_r = max(10.0, float(np.median(np.diff(sorted(template.id_rows_y))) * 0.32)) if len(template.id_rows_y) >= 2 else 14.0
        q_r  = max(10.0, float(np.median(np.diff(sorted(template.q_rows_y))) * 0.32)) if len(template.q_rows_y) >= 2 else 14.0

        def eval_cell(cx, cy, r_guess):
            cell_bin = self._extract_roi(b, cx, cy, r_guess)
            fill = self.fill_ratio(cell_bin)

            if for_key:
                return fill, 0.0, False  # ✅ disable X on key

            cell_gray = self._extract_roi(g, cx, cy, r_guess)
            xs = self.x_cancellation_score(cell_gray)
            # ✅ cancel only if X on empty-ish bubble
            cancelled = (xs >= self.x_score_thr_student) and (fill < (self.blank_fill_thr * 0.90))
            return fill, xs, cancelled

        # ID (4x10)
        for c in range(4):
            for r in range(10):
                cx = template.id_cols_x[c]
                cy = template.id_rows_y[r]
                fill, xs, cancelled = eval_cell(cx, cy, id_r)
                out[(100 + c, r)] = Bubble(cx, cy, id_r, fill, xs, cancelled)

        # Questions (N x choices)
        for rr in range(template.q_count):
            cy = template.q_rows_y[rr]
            for cc in range(template.choice_count):
                cx = template.q_cols_x[cc]
                fill, xs, cancelled = eval_cell(cx, cy, q_r)
                out[(rr, cc)] = Bubble(cx, cy, q_r, fill, xs, cancelled)

        return out

    # ---------- decision ----------

    def decide_row(self, bubbles_row: List[Bubble]) -> Dict:
        fills = np.array([b.fill for b in bubbles_row], dtype=float)
        canc  = np.array([1 if b.cancelled else 0 for b in bubbles_row], dtype=int)

        idx = [i for i in range(len(bubbles_row)) if canc[i] == 0]
        if not idx:
            return {"answer_idx": None, "status": "BLANK", "fills": fills.tolist()}

        sub = [(i, fills[i]) for i in idx]
        sub.sort(key=lambda t: t[1], reverse=True)
        top_i, top_f = sub[0]
        second_f = sub[1][1] if len(sub) > 1 else 0.0

        if top_f < self.blank_fill_thr:
            return {"answer_idx": None, "status": "BLANK", "fills": fills.tolist()}
        if second_f >= self.blank_fill_thr and (top_f - second_f) <= self.double_gap_thr:
            return {"answer_idx": None, "status": "DOUBLE", "fills": fills.tolist()}

        return {"answer_idx": top_i, "status": "OK", "fills": fills.tolist()}

    def decode_student_id(self, cell_map: Dict[Tuple[int, int], Bubble]) -> str:
        digits = []
        for col in range(4):
            fills = []
            for row in range(10):
                fills.append(cell_map[(100 + col, row)].fill)
            best_row = int(np.argmax(fills))
            best_fill = float(fills[best_row])
            digits.append("X" if best_fill < self.blank_fill_thr else str(best_row))
        return "".join(digits)

    def extract_answers(self, cell_map: Dict[Tuple[int, int], Bubble], template: AutoTemplate) -> Tuple[Dict[int, str], pd.DataFrame]:
        choices = "ABCDE"[:template.choice_count]
        answers = {}
        rows = []
        for r in range(template.q_count):
            row_bubs = [cell_map[(r, c)] for c in range(template.choice_count)]
            d = self.decide_row(row_bubs)
            ans = ""
            if d["status"] == "OK":
                ans = choices[d["answer_idx"]]
                answers[r + 1] = ans
            rows.append({
                "Q": r + 1,
                "Answer": ans,
                "Status": d["status"],
                "Cancelled": [bool(b.cancelled) for b in row_bubs],
                "Fills": d["fills"],
            })
        return answers, pd.DataFrame(rows)[["Q", "Answer", "Status"]]

    def draw_overlay(self, img_bgr: np.ndarray, template: AutoTemplate, cell_map: Dict[Tuple[int, int], Bubble]) -> np.ndarray:
        img = resize_max(img_bgr, 2400).copy()

        # column lines (blue)
        for x in template.id_cols_x:
            cv2.line(img, (int(x), 0), (int(x), img.shape[0]-1), (255, 0, 0), 2)
        for x in template.q_cols_x:
            cv2.line(img, (int(x), 0), (int(x), img.shape[0]-1), (255, 0, 0), 2)

        def draw_b(b: Bubble, color):
            cv2.circle(img, (int(b.x), int(b.y)), int(b.r), color, 2)
            cv2.circle(img, (int(b.x), int(b.y)), 3, color, -1)

        # ID bubbles green
        for c in range(4):
            for r in range(10):
                draw_b(cell_map[(100+c, r)], (0, 255, 0))

        # Question bubbles:
        # red filled, yellow cancelled, green blank
        for r in range(template.q_count):
            for c in range(template.choice_count):
                b = cell_map[(r, c)]
                if b.cancelled:
                    col = (0, 255, 255)
                elif b.fill >= self.blank_fill_thr:
                    col = (0, 0, 255)
                else:
                    col = (0, 255, 0)
                draw_b(b, col)

        return img


# ----------------------------
# Roster loader
# ----------------------------

def load_roster(roster_up) -> Dict[str, str]:
    b = read_uploaded_bytes(roster_up)
    if b is None:
        return {}
    name = roster_up.name.lower()
    df = pd.read_csv(io.BytesIO(b)) if name.endswith(".csv") else pd.read_excel(io.BytesIO(b))
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "student_code" not in df.columns or "student_name" not in df.columns:
        raise ValueError("ملف قائمة الطلاب يجب أن يحتوي عمودين: student_code و student_name")
    return dict(zip(df["student_code"].astype(str).str.strip(), df["student_name"].astype(str).str.strip()))


# ----------------------------
# Streamlit app
# ----------------------------

def main():
    st.set_page_config(page_title="OMR Bubble Sheet", layout="wide")
    st.title("✅ OMR Bubble Sheet — تدريب تلقائي + تصحيح")
    st.caption("إصلاح خلط كود الطالب مع الأسئلة + منع X الكاذب في Answer Key")

    c1, c2, c3 = st.columns(3)
    with c1:
        dpi = st.selectbox("DPI للـ PDF", [200, 250, 300, 350], index=2)
    with c2:
        debug = st.checkbox("Debug", value=True)
    with c3:
        strict_mode = st.checkbox("وضع صارم (BLANK/DOUBLE = خطأ)", value=True)

    st.divider()

    colA, colB, colC = st.columns(3)
    with colA:
        roster_up = st.file_uploader("📋 ملف الطلاب (Excel/CSV)", type=["xlsx", "xls", "csv"])
    with colB:
        key_up = st.file_uploader("🔑 Answer Key (PDF/صورة)", type=["pdf", "png", "jpg", "jpeg"])
    with colC:
        student_up = st.file_uploader("📚 أوراق الطلاب (PDF/صور)", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)

    st.divider()

    st.subheader("Thresholds")
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        blank_thr = st.slider("Blank fill threshold", 0.05, 0.35, 0.14, 0.01)
    with t2:
        double_gap = st.slider("Double gap threshold", 0.00, 0.15, 0.03, 0.01)
    with t3:
        x_votes = st.slider("X Hough votes", 6, 30, 12, 1)
    with t4:
        x_thr_student = st.slider("X score (student)", 0.5, 6.0, 2.5, 0.1)

    engine = OMRSmartEngine(
        blank_fill_thr=blank_thr,
        double_gap_thr=double_gap,
        x_hough_min_votes=x_votes,
        x_score_thr_student=x_thr_student,
        bubble_margin_ratio=0.22
    )

    # ----------------------------
    # 1) Train on Answer Key
    # ----------------------------
    st.subheader("1) Answer Key (استخراج قبل التصحيح)")
    if key_up is None:
        st.info("ارفع Answer Key للبدء.")
        st.stop()

    key_bytes = read_uploaded_bytes(key_up)
    key_pages = pil_from_bytes(key_bytes, key_up.name, dpi=int(dpi))
    if not key_pages:
        st.error("تعذر قراءة Answer Key.")
        st.stop()

    key_bgr = to_bgr(key_pages[0])
    bubbles, dbg_b = engine.detect_bubbles(key_bgr)

    if debug:
        st.write("Bubble detection debug:", dbg_b)

    try:
        template = engine.infer_template(bubbles, img_shape=resize_max(key_bgr, 2400).shape[:2])
    except Exception as e:
        st.error(f"❌ فشل التدريب/الاستخراج: {e}")
        if debug:
            st.write("Template debug (partial):", dbg_b)
        st.stop()

    if debug:
        st.write("Template debug:", template.debug)

    cell_map_key = engine.evaluate_cells(key_bgr, template, for_key=True)
    answer_key, df_key_view = engine.extract_answers(cell_map_key, template)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("عدد الأسئلة المكتشف", template.q_count)
    with k2:
        st.metric("عدد الخيارات المكتشف", template.choice_count)
    with k3:
        st.metric("أعمدة كود الطالب", 4)
    with k4:
        st.metric("صفوف كود الطالب", 10)

    st.markdown("### 🔑 Answer Key المستخرج (تأكد قبل التصحيح)")
    st.dataframe(df_key_view, width="stretch", height=320)

    if debug:
        st.markdown("### Overlay (أخضر=كود، أحمر=مظلّل، أصفر=ملغي X، أزرق=أعمدة)")
        ov = engine.draw_overlay(key_bgr, template, cell_map_key)
        st.image(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB), use_container_width=True)

    if len(answer_key) == 0:
        st.error("لم يتم استخراج أي إجابة من المفتاح. تأكد أن المفتاح مظلل فعلاً وبـ DPI مناسب.")
        st.stop()

    st.success("✅ تم استخراج Answer Key بنجاح.")
    st.divider()

    # ----------------------------
    # 2) Grading
    # ----------------------------
    st.subheader("2) التصحيح")
    if roster_up is None or not student_up:
        st.info("ارفع ملف الطلاب + أوراق الطلاب لإكمال التصحيح.")
        st.stop()

    try:
        roster = load_roster(roster_up)
        st.success(f"تم تحميل قائمة الطلاب: {len(roster)} طالب")
    except Exception as e:
        st.error(str(e))
        st.stop()

    if st.button("🚀 ابدأ التصحيح الآن", type="primary"):
        results = []
        export_rows = []

        for up in student_up:
            b = read_uploaded_bytes(up)
            pages = pil_from_bytes(b, up.name, dpi=int(dpi))
            if not pages:
                continue

            for pi, page in enumerate(pages, start=1):
                img_bgr = to_bgr(page)

                cell_map_student = engine.evaluate_cells(img_bgr, template, for_key=False)
                sid = engine.decode_student_id(cell_map_student)
                sname = roster.get(sid, "غير موجود")

                s_answers, _ = engine.extract_answers(cell_map_student, template)

                correct = 0
                total = len(answer_key)

                for q, k_ans in answer_key.items():
                    s_ans = s_answers.get(q, None)
                    if s_ans is None:
                        continue
                    if s_ans == k_ans:
                        correct += 1

                percent = (correct / total * 100.0) if total else 0.0
                passed = percent >= 50.0

                results.append({
                    "file": up.name,
                    "page": pi,
                    "student_code": sid,
                    "student_name": sname,
                    "score": correct,
                    "total": total,
                    "percentage": round(percent, 2),
                    "status": "ناجح" if passed else "راسب"
                })

                er = {"file": up.name, "page": pi, "student_code": sid, "student_name": sname}
                for q in range(1, template.q_count + 1):
                    er[f"Q{q}"] = s_answers.get(q, "")
                export_rows.append(er)

                if debug:
                    st.markdown(f"#### Overlay: {up.name} (page {pi})")
                    ov2 = engine.draw_overlay(img_bgr, template, cell_map_student)
                    st.image(cv2.cvtColor(ov2, cv2.COLOR_BGR2RGB), use_container_width=True)

        if not results:
            st.error("لم يتم إنتاج نتائج. تأكد من الملفات و DPI.")
            st.stop()

        df_res = pd.DataFrame(results)
        st.success("✅ اكتمل التصحيح")
        st.dataframe(df_res, width="stretch", height=300)

        st.markdown("### 📥 تحميل النتائج")
        out_xlsx = io.BytesIO()
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            df_res.to_excel(writer, index=False, sheet_name="Summary")
            pd.DataFrame(export_rows).to_excel(writer, index=False, sheet_name="Answers")

        st.download_button(
            "⬇️ تحميل النتائج (Excel)",
            data=out_xlsx.getvalue(),
            file_name="results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


if __name__ == "__main__":
    main()
