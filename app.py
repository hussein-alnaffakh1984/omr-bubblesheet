# ============================================================
# ✅ OMR Bubble Sheet (Streamlit) — FINAL FIX
# Key Training: detects ONLY question grid (no student ID splitting)
# Student Grading: uses learned question grid + tries to read ID (optional)
#
# Fixes:
# 1) Bubble detection improved: Contours + HoughCircles + merge
# 2) Key inference ignores student ID entirely (prevents wrong split)
# 3) ID extraction on students is best-effort (never crashes grading)
# 4) X-cancel only on students and only if bubble is not filled
# ============================================================

import io
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

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

def preprocess_gray(img_bgr: np.ndarray) -> np.ndarray:
    """More robust gray preprocessing."""
    img = resize_max(img_bgr, 2400)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # CLAHE improves contrast for faint circles
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(g)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    return g

def adaptive_bin_inv(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 7
    )

def merge_circles(circles: List[Tuple[float, float, float]], eps: float) -> List[Tuple[float, float, float]]:
    """Merge near-duplicate circles by simple clustering."""
    if not circles:
        return []
    circles = sorted(circles, key=lambda t: (t[0], t[1]))
    used = [False] * len(circles)
    out = []

    for i, (x, y, r) in enumerate(circles):
        if used[i]:
            continue
        grp = [(x, y, r)]
        used[i] = True
        for j in range(i + 1, len(circles)):
            if used[j]:
                continue
            x2, y2, r2 = circles[j]
            if (abs(x2 - x) <= eps) and (abs(y2 - y) <= eps):
                grp.append((x2, y2, r2))
                used[j] = True
        mx = float(np.mean([t[0] for t in grp]))
        my = float(np.mean([t[1] for t in grp]))
        mr = float(np.mean([t[2] for t in grp]))
        out.append((mx, my, mr))
    return out


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
class TemplateQ:
    q_cols_x: List[float]     # 2/4/5
    q_rows_y: List[float]     # N
    choice_count: int
    q_count: int
    debug: Dict

@dataclass
class TemplateFull:
    # Question template always exists
    q: TemplateQ
    # ID template optional (for students)
    id_cols_x: Optional[List[float]] = None
    id_rows_y: Optional[List[float]] = None


# ----------------------------
# Engine
# ----------------------------

class OMRSmartEngine:

    def __init__(
        self,
        blank_fill_thr: float = 0.14,
        double_gap_thr: float = 0.03,
        bubble_margin_ratio: float = 0.22,
        x_hough_min_votes: int = 12,
        x_score_thr_student: float = 2.5,
    ):
        self.blank_fill_thr = float(blank_fill_thr)
        self.double_gap_thr = float(double_gap_thr)
        self.bubble_margin_ratio = float(bubble_margin_ratio)
        self.x_hough_min_votes = int(x_hough_min_votes)
        self.x_score_thr_student = float(x_score_thr_student)

    # ---------- Bubble detection (Contours + HoughCircles) ----------

    def detect_bubbles(self, img_bgr: np.ndarray) -> Tuple[List[Bubble], Dict]:
        img = resize_max(img_bgr, 2400)
        g = preprocess_gray(img)

        b = adaptive_bin_inv(g)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        b_open = cv2.morphologyEx(b, cv2.MORPH_OPEN, k, iterations=1)

        h, w = b_open.shape[:2]
        dbg = {"img_shape": (h, w)}

        # ---- Contour circles ----
        contours, _ = cv2.findContours(b_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = max(70, int(0.000015 * (h * w)))
        max_area = int(0.0040 * (h * w))
        dbg["min_area"] = min_area
        dbg["max_area"] = max_area

        contour_circles = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue
            per = cv2.arcLength(c, True)
            if per <= 0:
                continue
            circularity = 4 * math.pi * (area / (per * per + 1e-9))
            # ✅ relaxed (your previous 0.45 كان قاسي)
            if circularity < 0.30:
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            if r < 7:
                continue
            contour_circles.append((float(x), float(y), float(r)))

        # ---- Hough circles ----
        # estimate radius range from contours if available
        rs = [c[2] for c in contour_circles]
        r_med = float(np.median(rs)) if rs else 20.0
        r_min = max(8, int(r_med * 0.70))
        r_max = int(r_med * 1.60)

        # param2 controls sensitivity (lower => more circles)
        hough = cv2.HoughCircles(
            g,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(10, int(r_med * 1.6)),
            param1=120,
            param2=22,
            minRadius=r_min,
            maxRadius=r_max
        )

        hough_circles = []
        if hough is not None:
            hough = np.squeeze(hough, axis=0)
            for (x, y, r) in hough:
                hough_circles.append((float(x), float(y), float(r)))

        # ---- Merge both ----
        all_circles = contour_circles + hough_circles
        if not all_circles:
            dbg.update({
                "bubbles_found_raw": 0,
                "bubbles_found_filtered": 0,
                "r_med": r_med,
                "r_band": (r_min, r_max),
            })
            return [], dbg

        merged = merge_circles(all_circles, eps=max(8.0, r_med * 0.7))

        # radius band filter (final)
        rr = np.array([c[2] for c in merged], dtype=float)
        r_med2 = float(np.median(rr)) if len(rr) else r_med
        r_lo = 0.65 * r_med2
        r_hi = 1.65 * r_med2

        final = [(x, y, r) for (x, y, r) in merged if (r_lo <= r <= r_hi)]
        bubbles = [Bubble(x=x, y=y, r=r) for (x, y, r) in final]

        dbg.update({
            "bubbles_found_raw": len(all_circles),
            "bubbles_found_merged": len(merged),
            "bubbles_found_filtered": len(bubbles),
            "r_med": r_med2,
            "r_band": (r_lo, r_hi),
            "contours_kept": len(contour_circles),
            "hough_kept": len(hough_circles),
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

    # ---------- Template inference for KEY (Questions only) ----------

    def infer_key_questions(self, bubbles: List[Bubble], img_shape: Tuple[int, int]) -> TemplateQ:
        h, w = img_shape
        if len(bubbles) < 30:
            raise ValueError("عدد الفقاعات قليل جداً لاستخراج الأسئلة من الـ Key. ارفع DPI أو حسّن وضوح الـ PDF.")

        xs = np.array([b.x for b in bubbles], dtype=float)
        ys = np.array([b.y for b in bubbles], dtype=float)
        rs = np.array([b.r for b in bubbles], dtype=float)

        r_med = float(np.median(rs))
        eps_x = max(10.0, r_med * 2.2)
        eps_y = max(10.0, r_med * 2.2)

        # ✅ KEY: ignore far-right ID area by default
        left_bubbles = [b for b in bubbles if b.x < w * 0.75]
        if len(left_bubbles) < 20:
            left_bubbles = bubbles

        q_xs = np.array([b.x for b in left_bubbles], dtype=float)
        col_centers = sorted(self._cluster_1d(q_xs, eps=eps_x))

        debug = {
            "r_med": r_med, "eps_x": eps_x, "eps_y": eps_y,
            "bubbles_total": len(bubbles),
            "bubbles_left": len(left_bubbles),
            "col_centers_total": len(col_centers),
        }

        if len(col_centers) < 2:
            raise ValueError("فشل تكوين أعمدة للأسئلة. ربما الصفحة مقصوصة.")

        from itertools import combinations

        def score_cols(cols_k: List[float]) -> Tuple[float, List[float]]:
            idx = []
            for i, b in enumerate(left_bubbles):
                if min(abs(b.x - c) for c in cols_k) <= eps_x:
                    idx.append(i)
            if len(idx) < 10 * len(cols_k):
                return -1e9, []

            y_local = np.array([left_bubbles[i].y for i in idx], dtype=float)
            y_centers = sorted(self._cluster_1d(y_local, eps=eps_y))
            if len(y_centers) < 5:
                return -1e9, y_centers

            diffs = np.diff(y_centers) if len(y_centers) >= 2 else np.array([9999.0])
            cv = float(np.std(diffs) / (np.mean(diffs) + 1e-9))
            density = len(idx) / (len(cols_k) * max(len(y_centers), 1))
            # prefer more rows (more questions) and stable spacing
            score = -cv * 10.0 + density * 4.0 + len(y_centers) * 0.15 + len(idx) * 0.01
            return score, y_centers

        best = None
        best_score = -1e9

        # ✅ try 4 first (most common), then 5, then 2
        for k in [4, 5, 2]:
            if len(col_centers) < k:
                continue
            for comb in combinations(col_centers[:14], k):
                cols_k = sorted(list(comb))
                s, y_centers = score_cols(cols_k)
                if s > best_score:
                    best_score = s
                    best = (cols_k, y_centers, k)

        if best is None:
            raise ValueError("فشل استخراج شبكة الأسئلة من Answer Key. جرّب DPI=350 أو PDF أوضح.")

        q_cols_x, q_rows_y, choice_count = best
        q_count = len(q_rows_y)
        debug.update({
            "q_cols_x": q_cols_x,
            "choice_count": choice_count,
            "q_count": q_count,
        })

        return TemplateQ(
            q_cols_x=q_cols_x,
            q_rows_y=q_rows_y,
            choice_count=choice_count,
            q_count=q_count,
            debug=debug
        )

    # ---------- OPTIONAL ID inference on students ----------

    def infer_student_id_grid(self, bubbles: List[Bubble], img_shape: Tuple[int, int]) -> Optional[Tuple[List[float], List[float]]]:
        """Best-effort read ID grid from FAR RIGHT. Returns (id_cols_x, id_rows_y) or None."""
        h, w = img_shape
        if len(bubbles) < 30:
            return None

        xs = np.array([b.x for b in bubbles], dtype=float)
        ys = np.array([b.y for b in bubbles], dtype=float)
        rs = np.array([b.r for b in bubbles], dtype=float)

        r_med = float(np.median(rs))
        eps_x = max(10.0, r_med * 2.2)
        eps_y = max(10.0, r_med * 2.2)

        # far right only
        right_b = [b for b in bubbles if b.x >= w * 0.72]
        if len(right_b) < 20:
            right_b = [b for b in bubbles if b.x >= w * 0.65]
        if len(right_b) < 20:
            return None

        col_centers = sorted(self._cluster_1d(np.array([b.x for b in right_b], dtype=float), eps=eps_x))
        if len(col_centers) < 4:
            return None

        from itertools import combinations

        def score_id(cols4: List[float]) -> Tuple[float, List[float]]:
            idx = []
            for i, b in enumerate(right_b):
                if min(abs(b.x - c) for c in cols4) <= eps_x:
                    idx.append(i)
            if len(idx) < 25:
                return -1e9, []

            y_local = np.array([right_b[i].y for i in idx], dtype=float)
            y_centers = sorted(self._cluster_1d(y_local, eps=eps_y))
            rc = len(y_centers)
            if rc < 8 or rc > 12:
                return -1e9, y_centers

            diffs = np.diff(y_centers) if len(y_centers) >= 2 else np.array([9999.0])
            cv = float(np.std(diffs) / (np.mean(diffs) + 1e-9))
            width = max(cols4) - min(cols4)
            score = -abs(rc - 10) * 6.0 - cv * 10.0 - width * 0.02 + len(idx) * 0.05
            return score, y_centers

        best = None
        best_s = -1e9
        for comb in combinations(col_centers[-10:], 4):
            cols4 = sorted(list(comb))
            s, y_centers = score_id(cols4)
            # ✅ additional safety: must be really on the right
            if min(cols4) < w * 0.60:
                continue
            if s > best_s:
                best_s = s
                best = (cols4, y_centers)

        if best is None:
            return None

        id_cols_x, id_rows_y = best
        if len(id_rows_y) > 10:
            id_rows_y = id_rows_y[:10]
        if len(id_rows_y) < 10:
            return None

        return id_cols_x, id_rows_y

    # ---------- cell scoring ----------

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
            return float(min(pos, 3) + min(neg, 3))
        return 0.0

    # ---------- evaluate question cells ----------

    def evaluate_questions(self, img_bgr: np.ndarray, tq: TemplateQ, for_key: bool) -> Dict[Tuple[int, int], Bubble]:
        img = resize_max(img_bgr, 2400)
        g = preprocess_gray(img)
        b = adaptive_bin_inv(g)

        # radius guess from row spacing
        q_r = max(10.0, float(np.median(np.diff(sorted(tq.q_rows_y))) * 0.32)) if len(tq.q_rows_y) >= 2 else 14.0

        out: Dict[Tuple[int, int], Bubble] = {}
        for rr in range(tq.q_count):
            cy = tq.q_rows_y[rr]
            for cc in range(tq.choice_count):
                cx = tq.q_cols_x[cc]
                cell_bin = self._extract_roi(b, cx, cy, q_r)
                fill = self.fill_ratio(cell_bin)

                if for_key:
                    xs = 0.0
                    cancelled = False  # ✅ Key: never apply X
                else:
                    cell_gray = self._extract_roi(g, cx, cy, q_r)
                    xs = self.x_cancellation_score(cell_gray)
                    # ✅ cancel only if X and bubble is NOT filled
                    cancelled = (xs >= self.x_score_thr_student) and (fill < self.blank_fill_thr * 0.9)

                out[(rr, cc)] = Bubble(cx, cy, q_r, fill, xs, cancelled)
        return out

    # ---------- decision logic ----------

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

    def extract_answers(self, cell_map: Dict[Tuple[int, int], Bubble], tq: TemplateQ) -> Tuple[Dict[int, str], pd.DataFrame]:
        choices = "ABCDE"[:tq.choice_count]
        answers = {}
        rows = []
        for r in range(tq.q_count):
            row_bubs = [cell_map[(r, c)] for c in range(tq.choice_count)]
            d = self.decide_row(row_bubs)

            ans = ""
            if d["status"] == "OK":
                ans = choices[d["answer_idx"]]
                answers[r + 1] = ans

            rows.append({"Q": r + 1, "Answer": ans, "Status": d["status"]})
        return answers, pd.DataFrame(rows)

    # ---------- decode student id (best-effort) ----------

    def decode_student_id(self, img_bgr: np.ndarray, bubbles: List[Bubble]) -> str:
        img = resize_max(img_bgr, 2400)
        h, w = img.shape[:2]
        grid = self.infer_student_id_grid(bubbles, (h, w))
        if grid is None:
            return "UNKNOWN"

        id_cols_x, id_rows_y = grid
        g = preprocess_gray(img)
        b = adaptive_bin_inv(g)

        id_r = max(10.0, float(np.median(np.diff(sorted(id_rows_y))) * 0.32)) if len(id_rows_y) >= 2 else 14.0

        digits = []
        for col in range(4):
            fills = []
            for row in range(10):
                cx = id_cols_x[col]
                cy = id_rows_y[row]
                cell_bin = self._extract_roi(b, cx, cy, id_r)
                fills.append(self.fill_ratio(cell_bin))
            best_row = int(np.argmax(fills))
            best_fill = float(fills[best_row])
            digits.append("X" if best_fill < self.blank_fill_thr else str(best_row))
        return "".join(digits)

    # ---------- overlay ----------

    def draw_overlay_questions(self, img_bgr: np.ndarray, tq: TemplateQ, cell_map: Dict[Tuple[int, int], Bubble]) -> np.ndarray:
        img = resize_max(img_bgr, 2400).copy()

        # columns (blue)
        for x in tq.q_cols_x:
            cv2.line(img, (int(x), 0), (int(x), img.shape[0]-1), (255, 0, 0), 2)

        def draw_b(b: Bubble, color):
            cv2.circle(img, (int(b.x), int(b.y)), int(b.r), color, 2)
            cv2.circle(img, (int(b.x), int(b.y)), 3, color, -1)

        for r in range(tq.q_count):
            for c in range(tq.choice_count):
                b = cell_map[(r, c)]
                if b.cancelled:
                    col = (0, 255, 255)   # yellow
                elif b.fill >= self.blank_fill_thr:
                    col = (0, 0, 255)     # red
                else:
                    col = (0, 255, 0)     # green
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
    st.title("✅ OMR Bubble Sheet — تدريب تلقائي + تصحيح (حل نهائي)")
    st.caption("التدريب من Answer Key يكتشف الأسئلة فقط (بدون كود) لمنع الخلط نهائيًا.")

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
        bubble_margin_ratio=0.22,
        x_hough_min_votes=x_votes,
        x_score_thr_student=x_thr_student,
    )

    # ----------------------------
    # 1) TRAIN / EXTRACT ANSWER KEY (Questions ONLY)
    # ----------------------------
    st.subheader("1) Answer Key (استخراج قبل التصحيح) — أسئلة فقط")
    if key_up is None:
        st.info("ارفع Answer Key للبدء.")
        st.stop()

    key_bytes = read_uploaded_bytes(key_up)
    key_pages = pil_from_bytes(key_bytes, key_up.name, dpi=int(dpi))
    if not key_pages:
        st.error("تعذر قراءة Answer Key.")
        st.stop()

    key_bgr = to_bgr(key_pages[0])
    bubbles_key, dbg_b = engine.detect_bubbles(key_bgr)

    if debug:
        st.write("Bubble detection debug:", dbg_b)

    if len(bubbles_key) < 25:
        st.error("الفقاعات المكتشفة قليلة جدًا. جرّب DPI=350 أو PDF أوضح/غير مقصوص.")
        st.stop()

    try:
        tq = engine.infer_key_questions(bubbles_key, img_shape=resize_max(key_bgr, 2400).shape[:2])
    except Exception as e:
        st.error(f"❌ فشل استخراج الأسئلة من Answer Key: {e}")
        st.stop()

    if debug:
        st.write("Key Question Template debug:", tq.debug)

    cell_map_key = engine.evaluate_questions(key_bgr, tq, for_key=True)
    answer_key, df_key_view = engine.extract_answers(cell_map_key, tq)

    k1, k2 = st.columns(2)
    with k1:
        st.metric("عدد الأسئلة المكتشف", tq.q_count)
    with k2:
        st.metric("عدد الخيارات المكتشف", tq.choice_count)

    st.markdown("### 🔑 Answer Key المستخرج (تأكد قبل التصحيح)")
    st.dataframe(df_key_view, width="stretch", height=320)

    if debug:
        st.markdown("### Overlay (أحمر=مظلّل، أصفر=ملغي X، أزرق=أعمدة الأسئلة)")
        ov = engine.draw_overlay_questions(key_bgr, tq, cell_map_key)
        st.image(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB), use_container_width=True)

    if len(answer_key) == 0:
        st.error("لم يتم استخراج أي إجابة من المفتاح. تأكد أن المفتاح مظلل فعلاً وبـ DPI مناسب.")
        st.stop()

    st.success("✅ تم استخراج Answer Key بنجاح.")
    st.divider()

    # ----------------------------
    # 2) GRADING
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

                bubbles_st, _ = engine.detect_bubbles(img_bgr)
                sid = engine.decode_student_id(img_bgr, bubbles_st)
                sname = roster.get(sid, "غير موجود") if sid != "UNKNOWN" else "UNKNOWN"

                cell_map_st = engine.evaluate_questions(img_bgr, tq, for_key=False)
                s_answers, _ = engine.extract_answers(cell_map_st, tq)

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
                for q in range(1, tq.q_count + 1):
                    er[f"Q{q}"] = s_answers.get(q, "")
                export_rows.append(er)

                if debug:
                    st.markdown(f"#### Overlay: {up.name} (page {pi})")
                    ov2 = engine.draw_overlay_questions(img_bgr, tq, cell_map_st)
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
