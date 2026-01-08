import io
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image


# ==============================
# File helpers
# ==============================
def read_bytes(uploaded_file) -> bytes:
    if uploaded_file is None:
        return b""
    try:
        return uploaded_file.getbuffer().tobytes()
    except Exception:
        try:
            return uploaded_file.read()
        except Exception:
            return b""


def load_pages(file_bytes: bytes, filename: str, dpi: int = 250) -> List[Image.Image]:
    if filename.lower().endswith(".pdf"):
        pages = convert_from_bytes(file_bytes, dpi=dpi)
        return [p.convert("RGB") for p in pages]
    return [Image.open(io.BytesIO(file_bytes)).convert("RGB")]


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ==============================
# Data models
# ==============================
@dataclass
class BubbleGrid:
    centers: np.ndarray  # (rows, cols, 2) float
    rows: int
    cols: int


@dataclass
class LearnedTemplate:
    ref_bgr: np.ndarray
    ref_w: int
    ref_h: int
    id_grid: BubbleGrid
    q_grid: BubbleGrid
    num_q: int
    num_choices: int
    id_rows: int
    id_digits: int


@dataclass
class AutoDetectedParams:
    """Parameters automatically detected from answer key"""
    num_questions: int
    num_choices: int
    id_digits: int
    id_rows: int
    answer_key: Dict[int, str]
    confidence: str  # "high", "medium", "low"
    detection_notes: List[str]


# ==============================
# Preprocess for bubble detection
# ==============================
def preprocess_binary_for_detection(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    bin_img = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 7
    )
    bin_img = cv2.medianBlur(bin_img, 3)
    return bin_img


def find_bubble_centers(bin_img: np.ndarray,
                        min_area: int,
                        max_area: int,
                        min_circularity: float) -> np.ndarray:
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        peri = cv2.arcLength(c, True)
        if peri <= 1e-6:
            continue
        circ = 4.0 * np.pi * area / (peri * peri)
        if circ < min_circularity:
            continue
        M = cv2.moments(c)
        if abs(M["m00"]) < 1e-6:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))

    if not centers:
        return np.zeros((0, 2), dtype=np.int32)
    return np.array(centers, dtype=np.int32)


# ==============================
# 🆕 Smart Region Detection
# ==============================
def detect_bubble_regions(centers: np.ndarray, w: int, h: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Automatically detect ID and Question regions using clustering
    Filters out question numbers on the far left
    Returns: (id_centers, q_centers, debug_info)
    """
    if centers.shape[0] < 20:
        raise ValueError("عدد الفقاعات قليل جداً - تأكد من جودة الصورة")
    
    xs = centers[:, 0]
    ys = centers[:, 1]
    
    # Split into quadrants for initial analysis
    x_mid = np.median(xs)
    y_mid = np.median(ys)
    
    # Count bubbles in each quadrant
    q1 = np.sum((xs > x_mid) & (ys < y_mid))  # top-right (usually ID)
    q2 = np.sum((xs < x_mid) & (ys < y_mid))  # top-left
    q3 = np.sum((xs < x_mid) & (ys > y_mid))  # bottom-left (usually Questions)
    q4 = np.sum((xs > x_mid) & (ys > y_mid))  # bottom-right
    
    debug = {
        "quadrant_counts": {"TR": q1, "TL": q2, "BL": q3, "BR": q4},
        "x_median": x_mid,
        "y_median": y_mid
    }
    
    # More precise splitting using better boundaries
    # ID section: typically top-right
    id_mask = (xs > 0.6 * w) & (ys < 0.5 * h)
    
    # Questions section: Find the leftmost answer bubble, not the question numbers
    # Strategy: Question numbers are isolated on far left, answer bubbles form a dense cluster
    # Let's find where the main cluster starts
    
    # Get left half bubbles
    left_bubbles = centers[(xs < x_mid) & (ys > 0.4 * h)]
    
    if left_bubbles.shape[0] > 0:
        # Sort by x position
        left_xs = np.sort(left_bubbles[:, 0])
        
        # Find the gap between question numbers and answer bubbles
        # Question numbers are typically 1-10 bubbles on far left
        # Then there's a gap, then answer bubbles (40-60 bubbles) start
        
        if len(left_xs) > 15:
            # Look at x-position differences
            x_diffs = np.diff(left_xs)
            
            # Find the largest gap in first 30% of bubbles (should be between Q nums and answers)
            search_range = min(15, len(x_diffs) // 3)
            if search_range > 0:
                max_gap_idx = np.argmax(x_diffs[:search_range])
                # The boundary should be after this gap
                left_boundary = (left_xs[max_gap_idx] + left_xs[max_gap_idx + 1]) / 2
            else:
                left_boundary = 0.08 * w
        else:
            left_boundary = 0.08 * w
    else:
        left_boundary = 0.08 * w
    
    # Make sure boundary is reasonable
    left_boundary = max(0.05 * w, min(left_boundary, 0.15 * w))
    
    # Questions section: after the boundary, in left half
    right_boundary = 0.5 * w  # Questions are in left half
    q_mask = (xs > left_boundary) & (xs < right_boundary) & (ys > 0.45 * h)
    
    id_centers = centers[id_mask]
    q_centers = centers[q_mask]
    
    # Fallback if regions produce odd counts
    if id_centers.shape[0] < 20 or id_centers.shape[0] > 100:
        # Try quadrant-based
        id_centers = centers[(xs > x_mid) & (ys < y_mid)]
    
    if q_centers.shape[0] < 20:
        # More lenient
        left_boundary = 0.08 * w
        q_centers = centers[(xs > left_boundary) & (xs < right_boundary) & (ys > 0.4 * h)]
    
    if q_centers.shape[0] < 20:
        # Last resort: quadrant-based but still filter obvious left edge
        left_boundary = 0.05 * w
        q_centers = centers[(xs > left_boundary) & (xs < x_mid) & (ys > y_mid)]
    
    debug["id_count"] = id_centers.shape[0]
    debug["q_count"] = q_centers.shape[0]
    debug["left_boundary"] = left_boundary
    debug["filtered_out"] = centers.shape[0] - id_centers.shape[0] - q_centers.shape[0]
    debug["id_expected_multiples"] = [30, 40, 50, 60, 70, 80]
    debug["q_expected_multiples"] = [40, 50, 60]
    
    return id_centers, q_centers, debug


# ==============================
# 🆕 Auto-detect grid dimensions
# ==============================
def estimate_grid_dimensions(centers: np.ndarray, is_questions: bool = True) -> Tuple[int, int, float]:
    """
    Estimate rows and columns by analyzing spacing patterns
    Returns: (estimated_rows, estimated_cols, confidence)
    """
    if centers.shape[0] < 4:
        return 1, 1, 0.0
    
    xs = centers[:, 0]
    ys = centers[:, 1]
    
    # Find typical spacing
    xs_sorted = np.sort(xs)
    ys_sorted = np.sort(ys)
    
    # Calculate differences
    x_diffs = np.diff(xs_sorted)
    y_diffs = np.diff(ys_sorted)
    
    # Find modal spacing (most common distance) - ignore very small diffs
    x_diffs_clean = x_diffs[x_diffs > 5]
    y_diffs_clean = y_diffs[y_diffs > 5]
    
    if len(x_diffs_clean) == 0 or len(y_diffs_clean) == 0:
        return 1, 1, 0.0
    
    x_spacing = np.median(x_diffs_clean)
    y_spacing = np.median(y_diffs_clean)
    
    # Estimate dimensions
    x_range = xs.max() - xs.min()
    y_range = ys.max() - ys.min()
    
    cols_raw = int(round(x_range / x_spacing)) + 1 if x_spacing > 0 else 1
    rows_raw = int(round(y_range / y_spacing)) + 1 if y_spacing > 0 else 1
    
    # Snap to expected values based on typical OMR sheets
    if is_questions:
        # Questions typically: 10 rows × (4, 5, or 6 choices)
        # Find closest valid configuration
        expected_configs = [
            (10, 4), (10, 5), (10, 6),
            (20, 4), (20, 5), (20, 6),
            (15, 4), (15, 5), (15, 6),
            (25, 4), (25, 5), (25, 6),
            (30, 4), (30, 5), (30, 6),
        ]
    else:
        # ID typically: 10 rows (0-9) × (3-8 digit columns)
        expected_configs = [
            (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8),
            (11, 3), (11, 4), (11, 5), (11, 6),
        ]
    
    # Find best match
    total_bubbles = centers.shape[0]
    best_config = (rows_raw, cols_raw)
    best_diff = float('inf')
    
    for exp_rows, exp_cols in expected_configs:
        expected_total = exp_rows * exp_cols
        diff = abs(expected_total - total_bubbles)
        if diff < best_diff:
            best_diff = diff
            best_config = (exp_rows, exp_cols)
    
    rows, cols = best_config
    
    # Sanity bounds
    cols = max(1, min(cols, 20))
    rows = max(1, min(rows, 100))
    
    # Confidence based on how close we are to expected
    expected_bubbles = rows * cols
    actual_bubbles = centers.shape[0]
    diff_ratio = abs(expected_bubbles - actual_bubbles) / expected_bubbles if expected_bubbles > 0 else 1.0
    confidence = 1.0 - min(1.0, diff_ratio)
    
    return rows, cols, confidence


# ==============================
# Build grids
# ==============================
def cluster_1d_equal_bins(values: np.ndarray, k: int) -> np.ndarray:
    idx = np.argsort(values)
    labels = np.zeros(len(values), dtype=np.int32)
    n = len(values)
    for j in range(k):
        s = int(j * n / k)
        e = int((j + 1) * n / k)
        labels[idx[s:e]] = j
    return labels


def build_grid(centers: np.ndarray, rows: int, cols: int) -> Optional[BubbleGrid]:
    """
    Build grid even with some missing bubbles by using interpolation
    """
    if centers.shape[0] < int(rows * cols * 0.5):  # Need at least 50%
        return None

    xs = centers[:, 0].astype(np.float32)
    ys = centers[:, 1].astype(np.float32)

    rlab = cluster_1d_equal_bins(ys, rows)
    clab = cluster_1d_equal_bins(xs, cols)

    grid = np.zeros((rows, cols, 2), dtype=np.float32)
    cnt = np.zeros((rows, cols), dtype=np.int32)

    # Collect bubbles into grid cells
    for (x, y), r, c in zip(centers, rlab, clab):
        grid[r, c, 0] += x
        grid[r, c, 1] += y
        cnt[r, c] += 1

    # Average positions for cells with bubbles
    for r in range(rows):
        for c in range(cols):
            if cnt[r, c] > 0:
                grid[r, c] /= cnt[r, c]

    # Interpolate missing bubbles
    # First, calculate row and column medians
    row_meds_y = []
    col_meds_x = []
    
    for r in range(rows):
        filled_in_row = [grid[r, c, 1] for c in range(cols) if cnt[r, c] > 0]
        if filled_in_row:
            row_meds_y.append(np.median(filled_in_row))
        else:
            row_meds_y.append(0)  # Will fix later
    
    for c in range(cols):
        filled_in_col = [grid[r, c, 0] for r in range(rows) if cnt[r, c] > 0]
        if filled_in_col:
            col_meds_x.append(np.median(filled_in_col))
        else:
            col_meds_x.append(0)  # Will fix later
    
    # Fix empty row/col medians
    if any(x == 0 for x in row_meds_y):
        valid_y = [y for y in row_meds_y if y > 0]
        if valid_y:
            y_spacing = np.median(np.diff(sorted(valid_y))) if len(valid_y) > 1 else 50
            for i in range(len(row_meds_y)):
                if row_meds_y[i] == 0:
                    row_meds_y[i] = min(valid_y) + i * y_spacing
    
    if any(x == 0 for x in col_meds_x):
        valid_x = [x for x in col_meds_x if x > 0]
        if valid_x:
            x_spacing = np.median(np.diff(sorted(valid_x))) if len(valid_x) > 1 else 50
            for i in range(len(col_meds_x)):
                if col_meds_x[i] == 0:
                    col_meds_x[i] = min(valid_x) + i * x_spacing
    
    # Fill missing cells using row/col medians
    for r in range(rows):
        for c in range(cols):
            if cnt[r, c] == 0:
                grid[r, c, 0] = col_meds_x[c]
                grid[r, c, 1] = row_meds_y[r]

    # Ensure proper ordering
    row_order = np.argsort([row_meds_y[r] for r in range(rows)])
    col_order = np.argsort([col_meds_x[c] for c in range(cols)])
    
    grid = grid[row_order][:, col_order]

    return BubbleGrid(centers=grid, rows=rows, cols=cols)


# ==============================
# ✅ Correct shading measure (GRAY DARKNESS)
# ==============================
def mean_darkness(gray: np.ndarray, cx: int, cy: int, win: int = 18) -> float:
    h, w = gray.shape[:2]
    x1 = max(0, cx - win)
    x2 = min(w, cx + win)
    y1 = max(0, cy - win)
    y2 = min(h, cy + win)
    patch = gray[y1:y2, x1:x2]
    if patch.size == 0:
        return 255.0

    ph, pw = patch.shape
    mh = int(ph * 0.25)
    mw = int(pw * 0.25)
    inner = patch[mh:ph - mh, mw:pw - mw]
    if inner.size == 0:
        inner = patch

    return float(np.mean(inner))


def pick_by_darkness(means: List[float], labels: List[str],
                     blank_mean_thresh: float, diff_thresh: float) -> Tuple[str, str]:
    if not means:
        return "?", "BLANK"
    
    idx = np.argsort(means)
    best = int(idx[0])
    second = int(idx[1]) if len(idx) > 1 else best
    best_m = means[best]
    second_m = means[second]

    if best_m > blank_mean_thresh:
        return "?", "BLANK"
    if (second_m - best_m) < diff_thresh:
        return "!", "DOUBLE"
    return labels[best], "OK"


# ==============================
# 🆕 AUTO-DETECT ALL PARAMETERS
# ==============================
def auto_detect_from_answer_key(key_bgr: np.ndarray,
                                 min_area: int = 120,
                                 max_area: int = 9000,
                                 min_circ: float = 0.55,
                                 blank_thresh: float = 170,
                                 diff_thresh: float = 12) -> Tuple[AutoDetectedParams, pd.DataFrame]:
    """
    Automatically detect ALL parameters from answer key image
    Returns: (params, debug_dataframe)
    """
    h, w = key_bgr.shape[:2]
    gray = cv2.cvtColor(key_bgr, cv2.COLOR_BGR2GRAY)
    bin_key = preprocess_binary_for_detection(key_bgr)
    centers = find_bubble_centers(bin_key, min_area, max_area, min_circ)
    
    notes = []
    
    if centers.shape[0] < 20:
        raise ValueError(f"عدد الفقاعات المكتشفة قليل جداً ({centers.shape[0]}). جرب تعديل min_area/max_area")
    
    notes.append(f"✓ تم اكتشاف {centers.shape[0]} فقاعة إجمالاً")
    
    # Detect regions
    id_centers, q_centers, region_debug = detect_bubble_regions(centers, w, h)
    notes.append(f"✓ منطقة الكود: {id_centers.shape[0]} فقاعة")
    notes.append(f"✓ منطقة الأسئلة: {q_centers.shape[0]} فقاعة")
    notes.append(f"✓ فقاعات متجاهلة (أرقام الأسئلة): {region_debug['filtered_out']}")
    notes.append(f"  → الحد الفاصل التلقائي: {region_debug['left_boundary']:.1f} بكسل")
    
    # Estimate ID grid dimensions
    id_rows_est, id_cols_est, id_conf = estimate_grid_dimensions(id_centers, is_questions=False)
    notes.append(f"✓ تقدير شبكة الكود: {id_rows_est} صفوف × {id_cols_est} أعمدة (ثقة: {id_conf:.1%})")
    notes.append(f"  → إجمالي فقاعات الكود المتوقعة: {id_rows_est * id_cols_est} (فعلياً: {id_centers.shape[0]})")
    
    # Estimate Question grid dimensions
    q_rows_est, q_cols_est, q_conf = estimate_grid_dimensions(q_centers, is_questions=True)
    notes.append(f"✓ تقدير شبكة الأسئلة: {q_rows_est} أسئلة × {q_cols_est} خيارات (ثقة: {q_conf:.1%})")
    notes.append(f"  → إجمالي فقاعات الأسئلة المتوقعة: {q_rows_est * q_cols_est} (فعلياً: {q_centers.shape[0]})")
    
    # Build grids with estimated dimensions
    id_grid = build_grid(id_centers, rows=id_rows_est, cols=id_cols_est)
    q_grid = build_grid(q_centers, rows=q_rows_est, cols=q_cols_est)
    
    if id_grid is None:
        raise ValueError("فشل بناء شبكة الكود. جرب تعديل معايير الكشف.")
    if q_grid is None:
        raise ValueError("فشل بناء شبكة الأسئلة. جرب تعديل معايير الكشف.")
    
    # Check for missing bubbles and warn
    id_expected = id_rows_est * id_cols_est
    id_actual = id_centers.shape[0]
    if id_actual < id_expected:
        notes.append(f"⚠️ ناقص {id_expected - id_actual} فقاعة من الكود - سيتم التقدير")
    
    q_expected = q_rows_est * q_cols_est
    q_actual = q_centers.shape[0]
    if q_actual < q_expected:
        notes.append(f"⚠️ ناقص {q_expected - q_actual} فقاعة من الأسئلة - سيتم التقدير")
        notes.append("   💡 جرب تقليل min_area أو زيادة max_area لكشف المزيد من الفقاعات")
    
    # Extract answer key by reading filled bubbles with DETAILED DEBUG
    answer_key = {}
    choices = list("ABCDEFGHIJ"[:q_cols_est])
    debug_rows = []
    
    for r in range(q_rows_est):
        means = []
        for c in range(q_cols_est):
            cx, cy = q_grid.centers[r, c]
            darkness = mean_darkness(gray, int(cx), int(cy), win=18)
            means.append(darkness)
        
        # Find darkest (filled) bubble
        min_idx = int(np.argmin(means))
        darkest_val = means[min_idx]
        
        # Get second darkest for comparison
        means_sorted = sorted(means)
        second_darkest = means_sorted[1] if len(means_sorted) > 1 else 255
        
        # Decision logic with more lenient thresholds
        if darkest_val > blank_thresh:
            # Too light - probably blank
            ans = "?"
            status = "BLANK"
        elif (second_darkest - darkest_val) < diff_thresh:
            # Two bubbles too close - double mark
            ans = "!"
            status = "DOUBLE"
        else:
            # Clear winner
            ans = choices[min_idx]
            status = "OK"
            answer_key[r + 1] = ans
        
        # Build debug row with ALL darkness values
        debug_row = {
            "Q": r + 1,
            "Picked": ans,
            "Status": status,
            "Darkest": round(darkest_val, 1),
            "2nd_Dark": round(second_darkest, 1),
            "Diff": round(second_darkest - darkest_val, 1)
        }
        # Add individual choice darkness values
        for i, choice in enumerate(choices):
            debug_row[choice] = round(means[i], 1)
        
        debug_rows.append(debug_row)
    
    debug_df = pd.DataFrame(debug_rows)
    
    notes.append(f"✓ تم استخراج {len(answer_key)} إجابة صحيحة من {q_rows_est} سؤال")
    
    if len(answer_key) == 0:
        notes.append("⚠️ لم يتم اكتشاف أي إجابات مظللة! تحقق من:")
        notes.append("  - هل الأنسر مظلل بشكل واضح؟")
        notes.append("  - جرب تقليل blank_threshold")
        notes.append("  - جرب تقليل diff_threshold")
    
    # Determine confidence
    avg_conf = (id_conf + q_conf) / 2
    if avg_conf > 0.8 and len(answer_key) > q_rows_est * 0.7:
        confidence = "high"
    elif avg_conf > 0.6 and len(answer_key) > q_rows_est * 0.5:
        confidence = "medium"
    else:
        confidence = "low"
    
    params = AutoDetectedParams(
        num_questions=q_rows_est,
        num_choices=q_cols_est,
        id_digits=id_cols_est,
        id_rows=id_rows_est,
        answer_key=answer_key,
        confidence=confidence,
        detection_notes=notes
    )
    
    return params, debug_df


# ==============================
# ORB alignment
# ==============================
def orb_align(student_bgr: np.ndarray, ref_bgr: np.ndarray) -> Tuple[np.ndarray, bool, int]:
    h, w = ref_bgr.shape[:2]
    orb = cv2.ORB_create(5000)

    g1 = cv2.cvtColor(student_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)

    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)

    if d1 is None or d2 is None or len(k1) < 25 or len(k2) < 25:
        return cv2.resize(student_bgr, (w, h)), False, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(d1, d2, k=2)

    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    if len(good) < 25:
        return cv2.resize(student_bgr, (w, h)), False, len(good)

    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return cv2.resize(student_bgr, (w, h)), False, len(good)

    warped = cv2.warpPerspective(student_bgr, H, (w, h))
    return warped, True, len(good)


# ==============================
# Roster
# ==============================
def load_roster(file, id_digits: int) -> Dict[str, str]:
    if file.name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "student_code" not in df.columns or "student_name" not in df.columns:
        raise ValueError("ملف الطلاب لازم يحتوي: student_code و student_name")

    codes = (
        df["student_code"]
        .astype(str).str.strip()
        .str.replace(".0", "", regex=False)
        .str.zfill(id_digits)
    )
    names = df["student_name"].astype(str).str.strip()
    return dict(zip(codes, names))


# ==============================
# Manual template learning (with known params)
# ==============================
def learn_template(key_bgr: np.ndarray,
                   id_rows: int, id_digits: int,
                   num_q: int, num_choices: int,
                   min_area: int, max_area: int, min_circ: float) -> Tuple[LearnedTemplate, Dict]:

    h, w = key_bgr.shape[:2]
    bin_key = preprocess_binary_for_detection(key_bgr)
    centers = find_bubble_centers(bin_key, min_area, max_area, min_circ)

    id_centers, q_centers, _ = detect_bubble_regions(centers, w, h)

    id_grid = build_grid(id_centers, rows=id_rows, cols=id_digits)
    q_grid = build_grid(q_centers, rows=num_q, cols=num_choices)

    if id_grid is None:
        raise ValueError("فشل تعلم شبكة الكود")
    if q_grid is None:
        raise ValueError("فشل تعلم شبكة الأسئلة")

    template = LearnedTemplate(
        ref_bgr=key_bgr, ref_w=w, ref_h=h,
        id_grid=id_grid, q_grid=q_grid,
        num_q=num_q, num_choices=num_choices,
        id_rows=id_rows, id_digits=id_digits
    )

    dbg = {
        "bin_key": bin_key,
        "centers": centers,
        "id_centers": id_centers,
        "q_centers": q_centers
    }
    return template, dbg


def extract_answer_key(template: LearnedTemplate,
                       blank_mean_thresh: float,
                       diff_thresh: float,
                       choices: List[str]) -> Tuple[Dict[int, str], pd.DataFrame]:

    gray = cv2.cvtColor(template.ref_bgr, cv2.COLOR_BGR2GRAY)
    rows_dbg = []
    key = {}

    for r in range(template.q_grid.rows):
        means = []
        for c in range(template.q_grid.cols):
            cx, cy = template.q_grid.centers[r, c]
            means.append(mean_darkness(gray, int(cx), int(cy), win=18))

        ans, status = pick_by_darkness(means, choices, blank_mean_thresh, diff_thresh)
        if status == "OK":
            key[r + 1] = ans

        rows_dbg.append([r + 1, ans, status] + [round(m, 1) for m in means])

    df_dbg = pd.DataFrame(rows_dbg, columns=["Q", "Picked", "Status"] + choices)
    return key, df_dbg


def read_student_id(template: LearnedTemplate,
                    gray: np.ndarray,
                    blank_mean_thresh: float,
                    diff_thresh: float) -> Tuple[str, pd.DataFrame]:

    digits = []
    dbg_rows = []

    for c in range(template.id_grid.cols):
        means = []
        for r in range(template.id_grid.rows):
            cx, cy = template.id_grid.centers[r, c]
            means.append(mean_darkness(gray, int(cx), int(cy), win=16))

        labels = [str(i) for i in range(template.id_grid.rows)]
        digit, status = pick_by_darkness(means, labels, blank_mean_thresh, diff_thresh)
        digits.append(digit if status == "OK" else "X")

        dbg_rows.append([c + 1, digit, status] + [round(m, 1) for m in means])

    df_dbg = pd.DataFrame(dbg_rows, columns=["DigitCol", "Picked", "Status"] + [str(i) for i in range(template.id_rows)])
    return "".join(digits), df_dbg


def read_student_answers(template: LearnedTemplate,
                         gray: np.ndarray,
                         blank_mean_thresh: float,
                         diff_thresh: float,
                         choices: List[str]) -> pd.DataFrame:

    rows_out = []
    for r in range(template.q_grid.rows):
        means = []
        for c in range(template.q_grid.cols):
            cx, cy = template.q_grid.centers[r, c]
            means.append(mean_darkness(gray, int(cx), int(cy), win=18))

        ans, status = pick_by_darkness(means, choices, blank_mean_thresh, diff_thresh)
        rows_out.append([r + 1, ans, status] + [round(m, 1) for m in means])

    return pd.DataFrame(rows_out, columns=["Q", "Picked", "Status"] + choices)


# ==============================
# Streamlit app
# ==============================
def main():
    st.set_page_config(page_title="🤖 Smart OMR - Auto Detect", layout="wide")
    st.title("🤖 OMR ذكي - يكتشف كل شيء تلقائياً!")
    
    st.info("🎯 **جديد!** الآن النظام يكتشف تلقائياً: عدد الأسئلة، عدد الخيارات، خانات الكود، والإجابات الصحيحة!")

    # Mode selection
    mode = st.radio(
        "اختر طريقة العمل:",
        ["🤖 اكتشاف تلقائي (Smart)", "⚙️ إعدادات يدوية (Manual)"],
        horizontal=True
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        roster_file = st.file_uploader("📋 ملف الطلاب (Excel/CSV)", type=["xlsx", "xls", "csv"])
    with col2:
        key_file = st.file_uploader("🔑 Answer Key (PDF/صورة)", type=["pdf", "png", "jpg", "jpeg"])
    with col3:
        sheets_file = st.file_uploader("📚 أوراق الطلاب (PDF/صورة)", type=["pdf", "png", "jpg", "jpeg"])

    st.markdown("---")

    # DPI setting (always needed)
    dpi = st.slider("DPI (جودة المسح)", 150, 400, 250, 10)

    # Detection parameters (for both modes)
    with st.expander("⚙️ إعدادات اكتشاف الفقاعات - **هام جداً!**", expanded=True):
        st.warning("⚠️ إذا كان عدد الفقاعات المكتشفة ناقص، عدّل هذه الإعدادات:")
        d1, d2, d3 = st.columns(3)
        with d1:
            min_area = st.number_input("min_area (كلما قل = يكشف فقاعات أصغر)", 20, 2000, 100, 10,
                                      help="الحد الأدنى لمساحة الفقاعة - قلله إذا كانت الفقاعات صغيرة")
        with d2:
            max_area = st.number_input("max_area (كلما كبر = يكشف فقاعات أكبر)", 1000, 30000, 10000, 500,
                                      help="الحد الأقصى لمساحة الفقاعة - زوده إذا كانت الفقاعات كبيرة")
        with d3:
            min_circ = st.slider("min_circularity (كلما قل = يقبل أشكال أقل استدارة)", 0.30, 0.95, 0.50, 0.01,
                               help="يقيس مدى استدارة الشكل - قلله إذا كانت الفقاعات بيضاوية")
        
        st.info("💡 **نصيحة:** ابدأ بتقليل min_area إلى 80-90 إذا كان عدد الفقاعات ناقص")

    # Reading parameters
    with st.expander("✅ إعدادات قراءة التظليل"):
        r1, r2 = st.columns(2)
        with r1:
            blank_mean_thresh = st.slider("Blank threshold (كلما قل = يقرأ تظليل أخف)", 120, 240, 185, 1,
                                         help="الفقاعة تعتبر فارغة إذا كان متوسط الظلام أكبر من هذه القيمة")
        with r2:
            diff_thresh = st.slider("Diff threshold (كلما قل = يقبل فروق أقل)", 3, 60, 8, 1,
                                   help="الفرق المطلوب بين أغمق فقاعة والثانية لاعتبارها مظللة بوضوح")

    debug = st.checkbox("🔍 عرض تفاصيل Debug", value=True)

    # Manual mode settings
    if mode == "⚙️ إعدادات يدوية (Manual)":
        st.subheader("📝 إعدادات يدوية")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            id_rows = st.number_input("صفوف الكود (0-9)", 10, 15, 10, 1)
        with c2:
            id_digits = st.number_input("خانات الكود", 1, 12, 4, 1)
        with c3:
            num_q = st.number_input("عدد الأسئلة", 1, 200, 10, 1)
        with c4:
            num_choices = st.selectbox("عدد الخيارات", [4, 5, 6], index=0)

    if not (roster_file and key_file and sheets_file):
        st.info("📤 ارفع الملفات الثلاثة حتى يبدأ البرنامج")
        return

    # Load Answer Key
    key_bytes = read_bytes(key_file)
    key_pages = load_pages(key_bytes, key_file.name, dpi=int(dpi))
    if not key_pages:
        st.error("❌ فشل قراءة Answer Key")
        return
    key_bgr = pil_to_bgr(key_pages[0])

    # AUTO-DETECT MODE
    if mode == "🤖 اكتشاف تلقائي (Smart)":
        st.markdown("---")
        st.subheader("🔍 جاري الكشف التلقائي...")
        
        try:
            with st.spinner("⏳ يتم تحليل ورقة الإجابة النموذجية..."):
                auto_params, debug_df = auto_detect_from_answer_key(
                    key_bgr,
                    min_area=int(min_area),
                    max_area=int(max_area),
                    min_circ=float(min_circ),
                    blank_thresh=float(blank_mean_thresh),
                    diff_thresh=float(diff_thresh)
                )
            
            # Display detected parameters
            st.success("✅ تم الكشف التلقائي بنجاح!")
            
            conf_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}
            st.metric("مستوى الثقة", f"{conf_color[auto_params.confidence]} {auto_params.confidence.upper()}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("عدد الأسئلة", auto_params.num_questions)
            with col2:
                st.metric("عدد الخيارات", auto_params.num_choices)
            with col3:
                st.metric("خانات الكود", auto_params.id_digits)
            with col4:
                st.metric("صفوف الكود", auto_params.id_rows)
            
            with st.expander("📋 ملاحظات الكشف التلقائي", expanded=True):
                for note in auto_params.detection_notes:
                    st.write(note)
            
            # Show answer key with better formatting
            st.subheader("🔑 الإجابات الصحيحة المكتشفة:")
            if len(auto_params.answer_key) > 0:
                # Display as a nice formatted dict
                key_display = " | ".join([f"Q{q}: {ans}" for q, ans in sorted(auto_params.answer_key.items())])
                st.success(key_display)
                
                # Show detailed detection table
                with st.expander("📊 جدول تفصيلي لقراءة كل سؤال (الأقل = المظلل)", expanded=True):
                    st.dataframe(debug_df, use_container_width=True, height=400)
                    st.info("💡 **نصيحة:** العمود 'Darkest' يجب أن يكون أقل من 'Blank threshold' للإجابات الصحيحة")
                
                # Visual representation of answer key
                if debug:
                    with st.expander("🖼️ تصور مرئي للكشف", expanded=True):
                        # Show original with detected regions
                        vis = key_bgr.copy()
                        
                        # Draw all detected centers
                        bin_key = preprocess_binary_for_detection(key_bgr)
                        all_centers = find_bubble_centers(bin_key, int(min_area), int(max_area), float(min_circ))
                        id_ctrs, q_ctrs, reg_debug = detect_bubble_regions(all_centers, key_bgr.shape[1], key_bgr.shape[0])
                        
                        # Draw ID bubbles in RED
                        for (x, y) in id_ctrs:
                            cv2.circle(vis, (int(x), int(y)), 8, (0, 0, 255), 2)
                        
                        # Draw Question bubbles in GREEN
                        for (x, y) in q_ctrs:
                            cv2.circle(vis, (int(x), int(y)), 8, (0, 255, 0), 2)
                        
                        # Draw filtered bubbles (question numbers) in GRAY
                        all_xs = all_centers[:, 0]
                        left_bound = reg_debug.get("left_boundary", 0.08 * key_bgr.shape[1])
                        filtered = all_centers[all_xs <= left_bound]
                        for (x, y) in filtered:
                            cv2.circle(vis, (int(x), int(y)), 8, (128, 128, 128), 2)
                        
                        # Draw boundary line
                        cv2.line(vis, (int(left_bound), 0), (int(left_bound), key_bgr.shape[0]), (255, 0, 255), 3)
                        cv2.putText(vis, f"Boundary", (int(left_bound) + 5, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                        
                        # Add text labels
                        cv2.putText(vis, f"ID: {id_ctrs.shape[0]}/{auto_params.id_rows * auto_params.id_digits}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(vis, f"Q: {q_ctrs.shape[0]}/{auto_params.num_questions * auto_params.num_choices}", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(vis, f"Filtered: {filtered.shape[0]}", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
                        
                        # Show missing count
                        q_missing = (auto_params.num_questions * auto_params.num_choices) - q_ctrs.shape[0]
                        if q_missing > 0:
                            cv2.putText(vis, f"Missing: {q_missing} bubbles!", (10, 90), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(bgr_to_rgb(key_bgr), caption="Answer Key الأصلي", use_container_width=True)
                        with col2:
                            st.image(bgr_to_rgb(vis), caption="المناطق المكتشفة: 🔴 ID | 🟢 أسئلة", use_container_width=True)
                        
                        missing_info = f"📊 توزيع الفقاعات: ID={id_ctrs.shape[0]} | Questions={q_ctrs.shape[0]} | Filtered={filtered.shape[0]} | Total={all_centers.shape[0]}"
                        if q_missing > 0:
                            st.error(f"{missing_info} | ⚠️ ناقص {q_missing} فقاعة!")
                            st.warning("🔧 **لإصلاح المشكلة:** قلل min_area إلى 80-100 أو قلل min_circularity إلى 0.45")
                        else:
                            st.success(missing_info)
                        
                        st.info("🔵 **الألوان:** 🔴 الكود | 🟢 الأسئلة | ⚪ أرقام الأسئلة (مُتجاهلة) | 🟣 الخط الفاصل")
                        
                        # Show binary image for debugging
                        st.image(bin_key, caption="الصورة الثنائية (Binary) - الفقاعات تظهر بيضاء", use_container_width=True)
                        st.info("💡 تحقق من الصورة الثنائية: هل كل الفقاعات تظهر بوضوح؟")
            else:
                st.error("❌ لم يتم اكتشاف أي إجابات مظللة!")
                st.warning("🔧 **حلول مقترحة:**")
                st.write("1. قلل قيمة **Blank threshold** (حالياً:", blank_mean_thresh, ")")
                st.write("2. قلل قيمة **Diff threshold** (حالياً:", diff_thresh, ")")
                st.write("3. تأكد أن الأنسر مظلل بقلم رصاص أسود غامق")
                st.write("4. تأكد أن جودة الصورة عالية (DPI حالياً:", dpi, ")")
                
                # Show the detection table anyway
                st.subheader("📊 جدول القراءة (للتشخيص):")
                st.dataframe(debug_df, use_container_width=True, height=400)
                st.info("💡 ابحث عن السطور التي 'Darkest' فيها أقل رقم - هذه المفروض تكون المظللة")
            
            choices = list("ABCDEFGHIJ"[:auto_params.num_choices])
            
            # Use detected parameters
            id_rows = auto_params.id_rows
            id_digits = auto_params.id_digits
            num_q = auto_params.num_questions
            num_choices = auto_params.num_choices
            answer_key = auto_params.answer_key
            
            # Show warning if no answers detected
            if len(answer_key) == 0:
                st.error("⚠️ لا يمكن المتابعة بدون إجابات صحيحة. عدّل الإعدادات أو استخدم الوضع اليدوي.")
                return
            
            # Show warning if confidence is low
            if auto_params.confidence == "low":
                st.warning("⚠️ مستوى الثقة منخفض! يُنصح بمراجعة النتائج أو استخدام الوضع اليدوي.")
            
        except Exception as e:
            st.error(f"❌ فشل الكشف التلقائي: {e}")
            st.info("💡 جرب تعديل معايير الكشف أو استخدم الوضع اليدوي")
            return
    
    # MANUAL MODE
    else:
        try:
            choices = list("ABCDEF"[:int(num_choices)])
            template, dbg = learn_template(
                key_bgr,
                id_rows=int(id_rows), id_digits=int(id_digits),
                num_q=int(num_q), num_choices=int(num_choices),
                min_area=int(min_area), max_area=int(max_area), min_circ=float(min_circ)
            )
            st.success("✅ تم تعلم القالب (يدوي)")
            
            answer_key, df_key_dbg = extract_answer_key(
                template,
                blank_mean_thresh=float(blank_mean_thresh),
                diff_thresh=float(diff_thresh),
                choices=choices
            )
            st.write("🔑 Answer Key:")
            st.write(answer_key)
            
            if debug:
                with st.expander("🔍 Debug: Answer Key"):
                    st.image(bgr_to_rgb(key_bgr), caption="Answer Key", width=800)
                    st.dataframe(df_key_dbg, width=800, height=400)
                    
        except Exception as e:
            st.error(f"❌ فشل التعلم اليدوي: {e}")
            return

    # Load Roster
    try:
        roster = load_roster(roster_file, id_digits=int(id_digits))
        st.success(f"✅ تم تحميل قائمة الطلاب: {len(roster)} طالب")
    except Exception as e:
        st.error(f"❌ خطأ في ملف الطلاب: {e}")
        return

    # Grade Students
    st.markdown("---")
    if st.button("🚀 ابدأ التصحيح", type="primary", use_container_width=True):
        sheets_bytes = read_bytes(sheets_file)
        pages = load_pages(sheets_bytes, sheets_file.name, dpi=int(dpi))
        if not pages:
            st.error("❌ فشل قراءة أوراق الطلاب")
            return

        # Build template for manual mode
        if mode == "⚙️ إعدادات يدوية (Manual)":
            try:
                template, _ = learn_template(
                    key_bgr, int(id_rows), int(id_digits),
                    int(num_q), int(num_choices),
                    int(min_area), int(max_area), float(min_circ)
                )
            except Exception as e:
                st.error(f"❌ خطأ في بناء القالب: {e}")
                return
        else:
            # Build template from auto-detected params
            try:
                template, _ = learn_template(
                    key_bgr, id_rows, id_digits,
                    num_q, num_choices,
                    int(min_area), int(max_area), float(min_circ)
                )
            except Exception as e:
                st.error(f"❌ خطأ في بناء القالب: {e}")
                return

        results = []
        debug_samples = []

        prog = st.progress(0)
        for i, pil_page in enumerate(pages, start=1):
            page_bgr = pil_to_bgr(pil_page)
            aligned, ok, good_matches = orb_align(page_bgr, template.ref_bgr)

            gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

            student_code, df_id_dbg = read_student_id(
                template, gray,
                float(blank_mean_thresh), float(diff_thresh)
            )
            student_code = str(student_code).zfill(int(id_digits))
            student_name = roster.get(student_code, "غير موجود")

            df_ans = read_student_answers(
                template, gray,
                float(blank_mean_thresh), float(diff_thresh),
                choices
            )

            correct = 0
            total = len(answer_key)
            for _, row in df_ans.iterrows():
                q = int(row["Q"])
                if q not in answer_key:
                    continue
                if row["Status"] != "OK":
                    continue
                if row["Picked"] == answer_key[q]:
                    correct += 1

            pct = (correct / total * 100.0) if total else 0.0
            status = "ناجح ✓" if pct >= 50 else "راسب ✗"

            results.append({
                "page": i,
                "aligned_ok": ok,
                "good_matches": good_matches,
                "student_code": student_code,
                "student_name": student_name,
                "score": correct,
                "total": total,
                "percentage": round(pct, 2),
                "status": status,
            })

            if debug and len(debug_samples) < 2:
                debug_samples.append((i, aligned, df_id_dbg, df_ans))

            prog.progress(int(i / len(pages) * 100))

        df_res = pd.DataFrame(results)
        st.success("✅ اكتمل التصحيح!")
        st.dataframe(df_res, use_container_width=True, height=420)

        # Download results
        out = io.BytesIO()
        df_res.to_excel(out, index=False, engine="openpyxl")
        st.download_button(
            "⬇️ تحميل النتائج Excel",
            data=out.getvalue(),
            file_name="results_smart_omr.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        # Debug samples
        if debug and debug_samples:
            st.markdown("---")
            st.header("🔍 Debug Samples")
            for page_no, aligned, df_id_dbg, df_ans in debug_samples:
                with st.expander(f"صفحة {page_no}"):
                    st.image(bgr_to_rgb(aligned), caption="Aligned", width=800)
                    st.subheader("ID Detection")
                    st.dataframe(df_id_dbg, width=800)
                    st.subheader("Answers Detection")
                    st.dataframe(df_ans, width=800, height=400)


if __name__ == "__main__":
    main()
