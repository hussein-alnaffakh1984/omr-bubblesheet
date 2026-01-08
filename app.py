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
    confidence: str
    detection_notes: List[str]


# ==============================
# 🆕 HUMAN-LIKE BUBBLE DETECTION
# ==============================
def preprocess_binary_for_detection(bgr: np.ndarray) -> np.ndarray:
    """Enhanced preprocessing for better bubble detection"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # Multiple preprocessing attempts
    # Method 1: Adaptive threshold
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary1 = cv2.adaptiveThreshold(
        gray_blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 7
    )
    
    # Method 2: Otsu's threshold
    _, binary2 = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Combine both methods
    binary = cv2.bitwise_or(binary1, binary2)
    
    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.medianBlur(binary, 3)
    
    return binary


def find_bubble_centers_smart(bin_img: np.ndarray,
                              min_area: int,
                              max_area: int,
                              min_circularity: float) -> np.ndarray:
    """Find bubbles with multiple validation passes"""
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
            
        peri = cv2.arcLength(c, True)
        if peri <= 1e-6:
            continue
            
        # Circularity check
        circ = 4.0 * np.pi * area / (peri * peri)
        if circ < min_circularity:
            continue
        
        # Additional checks for bubble-like shapes
        # Check aspect ratio
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h if h > 0 else 0
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            continue
        
        # Get centroid
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
# 🆕 SMART REGION DETECTION
# ==============================
def detect_bubble_regions_smart(centers: np.ndarray, w: int, h: int, 
                                left_boundary_percent: float = 8.0) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Smart region detection like a human would do:
    1. Find question numbers area (far left, sparse)
    2. Find answer bubbles (dense cluster after numbers)
    3. Find ID bubbles (top right)
    """
    if centers.shape[0] < 20:
        raise ValueError("عدد الفقاعات قليل جداً")
    
    xs = centers[:, 0]
    ys = centers[:, 1]
    
    x_mid = np.median(xs)
    y_mid = np.median(ys)
    
    debug = {
        "x_median": x_mid,
        "y_median": y_mid
    }
    
    # Calculate left boundary
    left_boundary = (left_boundary_percent / 100.0) * w
    
    # ID section: top-right quadrant
    id_mask = (xs > 0.6 * w) & (ys < 0.5 * h)
    id_centers = centers[id_mask]
    
    # Fallback for ID
    if id_centers.shape[0] < 20:
        id_centers = centers[(xs > x_mid) & (ys < y_mid)]
    
    # Questions section: after left boundary, in left half, bottom area
    q_mask = (xs > left_boundary) & (xs < 0.55 * w) & (ys > 0.4 * h)
    q_centers = centers[q_mask]
    
    # Fallback for questions
    if q_centers.shape[0] < 20:
        q_centers = centers[(xs > left_boundary) & (xs < x_mid) & (ys > y_mid)]
    
    debug["id_count"] = id_centers.shape[0]
    debug["q_count"] = q_centers.shape[0]
    debug["left_boundary"] = left_boundary
    debug["filtered_out"] = centers.shape[0] - id_centers.shape[0] - q_centers.shape[0]
    
    return id_centers, q_centers, debug


# ==============================
# 🆕 SMART GRID ESTIMATION
# ==============================
def estimate_grid_smart(centers: np.ndarray, is_questions: bool = True) -> Tuple[int, int, float]:
    """
    Smart grid estimation based on common OMR patterns
    """
    if centers.shape[0] < 4:
        return 10, 4, 0.0  # Default guess
    
    total_bubbles = centers.shape[0]
    
    # Common configurations
    if is_questions:
        configs = [
            (10, 4), (10, 5), (10, 6),  # Most common
            (20, 4), (20, 5), (20, 6),
            (15, 4), (15, 5), (25, 4),
            (30, 4), (30, 5),
        ]
    else:
        configs = [
            (10, 4), (10, 5), (10, 6),  # ID with 4-6 digits
            (10, 3), (10, 7), (10, 8),
            (11, 4), (11, 5),
        ]
    
    # Find best match
    best_config = (10, 4)
    best_score = float('inf')
    
    for rows, cols in configs:
        expected = rows * cols
        diff = abs(expected - total_bubbles)
        score = diff / expected if expected > 0 else 1.0
        
        if score < best_score:
            best_score = score
            best_config = (rows, cols)
    
    rows, cols = best_config
    confidence = max(0.0, 1.0 - best_score)
    
    return rows, cols, confidence


# ==============================
# 🆕 BUILD GRID WITH INTERPOLATION
# ==============================
def cluster_1d_equal_bins(values: np.ndarray, k: int) -> np.ndarray:
    """Cluster values into k equal bins"""
    idx = np.argsort(values)
    labels = np.zeros(len(values), dtype=np.int32)
    n = len(values)
    for j in range(k):
        s = int(j * n / k)
        e = int((j + 1) * n / k)
        labels[idx[s:e]] = j
    return labels


def build_grid_smart(centers: np.ndarray, rows: int, cols: int) -> Optional[BubbleGrid]:
    """
    Build grid with smart interpolation for missing bubbles
    """
    if centers.shape[0] < int(rows * cols * 0.4):
        return None

    xs = centers[:, 0].astype(np.float32)
    ys = centers[:, 1].astype(np.float32)

    # Cluster into rows and columns
    rlab = cluster_1d_equal_bins(ys, rows)
    clab = cluster_1d_equal_bins(xs, cols)

    grid = np.zeros((rows, cols, 2), dtype=np.float32)
    cnt = np.zeros((rows, cols), dtype=np.int32)

    # Place bubbles in grid
    for (x, y), r, c in zip(centers, rlab, clab):
        grid[r, c, 0] += x
        grid[r, c, 1] += y
        cnt[r, c] += 1

    # Average for cells with bubbles
    for r in range(rows):
        for c in range(cols):
            if cnt[r, c] > 0:
                grid[r, c] /= cnt[r, c]

    # Interpolate missing cells
    # Calculate row and column medians
    row_ys = []
    col_xs = []
    
    for r in range(rows):
        valid_y = [grid[r, c, 1] for c in range(cols) if cnt[r, c] > 0]
        row_ys.append(np.median(valid_y) if valid_y else 0)
    
    for c in range(cols):
        valid_x = [grid[r, c, 0] for r in range(rows) if cnt[r, c] > 0]
        col_xs.append(np.median(valid_x) if valid_x else 0)
    
    # Fill zeros with interpolated values
    for i, val in enumerate(row_ys):
        if val == 0 and i > 0:
            row_ys[i] = row_ys[i-1] + 50
    for i, val in enumerate(col_xs):
        if val == 0 and i > 0:
            col_xs[i] = col_xs[i-1] + 50
    
    # Fill missing cells
    for r in range(rows):
        for c in range(cols):
            if cnt[r, c] == 0:
                grid[r, c, 0] = col_xs[c]
                grid[r, c, 1] = row_ys[r]

    # Sort by position
    row_order = np.argsort(row_ys)
    col_order = np.argsort(col_xs)
    grid = grid[row_order][:, col_order]

    return BubbleGrid(centers=grid, rows=rows, cols=cols)


# ==============================
# 🆕 HUMAN-LIKE DARKNESS READING
# ==============================
def mean_darkness_smart(gray: np.ndarray, cx: int, cy: int, win: int = 18) -> float:
    """
    Read darkness like a human would - focus on the center of the bubble
    """
    h, w = gray.shape[:2]
    x1 = max(0, cx - win)
    x2 = min(w, cx + win)
    y1 = max(0, cy - win)
    y2 = min(h, cy + win)
    patch = gray[y1:y2, x1:x2]
    
    if patch.size == 0:
        return 255.0

    # Focus on inner circle (exclude edges)
    ph, pw = patch.shape
    margin_h = max(1, int(ph * 0.3))
    margin_w = max(1, int(pw * 0.3))
    inner = patch[margin_h:ph-margin_h, margin_w:pw-margin_w]
    
    if inner.size == 0:
        inner = patch

    # Return mean darkness (lower = darker = filled)
    return float(np.mean(inner))


def pick_answer_smart(means: List[float], labels: List[str],
                     blank_thresh: float, diff_thresh: float) -> Tuple[str, str, Dict]:
    """
    Pick answer like a human would:
    1. Find darkest bubble
    2. Check if it's dark enough (filled)
    3. Check if second darkest is too close (double mark)
    """
    if not means:
        return "?", "BLANK", {}
    
    # Sort by darkness
    sorted_indices = np.argsort(means)
    darkest_idx = int(sorted_indices[0])
    darkest_val = means[darkest_idx]
    
    second_darkest_val = means[int(sorted_indices[1])] if len(sorted_indices) > 1 else 255
    
    info = {
        "darkest": round(darkest_val, 1),
        "second": round(second_darkest_val, 1),
        "diff": round(second_darkest_val - darkest_val, 1),
        "all_values": [round(m, 1) for m in means]
    }
    
    # Decision logic
    if darkest_val > blank_thresh:
        return "?", "BLANK", info
    
    if (second_darkest_val - darkest_val) < diff_thresh:
        return "!", "DOUBLE", info
    
    return labels[darkest_idx], "OK", info


# ==============================
# 🎯 MAIN AUTO-DETECTION
# ==============================
def auto_detect_smart(key_bgr: np.ndarray,
                     min_area: int = 80,
                     max_area: int = 10000,
                     min_circ: float = 0.45,
                     blank_thresh: float = 185,
                     diff_thresh: float = 8,
                     left_boundary_percent: float = 8.0) -> Tuple[AutoDetectedParams, pd.DataFrame, np.ndarray]:
    """
    Smart auto-detection that mimics human analysis
    Returns: (params, debug_dataframe, visualization_image)
    """
    h, w = key_bgr.shape[:2]
    gray = cv2.cvtColor(key_bgr, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Find all bubbles
    bin_key = preprocess_binary_for_detection(key_bgr)
    centers = find_bubble_centers_smart(bin_key, min_area, max_area, min_circ)
    
    notes = []
    notes.append(f"✅ اكتشاف {centers.shape[0]} فقاعة إجمالاً")
    
    if centers.shape[0] < 20:
        raise ValueError(f"عدد قليل جداً ({centers.shape[0]}) - عدّل الإعدادات")
    
    # Step 2: Separate regions
    id_centers, q_centers, reg_debug = detect_bubble_regions_smart(
        centers, w, h, left_boundary_percent
    )
    
    notes.append(f"✅ منطقة الكود: {id_centers.shape[0]} فقاعة")
    notes.append(f"✅ منطقة الأسئلة: {q_centers.shape[0]} فقاعة")
    notes.append(f"✅ متجاهلة (أرقام): {reg_debug['filtered_out']} فقاعة")
    
    # Step 3: Estimate grid dimensions
    id_rows, id_cols, id_conf = estimate_grid_smart(id_centers, is_questions=False)
    q_rows, q_cols, q_conf = estimate_grid_smart(q_centers, is_questions=True)
    
    notes.append(f"✅ شبكة الكود: {id_rows}×{id_cols} (ثقة: {id_conf:.0%})")
    notes.append(f"✅ شبكة الأسئلة: {q_rows}×{q_cols} (ثقة: {q_conf:.0%})")
    
    # Step 4: Build grids
    id_grid = build_grid_smart(id_centers, id_rows, id_cols)
    q_grid = build_grid_smart(q_centers, q_rows, q_cols)
    
    if not id_grid or not q_grid:
        raise ValueError("فشل بناء الشبكات - عدّل min_area أو min_circularity")
    
    # Check for missing bubbles
    id_expected = id_rows * id_cols
    q_expected = q_rows * q_cols
    
    if id_centers.shape[0] < id_expected:
        notes.append(f"⚠️ ناقص {id_expected - id_centers.shape[0]} فقاعة من الكود")
    if q_centers.shape[0] < q_expected:
        notes.append(f"⚠️ ناقص {q_expected - q_centers.shape[0]} فقاعة من الأسئلة")
    
    # Step 5: Extract answers (THE SMART PART!)
    choices = list("ABCDEFGHIJ"[:q_cols])
    answer_key = {}
    debug_rows = []
    
    for r in range(q_rows):
        # Read darkness of all choices
        means = []
        for c in range(q_cols):
            cx, cy = q_grid.centers[r, c]
            darkness = mean_darkness_smart(gray, int(cx), int(cy), win=18)
            means.append(darkness)
        
        # Pick answer smartly
        ans, status, info = pick_answer_smart(means, choices, blank_thresh, diff_thresh)
        
        if status == "OK":
            answer_key[r + 1] = ans
        
        # Build debug row
        debug_row = {
            "Q": r + 1,
            "Answer": ans,
            "Status": status,
            "Darkest": info.get("darkest", 0),
            "2nd_Dark": info.get("second", 0),
            "Diff": info.get("diff", 0),
        }
        for i, choice in enumerate(choices):
            debug_row[choice] = round(means[i], 1)
        
        debug_rows.append(debug_row)
    
    debug_df = pd.DataFrame(debug_rows)
    
    notes.append(f"✅ استخراج {len(answer_key)}/{q_rows} إجابة")
    
    # Step 6: Create visualization
    vis = key_bgr.copy()
    
    # Draw ID bubbles (RED)
    for (x, y) in id_centers:
        cv2.circle(vis, (int(x), int(y)), 7, (0, 0, 255), 2)
    
    # Draw Question bubbles (GREEN)
    for (x, y) in q_centers:
        cv2.circle(vis, (int(x), int(y)), 7, (0, 255, 0), 2)
    
    # Draw filtered bubbles (GRAY)
    left_bound = reg_debug['left_boundary']
    filtered = centers[centers[:, 0] <= left_bound]
    for (x, y) in filtered:
        cv2.circle(vis, (int(x), int(y)), 7, (128, 128, 128), 2)
    
    # Draw boundary line (MAGENTA)
    cv2.line(vis, (int(left_bound), 0), (int(left_bound), h), (255, 0, 255), 3)
    
    # Add labels
    cv2.putText(vis, f"ID: {id_centers.shape[0]}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(vis, f"Q: {q_centers.shape[0]}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(vis, f"Filtered: {filtered.shape[0]}", (10, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
    
    # Determine confidence
    avg_conf = (id_conf + q_conf) / 2
    if avg_conf > 0.8 and len(answer_key) >= q_rows * 0.8:
        confidence = "high"
    elif avg_conf > 0.6 and len(answer_key) >= q_rows * 0.6:
        confidence = "medium"
    else:
        confidence = "low"
    
    params = AutoDetectedParams(
        num_questions=q_rows,
        num_choices=q_cols,
        id_digits=id_cols,
        id_rows=id_rows,
        answer_key=answer_key,
        confidence=confidence,
        detection_notes=notes
    )
    
    return params, debug_df, vis


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
        raise ValueError("ملف الطلاب يحتاج: student_code و student_name")

    codes = (
        df["student_code"]
        .astype(str).str.strip()
        .str.replace(".0", "", regex=False)
        .str.zfill(id_digits)
    )
    names = df["student_name"].astype(str).str.strip()
    return dict(zip(codes, names))


# ==============================
# Template creation from auto params
# ==============================
def create_template_from_params(key_bgr: np.ndarray, params: AutoDetectedParams,
                                min_area: int, max_area: int, min_circ: float,
                                left_boundary_percent: float) -> LearnedTemplate:
    """Create a template from auto-detected parameters"""
    h, w = key_bgr.shape[:2]
    bin_key = preprocess_binary_for_detection(key_bgr)
    centers = find_bubble_centers_smart(bin_key, min_area, max_area, min_circ)
    
    id_centers, q_centers, _ = detect_bubble_regions_smart(centers, w, h, left_boundary_percent)
    
    id_grid = build_grid_smart(id_centers, params.id_rows, params.id_digits)
    q_grid = build_grid_smart(q_centers, params.num_questions, params.num_choices)
    
    if not id_grid or not q_grid:
        raise ValueError("فشل بناء Template")
    
    return LearnedTemplate(
        ref_bgr=key_bgr,
        ref_w=w,
        ref_h=h,
        id_grid=id_grid,
        q_grid=q_grid,
        num_q=params.num_questions,
        num_choices=params.num_choices,
        id_rows=params.id_rows,
        id_digits=params.id_digits
    )


def read_student_id(template: LearnedTemplate, gray: np.ndarray,
                   blank_thresh: float, diff_thresh: float) -> Tuple[str, pd.DataFrame]:
    digits = []
    dbg_rows = []
    
    for c in range(template.id_grid.cols):
        means = []
        for r in range(template.id_grid.rows):
            cx, cy = template.id_grid.centers[r, c]
            means.append(mean_darkness_smart(gray, int(cx), int(cy), win=16))
        
        labels = [str(i) for i in range(template.id_grid.rows)]
        digit, status, _ = pick_answer_smart(means, labels, blank_thresh, diff_thresh)
        digits.append(digit if status == "OK" else "X")
        
        dbg_rows.append([c + 1, digit, status] + [round(m, 1) for m in means])
    
    df_dbg = pd.DataFrame(dbg_rows, columns=["Digit", "Pick", "Status"] + 
                         [str(i) for i in range(template.id_rows)])
    return "".join(digits), df_dbg


def read_student_answers(template: LearnedTemplate, gray: np.ndarray,
                        blank_thresh: float, diff_thresh: float,
                        choices: List[str]) -> pd.DataFrame:
    rows_out = []
    for r in range(template.q_grid.rows):
        means = []
        for c in range(template.q_grid.cols):
            cx, cy = template.q_grid.centers[r, c]
            means.append(mean_darkness_smart(gray, int(cx), int(cy), win=18))
        
        ans, status, _ = pick_answer_smart(means, choices, blank_thresh, diff_thresh)
        rows_out.append([r + 1, ans, status] + [round(m, 1) for m in means])
    
    return pd.DataFrame(rows_out, columns=["Q", "Pick", "Status"] + choices)


# ==============================
# Streamlit app
# ==============================
def main():
    st.set_page_config(page_title="🧠 Smart OMR - Human-Like Intelligence", layout="wide")
    st.title("🧠 OMR ذكي - يفكر مثل الإنسان!")
    
    st.success("✨ **جديد!** النظام الآن يحلل الأنسر بنفس طريقة البشر - دقة عالية جداً!")

    # File uploads
    col1, col2, col3 = st.columns(3)
    with col1:
        roster_file = st.file_uploader("📋 ملف الطلاب", type=["xlsx", "xls", "csv"])
    with col2:
        key_file = st.file_uploader("🔑 Answer Key", type=["pdf", "png", "jpg", "jpeg"])
    with col3:
        sheets_file = st.file_uploader("📚 أوراق الطلاب", type=["pdf", "png", "jpg", "jpeg"])

    st.markdown("---")

    # Settings
    dpi = st.slider("DPI (جودة المسح)", 150, 400, 250, 10)

    with st.expander("⚙️ إعدادات اكتشاف الفقاعات", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            min_area = st.number_input("min_area", 20, 2000, 70, 5)
        with col2:
            max_area = st.number_input("max_area", 1000, 30000, 10000, 500)
        with col3:
            min_circ = st.slider("min_circularity", 0.30, 0.95, 0.40, 0.01)

    with st.expander("✅ إعدادات قراءة التظليل"):
        col1, col2 = st.columns(2)
        with col1:
            blank_thresh = st.slider("Blank threshold", 120, 240, 185, 1)
        with col2:
            diff_thresh = st.slider("Diff threshold", 3, 60, 8, 1)

    with st.expander("✂️ إعدادات فصل أرقام الأسئلة"):
        left_boundary = st.slider("موضع الخط الفاصل (%)", 5.0, 20.0, 8.0, 0.5)

    debug = st.checkbox("🔍 عرض Debug", value=True)

    if not (roster_file and key_file and sheets_file):
        st.info("📤 ارفع الملفات الثلاثة للبدء")
        return

    # Load answer key
    key_bytes = read_bytes(key_file)
    key_pages = load_pages(key_bytes, key_file.name, dpi=int(dpi))
    if not key_pages:
        st.error("❌ فشل قراءة Answer Key")
        return
    key_bgr = pil_to_bgr(key_pages[0])

    # Smart auto-detection
    st.markdown("---")
    st.subheader("🧠 التحليل الذكي للأنسر...")
    
    try:
        with st.spinner("⏳ جاري التحليل بذكاء..."):
            params, debug_df, vis_img = auto_detect_smart(
                key_bgr,
                min_area=int(min_area),
                max_area=int(max_area),
                min_circ=float(min_circ),
                blank_thresh=float(blank_thresh),
                diff_thresh=float(diff_thresh),
                left_boundary_percent=float(left_boundary)
            )
        
        st.success("✅ تم التحليل بنجاح!")
        
        # Display results
        conf_colors = {"high": "🟢", "medium": "🟡", "low": "🔴"}
        st.metric("مستوى الثقة", f"{conf_colors[params.confidence]} {params.confidence.upper()}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("الأسئلة", params.num_questions)
        with col2:
            st.metric("الخيارات", params.num_choices)
        with col3:
            st.metric("خانات الكود", params.id_digits)
        with col4:
            st.metric("الإجابات", len(params.answer_key))
        
        with st.expander("📋 تفاصيل الكشف", expanded=True):
            for note in params.detection_notes:
                st.write(note)
        
        # Show answer key
        st.subheader("🔑 الإجابات الصحيحة:")
        if params.answer_key:
            ans_display = " | ".join([f"Q{q}: **{a}**" for q, a in sorted(params.answer_key.items())])
            st.success(ans_display)
            
            with st.expander("📊 جدول التحليل التفصيلي", expanded=True):
                st.dataframe(debug_df, use_container_width=True, height=400)
                st.info("💡 العمود 'Darkest' = أغمق فقاعة | 'Diff' = الفرق مع الثانية")
        else:
            st.error("❌ لم يتم استخراج إجابات!")
            st.dataframe(debug_df, use_container_width=True)
        
        # Visualization
        if debug:
            with st.expander("🎨 التصور المرئي", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.image(bgr_to_rgb(key_bgr), caption="الأصلي", use_container_width=True)
                with col2:
                    st.image(bgr_to_rgb(vis_img), caption="المناطق المكتشفة", use_container_width=True)
                st.info("🔴 الكود | 🟢 الأسئلة | ⚪ أرقام متجاهلة | 🟣 الخط الفاصل")
        
        if not params.answer_key:
            st.error("لا يمكن المتابعة بدون إجابات")
            return
        
        # Load roster
        try:
            roster = load_roster(roster_file, params.id_digits)
            st.success(f"✅ قائمة الطلاب: {len(roster)}")
        except Exception as e:
            st.error(f"❌ خطأ: {e}")
            return
        
        # Grading
        st.markdown("---")
        if st.button("🚀 ابدأ التصحيح", type="primary", use_container_width=True):
            sheets_bytes = read_bytes(sheets_file)
            pages = load_pages(sheets_bytes, sheets_file.name, dpi=int(dpi))
            
            if not pages:
                st.error("❌ فشل قراءة الأوراق")
                return
            
            # Create template
            template = create_template_from_params(
                key_bgr, params, int(min_area), int(max_area), 
                float(min_circ), float(left_boundary)
            )
            
            choices = list("ABCDEFGHIJ"[:params.num_choices])
            results = []
            
            prog = st.progress(0)
            for i, pil_page in enumerate(pages, 1):
                page_bgr = pil_to_bgr(pil_page)
                aligned, ok, matches = orb_align(page_bgr, template.ref_bgr)
                gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
                
                student_code, _ = read_student_id(template, gray, float(blank_thresh), float(diff_thresh))
                student_code = str(student_code).zfill(params.id_digits)
                student_name = roster.get(student_code, "غير موجود")
                
                df_ans = read_student_answers(template, gray, float(blank_thresh), float(diff_thresh), choices)
                
                correct = sum(1 for _, row in df_ans.iterrows() 
                            if int(row["Q"]) in params.answer_key 
                            and row["Status"] == "OK" 
                            and row["Pick"] == params.answer_key[int(row["Q"])])
                
                total = len(params.answer_key)
                pct = (correct / total * 100) if total else 0
                status = "ناجح ✓" if pct >= 50 else "راسب ✗"
                
                results.append({
                    "page": i,
                    "aligned": ok,
                    "matches": matches,
                    "code": student_code,
                    "name": student_name,
                    "score": correct,
                    "total": total,
                    "percentage": round(pct, 2),
                    "status": status
                })
                
                prog.progress(i / len(pages))
            
            df_results = pd.DataFrame(results)
            st.success("✅ اكتمل التصحيح!")
            st.dataframe(df_results, use_container_width=True, height=420)
            
            # Download
            out = io.BytesIO()
            df_results.to_excel(out, index=False, engine="openpyxl")
            st.download_button(
                "⬇️ تحميل النتائج",
                data=out.getvalue(),
                file_name="results_smart.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    
    except Exception as e:
        st.error(f"❌ خطأ: {e}")
        st.info("💡 جرب تعديل الإعدادات")


if __name__ == "__main__":
    main()
