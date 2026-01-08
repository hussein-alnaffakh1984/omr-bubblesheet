"""
🧠 ULTIMATE SMART OMR - Reads Like a Human!
This version literally mimics how I (Claude) read the answer key image.
"""
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
# 🧠 HUMAN-LIKE VISUAL ANALYSIS
# ==============================
def analyze_image_like_human(bgr: np.ndarray) -> Dict:
    """
    Analyze the image EXACTLY like a human would:
    1. Look at the WHOLE image
    2. Identify distinct regions visually
    3. Find patterns and groupings
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # Create a visual density map
    # Where are the dark areas? (bubbles)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Divide image into vertical strips and count dark pixels
    num_strips = 50
    strip_width = w // num_strips
    density = []
    
    for i in range(num_strips):
        x_start = i * strip_width
        x_end = min((i + 1) * strip_width, w)
        strip = thresh[:, x_start:x_end]
        # Count dark pixels (bubbles)
        dark_count = np.sum(strip > 0)
        density.append(dark_count)
    
    # Find the valley (gap) between question numbers and answer bubbles
    # Question numbers: LOW density (sparse, ~10 bubbles)
    # Gap: VERY LOW density (almost empty)
    # Answer bubbles: HIGH density (dense, ~40 bubbles)
    
    # Smooth the density curve
    density_smooth = np.convolve(density, np.ones(3)/3, mode='same')
    
    # Find local minima (valleys)
    valleys = []
    for i in range(5, 20):  # Look in first 20% of image
        if density_smooth[i] < density_smooth[i-1] and density_smooth[i] < density_smooth[i+1]:
            if density_smooth[i] < np.mean(density_smooth[:25]) * 0.3:  # Significant valley
                valleys.append((i, density_smooth[i]))
    
    if valleys:
        # Take the first significant valley
        valley_idx = valleys[0][0]
        boundary_x = valley_idx * strip_width
    else:
        # Fallback: find biggest drop
        diffs = -np.diff(density_smooth[:20])
        valley_idx = np.argmax(diffs)
        boundary_x = valley_idx * strip_width
    
    return {
        'boundary_x': boundary_x,
        'boundary_percent': (boundary_x / w) * 100,
        'density_profile': density,
        'valley_found': len(valleys) > 0,
        'confidence': 'high' if len(valleys) > 0 else 'medium'
    }


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


@dataclass
class BubbleGrid:
    centers: np.ndarray
    rows: int
    cols: int


@dataclass
class AutoDetectedParams:
    num_questions: int
    num_choices: int
    id_digits: int
    id_rows: int
    answer_key: Dict[int, str]
    confidence: str
    detection_notes: List[str]
    visual_boundary: float


# ==============================
# Preprocessing
# ==============================
def preprocess_binary(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive threshold
    binary1 = cv2.adaptiveThreshold(
        gray_blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 7
    )
    
    # Otsu's threshold
    _, binary2 = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Combine
    binary = cv2.bitwise_or(binary1, binary2)
    
    # Clean
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.medianBlur(binary, 3)
    
    return binary


def find_bubbles(bin_img: np.ndarray, min_area: int, max_area: int, min_circ: float) -> np.ndarray:
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
        if circ < min_circ:
            continue
        
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h if h > 0 else 0
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
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
# 🎯 HUMAN-LIKE REGION DETECTION
# ==============================
def separate_regions_visually(centers: np.ndarray, w: int, h: int, 
                              visual_analysis: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Use VISUAL analysis boundary instead of guessing!
    """
    if centers.shape[0] < 20:
        raise ValueError("عدد الفقاعات قليل جداً")
    
    xs = centers[:, 0]
    ys = centers[:, 1]
    
    x_mid = np.median(xs)
    y_mid = np.median(ys)
    
    # Use the VISUALLY detected boundary!
    left_boundary = visual_analysis['boundary_x']
    
    # ID section: top-right
    id_mask = (xs > 0.6 * w) & (ys < 0.5 * h)
    id_centers = centers[id_mask]
    
    if id_centers.shape[0] < 20:
        id_centers = centers[(xs > x_mid) & (ys < y_mid)]
    
    # Questions: after visual boundary
    q_mask = (xs > left_boundary) & (xs < 0.55 * w) & (ys > 0.4 * h)
    q_centers = centers[q_mask]
    
    if q_centers.shape[0] < 20:
        q_centers = centers[(xs > left_boundary) & (xs < x_mid) & (ys > y_mid)]
    
    # Filtered (question numbers)
    filtered_mask = (xs <= left_boundary) & (ys > 0.4 * h)
    filtered_count = np.sum(filtered_mask)
    
    debug = {
        'id_count': id_centers.shape[0],
        'q_count': q_centers.shape[0],
        'filtered_out': filtered_count,
        'boundary_x': left_boundary,
        'boundary_percent': visual_analysis['boundary_percent'],
        'visual_confidence': visual_analysis['confidence']
    }
    
    return id_centers, q_centers, debug


def estimate_grid(centers: np.ndarray, is_questions: bool = True) -> Tuple[int, int, float]:
    if centers.shape[0] < 4:
        return 10, 4, 0.0
    
    total = centers.shape[0]
    
    configs = [
        (10, 4), (10, 5), (10, 6),
        (20, 4), (20, 5), (15, 4),
    ] if is_questions else [
        (10, 4), (10, 5), (10, 6),
        (10, 3), (10, 7), (11, 4),
    ]
    
    best = (10, 4)
    best_score = float('inf')
    
    for r, c in configs:
        expected = r * c
        diff = abs(expected - total)
        score = diff / expected if expected > 0 else 1.0
        if score < best_score:
            best_score = score
            best = (r, c)
    
    rows, cols = best
    confidence = max(0.0, 1.0 - best_score)
    return rows, cols, confidence


def cluster_bins(values: np.ndarray, k: int) -> np.ndarray:
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
    Build grid with MAXIMUM tolerance for missing bubbles
    Humans can see a pattern even with missing parts!
    """
    expected = rows * cols
    actual = centers.shape[0]
    
    # VERY lenient threshold - accept even if 40% missing!
    if actual < int(expected * 0.3):
        st.warning(f"⚠️ فقط {actual} من {expected} فقاعات متوقعة - محاولة البناء...")
        return None

    xs = centers[:, 0].astype(np.float32)
    ys = centers[:, 1].astype(np.float32)

    # Try to cluster into rows and columns
    rlab = cluster_bins(ys, rows)
    clab = cluster_bins(xs, cols)

    grid = np.zeros((rows, cols, 2), dtype=np.float32)
    cnt = np.zeros((rows, cols), dtype=np.int32)

    # Place known bubbles
    for (x, y), r, c in zip(centers, rlab, clab):
        grid[r, c, 0] += x
        grid[r, c, 1] += y
        cnt[r, c] += 1

    # Average cells with bubbles
    for r in range(rows):
        for c in range(cols):
            if cnt[r, c] > 0:
                grid[r, c] /= cnt[r, c]

    # SMART INTERPOLATION for missing bubbles
    # Calculate row centers (Y positions)
    row_ys = []
    for r in range(rows):
        valid_y = [grid[r, c, 1] for c in range(cols) if cnt[r, c] > 0]
        if valid_y:
            row_ys.append(np.median(valid_y))
        else:
            row_ys.append(None)
    
    # Fill missing row positions using linear interpolation
    for i in range(len(row_ys)):
        if row_ys[i] is None:
            # Find nearest valid rows
            prev_valid = None
            next_valid = None
            for j in range(i-1, -1, -1):
                if row_ys[j] is not None:
                    prev_valid = (j, row_ys[j])
                    break
            for j in range(i+1, len(row_ys)):
                if row_ys[j] is not None:
                    next_valid = (j, row_ys[j])
                    break
            
            if prev_valid and next_valid:
                # Interpolate
                ratio = (i - prev_valid[0]) / (next_valid[0] - prev_valid[0])
                row_ys[i] = prev_valid[1] + ratio * (next_valid[1] - prev_valid[1])
            elif prev_valid:
                # Extrapolate forward
                row_ys[i] = prev_valid[1] + 50 * (i - prev_valid[0])
            elif next_valid:
                # Extrapolate backward
                row_ys[i] = next_valid[1] - 50 * (next_valid[0] - i)
            else:
                # Last resort
                row_ys[i] = 100 + i * 50
    
    # Calculate column centers (X positions)
    col_xs = []
    for c in range(cols):
        valid_x = [grid[r, c, 0] for r in range(rows) if cnt[r, c] > 0]
        if valid_x:
            col_xs.append(np.median(valid_x))
        else:
            col_xs.append(None)
    
    # Fill missing column positions
    for i in range(len(col_xs)):
        if col_xs[i] is None:
            prev_valid = None
            next_valid = None
            for j in range(i-1, -1, -1):
                if col_xs[j] is not None:
                    prev_valid = (j, col_xs[j])
                    break
            for j in range(i+1, len(col_xs)):
                if col_xs[j] is not None:
                    next_valid = (j, col_xs[j])
                    break
            
            if prev_valid and next_valid:
                ratio = (i - prev_valid[0]) / (next_valid[0] - prev_valid[0])
                col_xs[i] = prev_valid[1] + ratio * (next_valid[1] - prev_valid[1])
            elif prev_valid:
                col_xs[i] = prev_valid[1] + 50 * (i - prev_valid[0])
            elif next_valid:
                col_xs[i] = next_valid[1] - 50 * (next_valid[0] - i)
            else:
                col_xs[i] = 100 + i * 50
    
    # Fill ALL grid positions (even missing ones)
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
# Reading darkness
# ==============================
def read_darkness(gray: np.ndarray, cx: int, cy: int, win: int = 18) -> float:
    h, w = gray.shape[:2]
    x1 = max(0, cx - win)
    x2 = min(w, cx + win)
    y1 = max(0, cy - win)
    y2 = min(h, cy + win)
    patch = gray[y1:y2, x1:x2]
    
    if patch.size == 0:
        return 255.0

    ph, pw = patch.shape
    margin_h = max(2, int(ph * 0.35))
    margin_w = max(2, int(pw * 0.35))
    inner = patch[margin_h:ph-margin_h, margin_w:pw-margin_w]
    
    if inner.size == 0:
        margin_h = max(1, int(ph * 0.25))
        margin_w = max(1, int(pw * 0.25))
        inner = patch[margin_h:ph-margin_h, margin_w:pw-margin_w]
    
    if inner.size == 0:
        inner = patch
    
    flat = inner.flatten()
    if len(flat) > 10:
        sorted_pixels = np.sort(flat)
        darkest_half = sorted_pixels[:len(sorted_pixels)//2]
        return float(np.mean(darkest_half))
    
    return float(np.mean(inner))


def pick_answer(means: List[float], labels: List[str],
               blank_thresh: float, diff_thresh: float) -> Tuple[str, str, Dict]:
    if not means:
        return "?", "BLANK", {}
    
    sorted_idx = np.argsort(means)
    darkest_idx = int(sorted_idx[0])
    darkest_val = means[darkest_idx]
    
    second_val = means[int(sorted_idx[1])] if len(sorted_idx) > 1 else 255
    
    info = {
        "darkest": round(darkest_val, 1),
        "second": round(second_val, 1),
        "diff": round(second_val - darkest_val, 1),
    }
    
    if darkest_val > blank_thresh:
        return "?", "BLANK", info
    
    if (second_val - darkest_val) < diff_thresh:
        return "!", "DOUBLE", info
    
    return labels[darkest_idx], "OK", info


# ==============================
# 🧠 MAIN: READ LIKE HUMAN
# ==============================
def read_answer_key_like_human(key_bgr: np.ndarray,
                               min_area: int = 70,
                               max_area: int = 10000,
                               min_circ: float = 0.40,
                               blank_thresh: float = 185,
                               diff_thresh: float = 8) -> Tuple[AutoDetectedParams, pd.DataFrame, np.ndarray]:
    """
    Read answer key EXACTLY like a human would!
    """
    h, w = key_bgr.shape[:2]
    gray = cv2.cvtColor(key_bgr, cv2.COLOR_BGR2GRAY)
    
    notes = []
    
    # Step 1: VISUAL ANALYSIS (like looking at the whole page first)
    notes.append("👁️ **التحليل البصري**: فحص الصورة كاملة...")
    visual_info = analyze_image_like_human(key_bgr)
    notes.append(f"✅ كشف بصري: وجدت الحد الفاصل عند {visual_info['boundary_percent']:.1f}%")
    notes.append(f"   → ثقة الكشف: {visual_info['confidence']}")
    
    # Step 2: Find bubbles
    bin_key = preprocess_binary(key_bgr)
    centers = find_bubbles(bin_key, min_area, max_area, min_circ)
    notes.append(f"✅ اكتشاف {centers.shape[0]} فقاعة إجمالاً")
    
    if centers.shape[0] < 20:
        raise ValueError(f"عدد قليل ({centers.shape[0]})")
    
    # Step 3: Separate using VISUAL boundary
    id_centers, q_centers, reg_debug = separate_regions_visually(
        centers, w, h, visual_info
    )
    
    notes.append(f"✅ منطقة الكود: {id_centers.shape[0]} فقاعة")
    notes.append(f"✅ منطقة الأسئلة: {q_centers.shape[0]} فقاعة")
    notes.append(f"✅ متجاهلة (أرقام): {reg_debug['filtered_out']} فقاعة")
    
    # Validate
    if reg_debug['filtered_out'] < 8 or reg_debug['filtered_out'] > 12:
        notes.append(f"⚠️ متوقع ~10 أرقام، وجدت {reg_debug['filtered_out']}")
    if q_centers.shape[0] < 38 or q_centers.shape[0] > 42:
        notes.append(f"⚠️ متوقع ~40 فقاعة أسئلة، وجدت {q_centers.shape[0]}")
    
    # Step 4: Build grids
    id_rows, id_cols, id_conf = estimate_grid(id_centers, False)
    q_rows, q_cols, q_conf = estimate_grid(q_centers, True)
    
    notes.append(f"✅ شبكة الكود: {id_rows}×{id_cols} (ثقة: {id_conf:.0%})")
    notes.append(f"✅ شبكة الأسئلة: {q_rows}×{q_cols} (ثقة: {q_conf:.0%})")
    
    # Show expected vs actual
    id_expected = id_rows * id_cols
    q_expected = q_rows * q_cols
    notes.append(f"   → الكود: {id_centers.shape[0]}/{id_expected} فقاعات")
    notes.append(f"   → الأسئلة: {q_centers.shape[0]}/{q_expected} فقاعات")
    
    # Build with tolerance
    id_grid = build_grid(id_centers, id_rows, id_cols)
    q_grid = build_grid(q_centers, q_rows, q_cols)
    
    if not id_grid:
        notes.append(f"⚠️ فشل بناء شبكة الكود - ناقص {id_expected - id_centers.shape[0]} فقاعات")
        notes.append(f"   💡 جرب: تقليل min_area إلى {max(20, min_area - 10)} أو تقليل min_circularity")
        raise ValueError(f"فشل بناء شبكة الكود ({id_centers.shape[0]}/{id_expected})")
    
    if not q_grid:
        notes.append(f"⚠️ فشل بناء شبكة الأسئلة - ناقص {q_expected - q_centers.shape[0]} فقاعات")
        notes.append(f"   💡 جرب: تقليل min_area إلى {max(20, min_area - 10)} أو تقليل min_circularity")
        raise ValueError(f"فشل بناء شبكة الأسئلة ({q_centers.shape[0]}/{q_expected})")
    
    notes.append(f"✅ تم بناء الشبكات بنجاح (مع تقدير {id_expected + q_expected - id_centers.shape[0] - q_centers.shape[0]} فقاعات مفقودة)")
    
    # Step 5: Read answers
    choices = list("ABCDEFGHIJ"[:q_cols])
    answer_key = {}
    debug_rows = []
    crossed = []
    
    for r in range(q_rows):
        means = []
        for c in range(q_cols):
            cx, cy = q_grid.centers[r, c]
            darkness = read_darkness(gray, int(cx), int(cy), 18)
            means.append(darkness)
        
        ans, status, info = pick_answer(means, choices, blank_thresh, diff_thresh)
        
        if status == "DOUBLE":
            sorted_idx = np.argsort(means)
            first = choices[sorted_idx[0]]
            second = choices[sorted_idx[1]]
            crossed.append(f"Q{r+1}: {first} أو {second} - راجع!")
        
        if status == "OK":
            answer_key[r + 1] = ans
        
        debug_row = {
            "Q": r + 1,
            "Answer": ans,
            "Status": status,
            "Darkest": info.get("darkest", 0),
            "2nd": info.get("second", 0),
            "Diff": info.get("diff", 0),
        }
        for i, ch in enumerate(choices):
            debug_row[ch] = round(means[i], 1)
        
        debug_rows.append(debug_row)
    
    df = pd.DataFrame(debug_rows)
    notes.append(f"✅ استخراج {len(answer_key)}/{q_rows} إجابة")
    
    if crossed:
        notes.append("⚠️ **تحذير**: إجابات مشبوهة:")
        for c in crossed:
            notes.append(f"   • {c}")
    
    # Visualization
    vis = key_bgr.copy()
    
    # Draw detected bubbles
    for (x, y) in id_centers:
        cv2.circle(vis, (int(x), int(y)), 7, (0, 0, 255), 2)
    for (x, y) in q_centers:
        cv2.circle(vis, (int(x), int(y)), 7, (0, 255, 0), 2)
    
    # Draw interpolated (missing) bubbles in YELLOW
    id_expected = id_rows * id_cols
    q_expected = q_rows * q_cols
    if id_centers.shape[0] < id_expected:
        # Show interpolated ID positions
        for r in range(id_rows):
            for c in range(id_cols):
                cx, cy = id_grid.centers[r, c]
                # Check if this was interpolated (no nearby actual bubble)
                distances = np.sqrt(np.sum((id_centers - np.array([cx, cy]))**2, axis=1))
                if distances.min() > 20:  # No bubble within 20 pixels
                    cv2.circle(vis, (int(cx), int(cy)), 7, (0, 255, 255), 2)  # Yellow
    
    if q_centers.shape[0] < q_expected:
        # Show interpolated Q positions
        for r in range(q_rows):
            for c in range(q_cols):
                cx, cy = q_grid.centers[r, c]
                distances = np.sqrt(np.sum((q_centers - np.array([cx, cy]))**2, axis=1))
                if distances.min() > 20:
                    cv2.circle(vis, (int(cx), int(cy)), 7, (0, 255, 255), 2)  # Yellow
    
    # Draw filtered (question numbers)
    filtered = centers[(centers[:, 0] <= visual_info['boundary_x']) & (centers[:, 1] > 0.4 * h)]
    for (x, y) in filtered:
        cv2.circle(vis, (int(x), int(y)), 7, (128, 128, 128), 2)
    
    # Draw visual boundary
    bound_x = int(visual_info['boundary_x'])
    cv2.line(vis, (bound_x, 0), (bound_x, h), (255, 0, 255), 3)
    
    cv2.putText(vis, f"Visual Boundary: {visual_info['boundary_percent']:.1f}%", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    # Add legend
    legend_y = 60
    cv2.putText(vis, "Red=ID | Green=Questions | Yellow=Interpolated | Gray=Numbers", 
               (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    avg_conf = (id_conf + q_conf) / 2
    confidence = "high" if avg_conf > 0.8 and len(answer_key) >= q_rows * 0.8 else "medium" if avg_conf > 0.6 else "low"
    
    params = AutoDetectedParams(
        num_questions=q_rows,
        num_choices=q_cols,
        id_digits=id_cols,
        id_rows=id_rows,
        answer_key=answer_key,
        confidence=confidence,
        detection_notes=notes,
        visual_boundary=visual_info['boundary_x']
    )
    
    return params, df, vis


# ==============================
# Streamlit UI
# ==============================
def main():
    st.set_page_config(page_title="👁️ Human-Vision OMR", layout="wide")
    st.title("👁️ OMR بنظر بشري - يرى مثلك تماماً!")
    
    st.success("🆕 **النظام يستخدم التحليل البصري** - يرى الصفحة كاملة ويكتشف الأنماط مثل الإنسان!")

    col1, col2 = st.columns(2)
    with col1:
        key_file = st.file_uploader("🔑 Answer Key", type=["pdf", "png", "jpg"])
    with col2:
        dpi = st.slider("DPI", 150, 400, 250, 10)

    with st.expander("⚙️ إعدادات"):
        c1, c2, c3 = st.columns(3)
        with c1:
            min_area = st.number_input("min_area", 20, 2000, 70, 5)
        with c2:
            max_area = st.number_input("max_area", 1000, 30000, 10000, 500)
        with c3:
            min_circ = st.slider("min_circularity", 0.30, 0.95, 0.40, 0.01)
        
        c4, c5 = st.columns(2)
        with c4:
            blank_thresh = st.slider("Blank threshold", 120, 240, 185, 1)
        with c5:
            diff_thresh = st.slider("Diff threshold", 3, 60, 8, 1)

    if not key_file:
        st.info("📤 ارفع ملف Answer Key")
        return

    key_bytes = read_bytes(key_file)
    key_pages = load_pages(key_bytes, key_file.name, int(dpi))
    if not key_pages:
        st.error("❌ فشل قراءة الملف")
        return
    
    key_bgr = pil_to_bgr(key_pages[0])

    st.markdown("---")
    st.subheader("👁️ التحليل البصري...")
    
    try:
        with st.spinner("⏳ جاري القراءة مثل الإنسان..."):
            params, df, vis = read_answer_key_like_human(
                key_bgr,
                int(min_area), int(max_area), float(min_circ),
                float(blank_thresh), float(diff_thresh)
            )
        
        st.success("✅ تم القراءة بنجاح!")
        
        conf_colors = {"high": "🟢", "medium": "🟡", "low": "🔴"}
        st.metric("الثقة", f"{conf_colors[params.confidence]} {params.confidence.upper()}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("الأسئلة", params.num_questions)
        with col2:
            st.metric("الخيارات", params.num_choices)
        with col3:
            st.metric("خانات الكود", params.id_digits)
        with col4:
            st.metric("الإجابات", len(params.answer_key))
        
        with st.expander("📋 تفاصيل التحليل", expanded=True):
            for note in params.detection_notes:
                st.write(note)
        
        st.subheader("🔑 الإجابات:")
        if params.answer_key:
            ans_txt = " | ".join([f"Q{q}: **{a}**" for q, a in sorted(params.answer_key.items())])
            st.success(ans_txt)
            
            with st.expander("📊 الجدول التفصيلي", expanded=True):
                st.dataframe(df, use_container_width=True, height=400)
        else:
            st.error("❌ لم يتم استخراج إجابات")
            st.dataframe(df, use_container_width=True)
        
        with st.expander("🎨 التصور المرئي", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.image(bgr_to_rgb(key_bgr), caption="الأصلي", use_container_width=True)
            with col2:
                st.image(bgr_to_rgb(vis), caption="التحليل البصري", use_container_width=True)
            st.info("🔴 الكود | 🟢 الأسئلة | 🟡 مقدّرة (مفقودة) | ⚪ أرقام | 🟣 الحد البصري")
    
    except Exception as e:
        st.error(f"❌ خطأ: {e}")


if __name__ == "__main__":
    main()
