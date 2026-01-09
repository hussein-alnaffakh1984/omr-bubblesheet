# ============================================================
# engine_omr.py | Strong Template-Free OMR Engine (PDF)
# - Robust bubble detection (contours + circularity + radius band)
# - Auto column segmentation (multi-column sheets)
# - Auto question grouping (row clustering + option grouping)
# - Student code detection (vertical columns ~10 digits)
# - Adaptive thresholds per page (LIGHT/FULL) + Cancel (stroke/X)
# - Outputs: student_code, answers, num_questions, flags, debug
# ============================================================

import os
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import fitz  # PyMuPDF


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Bubble:
    x: float
    y: float
    r: float
    bbox: Tuple[int, int, int, int]  # (x,y,w,h)


@dataclass
class CodeBlock:
    cols: List[List[int]]            # list of columns (bubble indices)
    bbox: Tuple[int, int, int, int]  # (x,y,w,h)


# -----------------------------
# PDF -> images
# -----------------------------
def pdf_to_images(pdf_path: str, zoom: float = 2.6) -> List[np.ndarray]:
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(zoom, zoom)
    out = []
    for i in range(len(doc)):
        pix = doc.load_page(i).get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        out.append(img[:, :, ::-1].copy())  # RGB -> BGR
    return out


# -----------------------------
# Basic deskew (small angles)
# -----------------------------
def _deskew(gray: np.ndarray) -> np.ndarray:
    # estimate skew using Hough lines on edges (safe, small correction)
    edges = cv2.Canny(gray, 60, 160)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=220)
    if lines is None:
        return gray

    angles = []
    for rho_theta in lines[:200]:
        rho, theta = rho_theta[0]
        ang = (theta * 180 / np.pi) - 90
        # keep near horizontal lines
        if -15 <= ang <= 15:
            angles.append(ang)

    if not angles:
        return gray

    angle = float(np.median(angles))
    if abs(angle) < 0.4:
        return gray

    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    rot = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rot


# -----------------------------
# Preprocess
# -----------------------------
def _adaptive_bin(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 8
    )
    return th


# -----------------------------
# Bubble detection (robust)
# -----------------------------
def _find_bubbles(bin_img: np.ndarray) -> List[Bubble]:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)

    cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []

    areas = np.array([cv2.contourArea(c) for c in cnts], dtype=np.float32)
    a_med = float(np.median(areas))
    a_lo = max(25.0, 0.20 * a_med)
    a_hi = 8.0 * a_med

    bubbles: List[Bubble] = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < a_lo or area > a_hi:
            continue

        x, y, w, h = cv2.boundingRect(c)
        if w < 7 or h < 7:
            continue

        ar = w / float(h)
        if ar < 0.70 or ar > 1.35:
            continue

        per = cv2.arcLength(c, True)
        if per <= 0:
            continue

        circ = (4.0 * math.pi * area) / (per * per + 1e-6)
        if circ < 0.40:
            continue

        r = 0.25 * (w + h)
        bubbles.append(Bubble(x + w/2.0, y + h/2.0, r, (x, y, w, h)))

    return bubbles


def _r_median(bubbles: List[Bubble]) -> float:
    if not bubbles:
        return 12.0
    rs = np.array([b.r for b in bubbles], dtype=np.float32)
    return float(np.median(rs))


# -----------------------------
# 1D clustering (sorted, eps based)
# -----------------------------
def _cluster_sorted(vals_sorted: np.ndarray, idx_sorted: np.ndarray, eps: float) -> List[List[int]]:
    clusters = []
    cur = [int(idx_sorted[0])]
    last_v = float(vals_sorted[0])
    for v, idx in zip(vals_sorted[1:], idx_sorted[1:]):
        v = float(v); idx = int(idx)
        if abs(v - last_v) <= eps:
            cur.append(idx)
        else:
            clusters.append(cur)
            cur = [idx]
        last_v = v
    clusters.append(cur)
    return clusters


def _cluster_1d_indices(values: np.ndarray, eps: float) -> List[List[int]]:
    if len(values) == 0:
        return []
    order = np.argsort(values)
    vals_sorted = values[order]
    idx_sorted = order
    return _cluster_sorted(vals_sorted, idx_sorted, eps=eps)


# -----------------------------
# Detect student code block (vertical digit columns)
# -----------------------------
def detect_code_block(bubbles: List[Bubble], r_med: float) -> Optional[CodeBlock]:
    if len(bubbles) < 50:
        return None

    xs = np.array([b.x for b in bubbles], dtype=np.float32)
    eps_x = 2.6 * r_med
    x_clusters = _cluster_1d_indices(xs, eps=eps_x)

    # candidate columns: around 10 bubbles vertically (allow 7..13)
    candidate_cols = []
    for col in x_clusters:
        if 7 <= len(col) <= 13:
            ys = np.array([bubbles[i].y for i in col], dtype=np.float32)
            if (ys.max() - ys.min()) < 7.0 * r_med:
                continue
            # also require "mostly vertical"
            candidate_cols.append(sorted(col, key=lambda i: bubbles[i].y))

    if len(candidate_cols) < 2:
        return None

    # group columns that are close in x => one block
    centers = np.array([np.mean([bubbles[i].x for i in col]) for col in candidate_cols], dtype=np.float32)
    ordc = np.argsort(centers)

    blocks = []
    cur = [candidate_cols[int(ordc[0])]]
    for k in ordc[1:]:
        k = int(k)
        last_center = np.mean([bubbles[i].x for i in cur[-1]])
        if abs(centers[k] - last_center) <= 3.0 * r_med:
            cur.append(candidate_cols[k])
        else:
            blocks.append(cur)
            cur = [candidate_cols[k]]
    blocks.append(cur)

    # choose best block: many cols + near top
    best = None
    best_score = None
    for cols in blocks:
        # bbox
        xs_all = [bubbles[i].x for col in cols for i in col]
        ys_all = [bubbles[i].y for col in cols for i in col]
        x0 = int(max(0, min(xs_all) - 3*r_med))
        y0 = int(max(0, min(ys_all) - 3*r_med))
        x1 = int(max(xs_all) + 3*r_med)
        y1 = int(max(ys_all) + 3*r_med)
        bbox = (x0, y0, int(x1-x0), int(y1-y0))

        # score: prefer more columns + higher on page
        score = (len(cols) * 1000) - y0
        if best_score is None or score > best_score:
            best_score = score
            best = CodeBlock(cols=cols, bbox=bbox)

    return best


# -----------------------------
# Bubble fill ratio (inside inner circle)
# -----------------------------
def bubble_fill_ratio(gray: np.ndarray, bubble: Bubble) -> float:
    x, y, w, h = bubble.bbox
    pad = int(max(2, bubble.r * 0.35))
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(gray.shape[1], x + w + pad); y1 = min(gray.shape[0], y + h + pad)
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0

    patch = cv2.GaussianBlur(patch, (3, 3), 0)
    _, th = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cx = int(round(bubble.x - x0))
    cy = int(round(bubble.y - y0))
    rr = int(max(4, round(bubble.r * 0.65)))  # inner circle: avoid border ring

    mask = np.zeros_like(th, dtype=np.uint8)
    cv2.circle(mask, (cx, cy), rr, 255, -1)

    inside = th[mask == 255]
    if inside.size == 0:
        return 0.0
    return float(np.mean(inside > 0))


# -----------------------------
# Cancel detection (stroke/X) inside bubble
# -----------------------------
def bubble_cancel_strength(gray: np.ndarray, bubble: Bubble) -> float:
    x, y, w, h = bubble.bbox
    pad = int(max(2, bubble.r * 0.55))
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(gray.shape[1], x + w + pad); y1 = min(gray.shape[0], y + h + pad)
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0

    patch = cv2.GaussianBlur(patch, (3, 3), 0)
    edges = cv2.Canny(patch, 60, 160)

    H, W = edges.shape[:2]
    cx, cy = W // 2, H // 2
    rr = int(max(6, bubble.r * 0.85))

    mask = np.zeros_like(edges, dtype=np.uint8)
    cv2.circle(mask, (cx, cy), rr, 255, -1)

    inside = edges[mask == 255]
    if inside.size == 0:
        return 0.0
    return float(np.mean(inside > 0))


# -----------------------------
# Decode student code from code block
# -----------------------------
def decode_student_code(gray: np.ndarray, bubbles: List[Bubble], block: CodeBlock) -> str:
    # choose best digit per column by fill ratio
    digits = []
    min_fill = 0.22
    min_margin = 0.07

    for col in block.cols:
        col_sorted = sorted(col, key=lambda i: bubbles[i].y)
        scores = [bubble_fill_ratio(gray, bubbles[i]) for i in col_sorted]
        if not scores:
            digits.append("")
            continue
        best = int(np.argmax(scores))
        best_v = float(scores[best])
        s_sorted = sorted(scores, reverse=True)
        second = float(s_sorted[1]) if len(s_sorted) > 1 else 0.0

        if best_v < min_fill or (best_v - second) < min_margin:
            digits.append("")
        else:
            # map index to digit: if exact 10 rows => digit=best
            if len(col_sorted) == 10:
                digits.append(str(best))
            else:
                digits.append(str(int(round(best * 9 / max(1, (len(col_sorted)-1))))))

    code = "".join(digits).strip()
    return code if code else "UNKNOWN"


# -----------------------------
# Adaptive thresholds per page (auto)
# -----------------------------
def _adaptive_fill_thresholds(fill_values: List[float]) -> Tuple[float, float]:
    """
    Returns (light_thr, full_thr) based on distribution of fill ratios.
    Works across pens/scans.
    """
    if not fill_values:
        return 0.18, 0.52

    v = np.clip(np.array(fill_values, dtype=np.float32), 0, 1)
    # Otsu on scaled values
    x = np.clip((v * 255).astype(np.uint8), 0, 255)
    # histogram threshold
    _, thr = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = float(thr) / 255.0

    # light below otsu, full above otsu, but keep safe bounds
    full_thr = max(0.45, min(0.70, otsu + 0.08))
    light_thr = max(0.12, min(0.30, otsu * 0.65))
    return light_thr, full_thr


# -----------------------------
# Column segmentation for questions (multi-column)
# -----------------------------
def _segment_columns(bubbles: List[Bubble], exclude_bbox: Optional[Tuple[int,int,int,int]], r_med: float) -> List[List[int]]:
    """
    Returns list of columns, each is list of bubble indices (excluding code area).
    """
    idxs = []
    for i, b in enumerate(bubbles):
        if exclude_bbox is not None:
            x,y,w,h = exclude_bbox
            if x <= b.x <= x+w and y <= b.y <= y+h:
                continue
        idxs.append(i)

    if not idxs:
        return []

    xs = np.array([bubbles[i].x for i in idxs], dtype=np.float32)

    # cluster x into bands (columns) using eps based on r
    eps_x = 6.5 * r_med  # bigger than option spacing; separates distinct columns
    clusters_local = _cluster_1d_indices(xs, eps=eps_x)

    columns = []
    for cl in clusters_local:
        col_idxs = [idxs[k] for k in cl]
        if len(col_idxs) < 20:
            continue
        # sanity: y span should be big (questions run down)
        ys = np.array([bubbles[i].y for i in col_idxs], dtype=np.float32)
        if (ys.max() - ys.min()) < 15.0 * r_med:
            continue
        columns.append(col_idxs)

    # sort columns left->right
    columns.sort(key=lambda col: float(np.mean([bubbles[i].x for i in col])))
    return columns


# -----------------------------
# Build question groups within a column
# -----------------------------
def _build_question_groups_in_column(bubbles: List[Bubble], col_idxs: List[int], r_med: float) -> List[List[int]]:
    """
    Returns list of groups; each group = option bubble indices of one question.
    """
    ys = np.array([bubbles[i].y for i in col_idxs], dtype=np.float32)
    # cluster y into rows (questions)
    eps_y = 2.8 * r_med
    y_clusters_local = _cluster_1d_indices(ys, eps=eps_y)

    groups = []
    for cl in y_clusters_local:
        row_idxs = [col_idxs[k] for k in cl]
        if len(row_idxs) < 3:
            continue
        row_idxs = sorted(row_idxs, key=lambda i: bubbles[i].x)

        # within the row, split into groups if there are large gaps
        xs = np.array([bubbles[i].x for i in row_idxs], dtype=np.float32)
        gaps = np.diff(xs) if len(xs) > 1 else np.array([], dtype=np.float32)
        if len(gaps) == 0:
            continue

        gap_thr = 3.8 * r_med  # separates different question blocks (rare)
        cur = [row_idxs[0]]
        for j in range(1, len(row_idxs)):
            if (bubbles[row_idxs[j]].x - bubbles[cur[-1]].x) > gap_thr:
                groups.append(cur)
                cur = [row_idxs[j]]
            else:
                cur.append(row_idxs[j])
        groups.append(cur)

    # keep option-like sizes (3..6) and stable x-span
    cleaned = []
    for g in groups:
        if 3 <= len(g) <= 6:
            cleaned.append(g)

    # sort by vertical position then x
    cleaned.sort(key=lambda g: (float(np.mean([bubbles[i].y for i in g])), float(np.mean([bubbles[i].x for i in g]))))
    return cleaned


def _infer_num_options(all_groups: List[List[int]]) -> int:
    if not all_groups:
        return 4
    sizes = [len(g) for g in all_groups]
    med = int(round(float(np.median(sizes))))
    if med <= 4:
        return 4
    return 5


# -----------------------------
# Decode answers with your rules (FULL beats LIGHT, cancelled excluded)
# -----------------------------
def decode_answers(
    gray: np.ndarray,
    bubbles: List[Bubble],
    question_groups: List[List[int]],
    option_labels: str,
    num_opts: int,
    strong_margin: float = 0.12,
    cancel_thr: float = 0.06,
    light_thr: float = 0.18,
    full_thr: float = 0.52
) -> Tuple[List[str], Dict]:
    answers = []
    dbg = {"q": []}

    for qi, g in enumerate(question_groups, start=1):
        g_sorted = sorted(g, key=lambda i: bubbles[i].x)

        # if noise adds bubbles, choose most compact window of size num_opts
        if len(g_sorted) > num_opts:
            best_span = None
            best = None
            for s in range(0, len(g_sorted) - (num_opts - 1)):
                win = g_sorted[s:s+num_opts]
                span = bubbles[win[-1]].x - bubbles[win[0]].x
                if best_span is None or span < best_span:
                    best_span = span
                    best = win
            g_sorted = best

        # if still not enough
        if len(g_sorted) < 3:
            answers.append("")
            dbg["q"].append({"q": qi, "flag": "TOO_FEW_OPTIONS"})
            continue

        fills = [bubble_fill_ratio(gray, bubbles[i]) for i in g_sorted]
        cancS = [bubble_cancel_strength(gray, bubbles[i]) for i in g_sorted]
        cancelled = [cs >= cancel_thr for cs in cancS]

        # valid indices
        valid = [k for k in range(len(g_sorted)) if not cancelled[k]]
        if not valid:
            answers.append("")
            dbg["q"].append({"q": qi, "fills": fills, "cancelS": cancS, "flag": "ALL_CANCELLED"})
            continue

        ranked = sorted(valid, key=lambda k: fills[k], reverse=True)
        top = ranked[0]
        topv = float(fills[top])
        secondv = float(fills[ranked[1]]) if len(ranked) > 1 else 0.0

        # decision:
        if topv < light_thr:
            ans = ""
            flag = "BLANK"
        else:
            # if top is FULL and clearly stronger => take it (this fixes your Q5 case)
            if topv >= full_thr and (topv - secondv) >= strong_margin:
                ans = option_labels[top] if top < len(option_labels) else str(top)
                flag = "OK_STRONG_FULL"
            else:
                # if multiple FULL close => MULTI
                full_valid = [k for k in valid if fills[k] >= full_thr]
                if len(full_valid) >= 2:
                    ans = "MULTI"
                    flag = "MULTI_FULL"
                else:
                    # no clear FULL: if top beats second by small margin => choose as LIGHT_SELECTED else AMB
                    if (topv - secondv) >= 0.06:
                        ans = option_labels[top] if top < len(option_labels) else str(top)
                        flag = "LIGHT_SELECTED"
                    else:
                        ans = "AMB"
                        flag = "AMB_LOW_MARGIN"

        answers.append(ans)
        dbg["q"].append({
            "q": qi,
            "fills": [float(x) for x in fills],
            "cancelS": [float(x) for x in cancS],
            "cancelled": cancelled,
            "top": int(top),
            "topv": float(topv),
            "secondv": float(secondv),
            "flag": flag
        })

    return answers, dbg


# -----------------------------
# Process one page
# -----------------------------
def process_page(page_bgr: np.ndarray, strong_margin: float = 0.12) -> Dict:
    gray0 = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    gray = _deskew(gray0)
    bin_img = _adaptive_bin(gray)

    bubbles = _find_bubbles(bin_img)
    flags = []

    if len(bubbles) < 40:
        return {
            "student_code": "UNKNOWN",
            "answers": [],
            "num_questions": 0,
            "flags": ["TOO_FEW_BUBBLES"],
            "debug": {"bubbles_found": len(bubbles)}
        }

    r_med = _r_median(bubbles)

    # detect code block
    code_block = detect_code_block(bubbles, r_med=r_med)
    exclude_bbox = None
    if code_block is not None:
        # exclude only if bbox is not huge
        H, W = gray.shape[:2]
        x,y,w,h = code_block.bbox
        if (w*h) < 0.20 * (W*H):
            exclude_bbox = code_block.bbox

    # student code
    student_code = "UNKNOWN"
    if code_block is not None:
        student_code = decode_student_code(gray, bubbles, code_block)

    # segment question columns
    cols = _segment_columns(bubbles, exclude_bbox=exclude_bbox, r_med=r_med)
    if not cols:
        flags.append("NO_COLUMNS_FOUND")

    # build question groups from all columns
    all_groups: List[List[int]] = []
    for col in cols:
        gs = _build_question_groups_in_column(bubbles, col, r_med=r_med)
        all_groups.extend(gs)

    # if still low, try fallback without excluding bbox
    if len(all_groups) < 25 and exclude_bbox is not None:
        cols2 = _segment_columns(bubbles, exclude_bbox=None, r_med=r_med)
        all_groups2 = []
        for col in cols2:
            all_groups2.extend(_build_question_groups_in_column(bubbles, col, r_med=r_med))
        if len(all_groups2) > len(all_groups):
            all_groups = all_groups2
            flags.append("FALLBACK_NO_EXCLUDE_USED")

    if not all_groups:
        flags.append("NO_QUESTION_GROUPS_FOUND")
        return {
            "student_code": student_code,
            "answers": [],
            "num_questions": 0,
            "flags": flags,
            "debug": {"bubbles_found": len(bubbles), "r_med": r_med}
        }

    # infer options count (4 or 5)
    num_opts = _infer_num_options(all_groups)
    option_labels = "ABCDE"  # supports up to 5

    # adaptive fill thresholds using random sample of group bubbles
    sample_fills = []
    for g in all_groups[:min(120, len(all_groups))]:
        for i in g[:min(6, len(g))]:
            sample_fills.append(bubble_fill_ratio(gray, bubbles[i]))
    light_thr, full_thr = _adaptive_fill_thresholds(sample_fills)

    answers, ans_dbg = decode_answers(
        gray, bubbles, all_groups,
        option_labels=option_labels,
        num_opts=num_opts,
        strong_margin=strong_margin,
        cancel_thr=0.06,
        light_thr=light_thr,
        full_thr=full_thr
    )

    # sanity flags
    amb = sum(1 for a in answers if a == "AMB")
    multi = sum(1 for a in answers if a == "MULTI")
    if len(answers) > 0 and (amb + multi) / max(1, len(answers)) > 0.10:
        flags.append("MANY_AMB_OR_MULTI")
    if len(answers) < 0.6 * len(all_groups):
        flags.append("LOW_DECODE_RATIO")

    return {
        "student_code": student_code,
        "answers": answers,
        "num_questions": len(answers),
        "flags": flags,
        "debug": {
            "bubbles_found": len(bubbles),
            "r_med": r_med,
            "num_columns": len(cols),
            "groups_found": len(all_groups),
            "num_opts": num_opts,
            "light_thr": float(light_thr),
            "full_thr": float(full_thr),
            "answers_debug": ans_dbg
        }
    }


# -----------------------------
# Process PDF
# -----------------------------
def process_pdf(pdf_path: str, out_csv: str = "bubble_results.csv", zoom: float = 2.6, strong_margin: float = 0.12) -> pd.DataFrame:
    pages = pdf_to_images(pdf_path, zoom=zoom)
    rows_out = []

    for i, img in enumerate(pages):
        res = process_page(img, strong_margin=strong_margin)
        rows_out.append({
            "file": os.path.basename(pdf_path),
            "page": i + 1,
            "student_code": res["student_code"],
            "num_questions": res["num_questions"],
            "answers": " ".join(res["answers"]),
            "flags": ",".join(res["flags"]) if res["flags"] else ""
        })

    df = pd.DataFrame(rows_out)
    try:
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    except Exception:
        pass
    return df
