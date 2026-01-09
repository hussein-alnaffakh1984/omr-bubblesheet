"""
Template-Free (General) Bubble Sheet Reader
- Works without hardcoding "60/70/95" or fixed coordinates.
- Detects bubbles, groups them into rows/columns, separates:
  1) Student code blocks (vertical stacks of ~10 bubbles: 0..9)
  2) Answer blocks (rows with ~4/5 bubbles per question)
- Outputs per page:
  - student_code (best-effort)
  - answers list (e.g., ['B','D','', ...])
  - confidence flags

Tested idea: run on your PDFs (e.g., "التجميل.pdf", "قسم التخدير.pdf", ...).
You may need to tune only a few thresholds depending on scan quality.

Install (if needed):
  pip install opencv-python numpy pymupdf pandas
"""

import os
import math
import cv2
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


# -----------------------------
# PDF -> images
# -----------------------------
def pdf_to_images(pdf_path: str, zoom: float = 2.5) -> List[np.ndarray]:
    """
    Render each PDF page into a BGR image using PyMuPDF.
    zoom=2.5 ~ 180-220 dpi-ish depending on original.
    """
    doc = fitz.open(pdf_path)
    imgs = []
    mat = fitz.Matrix(zoom, zoom)
    for i in range(len(doc)):
        pix = doc.load_page(i).get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        imgs.append(img[:, :, ::-1].copy())  # RGB->BGR
    return imgs


# -----------------------------
# Utilities
# -----------------------------
@dataclass
class Bubble:
    x: float
    y: float
    r: float
    bbox: Tuple[int, int, int, int]  # (x,y,w,h)


def _adaptive_bin(gray: np.ndarray) -> np.ndarray:
    # Strong binarization for scanned sheets
    # Invert: bubbles become white-ish / easier for contour detection after cleaning
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 8
    )
    return th


def _find_bubbles(bin_img: np.ndarray) -> List[Bubble]:
    """
    Detect circular-ish contours, return bubble candidates.
    """
    # Clean tiny noise / connect broken circles slightly
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)

    cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubbles = []

    areas = [cv2.contourArea(c) for c in cnts]
    if not areas:
        return bubbles

    # Robust area band (ignore huge/very small)
    a = np.array(areas, dtype=np.float32)
    a_med = float(np.median(a))
    a_lo = max(20.0, 0.25 * a_med)
    a_hi = 6.0 * a_med

    for c in cnts:
        area = cv2.contourArea(c)
        if area < a_lo or area > a_hi:
            continue

        x, y, w, h = cv2.boundingRect(c)
        if w < 6 or h < 6:
            continue

        # circularity / aspect
        ar = w / float(h)
        if ar < 0.7 or ar > 1.35:
            continue

        per = cv2.arcLength(c, True)
        if per <= 0:
            continue
        circ = (4.0 * math.pi * area) / (per * per)  # 1.0 ideal circle
        if circ < 0.45:
            continue

        r = 0.5 * (w + h) / 2.0
        bubbles.append(Bubble(x + w / 2.0, y + h / 2.0, r, (x, y, w, h)))

    return bubbles


def _cluster_1d(vals: np.ndarray, tol: float) -> List[List[int]]:
    """
    Cluster indices by proximity on 1D axis.
    vals: shape (N,) floats
    tol: max distance to be in same cluster
    """
    if len(vals) == 0:
        return []
    order = np.argsort(vals)
    clusters = []
    cur = [int(order[0])]
    for idx in order[1:]:
        idx = int(idx)
        if abs(vals[idx] - vals[cur[-1]]) <= tol:
            cur.append(idx)
        else:
            clusters.append(cur)
            cur = [idx]
    clusters.append(cur)
    return clusters


def _estimate_spacing(bubbles: List[Bubble]) -> Dict[str, float]:
    """
    Estimate typical bubble radius and typical x/y neighbor spacing.
    """
    rs = np.array([b.r for b in bubbles], dtype=np.float32)
    r_med = float(np.median(rs)) if len(rs) else 10.0

    xs = np.sort(np.array([b.x for b in bubbles], dtype=np.float32))
    ys = np.sort(np.array([b.y for b in bubbles], dtype=np.float32))

    def median_diff(arr):
        if len(arr) < 2:
            return 20.0
        d = np.diff(arr)
        d = d[d > 1.0]
        return float(np.median(d)) if len(d) else 20.0

    dx = median_diff(xs)
    dy = median_diff(ys)

    # Clamp to sane
    dx = max(10.0, min(dx, 200.0))
    dy = max(10.0, min(dy, 200.0))

    return {"r": r_med, "dx": dx, "dy": dy}


def _roi_fill_score(gray: np.ndarray, bubble: Bubble) -> float:
    """
    Compute fill score inside bubble area.
    Higher = more filled/darker.
    Uses a circular mask; compares dark pixels ratio.
    """
    x, y, w, h = bubble.bbox
    pad = int(max(2, bubble.r * 0.25))
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(gray.shape[1], x + w + pad); y1 = min(gray.shape[0], y + h + pad)

    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0

    # local normalize
    patch_blur = cv2.GaussianBlur(patch, (3, 3), 0)
    # Otsu inside patch
    _, th = cv2.threshold(patch_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # circle mask centered at bubble center within patch
    cx = int(round(bubble.x - x0))
    cy = int(round(bubble.y - y0))
    rr = int(max(4, round(bubble.r * 0.75)))

    mask = np.zeros_like(th, dtype=np.uint8)
    cv2.circle(mask, (cx, cy), rr, 255, -1)

    # In TH (binary), ink is usually darker => 0. We want dark ratio.
    inside = th[mask == 255]
    if inside.size == 0:
        return 0.0
    dark_ratio = float(np.mean(inside == 0))
    return dark_ratio


# -----------------------------
# Structure detection
# -----------------------------
@dataclass
class CodeBlock:
    cols: List[List[int]]  # list of columns (each is list of bubble indices)
    bbox: Tuple[int, int, int, int]


@dataclass
class QuestionRow:
    groups: List[List[int]]  # each group are bubble indices representing options for a question
    y_mean: float


def detect_code_blocks(bubbles: List[Bubble], tol_x: float, tol_y: float) -> List[CodeBlock]:
    """
    Find student-code like blocks:
      - several vertical columns
      - each column has about 10 bubbles stacked with regular y spacing
    """
    if not bubbles:
        return []

    xs = np.array([b.x for b in bubbles], dtype=np.float32)
    x_clusters = _cluster_1d(xs, tol=tol_x)

    # Candidate columns are x-clusters with ~10 bubbles
    candidate_cols = []
    for col in x_clusters:
        if 7 <= len(col) <= 13:
            # check if y spread is large enough and roughly monotonic
            ys = np.array([bubbles[i].y for i in col], dtype=np.float32)
            ys_sorted = np.sort(ys)
            if (ys_sorted[-1] - ys_sorted[0]) < 6 * tol_y:
                continue
            candidate_cols.append(sorted(col, key=lambda i: bubbles[i].y))

    # Group nearby columns into blocks by x proximity
    blocks = []
    if not candidate_cols:
        return blocks

    col_centers = np.array([np.mean([bubbles[i].x for i in col]) for col in candidate_cols], dtype=np.float32)
    col_order = np.argsort(col_centers)

    cur_cols = [candidate_cols[int(col_order[0])]]
    for k in col_order[1:]:
        k = int(k)
        if abs(col_centers[k] - np.mean([bubbles[i].x for i in cur_cols[-1]])) <= 2.5 * tol_x:
            cur_cols.append(candidate_cols[k])
        else:
            blocks.append(cur_cols)
            cur_cols = [candidate_cols[k]]
    blocks.append(cur_cols)

    code_blocks = []
    for cols in blocks:
        # Must have at least 2 columns for a "code" region (common)
        if len(cols) < 2:
            continue

        xs_all, ys_all = [], []
        for col in cols:
            for i in col:
                xs_all.append(bubbles[i].x)
                ys_all.append(bubbles[i].y)

        x0 = int(max(0, min(xs_all) - 3 * tol_x))
        y0 = int(max(0, min(ys_all) - 3 * tol_y))
        x1 = int(max(xs_all) + 3 * tol_x)
        y1 = int(max(ys_all) + 3 * tol_y)

        code_blocks.append(CodeBlock(cols=cols, bbox=(x0, y0, x1 - x0, y1 - y0)))

    # Sort by "upper-right-ish" priority (common in many sheets)
    code_blocks.sort(key=lambda b: (b.bbox[1], -b.bbox[0]))
    return code_blocks


def detect_question_rows(bubbles: List[Bubble], exclude_bboxes: List[Tuple[int, int, int, int]], tol_y: float, tol_x: float) -> List[QuestionRow]:
    """
    Find rows of bubbles (questions) outside code blocks.
    Then split row bubbles into option-groups (each group ~4-5 bubbles).
    """
    if not bubbles:
        return []

    def in_any_bbox(b: Bubble) -> bool:
        for (x, y, w, h) in exclude_bboxes:
            if x <= b.x <= x + w and y <= b.y <= y + h:
                return True
        return False

    idxs = [i for i, b in enumerate(bubbles) if not in_any_bbox(b)]
    if not idxs:
        return []

    ys = np.array([bubbles[i].y for i in idxs], dtype=np.float32)
    # Cluster by y to form rows
    # tol_y should be around 0.6~1.0 of typical y spacing
    row_clusters_local = _cluster_1d(ys, tol=tol_y)

    rows: List[QuestionRow] = []
    for cluster in row_clusters_local:
        row_idxs = [idxs[i] for i in cluster]
        if len(row_idxs) < 3:
            continue

        # Sort by x
        row_idxs = sorted(row_idxs, key=lambda i: bubbles[i].x)
        y_mean = float(np.mean([bubbles[i].y for i in row_idxs]))

        # Split into groups by x gaps
        xs_row = np.array([bubbles[i].x for i in row_idxs], dtype=np.float32)
        gaps = np.diff(xs_row)
        if len(gaps) == 0:
            continue
        # A "big gap" likely separates question groups or columns
        gap_thr = max(2.2 * tol_x, float(np.median(gaps) * 2.5))

        groups = []
        cur = [row_idxs[0]]
        for j in range(1, len(row_idxs)):
            if (bubbles[row_idxs[j]].x - bubbles[cur[-1]].x) > gap_thr:
                groups.append(cur)
                cur = [row_idxs[j]]
            else:
                cur.append(row_idxs[j])
        groups.append(cur)

        # Keep groups that look like options: 3..6 bubbles (common 4 or 5)
        groups = [g for g in groups if 3 <= len(g) <= 6]
        if not groups:
            continue

        rows.append(QuestionRow(groups=groups, y_mean=y_mean))

    # sort rows top to bottom
    rows.sort(key=lambda r: r.y_mean)
    return rows


# -----------------------------
# Decode code + answers
# -----------------------------
def decode_student_code(gray: np.ndarray, bubbles: List[Bubble], code_block: CodeBlock) -> Tuple[str, Dict]:
    """
    For each column, pick the most filled bubble -> digit (0..9)
    Returns code string and debug info.
    """
    digits = []
    debug = {"cols": []}

    for col in code_block.cols:
        # Sort by y ascending; assume digits 0..9 in order top->bottom (common)
        col_sorted = sorted(col, key=lambda i: bubbles[i].y)
        scores = [(_roi_fill_score(gray, bubbles[i])) for i in col_sorted]

        # Best index
        best_k = int(np.argmax(scores))
        best_score = float(scores[best_k])

        # Decide with thresholds:
        # - require some minimum fill
        # - require separation from 2nd best
        s_sorted = sorted(scores, reverse=True)
        second = float(s_sorted[1]) if len(s_sorted) > 1 else 0.0

        # You can tune these:
        min_fill = 0.22
        min_margin = 0.08

        if best_score < min_fill or (best_score - second) < min_margin:
            digits.append("")  # uncertain/empty
            debug["cols"].append({"scores": scores, "digit": None, "best": best_score, "second": second})
        else:
            # map position -> digit
            # If column has 10 bubbles => positions 0..9
            # If not exactly 10, map by rank to nearest digit indices
            if len(col_sorted) == 10:
                digit = best_k
            else:
                digit = int(round(best_k * 9 / max(1, (len(col_sorted) - 1))))
            digits.append(str(digit))
            debug["cols"].append({"scores": scores, "digit": digit, "best": best_score, "second": second})

    code = "".join(digits).strip()
    if code == "":
        code = "UNKNOWN"
    return code, debug


def decode_answers(gray: np.ndarray, bubbles: List[Bubble], rows: List[QuestionRow], option_labels: str = "ABCDE") -> Tuple[List[str], Dict]:
    """
    For each question-group in each row: pick the most filled bubble -> label
    If uncertain or multiple: return "" or "MULTI"
    """
    answers = []
    dbg = {"questions": []}

    # Tuning
    min_fill = 0.22
    min_margin = 0.08
    multi_margin = 0.03  # if top two are too close, call ambiguous

    q_index = 0
    for r in rows:
        for g in r.groups:
            g_sorted = sorted(g, key=lambda i: bubbles[i].x)  # options left->right
            scores = [(_roi_fill_score(gray, bubbles[i])) for i in g_sorted]

            best = int(np.argmax(scores))
            best_score = float(scores[best])
            s_sorted = sorted(scores, reverse=True)
            second = float(s_sorted[1]) if len(s_sorted) > 1 else 0.0

            if best_score < min_fill:
                ans = ""  # blank
                flag = "BLANK_OR_WEAK"
            elif (best_score - second) < multi_margin:
                ans = "AMB"  # ambiguous
                flag = "AMBIGUOUS"
            elif (best_score - second) < min_margin:
                ans = "AMB"
                flag = "LOW_MARGIN"
            else:
                ans = option_labels[best] if best < len(option_labels) else str(best)
                flag = "OK"

            answers.append(ans)
            dbg["questions"].append({
                "q": q_index + 1,
                "scores": scores,
                "best": best_score,
                "second": second,
                "flag": flag
            })
            q_index += 1

    return answers, dbg


# -----------------------------
# Main pipeline per page
# -----------------------------
def process_page(page_bgr: np.ndarray) -> Dict:
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    bin_img = _adaptive_bin(gray)
    bubbles = _find_bubbles(bin_img)

    if len(bubbles) < 20:
        return {
            "student_code": "UNKNOWN",
            "num_questions": 0,
            "answers": [],
            "flags": ["TOO_FEW_BUBBLES"],
            "debug": {"bubbles_found": len(bubbles)}
        }

    spacing = _estimate_spacing(bubbles)
    r = spacing["r"]
    tol_x = max(8.0, 0.85 * spacing["dx"])
    tol_y = max(8.0, 0.85 * spacing["dy"])

    # 1) Detect code blocks
    code_blocks = detect_code_blocks(bubbles, tol_x=tol_x, tol_y=tol_y)
    exclude = [cb.bbox for cb in code_blocks[:1]]  # use best block only; you can expand later

    # 2) Detect question rows (excluding code bbox)
    rows = detect_question_rows(bubbles, exclude_bboxes=exclude, tol_y=tol_y, tol_x=tol_x)

    flags = []
    if not code_blocks:
        flags.append("NO_CODE_BLOCK_FOUND")
    if not rows:
        flags.append("NO_QUESTION_ROWS_FOUND")

    # 3) Decode
    if code_blocks:
        student_code, code_dbg = decode_student_code(gray, bubbles, code_blocks[0])
    else:
        student_code, code_dbg = "UNKNOWN", {}

    answers, ans_dbg = decode_answers(gray, bubbles, rows, option_labels="ABCDE")

    # 4) Consistency checks
    # If too many ambiguous, suggest rescan
    amb = sum(1 for a in answers if a in ("AMB",))
    if len(answers) > 0 and amb / max(1, len(answers)) > 0.10:
        flags.append("MANY_AMBIGUOUS_RESCan_SUGGESTED")

    return {
        "student_code": student_code,
        "num_questions": len(answers),
        "answers": answers,
        "flags": flags,
        "debug": {
            "spacing": spacing,
            "bubbles_found": len(bubbles),
            "code_debug": code_dbg,
            "answers_debug": ans_dbg,
            "code_bbox": exclude[0] if exclude else None
        }
    }


def process_pdf(pdf_path: str, out_csv: str = "results.csv") -> pd.DataFrame:
    pages = pdf_to_images(pdf_path, zoom=2.5)

    rows_out = []
    for i, img in enumerate(pages):
        res = process_page(img)
        rows_out.append({
            "file": os.path.basename(pdf_path),
            "page": i + 1,
            "student_code": res["student_code"],
            "num_questions": res["num_questions"],
            "answers": " ".join(res["answers"]),
            "flags": ",".join(res["flags"])
        })

    df = pd.DataFrame(rows_out)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return df


# -----------------------------
# Run on your file(s)
# -----------------------------
if __name__ == "__main__":
    # Change this to your file path:
    pdf_path = "التجميل.pdf"  # example
    # Or use an absolute path:
    # pdf_path = r"/mnt/data/التجميل.pdf"

    if not os.path.exists(pdf_path):
        # If you're in a notebook / server environment, your uploads may be in /mnt/data
        alt = os.path.join("/mnt/data", pdf_path)
        if os.path.exists(alt):
            pdf_path = alt

    df = process_pdf(pdf_path, out_csv="bubble_results.csv")
    print(df.head(10))
    print("\nSaved:", "bubble_results.csv")
