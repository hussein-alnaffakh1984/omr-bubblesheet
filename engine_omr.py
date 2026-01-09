import os
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

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
    cols: List[List[int]]  # list of columns (each column = bubble indices)
    bbox: Tuple[int, int, int, int]  # (x,y,w,h)


@dataclass
class QuestionRow:
    groups: List[List[int]]  # each group = bubble indices for options
    y_mean: float


# -----------------------------
# PDF -> images
# -----------------------------
def pdf_to_images(pdf_path: str, zoom: float = 2.5) -> List[np.ndarray]:
    doc = fitz.open(pdf_path)
    imgs = []
    mat = fitz.Matrix(zoom, zoom)
    for i in range(len(doc)):
        pix = doc.load_page(i).get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        imgs.append(img[:, :, ::-1].copy())  # RGB -> BGR
    return imgs


# -----------------------------
# Preprocess & bubble detection
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


def _find_bubbles(bin_img: np.ndarray) -> List[Bubble]:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)

    cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []

    areas = np.array([cv2.contourArea(c) for c in cnts], dtype=np.float32)
    a_med = float(np.median(areas))
    a_lo = max(20.0, 0.25 * a_med)
    a_hi = 6.0 * a_med

    bubbles: List[Bubble] = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < a_lo or area > a_hi:
            continue

        x, y, w, h = cv2.boundingRect(c)
        if w < 6 or h < 6:
            continue

        ar = w / float(h)
        if ar < 0.7 or ar > 1.35:
            continue

        per = cv2.arcLength(c, True)
        if per <= 0:
            continue

        circ = (4.0 * math.pi * area) / (per * per)
        if circ < 0.45:
            continue

        r = 0.5 * (w + h) / 2.0
        bubbles.append(Bubble(x + w / 2.0, y + h / 2.0, r, (x, y, w, h)))

    return bubbles


def _cluster_1d(vals: np.ndarray, tol: float) -> List[List[int]]:
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
    dx = max(10.0, min(dx, 200.0))
    dy = max(10.0, min(dy, 200.0))
    return {"r": r_med, "dx": dx, "dy": dy}


# -----------------------------
# Code block detection
# -----------------------------
def detect_code_blocks(bubbles: List[Bubble], tol_x: float, tol_y: float) -> List[CodeBlock]:
    if not bubbles:
        return []

    xs = np.array([b.x for b in bubbles], dtype=np.float32)
    x_clusters = _cluster_1d(xs, tol=tol_x)

    candidate_cols = []
    for col in x_clusters:
        if 7 <= len(col) <= 13:
            ys = np.array([bubbles[i].y for i in col], dtype=np.float32)
            ys_sorted = np.sort(ys)
            if (ys_sorted[-1] - ys_sorted[0]) < 6 * tol_y:
                continue
            candidate_cols.append(sorted(col, key=lambda i: bubbles[i].y))

    if not candidate_cols:
        return []

    col_centers = np.array([np.mean([bubbles[i].x for i in col]) for col in candidate_cols], dtype=np.float32)
    col_order = np.argsort(col_centers)

    blocks = []
    cur_cols = [candidate_cols[int(col_order[0])]]
    for k in col_order[1:]:
        k = int(k)
        last_center = np.mean([bubbles[i].x for i in cur_cols[-1]])
        if abs(col_centers[k] - last_center) <= 2.5 * tol_x:
            cur_cols.append(candidate_cols[k])
        else:
            blocks.append(cur_cols)
            cur_cols = [candidate_cols[k]]
    blocks.append(cur_cols)

    code_blocks: List[CodeBlock] = []
    for cols in blocks:
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

    code_blocks.sort(key=lambda b: (b.bbox[1], -b.bbox[0]))  # upper-right preference
    return code_blocks


# -----------------------------
# Question rows detection (template-free)
# -----------------------------
def detect_question_rows(bubbles: List[Bubble],
                         exclude_bboxes: List[Tuple[int, int, int, int]],
                         tol_y: float,
                         tol_x: float) -> List[QuestionRow]:
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
    row_clusters_local = _cluster_1d(ys, tol=tol_y)

    rows: List[QuestionRow] = []
    for cluster in row_clusters_local:
        row_idxs = [idxs[i] for i in cluster]
        if len(row_idxs) < 3:
            continue

        row_idxs = sorted(row_idxs, key=lambda i: bubbles[i].x)
        y_mean = float(np.mean([bubbles[i].y for i in row_idxs]))

        xs_row = np.array([bubbles[i].x for i in row_idxs], dtype=np.float32)
        gaps = np.diff(xs_row)
        if len(gaps) == 0:
            continue

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

        groups = [g for g in groups if 3 <= len(g) <= 6]
        if not groups:
            continue

        rows.append(QuestionRow(groups=groups, y_mean=y_mean))

    rows.sort(key=lambda r: r.y_mean)
    return rows


# -----------------------------
# Student code decode
# -----------------------------
def bubble_fill_ratio(gray: np.ndarray, bubble: Bubble) -> float:
    x, y, w, h = bubble.bbox
    pad = int(max(2, bubble.r * 0.30))
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(gray.shape[1], x + w + pad); y1 = min(gray.shape[0], y + h + pad)
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0

    patch = cv2.GaussianBlur(patch, (3, 3), 0)
    _, th = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cx = int(round(bubble.x - x0))
    cy = int(round(bubble.y - y0))
    rr = int(max(4, round(bubble.r * 0.75)))

    mask = np.zeros_like(th, dtype=np.uint8)
    cv2.circle(mask, (cx, cy), rr, 255, -1)
    inside = th[mask == 255]
    if inside.size == 0:
        return 0.0

    return float(np.mean(inside > 0))


def decode_student_code(gray: np.ndarray, bubbles: List[Bubble], code_block: CodeBlock) -> Tuple[str, Dict]:
    digits = []
    debug = {"cols": []}

    # thresholds
    min_fill = 0.22
    min_margin = 0.08

    for col in code_block.cols:
        col_sorted = sorted(col, key=lambda i: bubbles[i].y)
        scores = [bubble_fill_ratio(gray, bubbles[i]) for i in col_sorted]

        best_k = int(np.argmax(scores))
        best_score = float(scores[best_k])
        s_sorted = sorted(scores, reverse=True)
        second = float(s_sorted[1]) if len(s_sorted) > 1 else 0.0

        if best_score < min_fill or (best_score - second) < min_margin:
            digits.append("")
            debug["cols"].append({"scores": scores, "digit": None, "best": best_score, "second": second})
        else:
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


# -----------------------------
# Cancel detection (X/Stroke)
# -----------------------------
def bubble_is_cancelled(gray: np.ndarray, bubble: Bubble) -> bool:
    x, y, w, h = bubble.bbox
    pad = int(max(2, bubble.r * 0.55))
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(gray.shape[1], x + w + pad); y1 = min(gray.shape[0], y + h + pad)
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return False

    patch = cv2.GaussianBlur(patch, (3, 3), 0)
    edges = cv2.Canny(patch, 60, 160)

    min_len = int(max(10, bubble.r * 1.15))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=30,
        minLineLength=min_len,
        maxLineGap=6
    )
    if lines is None:
        return False

    H, W = edges.shape[:2]
    cx, cy = W / 2.0, H / 2.0
    hit = 0

    for (x1l, y1l, x2l, y2l) in lines[:, 0]:
        length = math.hypot(x2l - x1l, y2l - y1l)
        if length < min_len:
            continue
        mx, my = (x1l + x2l) / 2.0, (y1l + y2l) / 2.0
        if abs(mx - cx) < W * 0.28 and abs(my - cy) < H * 0.28:
            hit += 1

    return hit >= 1


# -----------------------------
# Answers decode (FINAL)
# -----------------------------
def decode_answers(gray: np.ndarray,
                  bubbles: List[Bubble],
                  rows: List[QuestionRow],
                  option_labels: str = "ABCDE") -> Tuple[List[str], Dict]:
    """
    يميّز:
    - BLANK
    - LIGHT (تظليل خفيف)
    - FULL (تظليل كامل)
    - CANCEL (ملغاة بخط/X)
    ويقرر إجابة السؤال بدون تخمين.
    """

    fill_light_thr = 0.18
    fill_full_thr  = 0.45

    answers: List[str] = []
    dbg = {"questions": []}

    q_index = 0
    for r in rows:
        for g in r.groups:
            g_sorted = sorted(g, key=lambda i: bubbles[i].x)

            # لو noise زاد فقاعات: نختار أقرب 4 (A-D). إذا عندك 5 خيارات غيّرها إلى 5.
            TARGET_OPTS = 4
            if len(g_sorted) > TARGET_OPTS:
                best_span = None
                best_window = None
                for s in range(0, len(g_sorted) - (TARGET_OPTS - 1)):
                    window = g_sorted[s:s + TARGET_OPTS]
                    span = bubbles[window[-1]].x - bubbles[window[0]].x
                    if best_span is None or span < best_span:
                        best_span = span
                        best_window = window
                g_sorted = best_window

            if len(g_sorted) < 3:
                continue

            fills = []
            states = []
            for idx in g_sorted:
                f = bubble_fill_ratio(gray, bubbles[idx])
                canc = bubble_is_cancelled(gray, bubbles[idx])
                fills.append(float(f))

                if canc:
                    states.append("CANCEL")
                elif f >= fill_full_thr:
                    states.append("FULL")
                elif f >= fill_light_thr:
                    states.append("LIGHT")
                else:
                    states.append("BLANK")

            full_idxs = [k for k, s in enumerate(states) if s == "FULL"]
            light_idxs = [k for k, s in enumerate(states) if s == "LIGHT"]

            if len(full_idxs) == 1:
                k = full_idxs[0]
                ans = option_labels[k] if k < len(option_labels) else str(k)
                flag = "OK_FULL"
            elif len(full_idxs) >= 2:
                ans = "MULTI"
                flag = "MULTI_FULL"
            else:
                if len(light_idxs) == 1:
                    k = light_idxs[0]
                    ans = option_labels[k] if k < len(option_labels) else str(k)
                    flag = "LIGHT_SELECTED"
                elif len(light_idxs) >= 2:
                    ans = "AMB"
                    flag = "AMB_LIGHT"
                else:
                    ans = ""
                    flag = "BLANK"

            answers.append(ans)
            dbg["questions"].append({"q": q_index + 1, "fills": fills, "states": states, "flag": flag})
            q_index += 1

    return answers, dbg


# -----------------------------
# Main per page
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
    tol_x = max(8.0, 0.85 * spacing["dx"])
    tol_y = max(8.0, 0.85 * spacing["dy"])

    code_blocks = detect_code_blocks(bubbles, tol_x=tol_x, tol_y=tol_y)
    exclude = [cb.bbox for cb in code_blocks[:1]]

    rows = detect_question_rows(bubbles, exclude_bboxes=exclude, tol_y=tol_y, tol_x=tol_x)

    flags = []
    if not code_blocks:
        flags.append("NO_CODE_BLOCK_FOUND")
    if not rows:
        flags.append("NO_QUESTION_ROWS_FOUND")

    if code_blocks:
        student_code, code_dbg = decode_student_code(gray, bubbles, code_blocks[0])
    else:
        student_code, code_dbg = "UNKNOWN", {}

    answers, ans_dbg = decode_answers(gray, bubbles, rows, option_labels="ABCDE")

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


def process_pdf(pdf_path: str, out_csv: str = "bubble_results.csv", zoom: float = 2.5) -> pd.DataFrame:
    pages = pdf_to_images(pdf_path, zoom=zoom)
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
    try:
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    except Exception:
        pass
    return df
