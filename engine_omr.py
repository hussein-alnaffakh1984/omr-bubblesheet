# =========================================================
# Smart Template-Free OMR Engine (FINAL VERSION)
# =========================================================

import os
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

import cv2
import numpy as np
import pandas as pd
import fitz  # PyMuPDF


# =========================================================
# Data Structures
# =========================================================

@dataclass
class Bubble:
    x: float
    y: float
    r: float
    bbox: Tuple[int, int, int, int]  # (x,y,w,h)


@dataclass
class CodeBlock:
    cols: List[List[int]]
    bbox: Tuple[int, int, int, int]


@dataclass
class QuestionRow:
    groups: List[List[int]]
    y_mean: float


# =========================================================
# PDF -> Images
# =========================================================

def pdf_to_images(pdf_path: str, zoom: float = 2.5) -> List[np.ndarray]:
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(zoom, zoom)
    images = []

    for i in range(len(doc)):
        pix = doc.load_page(i).get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3
        )
        images.append(img[:, :, ::-1].copy())  # RGB → BGR

    return images


# =========================================================
# Bubble Detection
# =========================================================

def _adaptive_bin(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 8
    )


def _find_bubbles(bin_img) -> List[Bubble]:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=1)

    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []

    areas = np.array([cv2.contourArea(c) for c in cnts], dtype=np.float32)
    med = np.median(areas)
    lo, hi = max(20, 0.25 * med), 6 * med

    bubbles = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < lo or area > hi:
            continue

        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h)
        if not (0.7 <= ar <= 1.35):
            continue

        per = cv2.arcLength(c, True)
        circ = (4 * math.pi * area) / (per * per + 1e-6)
        if circ < 0.45:
            continue

        r = (w + h) / 4.0
        bubbles.append(Bubble(x + w/2, y + h/2, r, (x, y, w, h)))

    return bubbles


# =========================================================
# Utility
# =========================================================

def _cluster_1d(vals, tol):
    order = np.argsort(vals)
    clusters = []
    cur = [order[0]]

    for i in order[1:]:
        if abs(vals[i] - vals[cur[-1]]) <= tol:
            cur.append(i)
        else:
            clusters.append(cur)
            cur = [i]
    clusters.append(cur)
    return clusters


# =========================================================
# Code Block Detection
# =========================================================

def detect_code_blocks(bubbles, tol_x, tol_y):
    xs = np.array([b.x for b in bubbles])
    x_clusters = _cluster_1d(xs, tol_x)

    cols = []
    for c in x_clusters:
        if 7 <= len(c) <= 13:
            cols.append(sorted(c, key=lambda i: bubbles[i].y))

    if len(cols) < 2:
        return []

    all_x = [bubbles[i].x for col in cols for i in col]
    all_y = [bubbles[i].y for col in cols for i in col]

    bbox = (
        int(min(all_x) - 3 * tol_x),
        int(min(all_y) - 3 * tol_y),
        int(max(all_x) - min(all_x) + 6 * tol_x),
        int(max(all_y) - min(all_y) + 6 * tol_y)
    )

    return [CodeBlock(cols=cols, bbox=bbox)]


# =========================================================
# Question Rows
# =========================================================

def detect_question_rows(bubbles, exclude_bbox, tol_x, tol_y):
    rows = []
    for i, b in enumerate(bubbles):
        if exclude_bbox:
            x, y, w, h = exclude_bbox
            if x <= b.x <= x + w and y <= b.y <= y + h:
                continue
        rows.append(i)

    ys = np.array([bubbles[i].y for i in rows])
    y_clusters = _cluster_1d(ys, tol_y)

    qrows = []
    for cl in y_clusters:
        idxs = [rows[i] for i in cl]
        if len(idxs) < 3:
            continue

        idxs = sorted(idxs, key=lambda i: bubbles[i].x)
        groups = [idxs]  # single group per row
        qrows.append(QuestionRow(groups=groups, y_mean=np.mean(ys[cl])))

    return qrows


# =========================================================
# Bubble Analysis
# =========================================================

def bubble_fill_ratio(gray, bubble):
    x, y, w, h = bubble.bbox
    patch = gray[y:y+h, x:x+w]
    if patch.size == 0:
        return 0.0

    _, th = cv2.threshold(
        cv2.GaussianBlur(patch, (3, 3), 0),
        0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return float(np.mean(th > 0))


def bubble_cancel_strength(gray, bubble):
    x, y, w, h = bubble.bbox
    patch = gray[y:y+h, x:x+w]
    if patch.size == 0:
        return 0.0

    edges = cv2.Canny(patch, 60, 160)
    return float(np.mean(edges > 0))


# =========================================================
# FINAL Answer Decoder
# =========================================================

def decode_answers(gray, bubbles, rows, option_labels="ABCDE"):
    fill_light = 0.18
    fill_full = 0.50
    cancel_thr = 0.06

    answers = []
    debug = []

    for r in rows:
        for g in r.groups:
            g = sorted(g, key=lambda i: bubbles[i].x)[:4]

            fills = [bubble_fill_ratio(gray, bubbles[i]) for i in g]
            cancels = [bubble_cancel_strength(gray, bubbles[i]) >= cancel_thr for i in g]

            states = []
            for f, c in zip(fills, cancels):
                if c:
                    states.append("CANCEL")
                elif f >= fill_full:
                    states.append("FULL")
                elif f >= fill_light:
                    states.append("LIGHT")
                else:
                    states.append("BLANK")

            # Remove cancelled
            full = [i for i,s in enumerate(states) if s=="FULL"]
            light = [i for i,s in enumerate(states) if s=="LIGHT"]

            if len(full) == 1:
                ans = option_labels[full[0]]
            elif len(full) > 1:
                ans = "MULTI"
            else:
                if len(light) == 1:
                    ans = option_labels[light[0]]
                elif len(light) > 1:
                    ans = "AMB"
                else:
                    ans = ""

            answers.append(ans)
            debug.append({"fills": fills, "states": states})

    return answers, debug


# =========================================================
# Main Processing
# =========================================================

def process_page(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin_img = _adaptive_bin(gray)
    bubbles = _find_bubbles(bin_img)

    tol_x = tol_y = 20
    code_blocks = detect_code_blocks(bubbles, tol_x, tol_y)
    exclude = code_blocks[0].bbox if code_blocks else None

    rows = detect_question_rows(bubbles, exclude, tol_x, tol_y)
    answers, _ = decode_answers(gray, bubbles, rows)

    return {
        "answers": answers,
        "num_questions": len(answers)
    }


def process_pdf(pdf_path, out_csv="bubble_results.csv", zoom=2.5):
    pages = pdf_to_images(pdf_path, zoom)
    rows = []

    for i, img in enumerate(pages):
        res = process_page(img)
        rows.append({
            "page": i+1,
            "num_questions": res["num_questions"],
            "answers": " ".join(res["answers"])
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return df
