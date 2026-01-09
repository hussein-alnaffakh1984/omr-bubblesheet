import math
import cv2
import numpy as np
from typing import List, Tuple, Dict


def bubble_fill_ratio(gray: np.ndarray, bubble) -> float:
    """
    نسبة الحبر داخل الدائرة (0..1).
    أعلى = تظليل أكثر.
    """
    x, y, w, h = bubble.bbox
    pad = int(max(2, bubble.r * 0.30))
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(gray.shape[1], x + w + pad); y1 = min(gray.shape[0], y + h + pad)
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0

    patch = cv2.GaussianBlur(patch, (3, 3), 0)
    # ثنائـية محلية (الحبر يصبح 1 بعد INVERT)
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


def bubble_is_cancelled(gray: np.ndarray, bubble) -> bool:
    """
    كشف الإلغاء (X/شخطة) داخل الفقاعة باستخدام HoughLinesP.
    """
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

    # نعتبرها ملغاة إذا وجدنا خط طويل يمر قرب مركز الفقاعة
    hit = 0
    for (x1l, y1l, x2l, y2l) in lines[:, 0]:
        length = math.hypot(x2l - x1l, y2l - y1l)
        if length < min_len:
            continue
        mx, my = (x1l + x2l) / 2.0, (y1l + y2l) / 2.0
        if abs(mx - cx) < W * 0.28 and abs(my - cy) < H * 0.28:
            hit += 1

    return hit >= 1


def decode_answers(gray: np.ndarray,
                  bubbles: List,
                  rows: List,
                  option_labels: str = "ABCDE") -> Tuple[List[str], Dict]:
    """
    Template-free decoding:
    - لكل مجموعة خيارات (سؤال): يصنّف كل فقاعة إلى BLANK/LIGHT/FULL/CANCEL
    - ثم يقرر إجابة السؤال:
        FULL واحد -> الخيار
        FULL متعدد -> MULTI
        لا FULL:
          LIGHT واحد -> LIGHT_SELECTED (اختيار ضعيف)
          LIGHT متعدد -> AMB (غامض)
          لا شيء -> BLANK
    - إذا خيار FULL لكنه CANCEL -> يُستبعد (كأنه غير مختار)
    """

    # عتبات عامة (قد تحتاج ضبط بسيط حسب DPI/القلم)
    fill_light_thr = 0.18   # أقل من هذا = فارغ
    fill_full_thr  = 0.45   # أكبر/يساوي = كامل

    answers: List[str] = []
    dbg = {"questions": []}

    q_index = 0
    for r in rows:
        for g in r.groups:
            # ترتيب الخيارات من اليسار لليمين
            g_sorted = sorted(g, key=lambda i: bubbles[i].x)

            # --- مهم: لو دخلت فقاعات زائدة بسبب noise، خذ أفضل 4 فقاعات متقاربة (للاختيارات A-D)
            # إذا ورقتك 5 خيارات فعلاً، غيّر 4 إلى 5.
            TARGET_OPTS = 4
            if len(g_sorted) > TARGET_OPTS:
                # نافذة منزلقة لاختيار أقرب TARGET_OPTS فقاعات
                best_span = None
                best_window = None
                for s in range(0, len(g_sorted) - (TARGET_OPTS - 1)):
                    window = g_sorted[s:s + TARGET_OPTS]
                    span = bubbles[window[-1]].x - bubbles[window[0]].x
                    if best_span is None or span < best_span:
                        best_span = span
                        best_window = window
                g_sorted = best_window

            # إن كانت أقل من 3 فقاعات لا نعتبرها سؤال صالح
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

            # استبعاد FULL الملغاة (CANCEL)
            full_idxs = [k for k, s in enumerate(states) if s == "FULL"]
            # إذا نفس الفقاعة ملغاة نعتبرها CANCEL فقط (حسب states بالفعل)

            light_idxs = [k for k, s in enumerate(states) if s == "LIGHT"]

            if len(full_idxs) == 1:
                k = full_idxs[0]
                ans = option_labels[k] if k < len(option_labels) else str(k)
                flag = "OK_FULL"
            elif len(full_idxs) >= 2:
                ans = "MULTI"
                flag = "MULTI_FULL"
            else:
                # لا يوجد FULL
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
            dbg["questions"].append({
                "q": q_index + 1,
                "fills": fills,
                "states": states,
                "flag": flag
            })
            q_index += 1

    return answers, dbg
