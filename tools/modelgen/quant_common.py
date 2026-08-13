"""Shared preprocessing that mirrors the C++ CPU OCR pipeline exactly, so that
calibration / eval crops have the same activation statistics the deployed engine
sees. References:
  - det preprocess + DB postproc: src/analysis/detection/cpu_paddle_det.cpp,
    src/analysis/detection/det_postprocess.cpp, include/.../det_config.h
  - rotate crop: src/analysis/recognition/crop_utils.cpp, include/.../perspective.h
  - rec preprocess: src/analysis/recognition/cpu_paddle_rec.cpp
"""
import glob
import math
import os

import cv2
import numpy as np
import onnxruntime as ort
import pyclipper

# repo root = up out of tools/modelgen/
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FUNSD = os.path.join(os.path.dirname(REPO), "compare-ocrs", "funsd_cache")

# ---- det constants (cpu_paddle_det.h / det_config.h) ----
DET_MAX_SIDE = 960
DET_DB_THRESH = 0.3
DET_DB_BOX_THRESH = 0.6
DET_DB_UNCLIP_RATIO = 1.5
MIN_BOX_SIDE = 3.0
MIN_UNCLIPPED_SIDE = 5.0

# ---- rec constants (cpu_paddle_rec.h) ----
REC_IMAGE_H = 48
REC_IMAGE_W = 320          # minimum target width
REC_MAX_WIDTH = 4000
VERTICAL_AR = 1.5          # box.h kVerticalAspectRatio


def list_pages(n=None):
    imgs = sorted(glob.glob(f"{FUNSD}/*.png"))
    return imgs if n is None else imgs[:n]


def make_cpu_session(model_path, intra=0, inter=1, opt_all=True):
    so = ort.SessionOptions()
    so.intra_op_num_threads = intra
    so.inter_op_num_threads = inter
    so.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if opt_all
        else ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    )
    return ort.InferenceSession(model_path, sess_options=so,
                                providers=["CPUExecutionProvider"])


# ---------------- detection ----------------
def det_preprocess(img_bgr):
    """Returns (nchw float32 input, resize_h, resize_w). Mirrors cpu_paddle_det.cpp."""
    h, w = img_bgr.shape[:2]
    ratio = 1.0
    if max(h, w) > DET_MAX_SIDE:
        ratio = DET_MAX_SIDE / h if h > w else DET_MAX_SIDE / w
    resize_h = max(int(round(h * ratio / 32.0) * 32), 32)
    resize_w = max(int(round(w * ratio / 32.0) * 32), 32)

    resized = cv2.resize(img_bgr, (resize_w, resize_h))
    f = resized.astype(np.float32) / 255.0
    # BGR channels normalized with ImageNet mean/std
    b = (f[:, :, 0] - 0.406) / 0.225
    g = (f[:, :, 1] - 0.456) / 0.224
    r = (f[:, :, 2] - 0.485) / 0.229
    # NCHW, RGB plane order (R,G,B)
    nchw = np.stack([r, g, b], axis=0)[None].astype(np.float32)
    return np.ascontiguousarray(nchw), resize_h, resize_w


def _get_mini_boxes(contour):
    """minAreaRect -> ordered [tl,tr,br,bl], min_side. Mirrors det_postprocess.cpp."""
    rect = cv2.minAreaRect(contour)
    min_side = min(rect[1][0], rect[1][1])
    pts = cv2.boxPoints(rect)  # 4x2 float
    pts = sorted(pts.tolist(), key=lambda p: p[0])  # sort by x
    p0, p1, p2, p3 = pts
    tl, bl = (p0, p1) if p0[1] < p1[1] else (p1, p0)
    tr, br = (p2, p3) if p2[1] < p3[1] else (p3, p2)
    box = np.array([tl, tr, br, bl], dtype=np.float32)
    return box, min_side


def _box_score_fast(pred_map, contour):
    h, w = pred_map.shape
    xs = contour[:, 0, 0]
    ys = contour[:, 0, 1]
    xmin = max(0, int(xs.min())); xmax = min(w - 1, int(xs.max()))
    ymin = max(0, int(ys.min())); ymax = min(h - 1, int(ys.max()))
    if xmax <= xmin or ymax <= ymin:
        return 0.0
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    shifted = contour.copy()
    shifted[:, 0, 0] -= xmin
    shifted[:, 0, 1] -= ymin
    cv2.fillPoly(mask, [shifted], 1)
    roi = pred_map[ymin:ymax + 1, xmin:xmax + 1]
    return float(cv2.mean(roi, mask)[0])


def _unclip(contour, ratio):
    poly = contour[:, 0, :]
    area = cv2.contourArea(poly)
    perimeter = cv2.arcLength(poly, True)
    if perimeter == 0:
        return poly
    distance = area * ratio / perimeter
    pco = pyclipper.PyclipperOffset()
    pco.AddPath([(int(p[0]), int(p[1])) for p in poly],
                pyclipper.JT_SQUARE, pyclipper.ET_CLOSEDPOLYGON)
    sol = pco.Execute(distance)
    if not sol:
        return poly
    best = max(sol, key=lambda p: abs(cv2.contourArea(np.array(p, dtype=np.int32))))
    return np.array(best, dtype=np.int32)


def det_postprocess(pred_map, orig_h, orig_w, resize_h, resize_w):
    """Returns list of boxes (4x2 int, [tl,tr,br,bl]) in original image coords."""
    bitmap = (pred_map > DET_DB_THRESH).astype(np.uint8) * 255
    ratio_h = resize_h / orig_h
    ratio_w = resize_w / orig_w
    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours[:1000]:
        if len(c) <= 2:
            continue
        br = cv2.boundingRect(c)
        if br[2] < 3 or br[3] < 3:
            continue
        if _box_score_fast(pred_map, c) < DET_DB_BOX_THRESH:
            continue
        _, ssid = _get_mini_boxes(c)
        if ssid < MIN_BOX_SIDE:
            continue
        unclipped = _unclip(c, DET_DB_UNCLIP_RATIO)
        if len(unclipped) < 3:
            continue
        uc = unclipped.reshape(-1, 1, 2).astype(np.int32)
        box, ssid2 = _get_mini_boxes(uc)
        if ssid2 < MIN_UNCLIPPED_SIDE:
            continue
        box[:, 0] = np.clip(np.round(box[:, 0] / ratio_w), 0, orig_w - 1)
        box[:, 1] = np.clip(np.round(box[:, 1] / ratio_h), 0, orig_h - 1)
        b = box.astype(np.int32)
        rw = int(math.hypot(b[0][0] - b[1][0], b[0][1] - b[1][1]))
        rh = int(math.hypot(b[0][0] - b[3][0], b[0][1] - b[3][1]))
        if rw <= 3 or rh <= 3:
            continue
        boxes.append(b)
    return boxes


# ---------------- rotate crop + rec preprocess ----------------
def get_rotate_crop_image(img_bgr, box):
    """Perspective warp. Mirrors compute_crop_geometry + get_rotate_crop_image."""
    b = box.astype(np.float32)
    crop_w = math.hypot(b[0][0] - b[1][0], b[0][1] - b[1][1])
    crop_h = math.hypot(b[0][0] - b[3][0], b[0][1] - b[3][1])
    vertical = crop_h >= crop_w * VERTICAL_AR
    if vertical:
        src = np.array([b[3], b[0], b[1], b[2]], dtype=np.float32)
        crop_w, crop_h = crop_h, crop_w
    else:
        src = np.array([b[0], b[1], b[2], b[3]], dtype=np.float32)
    dst_w = max(int(round(crop_w)), 1)
    dst_h = max(int(round(crop_h)), 1)
    dst = np.array([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img_bgr, M, (dst_w, dst_h),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def rec_target_width(crop):
    h, w = crop.shape[:2]
    ar = (w / h) if h > 0 else 0.0
    tw = min(int(math.ceil(REC_IMAGE_H * ar)), REC_MAX_WIDTH)
    return max(tw, REC_IMAGE_W)


def rec_preprocess(crop, target_w=None):
    """Resize to (48, target_w), /127.5-1, BGR->RGB, NCHW. Mirrors preprocess_crop."""
    if target_w is None:
        target_w = rec_target_width(crop)
    resized = cv2.resize(crop, (target_w, REC_IMAGE_H))
    f = resized.astype(np.float32) / 127.5 - 1.0
    b, g, r = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    nchw = np.stack([r, g, b], axis=0)[None].astype(np.float32)
    return np.ascontiguousarray(nchw), target_w


# ---------------- crop extraction driver ----------------
def extract_line_crops(det_sess, pages, max_crops=None, verbose=False):
    """Run det fp32 on pages, return list of BGR line crops (real text lines)."""
    in_name = det_sess.get_inputs()[0].name
    crops = []
    for p in pages:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        inp, rh, rw = det_preprocess(img)
        out = det_sess.run(None, {in_name: inp})[0]
        pred = out[0, 0].astype(np.float32)
        boxes = det_postprocess(pred, h, w, rh, rw)
        for box in boxes:
            crop = get_rotate_crop_image(img, box)
            if crop.shape[0] >= 4 and crop.shape[1] >= 4:
                crops.append(crop)
        if verbose:
            print(f"  {os.path.basename(p)}: {len(boxes)} boxes (total crops {len(crops)})")
        if max_crops and len(crops) >= max_crops:
            break
    return crops[:max_crops] if max_crops else crops


def load_keys(path=f"{REPO}/models/keys.txt"):
    """Returns label_list matching C++ load_label_dict: ['blank', <keys...>, ' ']."""
    labels = ["blank"]
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            labels.append(line)
    labels.append(" ")
    return labels


def ctc_greedy_decode(logits, label_list):
    """Greedy CTC decode. logits: (seq, num_classes). Mirrors ctc_greedy_decode_raw."""
    idx = logits.argmax(axis=1)
    text = []
    last = -1
    for i in idx:
        i = int(i)
        if i != last:
            if i != 0 and i < len(label_list):
                text.append(label_list[i])
        last = i
    return "".join(text)
