#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


# -------------------------
# IoU
# -------------------------
def iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter)


# -------------------------
# Matching
# -------------------------
def match_prf(gt, pred, thr):
    pairs = []

    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            if g["label"] != p["label"]:
                continue
            v = iou_xyxy(g["bbox_xyxy"], p["bbox_xyxy"])
            if v >= thr:
                pairs.append((v, gi, pi))

    pairs.sort(reverse=True, key=lambda x: x[0])

    used_gt = set()
    used_pr = set()
    tp = 0

    for _, gi, pi in pairs:
        if gi in used_gt or pi in used_pr:
            continue
        used_gt.add(gi)
        used_pr.add(pi)
        tp += 1

    fp = len(pred) - tp
    fn = len(gt) - tp
    return tp, fp, fn


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


# -------------------------
# Load GT / Pred
# -------------------------
def load_gt(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "label": str(x["label"]),
            "bbox_xyxy": x["bbox_xyxy"],
        }
        for x in data
        if "label" in x and "bbox_xyxy" in x
    ]


def load_pred_paddle(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results", [])

    pred = []
    for item in results:
        if not isinstance(item, list) or len(item) != 2:
            continue
        label, box = item
        if not isinstance(box, dict):
            continue

        x1 = box["left"]
        y1 = box["top"]
        x2 = box["left"] + box["width"]
        y2 = box["top"] + box["height"]

        pred.append({
            "label": str(label),
            "bbox_xyxy": [x1, y1, x2, y2]
        })
    return pred


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic_root", default="/home/root123/mount1/weilai/MIMIC",
                    help="/home/root123/mount1/weilai/MIMIC")
    ap.add_argument("--engine", default="paddle",
                    help="paddle / easyocr / etc")
    args = ap.parse_args()

    root = Path(args.mimic_root)
    data_root = root / "data"
    out_root = root / "output"
    thresholds = [i / 100 for i in range(15, 95, 5)]

    stats = {
        thr: {"TP": 0, "FP": 0, "FN": 0}
        for thr in thresholds
    }

    evaluated = 0
    skipped = 0

    for sample_dir in sorted(data_root.iterdir()):
        if not sample_dir.is_dir():
            continue

        gt_path = sample_dir / "gt_ocr.json"
        pred_path = out_root / sample_dir.name / f"{args.engine}.json"

        if not gt_path.exists() or not pred_path.exists():
            skipped += 1
            continue

        gt = load_gt(gt_path)
        pred = load_pred_paddle(pred_path)

        evaluated += 1

        for thr in stats:
            tp, fp, fn = match_prf(gt, pred, thr)
            stats[thr]["TP"] += tp
            stats[thr]["FP"] += fp
            stats[thr]["FN"] += fn

    print(f"Evaluated samples : {evaluated}")
    print(f"Skipped samples   : {skipped}\n")

    for thr, s in stats.items():
        p, r, f = prf(s["TP"], s["FP"], s["FN"])
        print(
            f"IoU@{int(thr*100):02d}  "
            f"TP={s['TP']:6d}  FP={s['FP']:6d}  FN={s['FN']:6d}  "
            f"P={p:.4f}  R={r:.4f}  F1={f:.4f}"
        )


if __name__ == "__main__":
    main()
