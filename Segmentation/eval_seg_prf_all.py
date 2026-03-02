#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional


# -------------------------
# IoU for xyxy boxes
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
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


# -------------------------
# Matching (greedy by IoU)
# -------------------------
def match_prf(gt: List[Dict[str, Any]], pred: List[Dict[str, Any]], thr: float) -> Tuple[int, int, int]:
    pairs = []

    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            # label match only if BOTH have label not None
            gl = g.get("label", None)
            pl = p.get("label", None)
            if gl is not None and pl is not None and str(gl) != str(pl):
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


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


# -------------------------
# Load GT (gt_instances.json)
# -------------------------
def xywh_to_xyxy(xywh: List[float]) -> List[float]:
    x, y, w, h = xywh
    return [x, y, x + w, y + h]


def load_gt_instances(path: Path) -> List[Dict[str, Any]]:
    """
    Expect list of instances, each may contain:
      - label (optional)
      - bbox_xyxy: [x1,y1,x2,y2]  (preferred)
      - bbox_xywh: [x,y,w,h]      (fallback)
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    gt = []
    if not isinstance(data, list):
        return gt

    for x in data:
        if not isinstance(x, dict):
            continue
        label = x.get("label", None)

        box_xyxy = None
        if "bbox_xyxy" in x and isinstance(x["bbox_xyxy"], list) and len(x["bbox_xyxy"]) == 4:
            box_xyxy = [float(v) for v in x["bbox_xyxy"]]
        elif "bbox_xywh" in x and isinstance(x["bbox_xywh"], list) and len(x["bbox_xywh"]) == 4:
            box_xyxy = xywh_to_xyxy([float(v) for v in x["bbox_xywh"]])

        if box_xyxy is None:
            continue

        gt.append({"label": str(label) if label is not None else None, "bbox_xyxy": box_xyxy})

    return gt


# -------------------------
# Load Pred ({engine}.json)
# -------------------------
def load_pred_engine(path: Path) -> List[Dict[str, Any]]:
    """
    Supports:
    1) SAM-style:
       {"bboxes": [[x1,y1,x2,y2], ...], "labels": [...] (optional)}
    2) Paddle-style (your old):
       {"results": [[label, {"left":..,"top":..,"width":..,"height":..}], ...]}
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    pred: List[Dict[str, Any]] = []

    # (1) SAM-style bboxes
    if isinstance(data, dict) and "bboxes" in data and isinstance(data["bboxes"], list):
        labels = data.get("labels", None)
        for i, bb in enumerate(data["bboxes"]):
            if not (isinstance(bb, list) and len(bb) == 4):
                continue
            x1, y1, x2, y2 = [float(v) for v in bb]
            lab = None
            if isinstance(labels, list) and i < len(labels):
                lab = labels[i]
            pred.append({"label": str(lab) if lab is not None else None, "bbox_xyxy": [x1, y1, x2, y2]})
        if pred:
            return pred

    # (2) Paddle-style results
    if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
        for item in data["results"]:
            if not (isinstance(item, list) and len(item) == 2):
                continue
            label, box = item
            if not isinstance(box, dict):
                continue
            if not all(k in box for k in ("left", "top", "width", "height")):
                continue

            x1 = float(box["left"])
            y1 = float(box["top"])
            x2 = x1 + float(box["width"])
            y2 = y1 + float(box["height"])

            pred.append({"label": str(label) if label is not None else None, "bbox_xyxy": [x1, y1, x2, y2]})

    return pred


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic_root", default="/home/root123/mount1/weilai/MIMIC",
                    help="/home/root123/mount1/weilai/MIMIC")
    ap.add_argument("--engine", default="SAM",
                    help="预测文件名 output/{id}/{engine}.json")
    ap.add_argument("--gt_name", default="gt_instances.json",
                    help="GT 文件名 data/{id}/{gt_name}")
    ap.add_argument("--ignore_label", action="store_true",
                    help="忽略 label，不做类别约束匹配")
    args = ap.parse_args()

    root = Path(args.mimic_root)
    data_root = root / "data"
    out_root = root / "output"
    thresholds = [i / 100 for i in range(15, 95, 5)]

    stats = {thr: {"TP": 0, "FP": 0, "FN": 0} for thr in thresholds}

    evaluated = 0
    skipped = 0

    for sample_dir in sorted(data_root.iterdir()):
        if not sample_dir.is_dir():
            continue

        gt_path = sample_dir / args.gt_name
        pred_path = out_root / sample_dir.name / f"{args.engine}.json"

        if not gt_path.exists() or not pred_path.exists():
            skipped += 1
            continue

        gt = load_gt_instances(gt_path)
        pred = load_pred_engine(pred_path)

        if args.ignore_label:
            for g in gt:
                g["label"] = None
            for p in pred:
                p["label"] = None

        evaluated += 1

        for thr in thresholds:
            tp, fp, fn = match_prf(gt, pred, thr)
            stats[thr]["TP"] += tp
            stats[thr]["FP"] += fp
            stats[thr]["FN"] += fn

    print(f"Evaluated samples : {evaluated}")
    print(f"Skipped samples   : {skipped}\n")

    for thr in thresholds:
        s = stats[thr]
        p, r, f = prf(s["TP"], s["FP"], s["FN"])
        print(
            f"IoU@{int(thr*100):02d}  "
            f"TP={s['TP']:6d}  FP={s['FP']:6d}  FN={s['FN']:6d}  "
            f"P={p:.4f}  R={r:.4f}  F1={f:.4f}"
        )


if __name__ == "__main__":
    main()
