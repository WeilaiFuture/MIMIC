#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path
import time
from typing import List, Dict, Any, Tuple, Optional
import math

IOU_THR = 0.45

# -------------------------
# Geometry
# -------------------------
Box = Tuple[float, float, float, float]      # xyxy
OCRItem = Tuple[str, Box]                    # (label, box)

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

def center(b: Box) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def l2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def prf(tp: int, fp: int, fn: int) -> Dict[str, Any]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "P": round(p, 6), "R": round(r, 6), "F1": round(f, 6)}

def xywh_to_xyxy(xywh: List[float]) -> List[float]:
    x, y, w, h = xywh
    return [x, y, x + w, y + h]


# -------------------------
# Load predictions
# -------------------------
def load_pred_ocr_ft(path: Path) -> List[OCRItem]:
    """
    ft.json:
      {"status":"ok","results":[["B2",{"top":3,"left":785,"height":43,"width":63}], ...]}
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results", [])
    out: List[OCRItem] = []
    if not isinstance(results, list):
        return out

    for item in results:
        if not (isinstance(item, list) and len(item) == 2):
            continue
        lab, box = item
        if not isinstance(box, dict):
            continue
        try:
            left = float(box["left"])
            top = float(box["top"])
            w = float(box["width"])
            h = float(box["height"])
        except Exception:
            continue
        xyxy = (left, top, left + w, top + h)
        out.append((str(lab).strip().upper(), xyxy))
    return out

def load_pred_seg_boxes(path: Path) -> List[Box]:
    """
    SAM/unet.json:
      {"status":"ok","boxes":[[x1,y1,x2,y2], ...]}
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    boxes = data.get("bboxes", [])
    out: List[Box] = []
    if not isinstance(boxes, list):
        return out
    for b in boxes:
        if isinstance(b, list) and len(b) == 4:
            x1, y1, x2, y2 = [float(v) for v in b]
            out.append((x1, y1, x2, y2))
    return out


# -------------------------
# Load GT & bind by instance_id
# -------------------------
def load_gt_instances(path: Path) -> Dict[int, Dict[str, Any]]:
    """
    gt_instances.json: list of instances
      instance_id, label, bbox_xyxy or bbox_xywh
    return: instance_id -> {"label","bbox_xyxy"}
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    mp: Dict[int, Dict[str, Any]] = {}
    if not isinstance(data, list):
        return mp

    for x in data:
        if not isinstance(x, dict):
            continue
        iid = x.get("instance_id", None)
        lab = x.get("label", None)
        if iid is None or lab is None:
            continue
        box = None
        if isinstance(x.get("bbox_xyxy"), list) and len(x["bbox_xyxy"]) == 4:
            box = [float(v) for v in x["bbox_xyxy"]]
        elif isinstance(x.get("bbox_xywh"), list) and len(x["bbox_xywh"]) == 4:
            box = xywh_to_xyxy([float(v) for v in x["bbox_xywh"]])
        if box is None:
            continue
        mp[int(iid)] = {"label": str(lab).strip().upper(), "bbox_xyxy": box}
    return mp

def load_gt_ocr(path: Path) -> Dict[int, Dict[str, Any]]:
    """
    gt_ocr.json: list of ocr entries
      instance_id, label, bbox_xyxy or bbox_xywh
    return: instance_id -> {"label","bbox_xyxy"}
    若同 instance_id 多条，保留面积最大的一条
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    best: Dict[int, Tuple[float, Dict[str, Any]]] = {}
    if not isinstance(data, list):
        return {}

    for x in data:
        if not isinstance(x, dict):
            continue
        iid = x.get("instance_id", None)
        lab = x.get("label", None)
        if iid is None or lab is None:
            continue

        box = None
        if isinstance(x.get("bbox_xyxy"), list) and len(x["bbox_xyxy"]) == 4:
            box = [float(v) for v in x["bbox_xyxy"]]
        elif isinstance(x.get("bbox_xywh"), list) and len(x["bbox_xywh"]) == 4:
            box = xywh_to_xyxy([float(v) for v in x["bbox_xywh"]])

        if box is None:
            continue

        area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        rec = {"label": str(lab).strip().upper(), "bbox_xyxy": box}
        iid_i = int(iid)
        if iid_i not in best or area > best[iid_i][0]:
            best[iid_i] = (area, rec)

    return {iid: rec for iid, (_, rec) in best.items()}

def build_gt_pairs(gt_inst: Dict[int, Dict[str, Any]], gt_ocr: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    pairs = []
    for iid, inst in gt_inst.items():
        if iid not in gt_ocr:
            continue
        o = gt_ocr[iid]
        pairs.append({
            "instance_id": iid,
            "label": o["label"],          # OCR label
            "inst_box": inst["bbox_xyxy"],
            "ocr_box": o["bbox_xyxy"],
        })
    return pairs


# -------------------------
# Matching strategies (seg -> ocr)
# produce predicted pairs: {"label","seg_box","ocr_box"}
# -------------------------
def match_static(seg_boxes: List[Box], ocr_boxes: List[OCRItem]) -> List[Dict[str, Any]]:
    out = []
    for sb in seg_boxes:
        sc = center(sb)
        best = None
        best_d = float("inf")
        for lab, ob in ocr_boxes:
            d = l2(sc, center(ob))
            if d < best_d:
                best_d = d
                best = (lab, ob)
        if best is not None:
            lab, ob = best
            out.append({"label": lab, "seg_box": list(sb), "ocr_box": list(ob)})
    return out

def match_dynamic(seg_boxes: List[Box], ocr_boxes: List[OCRItem]) -> List[Dict[str, Any]]:
    out = []
    remaining = list(ocr_boxes)
    for sb in seg_boxes:
        if not remaining:
            break
        sc = center(sb)
        best_j = -1
        best_d = float("inf")
        for j, (lab, ob) in enumerate(remaining):
            d = l2(sc, center(ob))
            if d < best_d:
                best_d = d
                best_j = j
        lab, ob = remaining.pop(best_j)
        out.append({"label": lab, "seg_box": list(sb), "ocr_box": list(ob)})
    return out

def match_km(seg_boxes: List[Box], ocr_boxes: List[OCRItem]) -> List[Dict[str, Any]]:
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except Exception:
        return match_dynamic(seg_boxes, ocr_boxes)

    if not seg_boxes or not ocr_boxes:
        return []

    cost = np.zeros((len(seg_boxes), len(ocr_boxes)), dtype=np.float32)
    for i, sb in enumerate(seg_boxes):
        sc = center(sb)
        for j, (_, ob) in enumerate(ocr_boxes):
            cost[i, j] = l2(sc, center(ob))

    row_ind, col_ind = linear_sum_assignment(cost)
    out = []
    for i, j in zip(row_ind, col_ind):
        lab, ob = ocr_boxes[j]
        sb = seg_boxes[i]
        out.append({"label": lab, "seg_box": list(sb), "ocr_box": list(ob)})
    return out


# -------------------------
# Voting fuse
# rule: majority; if all three different -> static
# fuse is per seg_box
# -------------------------
def fuse_vote(
    seg_boxes: List[Box],
    pred_s: List[Dict[str, Any]],
    pred_d: List[Dict[str, Any]],
    pred_k: List[Dict[str, Any]],
    default_method: str = "static",
) -> List[Dict[str, Any]]:
    """
    Vote fusion for (seg_box -> OCR match) from 3 strategies.

    default_method: "static" | "dynamic" | "km"
      - used when votes conflict (all different) OR when some votes missing and we need fallback.
    """

    def to_map(pred: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        mp = {}
        for it in pred:
            sb = it.get("seg_box")
            if not (isinstance(sb, list) and len(sb) == 4):
                continue
            key = f"{sb[0]},{sb[1]},{sb[2]},{sb[3]}"
            mp[key] = it
        return mp

    default_method = (default_method or "static").strip().lower()
    if default_method not in ("static", "dynamic", "km"):
        default_method = "static"

    ms = to_map(pred_s)
    md = to_map(pred_d)
    mk = to_map(pred_k)

    # helper: pick by method name
    def pick(method: str, cs, cd, ck):
        if method == "static":
            return cs
        if method == "dynamic":
            return cd
        if method == "km":
            return ck
        return cs  # fallback

    # helper: first available in a fixed order
    def first_available(order: List[str], cs, cd, ck):
        for m in order:
            ch = pick(m, cs, cd, ck)
            if ch is not None:
                return ch, m
        return None, "none"

    fused = []
    for sb in seg_boxes:
        key = f"{sb[0]},{sb[1]},{sb[2]},{sb[3]}"
        cs = ms.get(key)
        cd = md.get(key)
        ck = mk.get(key)

        ls = cs.get("label") if cs else None
        ld = cd.get("label") if cd else None
        lk = ck.get("label") if ck else None

        votes = {"static": ls, "dynamic": ld, "km": lk}
        present = {k: v for k, v in votes.items() if v is not None}

        chosen = None
        chosen_method = "none"

        if len(present) == 0:
            chosen = None
            chosen_method = "none"

        elif len(present) == 1:
            # only one vote exists -> choose it
            only_m = next(iter(present.keys()))
            chosen = pick(only_m, cs, cd, ck)
            chosen_method = only_m

        else:
            # check majority among existing votes
            # count labels
            cnt: Dict[str, int] = {}
            for v in present.values():
                cnt[v] = cnt.get(v, 0) + 1

            # best label by count
            best_label, best_count = None, -1
            for lab, c in cnt.items():
                if c > best_count:
                    best_label, best_count = lab, c

            if best_count >= 2:
                # majority exists: pick one of the methods that voted for best_label
                # if default_method is among them -> prioritize default_method, else prioritize static, dynamic, km
                candidate_methods = [m for m, lab in present.items() if lab == best_label]

                if default_method in candidate_methods:
                    chosen = pick(default_method, cs, cd, ck)
                    chosen_method = f"{default_method}(majority)"
                else:
                    # stable priority
                    for m in ("static", "dynamic", "km"):
                        if m in candidate_methods:
                            chosen = pick(m, cs, cd, ck)
                            chosen_method = f"{m}(majority)"
                            break
            else:
                # all different (or no majority because only 2 votes and different)
                # choose default_method if exists, else first available in static->dynamic->km
                preferred = pick(default_method, cs, cd, ck)
                if preferred is not None:
                    chosen = preferred
                    chosen_method = f"all_diff->{default_method}"
                else:
                    chosen, m = first_available(["static", "dynamic", "km"], cs, cd, ck)
                    chosen_method = f"all_diff->{m}"

        fused.append({
            "seg_box": list(sb),
            "votes": {
                "static": cs,
                "dynamic": cd,
                "km": ck
            },
            "chosen": {
                "method": chosen_method,
                "label": (chosen.get("label") if chosen else None),
                "ocr_box": (chosen.get("ocr_box") if chosen else None)
            }
        })

    return fused
# -------------------------
# Evaluate predicted pairs vs GT pairs (IoU=0.45, bound by instance_id via GT)
# TP if: label match + seg IoU + ocr IoU
# 1-to-1 greedy
# -------------------------
def eval_pairs_prf(gt_pairs: List[Dict[str, Any]], pred_pairs: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    candidates = []
    for pi, p in enumerate(pred_pairs):
        plab = str(p["label"]).upper()
        psb = p["seg_box"]
        pob = p["ocr_box"]
        for gi, g in enumerate(gt_pairs):
            if plab != str(g["label"]).upper():
                continue
            i_inst = iou_xyxy(psb, g["inst_box"])
            if i_inst < IOU_THR:
                continue
            i_ocr = iou_xyxy(pob, g["ocr_box"])
            if i_ocr < IOU_THR:
                continue
            score = i_inst * 10.0 + i_ocr
            candidates.append((score, pi, gi))

    candidates.sort(reverse=True, key=lambda x: x[0])

    used_p = set()
    used_g = set()
    tp = 0
    for _, pi, gi in candidates:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        tp += 1

    fp = len(pred_pairs) - tp
    fn = len(gt_pairs) - tp
    return tp, fp, fn


def eval_fused_prf(gt_pairs: List[Dict[str, Any]], fused: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    # turn fused into pred_pairs format
    pred = []
    for it in fused:
        ch = it.get("chosen", {})
        if not isinstance(ch, dict):
            continue
        lab = ch.get("label", None)
        ob = ch.get("ocr_box", None)
        sb = it.get("seg_box", None)
        if lab is None or not (isinstance(ob, list) and len(ob) == 4) or not (isinstance(sb, list) and len(sb) == 4):
            continue
        pred.append({"label": str(lab).upper(), "seg_box": sb, "ocr_box": ob})
    return eval_pairs_prf(gt_pairs, pred)


# -------------------------
# Save json helpers
# -------------------------
def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic_root", default="/home/root123/mount1/weilai/MIMIC")
    ap.add_argument("--seg_engines", default="unet,SAM", help="comma separated, e.g. unet,SAM")
    ap.add_argument("--ocr_engine", default="ft", help="default ft -> output/{id}/ft.json")
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--save_preds", action="store_true", default=True,
                    help="save pred_static/pred_dynamic/pred_km/fused into output/{id}/match_eval/... (default on)")
    ap.add_argument("--out_json", default="eval_ocr_seg_match_iou45_with_vote.json",
                    help="final summary json (relative to mimic_root)")
    args = ap.parse_args()

    global IOU_THR
    IOU_THR = float(args.iou)

    root = Path(args.mimic_root)
    data_root = root / "data"
    out_root = root / "output"

    seg_engines = [x.strip() for x in args.seg_engines.split(",") if x.strip()]
    ocr_engine = args.ocr_engine.strip()

    # aggregate: seg_engine -> algo -> TP/FP/FN
    agg = {
        se: {
            "static": [0, 0, 0],
            "dynamic": [0, 0, 0],
            "km": [0, 0, 0],
            "static_fused": [0, 0, 0],
            "dynamic_fused": [0, 0, 0],
            "km_fused": [0, 0, 0],
            
        } for se in seg_engines
    }
    evaluated = {se: 0 for se in seg_engines}
    skipped = {se: 0 for se in seg_engines}

    # traverse all samples
    sample_dirs = [p for p in sorted(data_root.iterdir()) if p.is_dir()]
    t0=0
    t1=0
    t2=0
    for sd in sample_dirs:
        sid = sd.name

        gt_inst_path = sd / "gt_instances.json"
        gt_ocr_path = sd / "gt_ocr.json"
        if not gt_inst_path.exists() or not gt_ocr_path.exists():
            for se in seg_engines:
                skipped[se] += 1
            continue

        gt_inst = load_gt_instances(gt_inst_path)
        gt_ocr = load_gt_ocr(gt_ocr_path)
        gt_pairs = build_gt_pairs(gt_inst, gt_ocr)
        if not gt_pairs:
            for se in seg_engines:
                skipped[se] += 1
            continue

        # pred OCR
        ocr_path = out_root / sid / f"{ocr_engine}.json"
        if not ocr_path.exists():
            for se in seg_engines:
                skipped[se] += 1
            continue
        pred_ocr = load_pred_ocr_ft(ocr_path)

        for se in seg_engines:
            seg_path = out_root / sid / f"{se}.json"
            if not seg_path.exists():
                skipped[se] += 1
                continue

            seg_boxes = load_pred_seg_boxes(seg_path)

            # three strategies
            t11=time.time()
            pred_static = match_static(seg_boxes, pred_ocr)
            t22=time.time()
            t0+=t22-t11
            pred_dynamic = match_dynamic(seg_boxes, pred_ocr)
            t33=time.time()
            t1+=t33-t22
            pred_km = match_km(seg_boxes, pred_ocr)
            t44=time.time()
            t2+=t44-t33

            # fuse by vote
            static_fused = fuse_vote(seg_boxes, pred_static, pred_dynamic, pred_km,default_method="static")
            dynamic_fused = fuse_vote(seg_boxes, pred_static, pred_dynamic, pred_km,default_method="dynamic")
            km_fused = fuse_vote(seg_boxes, pred_static, pred_dynamic, pred_km,default_method="km")


            # evaluate
            tp, fp, fn = eval_pairs_prf(gt_pairs, pred_static)
            agg[se]["static"][0] += tp; agg[se]["static"][1] += fp; agg[se]["static"][2] += fn

            tp, fp, fn = eval_pairs_prf(gt_pairs, pred_dynamic)
            agg[se]["dynamic"][0] += tp; agg[se]["dynamic"][1] += fp; agg[se]["dynamic"][2] += fn

            tp, fp, fn = eval_pairs_prf(gt_pairs, pred_km)
            agg[se]["km"][0] += tp; agg[se]["km"][1] += fp; agg[se]["km"][2] += fn

            tp, fp, fn = eval_fused_prf(gt_pairs, static_fused)
            agg[se]["static_fused"][0] += tp; agg[se]["static_fused"][1] += fp; agg[se]["static_fused"][2] += fn

            tp, fp, fn = eval_fused_prf(gt_pairs, dynamic_fused)
            agg[se]["dynamic_fused"][0] += tp; agg[se]["dynamic_fused"][1] += fp; agg[se]["dynamic_fused"][2] += fn

            tp, fp, fn = eval_fused_prf(gt_pairs, km_fused)
            agg[se]["km_fused"][0] += tp; agg[se]["km_fused"][1] += fp; agg[se]["km_fused"][2] += fn

            evaluated[se] += 1

            # save per-sample preds
            if args.save_preds:
                base_dir = out_root / sid / "match_eval" / f"{se}__{ocr_engine}"
                save_json(base_dir / "static.json", pred_static)
                save_json(base_dir / "dynamic.json", pred_dynamic)
                save_json(base_dir / "km.json", pred_km)
                save_json(base_dir / "static_fused.json", static_fused)
                save_json(base_dir / "dynamic_fused.json", dynamic_fused)
                save_json(base_dir / "km_fused.json", km_fused)
                save_json(base_dir / "meta.json", {
                    "sample_id": sid,
                    "seg_engine": se,
                    "ocr_engine": ocr_engine,
                    "iou": IOU_THR,
                    "paths": {
                        "seg": str(seg_path),
                        "ocr": str(ocr_path),
                        "gt_instances": str(gt_inst_path),
                        "gt_ocr": str(gt_ocr_path),
                    }
                })

    # final summary
    out = {
        "mimic_root": str(root),
        "ocr_engine": ocr_engine,
        "seg_engines": seg_engines,
        "iou": IOU_THR,
        "results": []
    }

    for se in seg_engines:
        se_item = {
            "seg_engine": se,
            "evaluated_samples": evaluated[se],
            "skipped_samples": skipped[se],
            "prf": {
                "static": prf(*agg[se]["static"]),
                "dynamic": prf(*agg[se]["dynamic"]),
                "km": prf(*agg[se]["km"]),
                "static_fused": prf(*agg[se]["static_fused"]),
                "dynamic_fused": prf(*agg[se]["dynamic_fused"]),
                "km_fused": prf(*agg[se]["km_fused"]),
            }
        }
        out["results"].append(se_item)

    out_path = root / args.out_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote summary: {out_path}")
    print(t0, t1, t2,len(sample_dirs))
    print(f"[TIME] match_static per sample: {(t0/len(sample_dirs)):.2f} sec, match_dynamic per sample: {(t1/len(sample_dirs)):.2f} sec, match_km per sample: {(t2/len(sample_dirs)):.2f} sec")


if __name__ == "__main__":
    main()
