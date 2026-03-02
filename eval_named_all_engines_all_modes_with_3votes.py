#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path
import time
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
import re


# =========================
# IoU / PRF
# =========================
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


def prf(tp: int, fp: int, fn: int) -> Dict[str, Any]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "P": round(p, 6), "R": round(r, 6), "F1": round(f, 6)}


def norm_name(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    t = str(s).strip()
    return t.lower() if t else None


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# =========================
# GT: gt_instances.json
# =========================
def load_gt_instances(path: Path) -> List[Dict[str, Any]]:
    """
    Need label, name, bbox_xyxy (use as segmentation bbox)
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    if not isinstance(data, list):
        return out
    for x in data:
        if not isinstance(x, dict):
            continue
        lab = x.get("label", None)
        name = x.get("name", None)
        box = x.get("bbox_xyxy", None)
        if lab is None or name is None or not (isinstance(box, list) and len(box) == 4):
            continue
        out.append({
            "label": str(lab).strip().upper(),
            "name_norm": norm_name(name),
            "bbox_xyxy": [float(v) for v in box],
        })
    return out


# =========================
# LLM: {engine}_{mode}.json
# =========================
def load_llm_label2name(path: Path) -> Dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    llm = data.get("llm", {})
    results = llm.get("results", {})
    if not isinstance(results, dict):
        return {}
    out = {}
    for k, v in results.items():
        out[str(k).strip().upper()] = "" if v is None else str(v).strip()
    return out


def discover_llm_variants(llm_output_dir: Path) -> List[Tuple[str, str, str]]:
    """
    Scan LLM_output/{sid} folders, collect (engine, mode, filename)
    filename is like: baichuan_base.json
    returns unique list.
    """
    variants = set()
    if not llm_output_dir.exists():
        return []
    for sid_dir in llm_output_dir.iterdir():
        if not sid_dir.is_dir():
            continue
        for fp in sid_dir.glob("*.json"):
            name = fp.name
            # split last "_"
            if "_" not in name:
                continue
            engine, mode_ext = name.rsplit("_", 1)
            mode = mode_ext.replace(".json", "")
            engine = engine.strip()
            mode = mode.strip()
            if engine and mode:
                variants.add((engine, mode, name))
    return sorted(list(variants))


# =========================
# match_eval files
# =========================
def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def extract_pred_pairs(algo: str, obj: Any) -> List[Dict[str, Any]]:
    """
    static/dynamic/km: list of {"label","seg_box","ocr_box"}
    fused not used here (we will rebuild fused with 3 defaults)
    """
    out = []
    if not isinstance(obj, list):
        return out
    for it in obj:
        if not isinstance(it, dict):
            continue
        lab = it.get("label", None)
        sb = it.get("seg_box", None)
        ob = it.get("ocr_box", None)
        if lab is None or not (isinstance(sb, list) and len(sb) == 4):
            continue
        rec = {
            "label": str(lab).strip().upper(),
            "seg_box": [float(v) for v in sb],
        }
        # ocr_box optional (for debug/save)
        if isinstance(ob, list) and len(ob) == 4:
            rec["ocr_box"] = [float(v) for v in ob]
        out.append(rec)
    return out


def attach_name(pred: List[Dict[str, Any]], llm_map: Dict[str, str]) -> List[Dict[str, Any]]:
    out = []
    for it in pred:
        lab = it["label"]
        name = llm_map.get(lab, "")
        out.append({
            **it,
            "name": name,
            "name_norm": norm_name(name),
        })
    return out


# =========================
# Vote fuse (3 defaults)
# =========================
def fuse_vote_default(
    seg_boxes: List[List[float]],
    pred_s: List[Dict[str, Any]],
    pred_d: List[Dict[str, Any]],
    pred_k: List[Dict[str, Any]],
    default_method: str = "static",
) -> List[Dict[str, Any]]:
    """
    Return fused list in "pred" format: {"label","name","name_norm","seg_box","ocr_box"?}
    Voting is per seg_box key. Majority by label; if tie/all different -> choose default_method.
    """
    default_method = (default_method or "static").strip().lower()
    if default_method not in ("static", "dynamic", "km"):
        default_method = "static"

    def key_box(sb: List[float]) -> str:
        return f"{sb[0]},{sb[1]},{sb[2]},{sb[3]}"

    def to_map(pred: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        mp = {}
        for it in pred:
            sb = it.get("seg_box")
            if not (isinstance(sb, list) and len(sb) == 4):
                continue
            mp[key_box(sb)] = it
        return mp

    ms = to_map(pred_s)
    md = to_map(pred_d)
    mk = to_map(pred_k)

    def pick(method: str, cs, cd, ck):
        if method == "static":
            return cs
        if method == "dynamic":
            return cd
        if method == "km":
            return ck
        return cs

    fused = []
    for sb in seg_boxes:
        k = key_box(sb)
        cs = ms.get(k)
        cd = md.get(k)
        ck = mk.get(k)

        votes = []
        if cs and cs.get("label") is not None: votes.append(("static", cs["label"]))
        if cd and cd.get("label") is not None: votes.append(("dynamic", cd["label"]))
        if ck and ck.get("label") is not None: votes.append(("km", ck["label"]))

        if not votes:
            continue

        cnt = Counter([lab for _, lab in votes])
        best_label, best_count = cnt.most_common(1)[0]

        # majority?
        if best_count >= 2:
            # prefer default_method if it voted for best_label
            preferred = pick(default_method, cs, cd, ck)
            if preferred is not None and preferred.get("label") == best_label:
                fused.append(preferred)
            else:
                # stable priority
                chosen = None
                for m in ("static", "dynamic", "km"):
                    cand = pick(m, cs, cd, ck)
                    if cand is not None and cand.get("label") == best_label:
                        chosen = cand
                        break
                if chosen is None:
                    chosen = pick("static", cs, cd, ck) or pick("dynamic", cs, cd, ck) or pick("km", cs, cd, ck)
                if chosen is not None:
                    fused.append(chosen)
        else:
            # all different / tie -> default_method if exists else static->dynamic->km
            preferred = pick(default_method, cs, cd, ck)
            if preferred is not None:
                fused.append(preferred)
            else:
                chosen = pick("static", cs, cd, ck) or pick("dynamic", cs, cd, ck) or pick("km", cs, cd, ck)
                if chosen is not None:
                    fused.append(chosen)

    return fused


# =========================
# Eval: label+name + IoU(seg_box, gt_bbox)>=thr
# =========================
def eval_named_prf(gt: List[Dict[str, Any]], pred: List[Dict[str, Any]], thr: float) -> Tuple[int, int, int]:
    candidates = []
    for pi, p in enumerate(pred):
        plab = p.get("label")
        pname = p.get("name_norm")
        psb = p.get("seg_box")
        if plab is None or pname is None or not (isinstance(psb, list) and len(psb) == 4):
            continue

        for gi, g in enumerate(gt):
            if plab != g["label"]:
                continue
            if pname != g["name_norm"]:
                continue
            i = iou_xyxy(psb, g["bbox_xyxy"])
            if i >= thr:
                candidates.append((i, pi, gi))

    candidates.sort(reverse=True, key=lambda x: x[0])
    used_p, used_g = set(), set()
    tp = 0
    for _, pi, gi in candidates:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        tp += 1

    fp = len(pred) - tp
    fn = len(gt) - tp
    return tp, fp, fn


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic_root", default="/home/root123/mount1/weilai/MIMIC")
    ap.add_argument("--iou", type=float, default=0.45)

    ap.add_argument("--match_eval_dirname", default="match_eval")
    ap.add_argument("--save_dirname", default="match_eval_named")  # output/{sid}/match_eval_named/...

    ap.add_argument("--out_json", default="eval_named_all_engines_all_modes_iou45.json")
    args = ap.parse_args()

    root = Path(args.mimic_root)
    data_root = root / "data"
    out_root = root / "output"
    llm_root = root / "LLM_output"
    thr = float(args.iou)

    # discover llm variants globally: (engine, mode, filename)
    llm_variants = discover_llm_variants(llm_root)
    if not llm_variants:
        raise SystemExit(f"[ERR] No LLM variants found under: {llm_root}")

    # aggregates:
    # key = (combo, llm_engine, llm_mode) where combo is like "unet__ft" or "SAM__ft"
    # algos: static/dynamic/km + fused@static/fused@dynamic/fused@km
    agg = defaultdict(lambda: {a: [0, 0, 0] for a in (
        "static", "dynamic", "km",
        "fused@static", "fused@dynamic", "fused@km"
    )})
    evaluated = defaultdict(int)
    skipped = defaultdict(int)

    # loop all samples
    for sd in sorted(data_root.iterdir()):
        if not sd.is_dir():
            continue
        sid = sd.name

        gt_path = sd / "gt_instances.json"
        if not gt_path.exists():
            continue
        gt = load_gt_instances(gt_path)
        if not gt:
            continue

        # match_eval root for this sample
        me_root = out_root / sid / args.match_eval_dirname
        if not me_root.exists():
            continue

        # pre-load all llm maps for this sample (only those that exist)
        llm_maps: Dict[Tuple[str, str], Dict[str, str]] = {}
        for eng, mode, fname in llm_variants:
            p = llm_root / sid / fname
            if p.exists():
                llm_maps[(eng, mode)] = load_llm_label2name(p)
            break

        if not llm_maps:
            # sample has no llm files -> cannot eval named
            continue
        t0=time.time()
        t1=time.time()
        t2=time.time()
        # each combo dir: unet__ft, SAM__ft, ...
        for combo_dir in sorted(me_root.iterdir()):
            if not combo_dir.is_dir():
                continue
            combo = combo_dir.name

            p_static = combo_dir / "static.json"
            p_dynamic = combo_dir / "dynamic.json"
            p_km = combo_dir / "km.json"

            if not (p_static.exists() and p_dynamic.exists() and p_km.exists()):
                # must have 3 to build 3-default fused robustly
                continue

            obj_s = load_json(p_static)
            obj_d = load_json(p_dynamic)
            obj_k = load_json(p_km)
            if obj_s is None or obj_d is None or obj_k is None:
                continue

            base_s = extract_pred_pairs("static", obj_s)
            base_d = extract_pred_pairs("dynamic", obj_d)
            base_k = extract_pred_pairs("km", obj_k)

            # collect seg_boxes from static entries (1-1 per seg_box in your generator)
            seg_boxes = [it["seg_box"] for it in base_s if isinstance(it.get("seg_box"), list) and len(it["seg_box"]) == 4]

            # for each llm variant that exists for this sample
            for (llm_engine, llm_mode), llm_map in llm_maps.items():
                key = (combo, llm_engine, llm_mode)
                time11=time.time()
                pred_s = attach_name(base_s, llm_map)
                time22=time.time()
                pred_d = attach_name(base_d, llm_map)
                time33=time.time()
                pred_k = attach_name(base_k, llm_map)
                time44=time.time()
                t0+=time22-time11
                t1+=time33-time22
                t2+=time44-time33

                # build three fused variants
                fused_static = fuse_vote_default(seg_boxes, pred_s, pred_d, pred_k, default_method="static")
                fused_dynamic = fuse_vote_default(seg_boxes, pred_s, pred_d, pred_k, default_method="dynamic")
                fused_km = fuse_vote_default(seg_boxes, pred_s, pred_d, pred_k, default_method="km")

                # save intermediate named preds
                save_base = out_root / sid / args.save_dirname / f"{combo}__{llm_engine}_{llm_mode}"
                save_json(save_base / "static.json", pred_s)
                save_json(save_base / "dynamic.json", pred_d)
                save_json(save_base / "km.json", pred_k)
                save_json(save_base / "fused@static.json", fused_static)
                save_json(save_base / "fused@dynamic.json", fused_dynamic)
                save_json(save_base / "fused@km.json", fused_km)

                # eval and accumulate
                tp, fp, fn = eval_named_prf(gt, pred_s, thr)
                agg[key]["static"][0] += tp; agg[key]["static"][1] += fp; agg[key]["static"][2] += fn

                tp, fp, fn = eval_named_prf(gt, pred_d, thr)
                agg[key]["dynamic"][0] += tp; agg[key]["dynamic"][1] += fp; agg[key]["dynamic"][2] += fn

                tp, fp, fn = eval_named_prf(gt, pred_k, thr)
                agg[key]["km"][0] += tp; agg[key]["km"][1] += fp; agg[key]["km"][2] += fn

                tp, fp, fn = eval_named_prf(gt, fused_static, thr)
                agg[key]["fused@static"][0] += tp; agg[key]["fused@static"][1] += fp; agg[key]["fused@static"][2] += fn

                tp, fp, fn = eval_named_prf(gt, fused_dynamic, thr)
                agg[key]["fused@dynamic"][0] += tp; agg[key]["fused@dynamic"][1] += fp; agg[key]["fused@dynamic"][2] += fn

                tp, fp, fn = eval_named_prf(gt, fused_km, thr)
                agg[key]["fused@km"][0] += tp; agg[key]["fused@km"][1] += fp; agg[key]["fused@km"][2] += fn

                evaluated[key] += 1

    # build summary json
    summary = {
        "mimic_root": str(root),
        "iou": thr,
        "match_eval_dirname": args.match_eval_dirname,
        "save_dirname": args.save_dirname,
        "results": []
    }

    # sort results for readability
    for (combo, llm_engine, llm_mode) in sorted(agg.keys(), key=lambda x: (x[0], x[1], x[2])):
        item = {
            "combo": combo,
            "llm_engine": llm_engine,
            "llm_mode": llm_mode,
            "evaluated_samples": evaluated.get((combo, llm_engine, llm_mode), 0),
            "prf": {}
        }
        for algo in ("static", "dynamic", "km", "fused@static", "fused@dynamic", "fused@km"):
            tp, fp, fn = agg[(combo, llm_engine, llm_mode)][algo]
            item["prf"][algo] = prf(tp, fp, fn)
        summary["results"].append(item)

    out_path = root / args.out_json
    save_json(out_path, summary)
    print(f"[OK] wrote summary: {out_path}")
    print(f"[TIME] attach_name_per_sample: {t0/228:.2f} sec, dynamic: {t1/228:.2f} sec, km: {t2/228:.2f} sec")    


if __name__ == "__main__":
    main()
