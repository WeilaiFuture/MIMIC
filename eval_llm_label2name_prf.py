#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import re
from pathlib import Path
from typing import Dict, Tuple, Any


def norm_text(s: Any) -> str:
    """
    轻量规范化：用于“字符串是否一致”的判断（不做模糊匹配）
    """
    if s is None:
        return ""
    s = str(s)
    s = s.strip()
    # 统一空白
    s = re.sub(r"\s+", " ", s)
    # 去掉两端常见标点（可按你需求调整）
    s = s.strip(" \t\r\n.,;:!?'\"，。；：！？“”‘’()（）[]【】")
    # 不区分大小写
    s = s.lower()
    return s


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_gt_label2name(gt_path: Path) -> Dict[str, str]:
    data = load_json(gt_path)
    if not isinstance(data, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in data.items():
        if k is None:
            continue
        out[str(k).strip()] = str(v) if v is not None else ""
    return out


def load_pred_label2name(pred_path: Path) -> Dict[str, str]:
    """
    读取 LLM_output/{id}/{engine}_{mode}.json
    期望结构：
    {
      ...
      "llm": {
        ...
        "results": { "A1": "...", ... }
      }
    }
    """
    data = load_json(pred_path)
    if not isinstance(data, dict):
        return {}

    llm = data.get("llm", {})
    if not isinstance(llm, dict):
        return {}

    results = llm.get("results", {})
    if not isinstance(results, dict):
        return {}

    out: Dict[str, str] = {}
    for k, v in results.items():
        if k is None:
            continue
        out[str(k).strip()] = str(v) if v is not None else ""
    return out


def prf_from_maps(gt: Dict[str, str], pred: Dict[str, str]) -> Tuple[int, int, int]:
    """
    定义：
    - TP：label 在 gt 和 pred 都存在，且 name(规范化后)完全一致
    - FP：pred 中的 label
          - 若 gt 不存在该 label：FP++
          - 若 gt 存在但 name 不一致：FP++
    - FN：gt 中的 label
          - 若 pred 不存在该 label：FN++
          - 若 pred 存在但 name 不一致：FN++   （错误预测同时算 FP+FN）
    """
    tp = fp = fn = 0

    all_keys = set(gt.keys()) | set(pred.keys())

    for k in all_keys:
        g_has = k in gt
        p_has = k in pred

        if g_has and p_has:
            g = norm_text(gt[k])
            p = norm_text(pred[k])
            if g != "" and p != "" and g == p:
                tp += 1
            else:
                # 预测了但不对（或空）=> 同时算 FP 和 FN
                fp += 1
                fn += 1
        elif (not g_has) and p_has:
            fp += 1
        elif g_has and (not p_has):
            fn += 1

    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic_root", default="/home/root123/mount1/weilai/MIMIC",
                    help="MIMIC 根目录，例如 /home/root123/mount1/weilai/MIMIC")
    ap.add_argument("--engine", required=True, help="例如 baichuan / ds / llama3 ...")
    ap.add_argument("--mode", required=True, help="例如 base / ft ...")
    ap.add_argument("--gt_name", default="gt_label2name.json", help="GT 文件名，位于 data/{id}/ 下")
    ap.add_argument("--show_per_sample", action="store_true", help="打印每个样本的 PRF")
    args = ap.parse_args()
    engines = [x.strip() for x in args.engine.split(",") if x.strip()]
    modes = [x.strip() for x in args.mode.split(",") if x.strip()]
    root = Path(args.mimic_root)
    data_root = root / "data"
    out_root = root / "LLM_output"

    total_tp = total_fp = total_fn = 0
    evaluated = 0
    skipped = 0

    for engine in engines:
        for mode in modes:
            for sample_dir in sorted(data_root.iterdir()):
                if not sample_dir.is_dir():
                    continue

                sid = sample_dir.name
                gt_path = sample_dir / args.gt_name
                pred_path = out_root / sid / f"{engine}_{mode}.json"

                if not gt_path.exists() or not pred_path.exists():
                    skipped += 1
                    continue

                gt = load_gt_label2name(gt_path)
                pred = load_pred_label2name(pred_path)

                tp, fp, fn = prf_from_maps(gt, pred)

                total_tp += tp
                total_fp += fp
                total_fn += fn
                evaluated += 1

                if args.show_per_sample:
                    p, r, f = prf(tp, fp, fn)
                    print(f"[{sid}] TP={tp:4d} FP={fp:4d} FN={fn:4d}  P={p:.4f} R={r:.4f} F1={f:.4f}")

            P, R, F = prf(total_tp, total_fp, total_fn)

            print("\n================ Summary ================")
            print(f"Evaluated samples : {evaluated}")
            print(f"Skipped samples   : {skipped}")
            print("-----------------------------------------")
            print(f"TOTAL TP={total_tp:6d}  FP={total_fp:6d}  FN={total_fn:6d}")
            print(f"Micro P={P:.4f}  R={R:.4f}  F1={F:.4f}")


if __name__ == "__main__":
    main()
