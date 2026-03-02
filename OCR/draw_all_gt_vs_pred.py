#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw
from tqdm import tqdm


# -------------------------
# Load GT
# -------------------------
def load_gt_boxes(gt_json: Path) -> List[Tuple[str, List[int]]]:
    data = json.loads(gt_json.read_text(encoding="utf-8"))
    boxes = []
    for x in data:
        if "label" not in x or "bbox_xyxy" not in x:
            continue
        boxes.append((str(x["label"]), list(map(int, x["bbox_xyxy"]))))
    return boxes


# -------------------------
# Load Pred (your OCR json)
# -------------------------
def load_pred_boxes(pred_json: Path) -> List[Tuple[str, List[int]]]:
    data = json.loads(pred_json.read_text(encoding="utf-8"))
    results = data.get("results", [])

    boxes = []
    for item in results:
        if not isinstance(item, list) or len(item) != 2:
            continue
        label, box = item
        if not isinstance(box, dict):
            continue
        if not all(k in box for k in ("left", "top", "width", "height")):
            continue

        x1 = int(box["left"])
        y1 = int(box["top"])
        x2 = int(box["left"] + box["width"])
        y2 = int(box["top"] + box["height"])
        boxes.append((str(label), [x1, y1, x2, y2]))
    return boxes


def draw_boxes(img: Image.Image, boxes: List[Tuple[str, List[int]]], color: str, width: int, draw_label: bool):
    draw = ImageDraw.Draw(img)
    for label, (x1, y1, x2, y2) in boxes:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        if draw_label:
            draw.text((x1 + 2, max(0, y1 - 12)), label, fill=color)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic_root", required=True, help="/home/root123/mount1/weilai/MIMIC")
    ap.add_argument("--engine", required=True, help="paddle / easyocr / doctr etc")
    ap.add_argument("--out_dir", default="", help="default: MIMIC/vis")
    ap.add_argument("--width", type=int, default=3, help="box line width")
    ap.add_argument("--draw_label", action="store_true", help="draw label text")
    args = ap.parse_args()

    root = Path(args.mimic_root).resolve()
    data_root = root / "data"
    out_root = root / "output"

    if not data_root.is_dir():
        raise FileNotFoundError(f"Not found: {data_root}")
    if not out_root.is_dir():
        raise FileNotFoundError(f"Not found: {out_root}")

    vis_root = Path(args.out_dir).resolve() if args.out_dir else (root / "vis")
    vis_root.mkdir(parents=True, exist_ok=True)

    # collect sample ids from data/*
    sample_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])

    total = 0
    saved = 0
    skipped = 0
    failed = 0

    with tqdm(total=len(sample_dirs), desc="Draw", ncols=90) as pbar:
        for sdir in sample_dirs:
            sid = sdir.name
            total += 1

            img_path = sdir / "merged.jpg"
            gt_json = sdir / "gt_ocr.json"
            pred_json = out_root / sid / f"{args.engine}.json"

            # 必需文件缺失则跳过
            if (not img_path.exists()) or (not gt_json.exists()) or (not pred_json.exists()):
                skipped += 1
                pbar.update(1)
                continue

            try:
                gt_boxes = load_gt_boxes(gt_json)
                pred_boxes = load_pred_boxes(pred_json)

                img = Image.open(img_path).convert("RGB")

                # GT 红，Pred 绿
                draw_boxes(img, gt_boxes, color="red", width=args.width, draw_label=args.draw_label)
                draw_boxes(img, pred_boxes, color="green", width=args.width, draw_label=args.draw_label)

                out_dir = vis_root / sid
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"gt_vs_{args.engine}.jpg"
                img.save(out_path, quality=95)

                saved += 1
            except Exception:
                failed += 1

            pbar.update(1)

    print(f"[DONE] total={total} saved={saved} skipped={skipped} failed={failed}")
    print(f"[OUT]  {vis_root}")


if __name__ == "__main__":
    main()
