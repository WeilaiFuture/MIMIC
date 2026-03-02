#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
compose_and_label.py

目的：合成 plate-level 图片，并生成后续论文评估所需的 GT 文件：
- OCR: PRF (IoU=0.5)
- mask: PRF (IoU=0.5) 这里用每个子图粘贴区域的矩形实例作为 GT
- name: PRF (基于 label->name 或 instance->name)
- OCR-mask 匹配准确率
- OCR-name 匹配准确率
- 三者（mask+OCR+name）同时匹配准确率

输出（每个 out_dir）：
- merged.jpg
- text.txt
- gt_label2name.json
- gt_instances.json
- gt_ocr.json
- gt_triples.json
- meta.json
"""

import json
import os
import random
import math
from typing import List, Dict, Tuple, Any
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from sentence import process_labels_dict


# ----------------------------
# Utils
# ----------------------------
def _xyxy_to_xywh(b: List[int]) -> List[int]:
    l, t, r, b2 = b
    return [int(l), int(t), int(r - l), int(b2 - t)]

def _bbox_poly_xyxy(b: List[int]) -> List[int]:
    l, t, r, b2 = b
    return [int(l), int(t), int(r), int(t), int(r), int(b2), int(l), int(b2)]

def _safe_int_bbox(bb) -> List[int]:
    return [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])]


# ----------------------------
# Label generation (reproducible)
# ----------------------------
def generate_labels(n: int, names: List[str], rng: random.Random) -> Tuple[List[str], str]:
    """
    返回 labels, label_type
    """
    name_count: Dict[str, int] = {}
    for name in names:
        name_count[name] = name_count.get(name, 0) + 1

    if len(name_count.keys()) > 15:
        label_types = ['numbers']
    elif n < 26:
        label_types = ['numbers', 'letters', 'mixed']
    else:
        label_types = ['numbers', 'mixed', 'double_letters']

    chosen_type = rng.choice(label_types)
    labels: List[str] = []

    if chosen_type == 'numbers':
        labels = [str(i) for i in range(1, n + 1)]
    elif chosen_type == 'letters':
        labels = [chr(65 + i) for i in range(n)]
    elif chosen_type == 'mixed':
        flag = any(v >= 10 for v in name_count.values())
        letter_prefixes = [chr(65 + i) for i in range(len(name_count))]
        if flag:
            for prefix, count in zip(letter_prefixes, name_count.values()):
                for i in range(count):
                    labels.append(f"{prefix}{chr(65 + i)}")
        else:
            for prefix, count in zip(letter_prefixes, name_count.values()):
                for i in range(1, count + 1):
                    labels.append(f"{prefix}{i}")
    else:  # double_letters
        letter_prefixes = [chr(65 + i) for i in range(len(name_count))]
        for prefix, count in zip(letter_prefixes, name_count.values()):
            for i in range(count):
                labels.append(f"{prefix}{chr(65 + i)}")

    return labels, chosen_type


# ----------------------------
# Drawing
# ----------------------------
def _load_font(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size)
    except OSError:
        print("Default font used.")
        return ImageFont.load_default()

def draw_label(draw: ImageDraw.ImageDraw,
               label: str,
               position: Tuple[int, int],  # top-left of pasted image
               img_width: int,
               img_height: int,
               fixed_font_size: int = 20):
    """
    返回 placed_text_bbox, ellipse_bbox（均为 [l, t, r, b]）
    OCR 评估主要使用 placed_text_bbox
    """
    font = _load_font(fixed_font_size)

    # measure text
    test_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = test_bbox[2] - test_bbox[0]
    text_height = test_bbox[3] - test_bbox[1]

    ascent, descent = font.getmetrics()
    real_height = ascent + descent

    # place in top-right
    circle_x = position[0] + img_width - text_width // 2 - 2
    circle_y = position[1] + real_height // 2 + 2
    radius_x = text_width // 2 + 2
    radius_y = real_height // 2 + 2

    ellipse_bbox = [
        circle_x - radius_x,
        circle_y - radius_y,
        circle_x + radius_x,
        circle_y + radius_y,
    ]
    draw.ellipse([(ellipse_bbox[0], ellipse_bbox[1]), (ellipse_bbox[2], ellipse_bbox[3])], fill="white")

    # baseline align
    text_x = circle_x - text_width // 2
    text_y = circle_y - ascent // 2 - descent // 2

    placed_text_bbox = draw.textbbox((text_x, text_y), label, font=font)
    draw.text((text_x, text_y), label, font=font, fill="black")

    return placed_text_bbox, ellipse_bbox


# ----------------------------
# Layout helpers (packed layout)
# ----------------------------
def calculate_total_area(image_paths: List[str]) -> Tuple[int, List[Tuple[int, int]]]:
    total_area = 0
    sizes = []
    for img_path in image_paths:
        with Image.open(img_path) as img:
            width, height = img.size
        sizes.append((width, height))
        total_area += width * height
    return total_area, sizes

def arrange_images_in_square_like_box(image_paths: List[str], max_size=2000):
    best_positions = None
    best_width = best_height = max_size

    while best_positions is None:
        total_area, sizes = calculate_total_area(image_paths)
        best_width = best_height = max_size
        best_positions = None
        estimated_side = math.isqrt(total_area)

        for trial_width in range(min(max_size, estimated_side), max_size):
            current_x, current_y = 0, 0
            row_height = 0
            positions = []
            for width, height in sizes:
                if current_x + width > trial_width:
                    current_x = 0
                    current_y += row_height
                    row_height = 0
                positions.append((current_x, current_y))
                current_x += width
                row_height = max(row_height, height)

            total_height = current_y + row_height
            if total_height < max_size and (trial_width * total_height < best_width * best_height):
                aspect_ratio = abs(trial_width - total_height)
                if aspect_ratio < abs(best_width - best_height) or best_width == best_height:
                    best_width = trial_width
                    best_height = total_height
                    best_positions = positions

        max_size += 1000

    return best_positions, (int(best_width) + 10, int(best_height) + 10)


# ----------------------------
# Compositing (packed layout) + GT outputs
# ----------------------------
def merge_images(selected_image_paths: List[str],
                 output_path: str,
                 max_size: int,
                 labels: List[str],
                 names: List[str],
                 spacing: int = 5):
    """
    输出：
    - gt_instances: list[dict]  用于 mask PRF (IoU=0.5)
    - gt_ocr:       list[dict]  用于 OCR PRF (IoU=0.5) & OCR-mask/OCR-name 匹配
    - gt_triples:   list[dict]  用于三者同时匹配
    """
    n = len(selected_image_paths)
    positions, (bounding_width, bounding_height) = arrange_images_in_square_like_box(selected_image_paths, max_size)

    new_im = Image.new('RGB', (bounding_width, bounding_height), 'white')
    draw = ImageDraw.Draw(new_im)

    # adaptive font size
    total_width = total_height = 0
    for img_path in selected_image_paths:
        with Image.open(img_path) as img:
            total_width += img.size[0]
            total_height += img.size[1]
    avg_width = max(1, total_width // max(1, n))
    avg_height = max(1, total_height // max(1, n))
    adaptive_font_size = max(20, avg_height // 15, avg_width // 15)

    gt_instances: List[Dict[str, Any]] = []
    gt_ocr: List[Dict[str, Any]] = []
    gt_triples: List[Dict[str, Any]] = []

    for i, img_path in enumerate(selected_image_paths):
        instance_id = i
        label = labels[i]
        name = names[i]

        with Image.open(img_path) as img:
            original_size = img.size
            try:
                resized_size = (max(1, original_size[0] - spacing), max(1, original_size[1] - spacing))
                img = img.resize(resized_size)
            except Exception:
                pass

            x_offset, y_offset = positions[i]
            x0 = x_offset + spacing // 2
            y0 = y_offset + spacing // 2
            new_im.paste(img, (x0, y0))

            # instance bbox (rectangle mask GT)
            inst_xyxy = [x0, y0, x0 + img.size[0], y0 + img.size[1]]
            inst_xyxy = _safe_int_bbox(inst_xyxy)

            # draw label -> OCR bbox
            text_bb, _ = draw_label(
                draw,
                label,
                (x0, y0),
                img.size[0],
                img.size[1],
                adaptive_font_size
            )
            ocr_xyxy = _safe_int_bbox(text_bb)

        gt_instances.append({
            "instance_id": int(instance_id),
            "src_path": img_path,
            "label": label,
            "name": name,
            "bbox_xyxy": inst_xyxy,
            "bbox_xywh": _xyxy_to_xywh(inst_xyxy),
            # rectangle polygon, compatible with segmentation format
            "segmentation": [_bbox_poly_xyxy(inst_xyxy)]
        })

        gt_ocr.append({
            "ocr_id": int(instance_id),  # 这里每个实例一个标签框，直接复用即可
            "instance_id": int(instance_id),
            "label": label,
            "name": name,
            "bbox_xyxy": ocr_xyxy,
            "bbox_xywh": _xyxy_to_xywh(ocr_xyxy)
        })

        gt_triples.append({
            "instance_id": int(instance_id),
            "label": label,
            "name": name
        })

    new_im.save(output_path)
    print(f"合成图片已保存至 {output_path}")
    return gt_instances, gt_ocr, gt_triples


# ----------------------------
# FS helpers
# ----------------------------
def gather_image_paths(folder_path: str) -> List[str]:
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.bmp')]


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    bmp_path = "/mnt/nas/牙形石-侯杰/PBDB牙形石-侯杰/new图版"
    base_path = "/home/root123/mount1/weilai/MIMIC/data"

    subdirectories = [d for d in os.listdir(bmp_path) if os.path.isdir(os.path.join(bmp_path, d))]
    index = 0

    subdirectories = [d for d in subdirectories if d.startswith('A')]
    subdirectories += ["17313880330", "37213956404", "55413908119"]

    for subdir in tqdm(subdirectories, desc="Processing subdirectories"):
        subfolder_path = os.path.join(bmp_path, subdir)
        sub_subdirectories = [d for d in os.listdir(subfolder_path) if os.path.isdir(os.path.join(subfolder_path, d))]

        for sub_subdir in sub_subdirectories:
            src_dir = os.path.join(subfolder_path, sub_subdir)
            image_paths = gather_image_paths(src_dir)
            n = len(image_paths)
            if n == 0:
                continue

            out_dir = os.path.join(base_path, str(index + 1))
            os.makedirs(out_dir, exist_ok=True)

            merged_img_path = os.path.join(out_dir, 'merged.jpg')
            text_path = os.path.join(out_dir, 'text.txt')

            gt_label2name_path = os.path.join(out_dir, 'gt_label2name.json')
            gt_instances_path = os.path.join(out_dir, 'gt_instances.json')
            gt_ocr_path = os.path.join(out_dir, 'gt_ocr.json')
            gt_triples_path = os.path.join(out_dir, 'gt_triples.json')
            meta_path = os.path.join(out_dir, 'meta.json')

            # names aligned with sorted file list
            names = []
            for p in image_paths:
                filename = os.path.basename(p)
                prefix = filename.split('-')[0]
                formatted_name = prefix.replace('_', ' ')
                names.append(formatted_name)

            image_paths, names = zip(*sorted(zip(image_paths, names)))
            image_paths = list(image_paths)
            names = list(names)

            # reproducible seed per sample (paper-friendly)
            seed = (hash(subdir) ^ hash(sub_subdir) ^ index) & 0xffffffff
            rng = random.Random(seed)

            labels, label_type = generate_labels(n, names, rng=rng)

            # label->name GT
            labels_dict = {labels[i]: names[i] for i in range(n)}
            with open(gt_label2name_path, 'w', encoding='utf-8') as f:
                json.dump(labels_dict, f, ensure_ascii=False, indent=2)

            # sentence generation (keep your original)
            # process_labels_dict currently expects a json_path; we write a temp json in out_dir
            tmp_json_path = os.path.join(out_dir, "tmp_label2name.json")
            with open(tmp_json_path, 'w', encoding='utf-8') as f:
                json.dump(labels_dict, f, ensure_ascii=False, indent=2)

            result = process_labels_dict(labels_dict, tmp_json_path)
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(result)

            # compositing + GT (instances/ocr/triples)
            max_size = 3000
            gt_instances, gt_ocr, gt_triples = merge_images(
                image_paths,
                merged_img_path,
                max_size,
                labels,
                names,
                spacing=5
            )

            with open(gt_instances_path, 'w', encoding='utf-8') as f:
                json.dump(gt_instances, f, ensure_ascii=False, indent=2)

            with open(gt_ocr_path, 'w', encoding='utf-8') as f:
                json.dump(gt_ocr, f, ensure_ascii=False, indent=2)

            with open(gt_triples_path, 'w', encoding='utf-8') as f:
                json.dump(gt_triples, f, ensure_ascii=False, indent=2)

            meta = {
                "index": int(index + 1),
                "subdir": subdir,
                "sub_subdir": sub_subdir,
                "seed": int(seed),
                "label_type": label_type,
                "n": int(n),
                "src_dir": src_dir,
                "merged_image": "merged.jpg",
                "gt_files": {
                    "gt_label2name": "gt_label2name.json",
                    "gt_instances": "gt_instances.json",
                    "gt_ocr": "gt_ocr.json",
                    "gt_triples": "gt_triples.json"
                }
            }
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            # cleanup temp (optional)
            try:
                os.remove(tmp_json_path)
            except Exception:
                pass

            index += 1
