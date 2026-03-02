#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time
from typing import List, Dict, Any, Tuple
import cv2
from tqdm import tqdm
def calculate_intersection_area(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x1 < x2 and y1 < y2:
        return (x2 - x1) * (y2 - y1)
    return 0
def merge_overlapping_boxes_with_ratio(boxes, threshold_ratio=0.4):
    n = len(boxes)
    if n == 0:
        return []
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            inter_area = calculate_intersection_area(boxes[i], boxes[j])
            area1 = max((boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]), 1e-6)
            area2 = max((boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1]), 1e-6)
            smaller = min(area1, area2)
            if inter_area / smaller >= threshold_ratio:
                adj[i].append(j)
                adj[j].append(i)
    visited = [False] * n
    merged_boxes = []
    def dfs(idx, cluster):
        visited[idx] = True
        cluster.append(boxes[idx])
        for nei in adj[idx]:
            if not visited[nei]:
                dfs(nei, cluster)
    for i in range(n):
        if not visited[i]:
            cluster = []
            dfs(i, cluster)
            xmin = min(b[0] for b in cluster)
            ymin = min(b[1] for b in cluster)
            xmax = max(b[2] for b in cluster)
            ymax = max(b[3] for b in cluster)
            merged_boxes.append((xmin, ymin, xmax, ymax))
    return merged_boxes

def get_bounding_boxes(mask, min_area=500):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h >= min_area:
            bounding_boxes.append((x, y, x + w, y + h))
    return bounding_boxes


# =========================
# Worker globals
# =========================
_WORKER_ENGINE = None
_WORKER_ENGINE_NAME = None
_WORKER_GPU_ID = None
_WORKER_BASE_DIR = None


def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-GPU local Seg inference for MIMIC-style folders -> MIMIC/output/<id>/<engine>.json"
    )
    p.add_argument("--input", type=str, default="MIMIC/data")
    p.add_argument("--engine", type=str, default="SAM", choices=["SAM", "unet"])
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--num_workers_per_gpu", type=int, default=1)
    p.add_argument("--pattern", type=str, default="*.jpg")
    p.add_argument("--skip_if_exists", action="store_true")
    p.add_argument(
    "--save_masks",
    action="store_true",
    help="Save original image and binary mask to output directory"
)

    return p.parse_args()


def locate_mimic_root_and_sample_id(img_path: Path) -> Tuple[Path, str]:
    img_path = img_path.resolve()
    for parent in img_path.parents:
        if parent.name == "data":
            return parent.parent, img_path.parent.name
    raise RuntimeError(f"Cannot locate MIMIC root for: {img_path}")


def list_images_and_prepare_jobs(
    input_path: Path,
    pattern: str,
    engine: str,
    skip_if_exists: bool,
    save_masks: bool,  
) -> List[Dict[str, Any]]:

    input_path = input_path.resolve()
    images: List[Path] = []

    if input_path.is_file():
        images = [input_path]
    else:
        data_dir = input_path
        if (input_path / "data").is_dir():
            data_dir = input_path / "data"
        images = sorted(data_dir.rglob(pattern))

    jobs = []
    for img in images:
        if img.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
            continue

        mimic_root, sample_id = locate_mimic_root_and_sample_id(img)
        out_json = mimic_root / "output" / sample_id / f"{engine}.json"

        if skip_if_exists and out_json.exists():
            continue

        jobs.append({
            "image": str(img),
            "mimic_root": str(mimic_root),
            "sample_id": sample_id,
            "out_json": str(out_json),
            "save_masks": bool(save_masks),
        })

    return jobs


def split_round_robin(jobs, gpu_list):
    buckets = {g: [] for g in gpu_list}
    for i, job in enumerate(jobs):
        buckets[gpu_list[i % len(gpu_list)]].append(job)
    return buckets


# =========================
# Worker init
# =========================
def _worker_init(gpu_id: str, engine_name: str):
    global _WORKER_ENGINE, _WORKER_ENGINE_NAME, _WORKER_GPU_ID, _WORKER_BASE_DIR

    _WORKER_GPU_ID = str(gpu_id)
    _WORKER_ENGINE_NAME = engine_name

    # 🔴 必须最先做
    os.environ["CUDA_VISIBLE_DEVICES"] = _WORKER_GPU_ID
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # 🔴 再 import torch
    import torch
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    _WORKER_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _WORKER_BASE_DIR not in sys.path:
        sys.path.append(_WORKER_BASE_DIR)

    from Segmentation.eam_sam.predict import FewShotSegmentation
    from Segmentation.unet.predict import UnetSegmentation

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    if engine_name == "SAM":
        _WORKER_ENGINE = FewShotSegmentation(
            name="xl0",
            weight_url="/home/root123/mount1/weilai/MIMIC/Segmentation/eam_sam/efficientvit/assets/checkpoints/sam/xl0.pt",
            train_path="/home/root123/mount1/weilai/MIMIC/Segmentation/eam_sam/few_shot",
            device=device,
        )
    elif engine_name == "unet":
        _WORKER_ENGINE = UnetSegmentation(
            weight_url="/home/root123/mount1/weilai/MIMIC/Segmentation/unet/best_model.pth",
            device=device,
        )
    else:
        raise ValueError(engine_name)


def _worker_seg_one(job: Dict[str, Any]) -> Dict[str, Any]:
    global _WORKER_ENGINE, _WORKER_ENGINE_NAME, _WORKER_GPU_ID

    img_path = job["image"]
    try:
        _, result = _WORKER_ENGINE.predict(img_path)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        boxes = get_bounding_boxes(gray, min_area=500)  # min_area 自己调
        merged_boxes = merge_overlapping_boxes_with_ratio(boxes, 0.4)
        if job.get("save_masks", False):
            out_json_path = Path(job["out_json"])
            out_dir = out_json_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)

            # ---- 1) save original image ----
            orig = cv2.imread(img_path)
            if orig is not None:
                orig_path = out_dir / "image.png"
                cv2.imwrite(str(orig_path), orig)

            # ---- 2) generate and save binary mask ----
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

            mask_path = out_dir / f"{_WORKER_ENGINE_NAME}.png"
            cv2.imwrite(str(mask_path), binary)
        return {
            "status": "ok",
            "engine": _WORKER_ENGINE_NAME,
            "gpu_id": _WORKER_GPU_ID,
            "pid": os.getpid(),
            "image": img_path,
            "sample_id": job["sample_id"],
            "bboxes": merged_boxes,
            "_out_json": job["out_json"],
        }
    except Exception as e:
        return {
            "status": "fail",
            "engine": _WORKER_ENGINE_NAME,
            "gpu_id": _WORKER_GPU_ID,
            "pid": os.getpid(),
            "image": img_path,
            "sample_id": job["sample_id"],
            "message": str(e),
            "_out_json": job["out_json"],
        }


def write_output_json(record: Dict[str, Any]):
    out_json = Path(record["_out_json"])
    out_json.parent.mkdir(parents=True, exist_ok=True)

    record = dict(record)
    record.pop("_out_json", None)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    input_path = Path(args.input).resolve()

    gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()]
    jobs = list_images_and_prepare_jobs(
    input_path,
    args.pattern,
    args.engine,
    args.skip_if_exists,
    args.save_masks
    )
    gpu_buckets = split_round_robin(jobs, gpu_list)

    ctx = mp.get_context("spawn")

    executors = {
        gpu: ProcessPoolExecutor(
            max_workers=max(1, args.num_workers_per_gpu),
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(gpu, args.engine),
        )
        for gpu in gpu_list if gpu_buckets.get(gpu)
    }

    futures = []
    for gpu, ex in executors.items():
        for job in gpu_buckets[gpu]:
            futures.append(ex.submit(_worker_seg_one, job))
    start=time.time()
    with tqdm(total=len(futures), desc="SEG") as pbar:
        for fu in as_completed(futures):
            rec = fu.result()
            print(rec)
            write_output_json(rec)
            pbar.update(1)

    for ex in executors.values():
        ex.shutdown()
    end=time.time()
    print(f"[TIME/PER] {(end-start)/len(futures):.4f} seconds")

if __name__ == "__main__":
    main()
