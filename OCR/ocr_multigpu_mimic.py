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
from tqdm import tqdm
import cv2


_WORKER_ENGINE = None
_WORKER_ENGINE_NAME = None
_WORKER_GPU_ID = None
_WORKER_BASE_DIR = None


def parse_args():
    p = argparse.ArgumentParser(description="Multi-GPU local OCR for MIMIC-style folders -> MIMIC/output/<id>/<engine>.json")
    p.add_argument("--input", type=str, required=True,
                   help="MIMIC root (contains data/) OR MIMIC/data OR a single jpg path")
    p.add_argument("--engine", type=str, default="paddle",
                   choices=["paddle", "ft", "easyocr", "doctrOCR"],
                   help="OCR engine name")
    p.add_argument("--gpus", type=str, default="0,1,2,3",
                   help="GPU ids, e.g. '0,1,2,3' or '0,2'")
    p.add_argument("--num_workers_per_gpu", type=int, default=1,
                   help="Processes per GPU (usually 1 for heavy OCR models)")
    p.add_argument("--pattern", type=str, default="*.jpg",
                   help="Glob pattern under data/ (default: *.jpg)")
    p.add_argument("--skip_if_exists", action="store_true",
                   help="Skip OCR if output json already exists")
    return p.parse_args()


def locate_mimic_root_and_sample_id(img_path: Path) -> Tuple[Path, str]:
    """
    img: .../MIMIC/data/1/merged.jpg
    return: (MIMIC_root, "1")
    """
    img_path = img_path.resolve()
    data_dir = None
    for parent in img_path.parents:
        if parent.name == "data":
            data_dir = parent
            break
    if data_dir is None:
        raise RuntimeError(f"Cannot locate MIMIC root (no 'data' parent) for: {img_path}")
    mimic_root = data_dir.parent
    sample_id = img_path.parent.name
    return mimic_root, sample_id


def list_images_and_prepare_jobs(input_path: Path, pattern: str, engine: str, skip_if_exists: bool) -> List[Dict[str, Any]]:
    """
    Build job list:
      job = {
        "image": "/.../MIMIC/data/1/merged.jpg",
        "mimic_root": "/.../MIMIC",
        "sample_id": "1",
        "out_json": "/.../MIMIC/output/1/<engine>.json"
      }
    """
    input_path = input_path.resolve()
    images: List[Path] = []

    if input_path.is_file():
        images = [input_path]
    else:
        data_dir = input_path
        images = sorted(data_dir.rglob(pattern))

    jobs: List[Dict[str, Any]] = []
    for img in images:
        # Only jpg-like for safety if user sets wide pattern
        if img.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
            continue

        mimic_root, sample_id = locate_mimic_root_and_sample_id(img)
        out_json = mimic_root / "output" / sample_id / f"{engine}.json"

        if skip_if_exists and out_json.exists():
            continue

        jobs.append({
            "image": str(img.resolve()),
            "mimic_root": str(mimic_root),
            "sample_id": sample_id,
            "out_json": str(out_json),
        })

    return jobs


def split_round_robin(jobs: List[Dict[str, Any]], gpu_list: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Round-robin assignment of jobs to GPUs.
    """
    buckets: Dict[str, List[Dict[str, Any]]] = {g: [] for g in gpu_list}
    for i, job in enumerate(jobs):
        g = gpu_list[i % len(gpu_list)]
        buckets[g].append(job)
    return buckets


# =========================
# Worker init + inference
# =========================
def _worker_init(gpu_id: str, engine_name: str):
    """
    Each process is pinned to a single GPU via CUDA_VISIBLE_DEVICES
    and loads OCR engine ONCE.
    """
    global _WORKER_ENGINE, _WORKER_ENGINE_NAME, _WORKER_GPU_ID, _WORKER_BASE_DIR

    _WORKER_GPU_ID = str(gpu_id)
    _WORKER_ENGINE_NAME = engine_name

    # Must be set before importing any DL libs
    os.environ["CUDA_VISIBLE_DEVICES"] = _WORKER_GPU_ID

    # Ensure imports work like your original code
    _WORKER_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _WORKER_BASE_DIR not in sys.path:
        sys.path.append(_WORKER_BASE_DIR)

    # Delayed import
    from OCR.ocr import myOCR
    from OCR.EasyOCR import MyEasyOCR
    from OCR.doctrOCR import doctrOCR

    def load_engine(name: str):
        if name == "paddle":
            return myOCR()
        if name == "ft":
            return myOCR(
                det_model_dir="/home/root123/mount1/weilai/MIMIC/OCR/inference/ch_PP-OCRv3_det_student_inference",
                cls_model_dir="/home/root123/mount1/weilai/MIMIC/OCR/inference/ch_ppocr_mobile_v2.0_cls_infer",
                rec_model_dir="/home/root123/mount1/weilai/MIMIC/OCR/inference/en_PP-OCRv3_rec_inference",
            )
        if name == "easyocr":
            return MyEasyOCR(["en", "ch_sim"], gpu=True)
        if name == "doctrOCR":
            return doctrOCR()
        raise ValueError(f"Unknown engine: {name}")

    _WORKER_ENGINE = load_engine(engine_name)
    print(f"[INIT] pid={os.getpid()} gpu={_WORKER_GPU_ID} engine={_WORKER_ENGINE_NAME}", flush=True)


def _worker_ocr_one(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    job includes:
      image, out_json, mimic_root, sample_id
    Return record to main process, main process writes file (avoid concurrent writes).
    """
    global _WORKER_ENGINE, _WORKER_ENGINE_NAME, _WORKER_GPU_ID

    img_path = job["image"]
    try:
        _, result = _WORKER_ENGINE.predict(img_path)
        return {
            "status": "ok",
            "engine": _WORKER_ENGINE_NAME,
            "gpu_id": _WORKER_GPU_ID,
            "pid": os.getpid(),
            "image": img_path,
            "mimic_root": job["mimic_root"],
            "sample_id": job["sample_id"],
            "results": result,
            "_out_json": job["out_json"],
        }
    except Exception as e:
        return {
            "status": "fail",
            "engine": _WORKER_ENGINE_NAME,
            "gpu_id": _WORKER_GPU_ID,
            "pid": os.getpid(),
            "image": img_path,
            "mimic_root": job["mimic_root"],
            "sample_id": job["sample_id"],
            "message": str(e),
            "_out_json": job["out_json"],
        }


def write_output_json(record: Dict[str, Any]) -> str:
    """
    Write:
      MIMIC/output/<id>/<engine>.json
    """
    out_json = Path(record["_out_json"]).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # Remove internal key before saving (optional)
    record_to_save = dict(record)
    record_to_save.pop("_out_json", None)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(record_to_save, f, ensure_ascii=False, indent=2)

    return str(out_json)


def main():
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    gpu_list = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if not gpu_list:
        raise RuntimeError("Empty --gpus")

    jobs = list_images_and_prepare_jobs(
        input_path=input_path,
        pattern=args.pattern,
        engine=args.engine,
        skip_if_exists=args.skip_if_exists,
    )
    if not jobs:
        print("[DONE] No jobs to run.")
        return

    gpu_buckets = split_round_robin(jobs, gpu_list)

    ctx = mp.get_context("spawn")
    workers_per_gpu = max(1, int(args.num_workers_per_gpu))

    # 1) 创建所有 GPU 的 executors（GPU 间并行）
    executors = {}
    for gpu_id in gpu_list:
        bucket = gpu_buckets.get(gpu_id, [])
        if not bucket:
            continue
        executors[gpu_id] = ProcessPoolExecutor(
            max_workers=workers_per_gpu,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(gpu_id, args.engine),
        )

    # 2) 提交所有任务
    all_futures = []
    for gpu_id, ex in executors.items():
        for job in gpu_buckets[gpu_id]:
            all_futures.append(ex.submit(_worker_ocr_one, job))

    total = len(all_futures)
    ok = fail = 0
    start=time.time()
    # 3) 进度条（唯一输出）
    try:
        with tqdm(total=total, desc="OCR", ncols=80) as pbar:
            for fu in as_completed(all_futures):
                rec = fu.result()
                write_output_json(rec)

                if rec.get("status") == "ok":
                    ok += 1
                else:
                    fail += 1

                pbar.update(1)
    finally:
        for ex in executors.values():
            ex.shutdown(wait=True, cancel_futures=False)
    end=time.time()
    print(f"[TIME/PER] {(end-start)/total:.4f} seconds")
    print(f"[DONE] total={total}, ok={ok}, fail={fail}")

if __name__ == "__main__":
    main()
