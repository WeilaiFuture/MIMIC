import os
import cv2
import uuid
import shutil
import asyncio
import threading
import itertools
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form

# =========================
# 你的项目 import
# =========================
from OCR.ocr import myOCR
from OCR.EasyOCR import MyEasyOCR
from OCR.doctrOCR import doctrOCR

from Segmentation.eam_sam.predict import FewShotSegmentation
from Segmentation.unet.predict import UnetSegmentation
from Text.predict import TextClassifier
from Vision.predict import ImageClassifier

# =========================
# 配置：GPU 分配策略
# =========================
GPU_OTHER_MODELS = [0]   # 0 卡跑 Mask / ImageCls /（可选）EasyOCR 等
GPU_PADDLE_BASE = 2      # 2 卡跑 baseline paddle
GPU_PADDLE_FT = 3        # 3 卡跑 finetuned paddle

UPLOAD_ROOT = Path("uploads")
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)


# =========================
# 通用工具：保存上传文件
# =========================
async def save_upload(file: UploadFile, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

def safe_rmtree(p: Path):
    try:
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass


# =========================
# Box helpers（你原来的逻辑）
# =========================
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
# ModelPool：只用 GPU 0（其他模型）
# =========================
class RoundRobin:
    def __init__(self, items: List[int]):
        self._cycle = itertools.cycle(items)
        self._lock = threading.Lock()
    def next(self) -> int:
        with self._lock:
            return next(self._cycle)

class ModelPool:
    def __init__(self, gpu_ids: List[int], per_gpu_concurrency: int = 1):
        self.gpu_ids = gpu_ids
        self.rr = RoundRobin(gpu_ids)
        self.sem: Dict[int, asyncio.Semaphore] = {
            gid: asyncio.Semaphore(per_gpu_concurrency) for gid in gpu_ids
        }
        self.eam_sam: Dict[int, FewShotSegmentation] = {}
        self.unet: Dict[int, UnetSegmentation] = {}
        self.img_cls: Dict[int, ImageClassifier] = {}

    def load_all(self):
        for gid in self.gpu_ids:
            device_str = f"cuda:{gid}"
            self.eam_sam[gid] = FewShotSegmentation(
                name="xl0",
                weight_url="/home/root123/mount1/weilai/project/Fossil_Extract/Segmentation/eam_sam/efficientvit/assets/checkpoints/sam/xl0.pt",
                train_path="/home/root123/mount1/weilai/project/Fossil_Extract/Segmentation/eam_sam/few_shot",
                device=torch.device(device_str),
            )
            self.unet[gid] = UnetSegmentation(
                weight_url="/home/root123/mount1/weilai/project/Fossil_Extract/Segmentation/unet/best_model.pth",
                device=torch.device(device_str),
            )
            self.img_cls[gid] = ImageClassifier(
                model_path="/home/root123/mount1/weilai/project/Fossil_Extract/Vision/model_epoch_1.pth",
                device=torch.device(device_str),
            )

    async def with_gpu(self, fn, model_type: str, *args, **kwargs):
        gid = self.rr.next()  # 这里只有 [0]，就是固定 0 卡
        sem = self.sem[gid]
        async with sem:
            if model_type == "eam_sam":
                model = self.eam_sam[gid]
            elif model_type == "unet":
                model = self.unet[gid]
            elif model_type == "image_classifier":
                model = self.img_cls[gid]
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            return await fn(model, gid, *args, **kwargs)


# =========================
# PaddleOCR Worker：进程级 GPU 隔离（核心）
# =========================
def _paddle_worker_loop(
    gpu_id: int,
    mode_name: str,
    in_q: "mp.Queue",
    out_q: "mp.Queue",
    det_model_dir: Optional[str],
    cls_model_dir: Optional[str],
    rec_model_dir: Optional[str],
):
    """
    子进程：绑定单一 GPU，加载 PaddleOCR（myOCR），循环处理任务
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # ⚠️ 必须在设置 CUDA_VISIBLE_DEVICES 后 import/初始化
    # 这里 myOCR 已在主进程 import，但 Paddle 内部真正初始化在构造里
    if det_model_dir and cls_model_dir and rec_model_dir:
        model = myOCR(det_model_dir=det_model_dir, cls_model_dir=cls_model_dir, rec_model_dir=rec_model_dir)
    else:
        model = myOCR()

    while True:
        item = in_q.get()
        if item is None:
            break

        job_id = item["job_id"]
        img_path = item["img_path"]

        try:
            _, result = model.predict(img_path)
            # result: [(text, {left,top,width,height}), ...]
            pred_ocr = []
            for txt, dic in (result or []):
                if not dic:
                    continue
                x1 = int(dic["left"])
                y1 = int(dic["top"])
                x2 = int(dic["left"] + dic["width"])
                y2 = int(dic["top"] + dic["height"])
                pred_ocr.append({"text": str(txt).upper(), "bbox_xyxy": [x1, y1, x2, y2]})

            out_q.put({
                "job_id": job_id,
                "status": "ok",
                "engine": mode_name,
                "gpu": gpu_id,
                "results": result,
                "pred_ocr": pred_ocr,
            })
        except Exception as e:
            out_q.put({
                "job_id": job_id,
                "status": "fail",
                "engine": mode_name,
                "gpu": gpu_id,
                "message": f"OCR 失败: {e}",
            })


class PaddleOCRProcessPool:
    """
    管理两个 PaddleOCR 子进程：
    - paddle -> GPU 2
    - fientuen -> GPU 3
    保证不同 GPU，且天然并行
    """
    def __init__(self):
        ctx = mp.get_context("spawn")  # 更安全（避免 CUDA fork 问题）
        self.ctx = ctx
        self.in_q: Dict[str, mp.Queue] = {
            "paddle": ctx.Queue(maxsize=64),
            "fientuen": ctx.Queue(maxsize=64),
        }
        self.out_q: mp.Queue = ctx.Queue(maxsize=256)

        self.procs: Dict[str, mp.Process] = {}

    def start(self):
        # baseline paddle (GPU 2)
        self.procs["paddle"] = self.ctx.Process(
            target=_paddle_worker_loop,
            args=(
                GPU_PADDLE_BASE,
                "paddle",
                self.in_q["paddle"],
                self.out_q,
                None, None, None,
            ),
            daemon=True
        )

        # finetuned paddle (GPU 3)
        self.procs["fientuen"] = self.ctx.Process(
            target=_paddle_worker_loop,
            args=(
                GPU_PADDLE_FT,
                "fientuen",
                self.in_q["fientuen"],
                self.out_q,
                '/home/root123/mount1/weilai/project/Fossil_Extract/inference/ch_PP-OCRv3_det_student_inference',
                '/home/root123/mount1/weilai/project/Fossil_Extract/inference/ch_ppocr_mobile_v2.0_cls_infer',
                '/home/root123/mount1/weilai/project/Fossil_Extract/inference/en_PP-OCRv3_rec_inference',
            ),
            daemon=True
        )

        for p in self.procs.values():
            p.start()

    def stop(self):
        for k, q in self.in_q.items():
            try:
                q.put_nowait(None)
            except Exception:
                pass
        for p in self.procs.values():
            try:
                p.join(timeout=2)
            except Exception:
                pass

    async def infer(self, ocr_type: str, img_path: str) -> Dict[str, Any]:
        """
        把任务丢给对应子进程，然后异步等待 out_q 返回 job_id 对应的结果
        """
        if ocr_type not in self.in_q:
            raise ValueError(f"Unsupported paddle ocr_type: {ocr_type}")

        job_id = uuid.uuid4().hex
        self.in_q[ocr_type].put({"job_id": job_id, "img_path": img_path})

        # 等待结果：用线程把阻塞 get 包起来，避免卡 event loop
        def _blocking_wait():
            while True:
                msg = self.out_q.get()
                if msg.get("job_id") == job_id:
                    return msg

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _blocking_wait)


# =========================
# FastAPI app + 初始化
# =========================
app = FastAPI(title="Fossil API (GPU split: other@0, paddle@2/3)")

# 其他模型（固定 GPU 0）
pool = ModelPool(GPU_OTHER_MODELS, per_gpu_concurrency=1)
pool.load_all()

# 文本分类（CPU）
text_classifier = TextClassifier("/home/root123/mount1/weilai/project/Fossil_Extract/Text/PBDB.txt")

# 其他 OCR（不强制；easyocr 可用 GPU0）
easyocr_model = MyEasyOCR(['en', 'ch_sim'], gpu=True)  # 会用默认 cuda:0（符合“0卡部署其他模型”）
doctr_model = doctrOCR()

# PaddleOCR 进程池（GPU2/GPU3）
paddle_pool = PaddleOCRProcessPool()
paddle_pool.start()


# =========================
# API: Mask
# =========================
@app.post("/process_mask")
async def process_mask(
    file: UploadFile = File(...),
    model_type: str = Form("eam_sam"),
    min_area: int = Form(1500),
    cleanup: int = Form(1),
):
    if model_type not in ["eam_sam", "unet"]:
        return {"status": "fail", "message": f"未知模型: {model_type}"}

    uid = uuid.uuid4().hex
    tmp_dir = UPLOAD_ROOT / uid
    tmp_path = tmp_dir / file.filename
    await save_upload(file, tmp_path)

    async def _infer(model, gid, image_path: Path):
        ok, result_image = model.predict(str(image_path))
        if not ok or result_image is None:
            return {"status": "fail", "message": "未检测到 mask", "gpu": gid}

        gray = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
        boxes = get_bounding_boxes(gray, min_area=min_area)
        merged = merge_overlapping_boxes_with_ratio(boxes, 0.4)

        pred_instances = [
            {"instance_id": int(i), "bbox_xyxy": [int(b[0]), int(b[1]), int(b[2]), int(b[3])]}
            for i, b in enumerate(merged)
        ]

        return {"status": "ok", "gpu": gid, "model": model_type, "boxes": merged, "pred_instances": pred_instances}

    out = await pool.with_gpu(_infer, model_type, tmp_path)

    if cleanup:
        safe_rmtree(tmp_dir)

    return out


# =========================
# API: OCR（关键：paddle / fientuen 用独立进程）
# =========================
@app.post("/ocr")
async def ocr(
    file: UploadFile = File(...),
    ocr_type: str = Form("paddle"),  # paddle / fientuen / easyocr / doctrOCR
    cleanup: int = Form(1),
):
    uid = uuid.uuid4().hex
    tmp_dir = UPLOAD_ROOT / uid
    tmp_path = tmp_dir / file.filename
    await save_upload(file, tmp_path)

    try:
        if ocr_type in ("paddle", "fientuen"):
            # 进程池推理：GPU2 / GPU3 真正隔离并行
            out = await paddle_pool.infer(ocr_type, str(tmp_path))

        elif ocr_type == "easyocr":
            _, result = easyocr_model.predict(str(tmp_path))
            pred_ocr = []
            for txt, dic in (result or []):
                if not dic:
                    continue
                x1 = int(dic["left"])
                y1 = int(dic["top"])
                x2 = int(dic["left"] + dic["width"])
                y2 = int(dic["top"] + dic["height"])
                pred_ocr.append({"text": str(txt).upper(), "bbox_xyxy": [x1, y1, x2, y2]})
            out = {"status": "ok", "engine": "easyocr", "results": result, "pred_ocr": pred_ocr}

        elif ocr_type == "doctrOCR":
            _, result = doctr_model.predict(str(tmp_path))
            pred_ocr = []
            for txt, dic in (result or []):
                if not dic:
                    continue
                x1 = int(dic["left"])
                y1 = int(dic["top"])
                x2 = int(dic["left"] + dic["width"])
                y2 = int(dic["top"] + dic["height"])
                pred_ocr.append({"text": str(txt).upper(), "bbox_xyxy": [x1, y1, x2, y2]})
            out = {"status": "ok", "engine": "doctrOCR", "results": result, "pred_ocr": pred_ocr}

        else:
            out = {"status": "fail", "message": f"未知 OCR 类型: {ocr_type}"}

    except Exception as e:
        out = {"status": "fail", "message": f"OCR 失败: {e}"}

    if cleanup:
        safe_rmtree(tmp_dir)

    return out


# =========================
# API: Text / Image
# =========================
@app.post("/classify_text")
async def classify_text(text: str = Form(...)):
    result = text_classifier.predict(text)
    return {"status": "ok", "input": text, "prediction": result}

@app.post("/classify_image")
async def classify_image(file: UploadFile = File(...), cleanup: int = Form(1)):
    uid = uuid.uuid4().hex
    tmp_dir = UPLOAD_ROOT / uid
    tmp_path = tmp_dir / file.filename
    await save_upload(file, tmp_path)

    async def _infer(model, gid, image_path: Path):
        result = model.predict(str(image_path))
        return {"status": "ok", "gpu": gid, "filename": image_path.name, "prediction": result}

    out = await pool.with_gpu(_infer, "image_classifier", tmp_path)

    if cleanup:
        safe_rmtree(tmp_dir)

    return out


# =========================
# LLM Router
# =========================
from llm_router import router as llm_router
app.include_router(llm_router)


# =========================
# Health
# =========================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "gpu_map": {
            "other_models": GPU_OTHER_MODELS,
            "paddle": GPU_PADDLE_BASE,
            "fientuen": GPU_PADDLE_FT,
        }
    }


# =========================
# Entry
# =========================
if __name__ == "__main__":
    # 强烈建议：不要用 uvicorn --workers 多进程
    # 否则每个 worker 都会再起两条 PaddleOCR 子进程，显存会爆
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
