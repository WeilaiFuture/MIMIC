#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List

import requests
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore


# =========================
# 1) vLLM 后端配置（与你现有一致）
# =========================
BACKENDS = {
    "baichuan": os.getenv("VLLM_URL_BAICHUAN", "http://10.126.126.209:9001/v1/completions"),
    "ds":       os.getenv("VLLM_URL_DS",       "http://10.126.126.209:9002/v1/completions"),
    "llama3":   os.getenv("VLLM_URL_LLAMA3",   "http://10.126.126.209:9003/v1/completions"),
    "mistral":  os.getenv("VLLM_URL_MISTRAL",  "http://10.126.126.209:9004/v1/completions"),
    "qwen":     os.getenv("VLLM_URL_QWEN",     "http://10.126.126.209:9005/v1/completions"),
}

samples_backends = {
    "baichuan": {"sample_count": 3, "max_tokens": 4096},
    "ds":       {"sample_count": 3, "max_tokens": 4096},
    "llama3":   {"sample_count": 4, "max_tokens": 51200},
    "mistral":  {"sample_count": 4, "max_tokens": 32768},
    "qwen":     {"sample_count": 4, "max_tokens": 32768},
}

BASE_MODEL = os.getenv("VLLM_BASE_MODEL", "llm")
LORA_MODEL = os.getenv("VLLM_LORA_MODEL", "seq_extract")
PROMPT_FILE = os.getenv("PROMPT_FILE", "prompts/example_prompts.json")


# =========================
# 2) 工具函数：engine / model 解析
# =========================
def resolve_backend(engine: str) -> str:
    engine = (engine or "llama3").lower()
    if engine not in BACKENDS:
        raise ValueError(f"unsupported engine: {engine}. valid={list(BACKENDS.keys())}")
    return BACKENDS[engine]


def resolve_model(mode: str) -> str:
    mode = (mode or "base").lower()
    if mode == "base":
        return BASE_MODEL
    if mode in ("lora", "base+lora"):
        return LORA_MODEL
    raise ValueError(f"unsupported mode: {mode}. valid=base|lora|base+lora")


# =========================
# 3) few-shot prompt 加载与构造
# =========================
def load_prompts(prompt_file: str = PROMPT_FILE) -> Dict[str, str]:
    with open(prompt_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("samples", {})
    if not isinstance(samples, dict) or not samples:
        raise RuntimeError("`samples` in prompt json must be a non-empty dict")
    return samples


def construct_prompt(sentence: str, engine: str, max_example_chars: int = 8000) -> str:
    samples_dict = load_prompts(PROMPT_FILE)
    samples_texts = list(samples_dict.values())

    selected_examples = samples_texts[:samples_backends[engine]["sample_count"]]
    joined_examples = "\n\n".join(selected_examples)

    instruction = (
        "You are an assistant specialized in analyzing fossil plate captions. Based on the examples below, "
        "extract all LABEL–FOSSIL NAME mappings from the input text.\n"
        "You must strictly follow these rules:\n"
        "1. Output only in the form LABEL:NAME, one mapping per line, with no spaces around the colon.\n"
        "2. LABEL must come from the original text (e.g., A, B, AA, AB, 1, 2, 3, A1, A2).\n"
        "   IMPORTANT:\n"
        "   - If the text contains a label RANGE (e.g., 2–5, 3-7, A1–A4), you MUST expand it into individual labels.\n"
        "   - Range labels MUST NOT appear in the output.\n"
        "   - For example, \"2–5\" MUST be output as:\n"
        "       2:NAME\n"
        "       3:NAME\n"
        "       4:NAME\n"
        "       5:NAME\n"
        "3. NAME must be a valid paleontological genus–species name (e.g., \"Masrasector nananubis sp. nov.\"). "
        "Do NOT output specimen numbers, locality codes, or catalog numbers (e.g., CUG123, DPC 9274).\n"
        "4. If multiple labels refer to different views of the same fossil specimen (e.g., A, B, C), then all such "
        "LABELs must map to the same NAME.\n"
        "5. If the caption describes only one fossil and contains no explicit labels (A/B/1/2/etc.), use \"all\" as the LABEL "
        "(e.g., all:Thylacinus cynocephalus).\n"
        "6. Follow the example format strictly. Do NOT output any explanations, reasoning, prefixes, or suffixes.\n"
        "7. Output only the LABEL:NAME lines. Do NOT output statements like \"Here is the output\" or \"The result is\".\n"
        "8. If the caption includes branching labels (e.g., 1, 1a, 1b), output each explicitly.\n\n"
        "Below are some examples:\n"
    )

    prompt = (
        instruction
        + joined_examples
        + "\n\n\n\n"
        +"现在开始执行我需要抽取的任务，不要提取sample中的样本，只要我接下来的句子，另外请回顾最开始的任务要求，尤其是2，一定要做好拆分\n"
        "<s>[INST]extract all LABEL–FOSSIL NAME mappings from the input text.[/INST]"
        f"[input]{sentence}[/input] [output]"
    )
    return prompt


# =========================
# 4) vLLM 调用 & 输出解析
# =========================
def generate_text(
    vllm_url: str,
    prompt: str,
    model: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.1,
    top_p: float = 0.9,
    timeout: int = 120,
) -> str:
    resp = requests.post(
        vllm_url,
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": ["[/output]", "</s>"],
        },
        timeout=timeout,
    )

    if resp.status_code != 200:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise RuntimeError(f"vLLM call failed: {resp.status_code} {detail}")

    data = resp.json()
    text = (data["choices"][0].get("text") or "").strip()

    start = text.rfind("[output]")
    start = start + len("[output]") if start != -1 else 0
    end = text.find("[/output]", start)
    end = len(text) if end == -1 else end
    return text[start:end].strip()


def parse_generated_text(generated_text: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}

    generated_text = generated_text.split("[/output]")[0]
    generated_text = generated_text.split("[output]")[-1]

    for line in generated_text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip()
        if len(k) == 0 or len(k) > 8:
            continue
        parsed[k] = v
    return parsed


def infer_one(
    text: str,
    engine: str,
    mode: str,
) -> Dict[str, Any]:
    vllm_url = resolve_backend(engine)
    model = resolve_model(mode)

    prompt = construct_prompt(text, engine=engine)

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "echo": True,
        "stop": ["</s>", "[/output]", "\n\n\n"],
    }

    r = requests.post(vllm_url, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")

    data = r.json()

    usage = data.get("usage") or {}
    pt = usage.get("prompt_tokens")
    t0 = time.time()

    raw_out = generate_text(
        vllm_url=vllm_url,
        prompt=prompt,
        model=model,
        max_new_tokens=samples_backends[engine]["max_tokens"] - pt,
    )
    dt = time.time() - t0

    return {
        "status": "ok",
        "engine": engine,
        "mode": mode,
        "model": model,
        "latency_sec": round(dt, 4),
        "results": parse_generated_text(raw_out),
        "raw_output": raw_out,
        "input_prompt": prompt,
    }


# =========================
# 5) MIMIC 遍历：data/<id>/text.txt
# =========================
def list_text_jobs(
    input_path: Path,
    out_root: Path,
    skip_if_exists: bool,
    engines: List[str],
    modes: List[str],
) -> List[Dict[str, Any]]:
    input_path = input_path.resolve()
    out_root = out_root.resolve()

    if not input_path.is_dir():
        raise ValueError(f"MIMIC data path must be a directory: {input_path}")

    jobs: List[Dict[str, Any]] = []

    for sample_dir in sorted(input_path.iterdir()):
        if not sample_dir.is_dir():
            continue

        sample_id = sample_dir.name
        txt_file = sample_dir / "text.txt"
        if not txt_file.exists():
            continue

        out_dir = out_root / sample_id

        if skip_if_exists:
            all_exist = True
            for eng in engines:
                for mode in modes:
                    out_json = out_dir / f"{eng}_{mode}.json"
                    if not out_json.exists():
                        all_exist = False
                        break
                if not all_exist:
                    break
            if all_exist:
                continue

        jobs.append({
            "text_file": str(txt_file),
            "sample_id": sample_id,
            "out_dir": str(out_dir),
        })

    return jobs


def read_text_file(path: str) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8", errors="ignore").strip()


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =========================
# ✅ 提升点：重试 + engine限流 + 全局任务池
# =========================
def _infer_with_retry(text: str, engine: str, mode: str, retries: int, backoff_sec: float) -> Dict[str, Any]:
    last_err = None
    for i in range(retries + 1):
        try:
            return infer_one(text=text, engine=engine, mode=mode)
        except Exception as e:
            last_err = e
            if i < retries:
                time.sleep(backoff_sec * (i + 1))
    raise RuntimeError(str(last_err))


def _run_one_task_limited(
    *,
    sem: Semaphore,
    text: str,
    text_file: str,
    sample_id: str,
    engine: str,
    mode: str,
    out_json: Path,
    retries: int,
    backoff_sec: float,
) -> Dict[str, Any]:
    # engine 级并发限流（避免打爆单个端口）
    with sem:
        try:
            res = _infer_with_retry(text=text, engine=engine, mode=mode, retries=retries, backoff_sec=backoff_sec)

            payload = {
                "text_file": text_file,
                "sample_id": sample_id,
                "engine": engine,
                "mode": mode,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "llm": res,
            }
            write_json(out_json, payload)

            return {
                "status": "ok",
                "text_file": text_file,
                "sample_id": sample_id,
                "engine": engine,
                "mode": mode,
                "out_json": str(out_json),
                "latency_sec": res.get("latency_sec"),
                "n_results": len(res.get("results", {})),
            }

        except Exception as e:
            payload = {
                "text_file": text_file,
                "sample_id": sample_id,
                "engine": engine,
                "mode": mode,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(e),
            }
            write_json(out_json, payload)

            return {
                "status": "fail",
                "text_file": text_file,
                "sample_id": sample_id,
                "engine": engine,
                "mode": mode,
                "out_json": str(out_json),
                "error": str(e),
            }


def parse_args():
    p = argparse.ArgumentParser(description="Batch LLM infer on MIMIC text files data/<id>/text.txt")
    p.add_argument("--input", type=str, default="/home/root123/mount1/weilai/MIMIC/data")
    p.add_argument("--out_root", type=str, default="/home/root123/mount1/weilai/MIMIC/output")
    p.add_argument("--skip_if_exists", action="store_true", default=False)

    p.add_argument("--engines", type=str, default="baichuan,ds,llama3,mistral,qwen")
    p.add_argument("--modes", type=str, default="base,lora")

    # ✅ 全局并发
    p.add_argument("--workers", type=int, default=int(os.getenv("WORKERS", "1")),
                   help="global thread workers")

    # ✅ engine 并发限流：每个engine最多同时跑多少个请求
    p.add_argument("--per_engine", type=int, default=int(os.getenv("PER_ENGINE", "1")),
                   help="max concurrent tasks per engine (rate limit)")

    # ✅ 失败重试
    p.add_argument("--retries", type=int, default=int(os.getenv("RETRIES", "1")),
                   help="retry times per task (default=1)")
    p.add_argument("--backoff", type=float, default=float(os.getenv("BACKOFF", "1.0")),
                   help="retry backoff seconds base (default=1.0)")

    return p.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    out_root = Path(args.out_root)

    engines = [x.strip().lower() for x in args.engines.split(",") if x.strip()]
    modes = [x.strip().lower() for x in args.modes.split(",") if x.strip()]

    for e in engines:
        if e not in BACKENDS:
            raise SystemExit(f"Unknown engine: {e}, valid={list(BACKENDS.keys())}")
    for m in modes:
        if m not in ("base", "lora", "base+lora"):
            raise SystemExit("modes must be base,lora (or base+lora)")

    jobs = list_text_jobs(
        input_path=input_path,
        out_root=out_root,
        skip_if_exists=args.skip_if_exists,
        engines=engines,
        modes=modes,
    )
    print(f"Found {len(jobs)} samples from: {input_path}")

    # ✅ 一次性读入所有文本（避免任务线程重复读盘）
    sample_text: Dict[str, str] = {}
    valid_samples: List[Dict[str, Any]] = []
    for job in tqdm(jobs, desc="READ"):
        text_file = job["text_file"]
        sample_id = job["sample_id"]
        try:
            txt = read_text_file(text_file)
        except Exception:
            txt = ""
        if not txt:
            continue
        sample_text[sample_id] = txt
        valid_samples.append(job)

    # ✅ engine semaphore：每个 engine 一个限流器
    semaphores: Dict[str, Semaphore] = {e: Semaphore(max(1, int(args.per_engine))) for e in engines}

    # ✅ 任务列表：sample×engine×mode
    tasks = []
    for job in valid_samples:
        text_file = job["text_file"]
        sample_id = job["sample_id"]
        out_dir = Path(job["out_dir"])

        for engine in engines:
            for mode in modes:
                out_json = out_dir / f"{engine}_{mode}.json"
                if args.skip_if_exists and out_json.exists():
                    continue
                tasks.append((sample_id, text_file, engine, mode, out_json))

    print(f"Total tasks = {len(tasks)} (sample×engine×mode), workers={args.workers}, per_engine={args.per_engine}")

    start=time.time()
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futures = []
        for sample_id, text_file, engine, mode, out_json in tasks:
            futures.append(
                ex.submit(
                    _run_one_task_limited,
                    sem=semaphores[engine],
                    text=sample_text[sample_id],
                    text_file=text_file,
                    sample_id=sample_id,
                    engine=engine,
                    mode=mode,
                    out_json=out_json,
                    retries=int(args.retries),
                    backoff_sec=float(args.backoff),
                )
            )

        for fu in tqdm(as_completed(futures), total=len(futures), desc="RUN"):
            rec = fu.result()
    end=time.time()
    print(f"[TIME/PER] {(end-start)/len(futures):.4f} seconds")
    print("[DONE] All tasks finished.")

if __name__ == "__main__":
    main()
