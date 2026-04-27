# MIMIC - Multimodal Intelligent Information Extraction System

An AI-powered system for extracting information from paleontological literature using multimodal deep learning and large language models.

## Overview

MIMIC is designed to automatically identify and extract information from paleontological academic literature, including fossil labels, species names, descriptions, and other relevant data.

## Features

### 1. OCR Text Recognition (`OCR/`)

Supports multiple OCR engines:

- **PaddleOCR** - Baidu Paddle framework (baseline and finetuned versions)
- **EasyOCR** - PyTorch-based OCR toolkit
- **doctrOCR** - Document understanding OCR engine

### 2. Image Segmentation (`Segmentation/`)

- **EAM-SAM** - Few-shot segmentation based on EfficientViT-SAM
- **Unet** - Traditional U-Net segmentation network

### 3. Text Classification (`Text/`)

- Text type classifier for identifying and categorizing extracted content

### 4. Image Classification (`Vision/`)

- Image type classifier for identifying literature figure types

### 5. LLM Integration (`llm_batch_text_infer.py`)

Supports multiple large language model backends:

- **Baichuan**
- **DeepSeek** (DS)
- **Llama3**
- **Mistral**
- **Qwen**

Supports both Base and LoRA fine-tuned model modes.

### 6. API Service (`newapi.py`)

FastAPI-based RESTful API providing:

- Image upload processing
- Multi-model collaborative inference
- Multi-GPU resource scheduling

## Directory Structure

```
MIMIC/
├── OCR/                    # OCR code and models
│   └── inference/         # PaddleOCR inference models
├── Segmentation/          # Image segmentation module
│   └── eam_sam/          # EfficientViT-SAM implementation
├── Text/                  # Text classification module
├── Vision/                # Image classification module
├── utils/                 # Utility functions
├── prompts/               # LLM prompt templates
├── data/                  # Dataset (230+ samples)
├── output/                # Model output results
├── LLM_output_*/          # LLM inference results
├── eval_*.py              # Evaluation scripts
├── newapi.py              # API service
└── PBDB.txt              # Paleobiology database labels
```

## Data Format

Each data sample contains:

- `merged.jpg` - Processed literature image
- `gt_instances.json` - Instance annotations
- `gt_label2name.json` - Label to name mapping
- `gt_ocr.json` - OCR annotations
- `gt_triples.json` - Triple annotations
- `text.txt` - Text content
- `meta.json` - Metadata

## Evaluation Metrics

The system supports multiple evaluation modes:

- **IoU Thresholds**: 0.45, 0.5, etc.
- **Metrics**: Precision, Recall, F1-score
- **Text Accuracy**: Accuracy on matched text
- **Multi-model Voting**: 3-vote voting mechanism for improved accuracy

### Evaluation Results

Performance comparison of different OCR engines (IoU=0.5):


| OCR Engine | Precision | Recall | F1-score | Text Accuracy |
| ---------- | --------- | ------ | -------- | ------------- |
| PaddleOCR  | 0.808     | 0.245  | 0.376    | 93.3%         |
| EasyOCR    | 0.798     | 0.340  | 0.477    | 81.1%         |
| doctrOCR   | 0.582     | 0.524  | 0.552    | 92.4%         |
| Fientuen   | 0.997     | 0.410  | 0.581    | 98.0%         |

## Usage

### 1. Start API Service

```bash
python newapi.py
```

Starts the FastAPI server with configured ports.

### 2. LLM Batch Inference

```bash
python llm_batch_text_infer.py \
    --prompt_file prompts/example_prompts.json \
    --engine llama3 \
    --mode base
```

### 3. Evaluation Scripts

```bash
# OCR segmentation matching evaluation
python eval_ocr_seg_match_prf_iou45.py

# Multi-engine multi-mode evaluation
python eval_named_all_engines_all_modes_iou45.py
```

## GPU Configuration

Default GPU allocation strategy:

- **GPU 0**: Mask/Image Classification/EasyOCR
- **GPU 2**: PaddleOCR baseline
- **GPU 3**: PaddleOCR finetuned

Modify in `newapi.py`:

```python
GPU_OTHER_MODELS = [0]
GPU_PADDLE_BASE = 2
GPU_PADDLE_FT = 3
```

## Dependencies

- PyTorch
- FastAPI + uvicorn
- PaddleOCR
- EasyOCR
- doctr
- OpenCV
- vLLM services for LLM backends

## References

- EfficientViT: [arXiv:2205.14756](https://arxiv.org/abs/2205.14756)
- EfficientViT-SAM: [arXiv:2402.05008](https://arxiv.org/abs/2402.05008)
- [EfficientViT GitHub](https://github.com/mit-han-lab/efficientvit)
- [Paleobiology Database (PBDB)](https://pbdb.org)

## Notes

1. Multi-GPU environment required
2. vLLM services need to be configured for LLM inference
3. Model weights need to be downloaded separately
4. Python 3.8+ recommended

## License

Copyright (c) 2026 Haijun Song

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This project includes third-party code (e.g., EfficientViT). Please comply with their respective licenses.
