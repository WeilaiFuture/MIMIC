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
| Seg | LLM | Mode | Static | Dynamic | KM | Vote@Static | Vote@Dynamic | Vote@KM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM | REG | Rule | 0.2146 | 0.1763 | 0.2198 | 0.2235 | 0.2179 | 0.2206 |
| SAM | NULL | -- | 0.6184 | 0.5264 | 0.6222 | 0.6355 | 0.6218 | 0.6266 |
| SAM | Baichuan-2 | base | 0.2651 | 0.2149 | 0.2656 | 0.2718 | 0.268 | 0.2695 |
| SAM | Baichuan-2 | lora | 0.2927 | 0.2538 | 0.2982 | 0.3016 | 0.2959 | 0.2968 |
| SAM | Deepseek | base | 0.3134 | 0.2593 | 0.3206 | 0.3219 | 0.3164 | 0.3181 |
| SAM | Deepseek | lora | 0.5688 | 0.4802 | 0.5772 | 0.5832 | 0.5726 | 0.5768 |
| SAM | LLaMA3 | base | 0.6353 | 0.5425 | 0.6399 | 0.6524 | 0.6395 | 0.6437 |
| SAM | LLaMA3 | lora | 0.6439 | 0.5482 | 0.6473 | 0.6615 | 0.6477 | 0.6522 |
| SAM | mistral | base | 0.6317 | 0.5357 | 0.634 | 0.6482 | 0.6347 | 0.6391 |
| SAM | mistral | lora | 0.0199 | 0.0237 | 0.0207 | 0.0207 | 0.02 | 0.0207 |
| SAM | Qwen-3 | base | 0.6205 | 0.5264 | 0.6245 | 0.6363 | 0.6237 | 0.6281 |
| SAM | Qwen-3 | lora | 0.6338 | 0.5405 | 0.6383 | 0.6509 | 0.6378 | 0.6423 |
| UNet | REG | -- | 0.2468 | 0.2037 | 0.2589 | 0.2605 | 0.2516 | 0.2554 |
| UNet | NULL | -- | 0.6418 | 0.5525 | 0.6674 | 0.6655 | 0.6536 | 0.6595 |
| UNet | Baichuan-2 | base | 0.2963 | 0.2416 | 0.3021 | 0.3045 | 0.2997 | 0.3009 |
| UNet | Baichuan-2 | lora | 0.3401 | 0.2891 | 0.3502 | 0.3504 | 0.3437 | 0.3453 |
| UNet | Deepseek | base | 0.349 | 0.2907 | 0.359 | 0.3589 | 0.3524 | 0.3543 |
| UNet | Deepseek | lora | 0.6002 | 0.5085 | 0.6118 | 0.6181 | 0.6073 | 0.6116 |
| UNet | LLaMA3 | base | 0.6569 | 0.564 | 0.6816 | 0.6815 | 0.67 | 0.6744 |
| UNet | LLaMA3 | lora | 0.667 | 0.5744 | 0.6927 | 0.692 | 0.6802 | 0.6851 |
| UNet | mistral | base | 0.654 | 0.5615 | 0.6785 | 0.6779 | 0.6662 | 0.6712 |
| UNet | mistral | lora | 0.0207 | 0.0245 | 0.0213 | 0.0218 | 0.021 | 0.0213 |
| UNet | Qwen-3 | base | 0.6414 | 0.547 | 0.669 | 0.6651 | 0.6538 | 0.6588 |
| UNet | Qwen-3 | lora | 0.6557 | 0.5633 | 0.6805 | 0.6802 | 0.6687 | 0.6737 |

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
