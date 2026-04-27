#!/usr/bin/env python3
"""
Quick test script for MIMIC system.
This script demonstrates how to run evaluation on sample data.

Usage:
    python examples/quick_test.py
"""

import json
import os
from pathlib import Path

def check_data_folder():
    """Check if sample data exists."""
    data_dir = Path(__file__).parent.parent / "data" / "1"
    if not data_dir.exists():
        print(f"Warning: Sample data not found at {data_dir}")
        print("Please ensure the data folder exists with test samples.")
        return False

    required_files = ["merged.jpg", "gt_ocr.json", "gt_instances.json"]
    for f in required_files:
        if not (data_dir / f).exists():
            print(f"Warning: Missing {f} in {data_dir}")
            return False

    print("Sample data found: data/1/")
    return True

def show_sample_annotation():
    """Display sample ground truth annotation."""
    gt_file = Path(__file__).parent.parent / "data" / "1" / "gt_ocr.json"
    if gt_file.exists():
        with open(gt_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"\nSample GT annotation (first 3 items):")
        items = data.get("results", [])[:3] if isinstance(data, dict) else data[:3]
        for item in items:
            print(f"  {item}")

def run_evaluation():
    """Run the OCR evaluation script."""
    eval_script = Path(__file__).parent.parent / "eval_ocr_seg_match_prf_iou45.py"
    if eval_script.exists():
        print(f"\nRunning evaluation script: {eval_script}")
        print("Please run: python eval_ocr_seg_match_prf_iou45.py")
    else:
        print(f"Evaluation script not found at {eval_script}")

def main():
    print("=" * 50)
    print("MIMIC System - Quick Test")
    print("=" * 50)

    print("\n1. Checking data folder...")
    has_data = check_data_folder()

    if has_data:
        print("\n2. Sample annotations...")
        show_sample_annotation()

    print("\n3. To run full evaluation:")
    print("   python eval_ocr_seg_match_prf_iou45.py")

    print("\n4. To start API service:")
    print("   python newapi.py")

    print("\n" + "=" * 50)
    print("Quick test complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
