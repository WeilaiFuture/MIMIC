import os
import re
import json
import numpy as np

def save_json_file(results,output_json_path):
    """
    保存为 JSON 文件
    """
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, indent=4, default=lambda obj: obj.tolist() if isinstance(obj, np.ndarray) else obj)

def read_json_file(json_file_path):
    """
    读取 JSON 文件并返回内容
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except Exception as e:
        print(f"读取 JSON 文件时出错: {e}")
        return None

def get_md_files(directory):
    """
    读取 directory 路径下的所有md文件
    """
    md_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".md"):
                md_files.append(os.path.join(root, file))
    return md_files
