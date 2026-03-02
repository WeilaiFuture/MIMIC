
from scipy.optimize import linear_sum_assignment
import os
import re

def remove_md(text):
    """
    清理 md 文件内容
    """
    text = re.sub(r'\$([^$]*)\$', r'\1', text)  # 移除数学公式中的 `$` 符号
    text = re.sub(r'\\mathrm\{(.*?)\}', r'\1', text)  # 处理 `\mathrm`
    text = re.sub(r'\\operatorname\{(.*?)\}', r'\1', text)  # 处理 `\operatorname`
    text = re.sub(r'\\[!,]', '', text)  # 移除 `\!` 和 `\,`
    text = re.sub(r'\{\\big\((.*?)\\big\)\}', r'(\1)', text)  # 修复 `\big(...)`
    text = re.sub(r'\\mathbf\{(.*?)\}', r'\1', text)  # 处理 `\mathbf`
    text = re.sub(r'\{\\ n=(.*?)\}', r'n=\1', text)  # 处理 `{\ n=16}`
    text = re.sub(r'\\phantom-?', '', text)  # 移除 `\phantom`
    text = re.sub(r'[{}]', '', text)  # 移除所有 `{}`
    return text


def extract_images_and_captions_proximity(markdown_file,folder,fig_len=3):
    """
    从Markdown文件中提取图片和标题的函数。

    参数:
    markdown_file (str): Markdown文件的路径。
    ocr_folder (str): OCR结果输出的文件夹路径。

    返回:
    image_caption_list (list): 包含图片路径和标题的字典列表。
    """
    # 用于存储图片和注释的列表
    image_caption_list = []
    # 读取 Markdown 文件内容
    with open(markdown_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 正则表达式匹配 Markdown 图片格式 ![alt](image_path)
    image_pattern = re.compile(r'!\[.*?\]\((.*?)\)')

    # 遍历文件行
    for i, line in enumerate(lines):
        line = line.strip()
        # 如果是图片行
        if image_pattern.search(line):
            img_path = image_pattern.search(line).group(1)
            # 拼接图片路径
            full_img_path = os.path.abspath(os.path.join(folder, img_path))

            # 获取上下三行的注释
            lines_above = lines[max(0, i - fig_len):i]  # 图片行上方的三行
            lines_below = lines[i + 1:min(len(lines), i + 1 +fig_len)]  # 图片行下方的三行

            # 从下方三行优先寻找包含关键词的注释
            caption = None
            for line in lines_below:  # 优先检查下方三行
                if line.lower().startswith("fig") or  line.lower().startswith("plate"):
                    caption = line.strip()
                    break

            # 如果下方没有找到合适的注释，再检查上方三行
            if not caption:
                for line in lines_above:
                    if line.lower().startswith("fig") or line.lower().startswith("plate"):
                        caption = line.strip()
                        break
            caption=remove_md(caption)
            # print(caption)
            # 如果仍然没有找到符合条件的注释，跳过当前图片
            if not caption:
                continue
            # 保存结果
            image_caption_list.append({
                "img_path": full_img_path,
                "caption": caption
            })

    return image_caption_list


def find_min_distance_matches(images, captions):
    """
    使用匈牙利算法找到 image 和 caption 的最小距离匹配
    :param images: list of int, image 的行号
    :param captions: list of int, caption 的行号
    :return: list of tuples, 每个 tuple 是 (image 行号, caption 行号)
    """
    if(not images) or (not captions):
        return []
    # 提取位置信息
    img_positions = [img[1] for img in images]
    cap_positions = [cap[1] for cap in captions]
    # 构建距离矩阵
    distance_matrix = []
    for img in img_positions:
        row = []
        for cap in cap_positions:
            row.append(abs(img - cap))  # 计算绝对距离
        distance_matrix.append(row)
    
    # 使用匈牙利算法找到最小权匹配
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    
    # 返回匹配结果
    matches = []
    for i, j in zip(row_ind, col_ind):
        matches.append((images[i][0], captions[j][0]))  # 返回内容信息
     # 处理未匹配的 images
    matched_images = set(row_ind)
    for i in range(len(images)):
        if i not in matched_images:
            # 找到最近的 caption
            min_distance = float('inf')
            nearest_caption = None
            for j in range(len(captions)):
                distance = abs(images[i][1] - captions[j][1])
                if distance < min_distance:
                    min_distance = distance
                    nearest_caption = captions[j][0]
            matches.append((images[i][0], nearest_caption))
    return matches


def extract_images_and_captions_global(markdown_file,folder):
    """
    从Markdown文件中提取图片和标题的函数。

    参数:
    markdown_file (str): Markdown文件的路径。
    ocr_folder (str): OCR结果输出的文件夹路径。

    返回:
    image_caption_list (list): 包含图片路径和标题的字典列表。
    """
    # 用于存储图片和注释的列表
    image_caption_list = []
    # 读取 Markdown 文件内容
    with open(markdown_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 正则表达式匹配 Markdown 图片格式 ![alt](image_path)
    image_pattern = re.compile(r'!\[.*?\]\((.*?)\)')

    images=[]
    captions=[]
    # 遍历文件行
    for i, line in enumerate(lines):
        line = line.strip()
        # 如果是图片行
        if image_pattern.search(line):
            img_path = image_pattern.search(line).group(1)
            # 拼接图片路径
            full_img_path = os.path.join(folder, img_path)
            images.append([full_img_path,i])

            # 获取上下三行的注释
        if line.lower().startswith("fig") or  line.lower().startswith("plate"):
            caption=remove_md(line)
            captions.append([caption,i])

    matches=find_min_distance_matches(images,captions)  
    for match in matches:
        # 保存结果
        image_caption_list.append({
            "img_path": match[0],
            "caption": match[1]
        })          
    return image_caption_list

