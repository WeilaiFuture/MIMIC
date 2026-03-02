
import cv2
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from scipy.optimize import linear_sum_assignment

    
def calculate_iou(box1, box2):
    """计算两个框的IoU"""
    x_left = max(box1['xmin'], box2['xmin'])
    y_top = max(box1['ymin'], box2['ymin'])
    x_right = min(box1['xmax'], box2['xmax'])
    y_bottom = min(box1['ymax'], box2['ymax'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # 没有交集

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1['xmax'] - box1['xmin']) * (box1['ymax'] - box1['ymin'])
    box2_area = (box2['xmax'] - box2['xmin']) * (box2['ymax'] - box2['ymin'])
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area

def evaluate(json_file, position_file, predict_boxes,predict_name, iou_threshold=0.5):
    """
    评估预测框与真实框之间的匹配：
    - IoU 需要超过 iou_threshold
    - 名称需要匹配
    """
    # 加载名称信息和位置信息
    json_data = load_json(json_file)
    position_data = load_json(position_file)

    keys = list(json_data.keys())
    values = list(json_data.values())

    total_ground_truth = len(keys)  # 真实框数量
    total_predictions = len(predict_boxes)  # 预测框数量
    true_positive = 0

    # 遍历所有真实框和名称，计算与预测框的IoU并匹配名称
    for i, true_box in enumerate(list(position_data.values())):
        true_name = keys[i]
        matched = False
        true_box_dict = {
        'xmin': true_box[0],
        'ymin': true_box[1],
        'xmax': true_box[2],
        'ymax': true_box[3]
        }

        for pred_box in predict_boxes:
            iou = calculate_iou(true_box_dict, pred_box['unet_box'])
            pred_name = pred_box['ocr_box']['text']
            if iou >= iou_threshold and true_name == pred_name and true_name in predict_name and values[i]==predict_name[true_name]:
                true_positive += 1
                matched = True
                break  # 找到一个匹配就退出

    false_positive = total_predictions - true_positive
    false_negative = total_ground_truth - true_positive

    # 计算准确率、召回率和F1分数
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative
    }

def expand_box(box, expand_pixels):
    """
    将给定的框向四周扩大指定的像素点数
    :param box: (xmin, ymin, xmax, ymax)
    :param expand_pixels: 扩大的像素点数
    :return: 扩大后的框
    """
    xmin, ymin, xmax, ymax = box
    return (xmin - expand_pixels, ymin - expand_pixels, xmax + expand_pixels, ymax + expand_pixels)

def is_box_contained(small_box, big_box):
    """
    检查第一个框是否完全被第二个框包含
    :param small_box: (xmin, ymin, xmax, ymax)
    :param big_box: (xmin, ymin, xmax, ymax)
    :return: 如果small_box被big_box完全包含，则返回True
    """
    return (small_box[0] >= big_box[0] and small_box[1] >= big_box[1] and
            small_box[2] <= big_box[2] and small_box[3] <= big_box[3])

def filter_ocr_boxes(ocr_boxes, unet_boxes, expand_pixels=30):
    """
    过滤掉被UNet目标框包含的OCR目标框
    :param ocr_boxes: OCR目标框列表 [(text, (xmin, ymin, xmax, ymax)), ...]
    :param unet_boxes: UNet目标框列表 [(xmin, ymin, xmax, ymax), ...]
    :param expand_pixels: 扩大的像素点数
    :return: 过滤后的OCR目标框列表
    """

    filtered_ocr_boxes = []

    for text,box in ocr_boxes:
        contained = any(is_box_contained(expand_box(box, expand_pixels), unet_box) for unet_box in unet_boxes)
        if not contained:
            filtered_ocr_boxes.append((text, box))

    return filtered_ocr_boxes

def apply_crf(original_image, mask_img):
    h, w = original_image.shape[:2]
    d = dcrf.DenseCRF2D(w, h, 2)
    unary = unary_from_softmax(mask_img)
    d.setUnaryEnergy(unary)
    
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=10, rgbim=original_image, compat=5)
    Q = d.inference(5)
    return np.argmax(Q, axis=0).reshape((h, w))

def get_bounding_boxes(mask, min_area=500):
    """
    获取UNet检测结果的外接矩形框，并过滤掉面积小于设定阈值的矩形框
    :param mask: UNet分割结果的掩码
    :param min_area: 最小面积阈值，忽略面积小于该值的矩形框
    :return: 过滤后的矩形框列表 [(xmin, ymin, xmax, ymax)]
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)  # 获取外接矩形框
        area = w * h  # 计算矩形框的面积
        if area >= min_area:  # 只保留面积大于等于阈值的矩形框
            bounding_boxes.append((x, y, x + w, y + h))  # 矩形框的坐标
    
    return bounding_boxes


def calculate_intersection(box1, box2):
    """
    计算两个矩形框的交集面积
    :param box1: (xmin1, ymin1, xmax1, ymax1)
    :param box2: (xmin2, ymin2, xmax2, ymax2)
    :return: 交集面积
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # 如果没有交集，返回 0
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    return (x_right - x_left) * (y_bottom - y_top)


def merge_overlapping_boxes_with_ratio(boxes, threshold_ratio=0.4):
    """
    合并相交面积占较小矩形框一定比例的检测框
    :param boxes: [(xmin, ymin, xmax, ymax), ...] 检测框列表
    :param threshold_ratio: 合并的面积占较小矩形框的阈值，默认40%
    :return: 合并后的检测框列表
    """
    merged_boxes = []
    used = [False] * len(boxes)
    to_merge = []  # 用于存储需要合并的框

    for i in range(len(boxes)):
        if used[i]:
            continue
        
        new_box = list(boxes[i])  # 初始化新的合并框
        used[i] = True
        to_merge.append(new_box)  # 将初始框加入待合并列表

        # 检查是否可以与其他框合并
        while True:
            merged = False
            for j in range(len(boxes)):
                if used[j]:
                    continue

                intersection_area = calculate_intersection(new_box, boxes[j])
                
                box1_area = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])
                box2_area = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
                
                smaller_area = min(box1_area, box2_area)

                # 判断交集面积是否占较小矩形框面积的阈值
                if intersection_area / smaller_area >= threshold_ratio:
                    # 合并矩形框
                    new_box[0] = min(new_box[0], boxes[j][0])
                    new_box[1] = min(new_box[1], boxes[j][1])
                    new_box[2] = max(new_box[2], boxes[j][2])
                    new_box[3] = max(new_box[3], boxes[j][3])
                    used[j] = True  # 标记 box2 为已合并
                    merged = True  # 标记发生了合并

            if not merged:  # 如果没有新的合并发生，跳出循环
                break

        merged_boxes.append(tuple(new_box))  # 将合并后的框添加到结果中

    return merged_boxes


def draw_boxes(image, boxes, color, labels=None):
    """
    在图片上绘制目标检测框
    :param image: 目标图片
    :param boxes: [(xmin, ymin, xmax, ymax)] 检测框列表
    :param color: 颜色
    :param label: 标签（可选）
    :return: 绘制目标框后的图片
    """
    for i in range(len(boxes)):
        box=boxes[i]
        if labels:
            label=labels[i]
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)  # 画出矩形框
        if labels:
            cv2.putText(image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return image


def calculate_center(xmin, ymin, xmax, ymax):
    """计算矩形框的中心点坐标"""
    return ((xmin + xmax) / 2, (ymin + ymax) / 2)

def match_boxes(unet_boxes, ocr_boxes):
    """
    按最短距离匹配 UNet 目标框和 OCR 字符框
    :param unet_boxes: UNet检测的图版目标框列表 [(xmin, ymin, xmax, ymax), ...]
    :param ocr_boxes: OCR识别的字符数字框列表 [(text, (xmin, ymin, xmax, ymax)), ...]
    :return: 匹配列表，每个元素为 (unet_box, ocr_box)
    """
    matches = []
    for unet_box in unet_boxes:
        unet_center = calculate_center(*unet_box)
        min_distance = float('inf')
        matched_ocr_box = None

        for ocr_box in ocr_boxes:
            ocr_center = calculate_center(*ocr_box[1])
            distance = np.linalg.norm(np.array(unet_center) - np.array(ocr_center))

            if distance < min_distance:
                min_distance = distance
                matched_ocr_box = ocr_box

        if matched_ocr_box:
            matches.append((unet_box, matched_ocr_box))

    return matches

def match_boxes_dynamic(unet_boxes, ocr_boxes):
    """
    动态匹配 UNet 目标框和 OCR 字符框，匹配过的目标框不参与后续匹配
    :param unet_boxes: UNet检测的图版目标框列表 [(xmin, ymin, xmax, ymax), ...]
    :param ocr_boxes: OCR识别的字符数字框列表 [(text, (xmin, ymin, xmax, ymax)), ...]
    :return: 匹配列表，每个元素为 (unet_box, ocr_box)
    """
    matches = []
    remaining_ocr_boxes = sorted(ocr_boxes, key=lambda x: x[0])  # 按照文本的字典顺序进行排序

    for unet_box in unet_boxes:
        unet_center = calculate_center(*unet_box)
        min_distance = float('inf')
        matched_ocr_box = None

        for ocr_box in remaining_ocr_boxes:
            ocr_center = calculate_center(*ocr_box[1])
            distance = np.linalg.norm(np.array(unet_center) - np.array(ocr_center))

            if distance < min_distance:
                min_distance = distance
                matched_ocr_box = ocr_box

        if matched_ocr_box:
            matches.append((unet_box, matched_ocr_box))
            remaining_ocr_boxes.remove(matched_ocr_box)  # 匹配后移除

    return matches


def match_boxes_km(unet_boxes, ocr_boxes):
    """
    使用KM算法（匈牙利算法）进行全局最优匹配
    :param unet_boxes: UNet检测的图版目标框列表 [(xmin, ymin, xmax, ymax), ...]
    :param ocr_boxes: OCR识别的字符数字框列表 [(text, (xmin, ymin, xmax, ymax)), ...]
    :return: 匹配列表，每个元素为 (unet_box, ocr_box)
    """
    cost_matrix = np.zeros((len(unet_boxes), len(ocr_boxes)))

    # 构造距离矩阵
    for i, unet_box in enumerate(unet_boxes):
        unet_center = calculate_center(*unet_box)
        for j, ocr_box in enumerate(ocr_boxes):
            ocr_center = calculate_center(*ocr_box[1])
            cost_matrix[i, j] = np.linalg.norm(np.array(unet_center) - np.array(ocr_center))

    # 使用匈牙利算法寻找最小匹配成本
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 返回匹配结果
    matches = [(unet_boxes[i], ocr_boxes[j]) for i, j in zip(row_ind, col_ind)]
    return matches
