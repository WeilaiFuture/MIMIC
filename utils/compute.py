import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from scipy.ndimage.measurements import label, center_of_mass, find_objects

def compute_caption_with_image(csv_path="test.csv",image_captions=None):
    # 初始化变量用于存储预测结果和真实标签
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    df = pd.read_csv(csv_path)
    vectorizer = TfidfVectorizer()
    for i in image_captions:
        img_path=os.path.basename(i["img_path"])
        caption=i["caption"]
        matched_row = df[df['path'] == img_path]
        texts=[matched_row['caption'].values[0],caption]

        tfidf_matrix = vectorizer.fit_transform(texts)

        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        # print(matched_row['caption'].values[0])
        # print(caption)
        # print(similarity[0][0])  # 值在 0~1 之间
         # 判断是否为正样本
        if similarity[0][0] > 0.75:
            predicted_label = "positive"
        else:
            predicted_label = "negative"
        
        # 假设真实标签为 "positive"（这里需要根据实际情况调整）
        true_label = "positive"
        
        # 更新混淆矩阵
        if predicted_label == "positive" and true_label == "positive":
            true_positives += 1
        elif predicted_label == "positive" and true_label == "negative":
            false_positives += 1
        elif predicted_label == "negative" and true_label == "positive":
            false_negatives += 1
    # 计算准确率
    precision = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return precision,true_positives,true_positives + false_negatives

def compute_fossil_filter(csv_path="test.csv",image_captions=None):
    # 初始化变量用于存储预测结果和真实标签
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    df = pd.read_csv(csv_path)
    for i in image_captions:
        img_path=os.path.basename(i["img_path"])
        label=i["fossil"]
        matched_row = df[df['path'] == img_path]
         # 判断是否为正样本
        if label=="1":
            predicted_label = "positive"
        else:
            predicted_label = "negative"
        
        # 假设真实标签为 "positive"（这里需要根据实际情况调整）
        if str(matched_row['fossil'].values[0])=="1":
            true_label = "positive"
        else:true_label = "negative"
        
        # print(i["caption"])   
        # print(true_label,label,matched_row['fossil'].values[0])
        
        # 更新混淆矩阵
        if predicted_label == "positive" and true_label == "positive":
            true_positives += 1
        elif predicted_label == "positive" and true_label == "negative":
            false_positives += 1
        elif predicted_label == "negative" and true_label == "positive":
            false_negatives += 1

    return true_positives,false_positives,false_negatives

def get_centroids_and_bounding_boxes(mask):
    labeled_mask, num_regions = label(mask)
    centroids = []
    bounding_boxes = []
    for i in range(1, num_regions + 1):
        region_mask = (labeled_mask == i)
        centroid = center_of_mass(region_mask)
        centroids.append([centroid[1], centroid[0]]) 
        slices = find_objects(labeled_mask == i)[0]
        y1, x1 = slices[0].start, slices[1].start
        y2, x2 = slices[0].stop - 1, slices[1].stop - 1  
        bounding_boxes.append([x1, y1, x2, y2])
    return centroids, bounding_boxes