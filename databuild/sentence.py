import random
import re
import json


# 合并段落，根据 labels_dict 中相同的 value 进行合并，并处理非连续段
def merge_segments(labels_dict ,singel_label=False):
    if(len(set(labels_dict.values())) == 1 and singel_label):
        return list(set(labels_dict.values()))
    # 按标签顺序排序
    sorted_items = sorted(labels_dict.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0])

    merged_segments = {}  # 存储合并后的段落
    current_value = None  # 当前的名称值
    current_range = []  # 当前范围的标签

    for label, value in sorted_items:
        if current_value is None:  # 初始化当前值
            current_value = value
            current_range.append(label)  # 将当前标签加入范围
        elif value == current_value:  # 如果值相同，继续扩展范围
            current_range.append(label)
        else:  # 值不同，处理合并
            # 将当前范围合并为字符串形式
            range_str = f"{current_range[0]}-{current_range[-1]}" if len(current_range) > 1 else current_range[0]
            if current_value not in merged_segments:
                merged_segments[current_value] = []
            merged_segments[current_value].append(range_str)
            
            # 重置范围
            current_value = value
            current_range = [label]  # 重新开始新的范围

    # 处理最后一个范围
    range_str = f"{current_range[0]}-{current_range[-1]}" if len(current_range) > 1 else current_range[0]
    if current_value not in merged_segments:
        merged_segments[current_value] = []
    merged_segments[current_value].append(range_str)

    # 创建包含两个字符串的结果数组
    result = []
    for value, labels in merged_segments.items():
        combined_labels = ', '.join(labels)
        result.append([f"{combined_labels}: ",f"{value}"])

    return result



def add_detailed_descriptions(merged_segments, labels_dict):
    result = []
    bert_result = []
    # 生成描述时直接引用 labels_dict
    for segment in merged_segments:
        # parts = segment.split()  # 将合并段分成标签和名称
        # labels = parts[0]  # 获取标签部分
        # name = parts[-1]  # 获取名称部分
        labels = segment[0]  # 获取第一类标签部分 (除去最后一个)
        name = segment[1]  # 获取第二类标签部分 (最后一个元素)
        result.append(f"{labels}{name}"+",")  # 添加原始合并段

        B=name+","
 # 为第一类标签添加 B1 和 I1 标签
        for i, label in enumerate(labels.split(' ')):
            if(len(label) > 0):
                if i == 0:
                    bert_result.append(f"{label} B1-START")  # 第一个元素标记为 B1-START
                else:
                    bert_result.append(f"{label} I1-START")  # 后续的元素标记为 I1-START

     
        # 为第二类标签添加 B2 和 I2 标签
        name_tokens = B.split(' ')  # 根据下划线分割第二类标签
        for i, token in enumerate(name_tokens):
            if i == 0:
                bert_result.append(f"{token} B2-START")  # 名称开始部分标记为 B2-START
            else:
                bert_result.append(f"{token} I2-START")  # 名称其余部分标记为 I2-START   # 获取所有与名称对应的标签
        relevant_labels = [key for key, value in labels_dict.items() if value == name]

        # 将标签按字典序排序
        relevant_labels.sort(key=lambda x: (x.isdigit(), x))  # 先按数字，后按字母排序

        # 为每个标签添加描述
        index=0
        for item in relevant_labels:
            index=index+1
            description = random.choice([
                'P1 element, upper and oblique upper views, specimen F6-4, USNM541102',
                'lower view CEAF-0005', 'lorem ipsum', 'Middle to Late Devonian diversity'
            ])
            result.append(f"{item}, {description},")  # 添加描述
            bert_result.append(f"{item}, O")  # 添加描述及其标签
            desc_tokens = description.split()  # 根据空格分割描述
            if(index==len(relevant_labels)):
                desc_tokens[-1]+="."
            else:desc_tokens[-1]+=","
            for desc in desc_tokens:
                bert_result.append(f"{desc} O")  # 添加描述及其标签
        result[-1] = result[-1][:-1] + "."  # 确保最后一个描述以句号结
    return result,bert_result



def add_BERT_descriptions(merged_segments, labels_dict):
    bert_result = []

    # 遍历合并后的段落
    for segment in merged_segments:
        labels = segment[0]  # 获取第一类标签部分 (除去最后一个)
        name = segment[1]  # 获取第二类标签部分 (最后一个元素)

        # 为第一类标签添加 B1 和 I1 标签
        for i, label in enumerate(labels.split(' ')):
            if(len(label) > 0):
                if i == 0:
                    bert_result.append(f"{label} B1-START")  # 第一个元素标记为 B1-START
                else:
                    bert_result.append(f"{label} I1-START")  # 后续的元素标记为 I1-START

        # 为第二类标签添加 B2 和 I2 标签
        name_tokens = name.split(' ')  # 根据下划线分割第二类标签
        for i, token in enumerate(name_tokens):
            if i == 0:
                bert_result.append(f"{token} B2-START")  # 名称开始部分标记为 B2-START
            else:
                bert_result.append(f"{token} I2-START")  # 名称其余部分标记为 I2-START

        # 添加描述部分（全为 O 标签）
        relevant_labels = [key for key, value in labels_dict.items() if value == name]
        for item in relevant_labels:
            description = random.choice([
                'P1 element, upper and oblique upper views, specimen F6-4, USNM541102',
                'lower view CEAF-0005', 'lorem ipsum', 'Middle to Late Devonian diversity'
            ])
            desc_tokens = description.split()  # 根据空格分割描述
            for desc in desc_tokens:
                bert_result.append(f"{desc} O")  # 添加描述及其标签


    return bert_result

# 处理 labels_dict
def process_labels_dict(labels_dict, json_path):
    # 第一步：合并段
    merged_segments=merge_segments(labels_dict)
    
    # 保存 labels_dict 到 JSON 文件
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(labels_dict, json_file, ensure_ascii=False, indent=4)

    
    # 第二步：插入详细描述
    final_segments,BERBERT_segments = add_detailed_descriptions(merged_segments, labels_dict)
    with open(json_path.replace("json.json","bert.json"), 'w', encoding='utf-8') as json_file:
        json.dump(BERBERT_segments, json_file, ensure_ascii=False, indent=4)
    # print(BERBERT_segments)
    # seg=[]
    # for b in BERBERT_segments:
    #     seg.append(b.split(" ")[0])
    
    # formatted_result = "Fig 1. " + " ".join(seg)   
    # print(formatted_result.replace(",,", ",").replace("_"," "))
    # 最终格式化输出
    formatted_result = "Fig 1. " + " ".join(final_segments)
    return formatted_result.replace(",,", ",").replace("_"," ")

if __name__ == "__main__":
    # 假设已经有了 labels_dict
    labels_dict = {
    'A1': 'Icriodus amabilis1',
    'A2': 'Icriodus amabilis',
    'A3': 'Icriodus_amabilis',
    'A4': 'Icriodus_amabilis',
    'A5': 'Icriodus_amabilis',
    'A6': 'Icriodus amabilis1',
}
    # 调用 process_labels_dict 处理 labels_dict，并生成结果
    result = process_labels_dict(labels_dict, 'json.json')
    print(result)
