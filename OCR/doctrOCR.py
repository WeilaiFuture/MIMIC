from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import Image

class doctrOCR:
    def __init__(self):
        """
        初始化 docTR OCR 模型，使用预训练的模型
        """
        # 加载 docTR 的预训练模型
        self.ocr = ocr_predictor(pretrained=True, assume_straight_pages=True)

    def predict(self, image_path):
        """
        处理单个图片并返回 (image_path, getresult)
        :param image_path: 输入图像的路径
        :return: 图像路径和识别结果（包括文本和边界框）
        """
        # 使用 docTR 进行 OCR 识别
        doc = DocumentFile.from_images(image_path)
        result = self.ocr(doc)

        # 打开图像文件并获取宽度和高度
        img = Image.open(image_path)
        img_width, img_height = img.size

        result_list = []

        # 遍历识别结果
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        # 获取每个单词的文本和对应的边界框
                        word_text = word.value
                        word_bbox = word.geometry

                        # 确保 word_bbox 是有效的（包含两个坐标）
                        if len(word_bbox) == 2:
                            # 获取两个对角坐标（相对坐标）
                            top_left = word_bbox[0]
                            bottom_right = word_bbox[1]

                            # 将相对坐标转换为像素坐标
                            top = int(top_left[1] * img_height)
                            left = int(top_left[0] * img_width)
                            width = int(abs(top_left[0] - bottom_right[0]) * img_width)
                            height = int(abs(top_left[1] - bottom_right[1]) * img_height)

                            # 保存文本和边界框信息
                            result_list.append([str(word_text).upper(), {"top": top, "left": left, "height": height, "width": width}])
                        else:
                            print(f"Skipping word with invalid bbox: {word_text}")

        return image_path, result_list


# 示例用法
if __name__ == "__main__":
    # 初始化 OCR 模型
    model = doctrOCR()

    # 处理图像并获取结果
    image_path = "/home/root123/mount1/weilai/MIMIC/data/7/merged.jpg"
    _, ocr_result = model.predict(image_path)

    # 打印识别结果
    print("OCR 结果：")
    for text, bbox in ocr_result:
        print(f"文本：{text}, 边界框：{bbox}")
