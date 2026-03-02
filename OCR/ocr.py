from paddleocr import PaddleOCR

class myOCR:
    def __init__(self,det_model_dir=None, cls_model_dir=None, rec_model_dir=None):
        self.ocr = PaddleOCR(
                det_model_dir=det_model_dir, 
                cls_model_dir=cls_model_dir, 
                rec_model_dir=rec_model_dir, 
                use_angle_cls=True, lang="en", det_limit_side_len=3200, use_gpu=True,rec_algorithm="CRNN",show_log=False)

    def predict(self,image_path):
    
        """
        处理单个图片并返回 (image_path, getresult)
        """
        result = self.ocr.ocr(image_path, cls=True)
        list = []
        for idx in range(len(result)):
            res = result[idx]
            if res:
                for line in res:
                    # OCR处理
                    line[0][0][0] = int(line[0][0][0])
                    line[0][0][1] = int(line[0][0][1])
                    line[0][1][0] = int(line[0][1][0])
                    line[0][1][1] = int(line[0][1][1])
                    line[0][2][0] = int(line[0][2][0])
                    line[0][2][1] = int(line[0][2][1])
                    line[0][3][0] = int(line[0][3][0])
                    line[0][3][1] = int(line[0][3][1])
                    
                    # 计算bounding box
                    top = min(line[0][0][1], line[0][1][1])
                    left = min(line[0][0][0], line[0][3][0])
                    height = max(line[0][3][1] - line[0][0][1], line[0][2][1] - line[0][1][1],
                                line[0][3][1] - line[0][1][1], line[0][2][1] - line[0][0][1])
                    width = max(line[0][1][0] - line[0][0][0], line[0][2][0] - line[0][3][0],
                                line[0][2][0] - line[0][0][0], line[0][1][0] - line[0][3][0])
                    list.append([str(line[1][0]).upper(), {"top": top, "left": left, "height": height, "width": width}])
        return image_path, list
    

# 示例用法
if __name__ == "__main__":
    model=myOCR(
        det_model_dir='/home/root123/mount1/weilai/MIMIC/OCR/inference/ch_PP-OCRv3_det_student_inference',
        cls_model_dir='/home/root123/mount1/weilai/MIMIC/OCR/inference/ch_ppocr_mobile_v2.0_cls_infer',
        rec_model_dir='/home/root123/mount1/weilai/MIMIC/OCR/inference/en_PP-OCRv3_rec_inference'
    )
    _, ocr_result = model.predict("/home/root123/mount1/weilai/MIMIC/data/1/merged.jpg")
    print(ocr_result)

    ocr = PaddleOCR()  # 仅识别英文
    for i in range(20):
        result = ocr.ocr("/home/root123/mount1/weilai/MIMIC/data/1/merged.jpg")
        print(ocr_result)