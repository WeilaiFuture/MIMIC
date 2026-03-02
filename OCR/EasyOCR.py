import os


class MyEasyOCR:
    def __init__(self, lang_list=['en', 'ch_sim'], gpu=True):
        import easyocr
        os.environ["EASYOCR_MODULE_PATH"] = "~.easyocr/model"
        self.reader = easyocr.Reader(lang_list, gpu=gpu)

    def predict(self, image_path):
        result = self.reader.readtext(image_path)
        items = []
        for bbox, text, conf in result:
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            left, top = int(min(xs)), int(min(ys))
            width, height = int(max(xs) - min(xs)), int(max(ys) - min(ys))
            items.append([str(text).upper(), {"top": top, "left": left, "height": height, "width": width}])

        return image_path, items


if __name__ == "__main__":
    model=MyEasyOCR()
    _, ocr_result = model.predict("/home/root123/mount1/weilai/MIMIC/data/1/merged.jpg")
    print(ocr_result)