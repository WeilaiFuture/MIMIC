
def load_phrases_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            phrases = [line.strip() for line in f if line.strip()]  # 忽略空行
        return phrases
    except Exception as e:
        print(f"读取短语文件时出错: {e}")
        return []
    
class TextClassifier:
    def __init__(self, phrases_path):
        self.phrases = load_phrases_from_file(phrases_path)

    def predict(self,text):
        # 检查 caption 是否包含词典中的任意短语
        # if(not text):
        #     return "fossil"
        # print(text)
        if any(phrase in text for phrase in self.phrases):
            return "fossil"
        else:
            return "other"