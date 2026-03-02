import torchvision.transforms as transforms
from torch.autograd import Variable
import os
from PIL import Image
import torch

class ImageClassifier:
    def __init__(self, model_path, device,classes=('fossil', 'other')):
        self.classes = classes
        self.transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.DEVICE = device
        self.model = torch.load(model_path)
        self.model.eval()
        self.model.to(self.DEVICE)

    def predict(self,file):
        img = Image.open(file)
        img = img.convert('RGB')
        img = self.transform_test(img)
        img.unsqueeze_(0)
        img = Variable(img).to(self.DEVICE)
        out = self.model(img)
        _, pred = torch.max(out.data, 1)
        return self.classes[pred.data.item()]
            # print('{}, predict: {}'.format(file, self.classes[pred.data.item()]))