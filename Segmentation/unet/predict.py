
# 获取当前文件所在目录的上一级目录
import os
import sys


unet_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(unet_dir, "../.."))

# 把 `mask_model/eam_sam/` 添加到 sys.path
sys.path.append(unet_dir)

# 把整个 `project` 目录添加到 sys.path
sys.path.append(project_root)
import cv2
import numpy as np
import torch
from model.unet_model import UNet
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from PIL import Image

def apply_crf(original_image, mask_img):
    h, w = original_image.shape[:2]
    d = dcrf.DenseCRF2D(w, h, 2)
    unary = unary_from_softmax(mask_img)
    d.setUnaryEnergy(unary)
    
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=10, rgbim=original_image, compat=5)
    Q = d.inference(5)
    return np.argmax(Q, axis=0).reshape((h, w))

def post_process(pred):
    if pred.dtype != np.uint8:
        pred = pred.astype(np.uint8)
    
    kernel = np.ones((3, 3), np.uint8)
    pred = cv2.erode(pred, kernel, iterations=2)
    pred = cv2.dilate(pred, kernel, iterations=1)
    return pred

class UnetSegmentation:
    def __init__(self, device: str = None, weight_url: str = "best_model.pth"):
        """
        Args:
            device: 'cuda' 或 'cpu'，默认自动检测
            weight_url: UNet 模型权重路径
        """
        # 1. 自动检测 device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # 2. 检查权重文件
        if not os.path.exists(weight_url):
            raise FileNotFoundError(f"❌ 权重文件不存在: {weight_url}")

        # 3. 初始化模型
        self.net = UNet(n_channels=1, n_classes=1).to(self.device)

        # 4. 加载权重
        try:
            state_dict = torch.load(weight_url, map_location=self.device)
            self.net.load_state_dict(state_dict)
        except Exception as e:
            raise RuntimeError(f"❌ 加载权重失败: {weight_url}, 错误: {e}")

        # 5. 设置为推理模式
        self.net.eval()

    def predict(self,img_path,  threshold=0.1):
        img = cv2.imread(img_path)
        origin_img = img.copy()
        origin_shape = img.shape
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (512, 512))
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        img_tensor = torch.from_numpy(img).to(device=self.device, dtype=torch.float32)
        
        with torch.no_grad():
            pred = self.net(img_tensor)
        
        pred = np.array(pred.data.cpu()[0])[0]
        pred = cv2.resize(pred, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)
        
        softmax_pred = np.stack([1-pred, pred], axis=0)
        crf_pred = apply_crf(origin_img, softmax_pred)
        
        crf_pred[crf_pred >= threshold] = 255
        crf_pred[crf_pred < threshold] = 0
        crf_pred = post_process(crf_pred)
        
        crf_pred = np.stack([crf_pred] * 3, axis=-1)
        result_image = Image.fromarray(crf_pred[:,:,0].astype(np.uint8), mode='L')
        return result_image,crf_pred
    
# if __name__ == "__main__":
#     mask_device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
#     text=UnetSegmentation(device=mask_device)
#     ans,crf_pred=text.predict(img_path="/home/root123/mount1/weilai/project/mask_model/eam_sam/your_data/9fc8160c7c196867b5eecf5397e51dfb1179b482e7184bc1f1e8a034ab6de95a.jpg")
#     ans.save("/home/root123/mount1/weilai/project/mask_model/eam_sam/output/img/1.png")