from super_gradients.training import models
import cv2
import torch

device = 'cuda' if torch.cuda.is_available() else "cpu"
model = models.get("yolo_nas_s", pretrained_weights="coco")

img_path = r"messi_ronaldo.jpg"
model.predict(img_path, conf=0.5).show()
