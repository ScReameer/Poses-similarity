import torch
from torchvision import models

class KeypointDetector(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.detector = models.detection.keypoint_rcnn.keypointrcnn_resnet50_fpn(
            weights=models.detection.keypoint_rcnn.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        ).eval().to(self.device)
    
    def forward(self, x: torch.Tensor) -> list:
        with torch.no_grad():
            return self.detector(x)