import cv2 as cv
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Abstract class
class Processor:
    def __init__(self) -> None:
        self.transforms = T.Compose([
            T.ToTensor()
        ])
        
    def __call__(self, path:str):
        return self._process(path)
    
    def _process(self, path:str):
        pass
    
# One image preprocessor
class ImagePreprocessor(Processor):
    def __init__(self) -> None:
        super().__init__()
    
    def _process(self, img_path:str):
        rgb_img = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2RGB)
        # [1, H, W, C]
        single_image = self.transforms(rgb_img).to(DEVICE)[None, ...]
        dataloader = DataLoader(single_image, batch_size=1, shuffle=False)
        return dataloader
    
# Video preprocessor
class VideoPreprocessor(Processor):
    def __init__(self) -> None:
        super().__init__()
    
    def _process(self, video_path:str):
        dataset = VideoDataset(video_path=video_path, transform=self.transforms)
        # [4, H, W, C]
        dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=False)
        return dataloader

# Auxiliary dataset class for video processing
class VideoDataset(Dataset):
    def __init__(self, video_path, frame_rate=1, transform=None):
        self.video_path = video_path
        self.cap = cv.VideoCapture(video_path)
        self.frame_rate = frame_rate
        self.transform = transform
        self.length = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx >= self.length:
            return None
        self.cap.set(cv.CAP_PROP_POS_FRAMES, idx)
        success, frame = self.cap.read()
        if not success:
            return None
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        if self.transform is not None:
            frame = self.transform(frame).to(DEVICE)
        return frame

    def close(self):
        self.cap.release()