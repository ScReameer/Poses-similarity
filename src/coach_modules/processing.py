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
        
    def __call__(self, *args, **kwds):
        return self._process(*args, **kwds)
    
    def _process(self, *args, **kwds):
        pass
    
# One image preprocessor
class ImagePreprocessor(Processor):
    def __init__(self) -> None:
        super().__init__()
    
    def _process(self, img_path: str) -> DataLoader:
        rgb_img = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2RGB)
        # [1, H, W, C]
        single_image = self.transforms(rgb_img).to(DEVICE)[None, ...]
        dataloader = DataLoader(single_image, batch_size=1, shuffle=False)
        return dataloader
    
# Video preprocessor
class VideoPreprocessor(Processor):
    def __init__(self) -> None:
        super().__init__()

    def _process(self, video_path: str, frame_rate=1, batch_size=1) -> DataLoader:
        dataset = VideoDataset(video_path=video_path, transform=self.transforms, frame_rate=frame_rate)
        # [1, H, W, C]
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
        return dataloader

# Auxiliary dataset class for video processing
class VideoDataset(Dataset):
    def __init__(self, video_path: str, frame_rate: int, transform=None):
        self.video_path = video_path
        self.cap = cv.VideoCapture(video_path)
        self.frame_rate = frame_rate
        self.transform = transform
        self.frames = []
        frame_counter = 0
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_counter += 1
            if frame_counter % frame_rate == 0:
                self.frames.append(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> torch.Tensor:
        frame = self.frames[idx]
        if self.transform:
            frame = self.transform(frame)
        return frame.to(DEVICE)