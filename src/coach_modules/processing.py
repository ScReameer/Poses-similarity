import cv2 as cv
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Abstract class
class Processor:
    def __init__(self) -> None:
        """Abstract processing class
        """
        self.transforms = T.Compose([
            T.ToTensor()
        ])
        
    def __call__(self, *args, **kwds):
        """Calls `self._process` for input

        Returns:
            `DataLoader`: converted path into `DataLoader`
        """
        return self._process(*args, **kwds)
    
    def _process(self, *args, **kwds):
        """Abstract processing function
        """
        pass
    
# One image preprocessor
class ImagePreprocessor(Processor):
    def __init__(self) -> None:
        """Image preprocessor class, converts image path into `DataLoader`
        """
        super().__init__()
    
    def _process(self, img_path: str, *args, **kwds) -> DataLoader:
        """Converts image path into `DataLoader`

        Args:
            `img_path` (`str`): relative or full path to image

        Returns:
            `dataloader` (`DataLoader`): dataloader with one RGB image of shape `[1, C, H, W]` 
        """
        rgb_img = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2RGB)
        # [1, C, H, W]
        single_image = self.transforms(rgb_img).to(DEVICE)[None, ...]
        dataloader = DataLoader(single_image, batch_size=1, shuffle=False)
        return dataloader
    
# Video preprocessor
class VideoPreprocessor(Processor):
    def __init__(self) -> None:
        """Video preprocessor class, converts video path into `DataLoader`
        """
        super().__init__()

    def _process(self, video_path: str, frame_skip: int, batch_size: int) -> DataLoader:
        """Converts video path into `DataLoader`

        Args:
            `video_path` (`str`): relative or full path to video
            `frame_skip` (`int`): affects the speed of the video, for example, 2 means speed 2x
            `batch_size` (`int`): the number of frames that will be combined into one batch.  
            It strongly affects the performance of the system, 
            it is not recommended to use large values if the device does not have enough video memory and/or RAM

        Returns:
            `dataloader` (`DataLoader`): dataloader with N RGB frames of shape `[batch_size, C, H, W]`
        """
        dataset = VideoDataset(video_path=video_path, transform=self.transforms, frame_skip=frame_skip)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
        return dataloader

# Auxiliary dataset class for video processing
class VideoDataset(Dataset):
    def __init__(self, video_path: str, frame_skip: int, transform=None):
        """Auxiliary class for video processing

        Args:
            `video_path` (`str`): relative or full path to video
            `frame_skip` (`int`): affects the speed of the video, for example, 2 means speed 2x
            `transform` (`None` or `torchvision.transforms.Compose`, optional): Transformations for each of the video frames. Defaults to `None`.
        """
        self.video_path = video_path
        self.cap = cv.VideoCapture(video_path)
        self.frame_skip = frame_skip
        self.transform = transform
        # All video frames as list
        self.frames = []
        # Counter for frame skipping
        frame_counter = 0
        # Read every video frame in loop
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            # End of video
            if not ret:
                break
            frame_counter += 1
            # Skip frames, if `frame_skip` != 1, affects video speed
            if frame_counter % frame_skip == 0:
                self.frames.append(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> torch.Tensor:
        frame = self.frames[idx]
        if self.transform:
            frame = self.transform(frame)
        return frame.to(DEVICE)