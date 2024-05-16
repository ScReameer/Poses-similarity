from .coach_modules.detector import KeypointDetector
from .coach_modules.metrics import  CosineSimilarity, ObjectKeypointSimilarity, RMSE
from .coach_modules.processing import ImagePreprocessor, VideoPreprocessor
from .coach_modules.writer import Writer
import cv2 as cv
import torch
from torch.utils.data import DataLoader

class VirtualCoach:
    def __init__(self) -> None:
        # Two base classes for preprocess images and videos
        self.preprocessors = {
            'image': ImagePreprocessor(),
            'video': VideoPreprocessor()
        }
        # Pretrained model
        self.keypoint_detector = KeypointDetector()
        # Metrics
        self.oks = ObjectKeypointSimilarity()
        self.cosine_similarity = CosineSimilarity()
        self.rmse = RMSE()
        # Output writer
        self.writer = Writer()
        
    def __call__(self, *args, **kwds):
        return self.compare_poses(*args, **kwds)
        
    def _predict(self, reference_batch: torch.Tensor, actual_batch: torch.Tensor) -> tuple:
        return (
            self.keypoint_detector(reference_batch),
            self.keypoint_detector(actual_batch)
        )
        
    def _compare(
        self, 
        ref_dl: DataLoader,
        actual_dl: DataLoader,
        mode: str,
        name: str
    ) -> dict:
        # Check how many iterations there will be in the zip operator. 
        # If there are 2 videos of different duration, 
        # then the number of iterations will be counted according to the minimum duration
        total_min_batches = min(map(len, [ref_dl, actual_dl]))
        # Cumulative variables for metrics
        oks_sum = .0
        cossim_sum = .0
        rmse_sum = .0
        if mode == 'video':
            height = min(next(iter(ref_dl)).shape[-2], next(iter(actual_dl)).shape[-2])
            width = min(next(iter(ref_dl)).shape[-1], next(iter(actual_dl)).shape[-1])
            output_width = width * 2
            fourcc = cv.VideoWriter_fourcc(*'XVID')
            video_writer = cv.VideoWriter(f'{name}.avi', fourcc, 30, (output_width, height))
        else:
            video_writer = None
        # Every batch have shape [B, C, H, W]
        for ref_batch, actual_batch in zip(ref_dl, actual_dl):
            # Forward pass through network
            nn_output = self._predict(ref_batch, actual_batch)
            # Compute metrics
            oks = self.oks(*nn_output)
            cossim = self.cosine_similarity(*nn_output)
            rmse = self.rmse(*nn_output)
            # Skip iteration if any metric is None
            # This means that the network has not found any pose on at least one of the frames
            if any(map(lambda x: x is None, [oks, cossim, rmse])):
                continue
            # Make new video with a drawn skeleton and metrics
            self.writer.write(ref_batch, actual_batch, nn_output, [oks, cossim, rmse], video_writer, name)
            # Summarize metrics
            oks_sum += oks.cpu().item()
            cossim_sum += cossim.cpu().item()
            rmse_sum += rmse.cpu().item()
        # Average metrics for all batches 
        oks_sum /= total_min_batches
        cossim_sum /= total_min_batches
        rmse_sum /= total_min_batches
        return {
            'OKS': oks_sum,
            'CosSim': cossim_sum,
            'RMSE': rmse_sum,
        }
    
    def compare_poses(
        self,
        reference_path: str, 
        actual_path: str, 
        mode: str,
        name='result',
        frame_skip=1, 
        batch_size=1
    ) -> dict:
        assert mode in ['video', 'image'], 'Wrong mode'
        preprocessor = self.preprocessors[mode]
        extension = {
            'video': '.avi',
            'image': '.png'
        }
        if not all(map(lambda path: isinstance(path, str), [reference_path, actual_path])):
            print('Path type should be string')
            
        else:
            try:
                reference_dl = preprocessor(reference_path, frame_skip, batch_size)
                actual_dl = preprocessor(actual_path, frame_skip, batch_size)
            except:
                print('Wrong path')
            else:
                metrics = self._compare(reference_dl, actual_dl, mode, name)
                print(f'Success! The results are saved to a file "{name+extension[mode]}"')
                return metrics
            