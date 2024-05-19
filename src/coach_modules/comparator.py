import torch
import cv2 as cv
from torch.utils.data import DataLoader
from .detector import KeypointDetector
from .metrics import RMSE, ObjectKeypointSimilarity, CosineSimilarity, WeightedDistance
from .writer import Writer

class Comparator:
    def __init__(self, oks_threshold) -> None:
        self.keypoint_detector = KeypointDetector()
        self.oks = ObjectKeypointSimilarity()
        self.rmse = RMSE()
        self.cossim = CosineSimilarity()
        self.wd = WeightedDistance()
        self.writer = Writer(oks_threshold)
        
    def _predict(self, reference_batch: torch.Tensor, actual_batch: torch.Tensor) -> tuple:
        return {
            'reference_output': self.keypoint_detector(reference_batch),
            'actual_output': self.keypoint_detector(actual_batch)
        }
    
    def compare(
        self, 
        ref_dl: DataLoader,
        actual_dl: DataLoader,
        mode: str,
        name: str,
    ) -> dict:
        # Check how many iterations there will be in the zip operator. 
        # If there are 2 videos of different duration, 
        # then the number of iterations will be counted according to the minimum duration
        total_min_batches = min(map(len, [ref_dl, actual_dl]))
        # Cumulative variables for metrics
        oks_sum = .0
        rmse_sum = .0
        cossim_sum = .0
        wd_sum = .0
        if mode == 'video':
            height = next(iter(ref_dl)).shape[-2]
            width = next(iter(ref_dl)).shape[-1]
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
            oks = self.oks(**nn_output)
            rmse = self.rmse(**nn_output)
            cossim = self.cossim(**nn_output)
            wd = self.wd(**nn_output)
            # Skip iteration if any metric is None
            # This means that the network has not found any pose on at least one of the frames
            if any(map(lambda x: x is None, [oks, rmse])):
                continue
            # Make new video with a drawn skeleton and metrics
            self.writer.write(ref_batch, actual_batch, nn_output, [oks, rmse, cossim, wd], video_writer, name)
            # Summarize metrics
            cossim_sum += cossim.cpu().item()
            oks_sum += oks.cpu().item()
            rmse_sum += rmse.cpu().item()
            wd_sum += wd.cpu().item()
        # Average metrics for all batches 
        wd_sum /= total_min_batches
        oks_sum /= total_min_batches
        rmse_sum /= total_min_batches
        cossim_sum /= total_min_batches
        return {
            'OKS': oks_sum,
            'RMSE': rmse_sum,
            'CosSim': cossim_sum,
            'WD': wd_sum
        }