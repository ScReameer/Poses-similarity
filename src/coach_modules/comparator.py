import torch
import cv2 as cv
from torch.utils.data import DataLoader
from .detector import KeypointDetector
from .writer import Writer
from .metrics_aggregator import MetricsAggregator

class Comparator:
    def __init__(self, oks_threshold: float, metrics: dict) -> None:
        self.keypoint_detector = KeypointDetector()
        self.metrics_aggregator = MetricsAggregator(metrics)
        self.writer = Writer(oks_threshold)
        
    def _predict(self, reference_batch: torch.Tensor, actual_batch: torch.Tensor) -> tuple:
        return {
            'reference_output': self.keypoint_detector(reference_batch),
            'actual_output': self.keypoint_detector(actual_batch)
        }
    
    def compare(
        self, 
        reference_dl: DataLoader,
        actual_dl: DataLoader,
        mode: str,
        name: str,
        fps: int
    ) -> dict:
        # Check how many iterations there will be in the zip operator. 
        # If there are 2 videos of different duration, 
        # then the number of iterations will be counted according to the minimum duration
        total_min_batches = min(map(len, [reference_dl, actual_dl]))
        self.metrics_aggregator.set_cumulative_zeros() # For averaging metrics
        if mode == 'video':
            height = next(iter(reference_dl)).shape[-2]
            width = next(iter(reference_dl)).shape[-1]
            output_width = width * 2
            fourcc = cv.VideoWriter_fourcc(*'XVID')
            video_writer = cv.VideoWriter(
                filename=f'{name}.avi', 
                fourcc=fourcc,
                fps=fps, 
                frameSize=(output_width, height)
            )
        else:
            video_writer = None
        # Every batch have shape [B, C, H, W]
        for reference_batch, actual_batch in zip(reference_dl, actual_dl):
            # Forward pass through network
            nn_output = self._predict(reference_batch, actual_batch)
            # Compute metrics
            metrics_dict_batch = self.metrics_aggregator.get_metrics_per_batch(nn_output)
            # Skip iteration if any metric is None
            # This means that the network has not found any pose on at least one of the frames
            if any(map(lambda x: x is None, metrics_dict_batch.values())):
                continue
            # Make new video or image with a drawn skeleton and metrics
            self.writer.write(
                reference_batch=reference_batch, 
                actual_batch=actual_batch, 
                nn_output=nn_output,
                metrics=metrics_dict_batch,
                video_writer=video_writer, 
                name=name
            )
            # Accumulate metrics 
            self.metrics_aggregator.accumulate_per_batch(metrics_dict_batch)
        # Return averaged metrics over total batches
        return self.metrics_aggregator.get_averaged_metrics(total_min_batches)