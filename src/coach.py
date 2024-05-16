from .coach_modules.detector import KeypointDetector
from .coach_modules.metrics import  CosineSimilarity, ObjectKeypointSimilarity, RMSE, MAE
from .coach_modules.processing import ImagePreprocessor, VideoPreprocessor
import plotly.express as px
import plotly.io as pio
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import draw_keypoints
pio.renderers.default = 'png'
pio.templates.default = 'plotly_dark'

class VirtualCoach:
    def __init__(self) -> None:
        # Two base classes for preprocess images and videos
        self.preprocessors = {
            'image': ImagePreprocessor(),
            'video': VideoPreprocessor()
        }
        # Keypoints order for correct skeleton visualization
        self.keypoints = [
            'nose','left_eye','right_eye',
            'left_ear','right_ear','left_shoulder',
            'right_shoulder','left_elbow','right_elbow',
            'left_wrist','right_wrist','left_hip',
            'right_hip','left_knee', 'right_knee',
            'left_ankle','right_ankle'
        ]
        # Pretrained model
        self.keypoint_detector = KeypointDetector()
        # Metrics
        self.metrics = {
            'OKS': ObjectKeypointSimilarity(),
            'CosSim': CosineSimilarity(),
            'RMSE': RMSE(),
            'MAE': MAE()
        }
        self.oks = ObjectKeypointSimilarity()
        self.cosine_similarity = CosineSimilarity()
        self.rmse = RMSE()
        self.mae = MAE()
        self.threshold = 0.7
        
    def __call__(self, *args, **kwds):
        return self.compare_poses(*args, **kwds)
        
    def _predict(self, reference_batch: torch.Tensor, actual_batch: torch.Tensor) -> tuple:
        return (
            self.keypoint_detector(reference_batch),
            self.keypoint_detector(actual_batch)
        )
        
    def _compare(self, ref_dl: DataLoader, actual_dl: DataLoader) -> dict:
        # Check how many iterations there will be in the zip operator. 
        # If there are 2 videos of different duration, 
        # then the number of iterations will be counted according to the minimum duration
        total_min_batches = min(map(len, [ref_dl, actual_dl]))
        # Cumulative variables for metrics
        oks_sum = .0
        cossim_sum = .0
        rmse_sum = .0
        mae_sum = .0
        # 
        height = min(next(iter(ref_dl)).shape[-2], next(iter(actual_dl)).shape[-2])
        width = min(next(iter(ref_dl)).shape[-1], next(iter(actual_dl)).shape[-1])
        output_width = width * 2
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        video_writer = cv.VideoWriter('output.avi', fourcc, 30, (output_width, height))
        # Every batch have shape [B, C, H, W]
        for ref_batch, actual_batch in zip(ref_dl, actual_dl):
            # Forward pass through network
            nn_output = self._predict(ref_batch, actual_batch)
            # Compute metrics
            oks = self.oks(*nn_output)
            cossim = self.cosine_similarity(*nn_output)
            rmse = self.rmse(*nn_output)
            mae = self.mae(*nn_output)
            # Skip iteration if any metric is None
            # This means that the network has not found any pose on at least one of the frames
            if any(map(lambda x: x is None, [oks, cossim, rmse, mae])):
                continue
            # Make new video with a drawn skeleton and metrics
            self._write(ref_batch, actual_batch, nn_output, video_writer, oks, cossim, rmse, mae)
            # Summarize metrics
            oks_sum += oks.cpu().item()
            cossim_sum += cossim.cpu().item()
            rmse_sum += rmse.cpu().item()
            mae_sum += mae.cpu().item()
        # Average metrics for all batches 
        oks_sum /= total_min_batches
        cossim_sum /= total_min_batches
        rmse_sum /= total_min_batches
        mae_sum /= total_min_batches
        return {
            'OKS': oks_sum,
            'CosSim': cossim_sum,
            'RMSE': rmse_sum,
            'MAE': mae_sum
        }
    
    def _write(
        self, 
        ref_batch: torch.Tensor,
        actual_batch: torch.Tensor,
        nn_output: dict,
        video_writer: cv.VideoWriter,
        *metrics
    ):
        metrics = list(map(lambda x: round(x.cpu().item(), 2), metrics))
        # for ref_frame, act_frame in zip(ref_batch, actual_batch):
        min_batch_size = min(len(ref_batch), len(actual_batch))
        for batch_idx in range(min_batch_size):
            ref_img = ref_batch[batch_idx].cpu()
            act_img = actual_batch[batch_idx].cpu()
            ref_img_skeleton = self._draw_skeleton(
                img=ref_img,
                all_keypoints=nn_output[0][batch_idx]['keypoints'],
            )
            act_img_skeleton = self._draw_skeleton(
                img=act_img,
                all_keypoints=nn_output[1][batch_idx]['keypoints'],
            )
            combined_frames = self._combine(ref_img_skeleton, act_img_skeleton, metrics)
            video_writer.write(combined_frames)
        
    def _combine(self, ref_img: np.ndarray, act_img: np.ndarray, metrics: tuple):
        act_img_resized = cv.resize(
            act_img,
            dsize=ref_img.shape[:-1][::-1],
            interpolation=cv.INTER_LINEAR
        )
        combined_img = np.hstack((ref_img[..., ::-1], act_img_resized[..., ::-1]))
        if metrics[0] > self.threshold:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        oks, cossim, rmse, mae = metrics
        cv.putText(
            combined_img, 
            text=f'OKS={oks}, CosSim={cossim}, RMSE={rmse}, MAE={mae}',
            org=(0, combined_img.shape[0] - 10), 
            color=color, 
            fontFace=cv.FONT_HERSHEY_TRIPLEX, 
            fontScale=1,
            thickness=2
        )
        return combined_img
    
    def _draw_skeleton(self, img: torch.Tensor, all_keypoints: torch.Tensor) -> np.ndarray:
        connectivity = self._get_limbs(self.keypoints)
        kpts = all_keypoints[..., :-1][0][None, ...]
        visibility = all_keypoints[..., -1:][0][None, ...]
        drawed = draw_keypoints(
            img,
            keypoints=kpts,
            visibility=visibility,
            connectivity=connectivity,
            radius=5,
            width=3,
            colors=(0, 255, 255)
        )
        # [3, H, W]: float -> [H, W, 3]: uint8
        return (drawed.permute(1, 2, 0).cpu() * 255).numpy().astype(np.uint8)
    
    def _get_limbs(self, keypoints: dict) -> list:
        limbs = [
            [keypoints.index("right_eye"), keypoints.index("nose")],
            [keypoints.index("right_eye"), keypoints.index("right_ear")],
            [keypoints.index("left_eye"), keypoints.index("nose")],
            [keypoints.index("left_eye"), keypoints.index("left_ear")],
            [keypoints.index("right_shoulder"), keypoints.index("right_elbow")],
            [keypoints.index("right_elbow"), keypoints.index("right_wrist")],
            [keypoints.index("left_shoulder"), keypoints.index("left_elbow")],
            [keypoints.index("left_elbow"), keypoints.index("left_wrist")],
            [keypoints.index("right_hip"), keypoints.index("right_knee")],
            [keypoints.index("right_knee"), keypoints.index("right_ankle")],
            [keypoints.index("left_hip"), keypoints.index("left_knee")],
            [keypoints.index("left_knee"), keypoints.index("left_ankle")],
            [keypoints.index("right_shoulder"), keypoints.index("left_shoulder")],
            [keypoints.index("right_hip"), keypoints.index("left_hip")],
            [keypoints.index("right_shoulder"), keypoints.index("right_hip")],
            [keypoints.index("left_shoulder"), keypoints.index("left_hip")],
        ]
        return limbs
    
    def compare_poses(self, reference_path: str, actual_path: str, mode: str, frame_rate=1, batch_size=1):
        assert mode in ['video', 'image'], 'Wrong mode'
        preprocessor = self.preprocessors[mode]
        
        if not all(map(lambda path: isinstance(path, str), [reference_path, actual_path])):
            print('Path type should be string')
            
        else:
            try:
                reference_dl = preprocessor(reference_path, frame_rate, batch_size)
                actual_dl = preprocessor(actual_path, frame_rate, batch_size)
            except:
                print('Wrong path')
            else:
                metric = self._compare(reference_dl, actual_dl)
                return metric
            