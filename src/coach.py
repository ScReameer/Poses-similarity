import torch.utils
from .coach_modules.detector import KeypointDetector
from .coach_modules.metrics import  CosineSimilarity, ObjectKeypointSimilarity, MSE
from .coach_modules.processing import ImagePreprocessor, VideoPreprocessor
import plotly.express as px
import plotly.io as pio
import cv2 as cv
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import draw_keypoints
pio.renderers.default = 'png'
pio.templates.default = 'plotly_dark'

class VirtualCoach:
    def __init__(self) -> None:
        self.preprocessors = {
            'image': ImagePreprocessor(),
            'video': VideoPreprocessor()
        }
        self.keypoints = [
            'nose','left_eye','right_eye',
            'left_ear','right_ear','left_shoulder',
            'right_shoulder','left_elbow','right_elbow',
            'left_wrist','right_wrist','left_hip',
            'right_hip','left_knee', 'right_knee',
            'left_ankle','right_ankle'
        ]
        self.keypoint_detector = KeypointDetector()
        self.oks = ObjectKeypointSimilarity()
        self.cosine_similarity = CosineSimilarity()
        self.mse = MSE()
        
        
    def __call__(self, reference_path:str, actual_path:str, mode:str, frame_rate=1):
        assert mode in ['video', 'image'], 'Wrong mode'
        preprocessor = self.preprocessors[mode]
        preprocessor.frame_rate = frame_rate
        
        if not all(map(lambda path: isinstance(path, str), [reference_path, actual_path])):
            print('Path type should be string')
            
        else:
            reference_dl = preprocessor(reference_path)
            actual_dl = preprocessor(actual_path)
            metric = self._compare(reference_dl, actual_dl)
            return metric
        
    def _predict(self, reference_batch, actual_batch):
        
        return (
            self.keypoint_detector(reference_batch),
            self.keypoint_detector(actual_batch)
        )
        
    def _compare(self, ref_dl, actual_dl):
        total_min_batches = min(map(len, [ref_dl, actual_dl]))
        oks_sum = .0
        cossim_sum = .0
        mse_sum = .0
        
        for ref_batch, actual_batch in zip(ref_dl, actual_dl):
            nn_output = self._predict(ref_batch, actual_batch)
            
            oks = self.oks(*nn_output)
            cossim = self.cosine_similarity(*nn_output)
            mse = self.mse(*nn_output)
            if any(map(lambda x: x is None, [oks, cossim, mse])):
                continue
            self._draw_comparison(ref_batch, actual_batch, nn_output)
            oks_sum += oks.cpu().item()
            cossim_sum += cossim.cpu().item()
            mse_sum += mse.cpu().item()

        
        oks_sum /= total_min_batches
        cossim_sum /= total_min_batches
        mse_sum /= total_min_batches
        return {
            'OKS': oks_sum,
            'CosSim': cossim_sum,
            'MSE': mse_sum
        }
    
    def _draw_comparison(self, ref_batch:torch.Tensor, actual_batch:torch.Tensor, nn_output):
        ref_img = ref_batch[0].cpu()
        act_img = actual_batch[0].cpu()
        px.imshow(
            self._draw_skeleton(
                img=ref_img,
                all_keypoints=nn_output[0][0]['keypoints'],
            ),
            width=ref_img.shape[2],
            height=ref_img.shape[1]
        ).show()
        px.imshow(
            self._draw_skeleton(
                img=act_img,
                all_keypoints=nn_output[1][0]['keypoints'],
            ),
            width=act_img.shape[2],
            height=act_img.shape[1]
        ).show()
    
    def _draw_skeleton(self, img, all_keypoints):
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

        return (drawed.permute(1, 2, 0).cpu() * 255).numpy().astype(np.uint8)
    
    def _get_limbs(self, keypoints):
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