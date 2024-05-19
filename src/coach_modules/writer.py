import numpy as np
import torch
import cv2 as cv
from torchvision.utils import draw_keypoints

class Writer:
    def __init__(self, oks_threshold) -> None:
        self.keypoints = [
            'nose','left_eye','right_eye',
            'left_ear','right_ear','left_shoulder',
            'right_shoulder','left_elbow','right_elbow',
            'left_wrist','right_wrist','left_hip',
            'right_hip','left_knee', 'right_knee',
            'left_ankle','right_ankle'
        ]
        self.oks_threshold = oks_threshold

    def _combine(
        self, 
        ref_img: np.ndarray,
        act_img: np.ndarray,
        metrics: tuple
    ) -> np.ndarray:
        # Resize actual frame to the reference frame size
        act_img_resized = cv.resize(
            act_img,
            dsize=ref_img.shape[:-1][::-1],
            interpolation=cv.INTER_LINEAR
        )
        # Stack images by width
        combined_img = np.hstack((ref_img[..., ::-1], act_img_resized[..., ::-1]))
        if metrics[0] >= self.oks_threshold:
            text_color = (0, 255, 0) # green
            success_text = 'Good'
        else:
            text_color = (0, 0, 255) # red
            success_text = 'Bad'
        outline_color = (0, 0, 0)
        oks, rmse, cossim, wd = metrics
        metrics_text = f'OKS={oks}, RMSE={rmse}, CosSim={cossim}, WD={wd}'
        metrics_position = (0, combined_img.shape[0] - 10) # 10px up from left down corner
        success_position = (0, combined_img.shape[0] - 50) # 50px up from left down corner
        # Add colored text with outline
        for text_to_add, position in zip(
            [success_text, metrics_text], 
            [success_position, metrics_position]
        ):
            # Add text outline
            cv.putText(
                combined_img,
                text=text_to_add,
                org=position,
                color=outline_color,
                fontFace=cv.FONT_HERSHEY_COMPLEX,
                fontScale=1,
                thickness=8
            )
            # Add colored text
            cv.putText(
                combined_img,
                text=text_to_add,
                org=position,
                color=text_color,
                fontFace=cv.FONT_HERSHEY_COMPLEX,
                fontScale=1,
                thickness=2
            )
        return combined_img
        
    def _get_joint_connections(self, keypoints: dict) -> list:
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
        
    def _draw_skeleton(
        self, 
        img: torch.Tensor,
        all_keypoints: torch.Tensor
    ) -> np.ndarray:
        
        joint_connections = self._get_joint_connections(self.keypoints)
        kpts = all_keypoints[..., :-1][0][None, ...] # [1, 17, 2]
        visibility = all_keypoints[..., -1:][0][None, ...] # [1, 17, 1]
        drawed = draw_keypoints(
            img,
            keypoints=kpts,
            visibility=visibility,
            connectivity=joint_connections,
            radius=5,
            width=3,
            colors=(0, 255, 255)
        )
        # [3, H, W]: float -> [H, W, 3]: uint8
        return (drawed.permute(1, 2, 0).cpu() * 255).numpy().astype(np.uint8)    
    
    def write(
        self, 
        ref_batch: torch.Tensor,
        actual_batch: torch.Tensor,
        nn_output: dict,
        metrics: list,
        video_writer: cv.VideoWriter,
        name: str
    ) -> None:
        # Round metrics to 2nd decimal
        metrics = list(map(lambda x: round(x.cpu().item(), 2), metrics))
        min_batch_size = min(len(ref_batch), len(actual_batch))
        for batch_idx in range(min_batch_size):
            # Take frames from batches
            ref_img = ref_batch[batch_idx].cpu()
            act_img = actual_batch[batch_idx].cpu()
            # Draw skeleton on frames
            ref_img_skeleton = self._draw_skeleton(
                img=ref_img,
                all_keypoints=nn_output['reference_output'][batch_idx]['keypoints'],
            )
            act_img_skeleton = self._draw_skeleton(
                img=act_img,
                all_keypoints=nn_output['actual_output'][batch_idx]['keypoints'],
            )
            # Draw metrics and merge 2 frames into one by width
            combined_frames = self._combine(ref_img_skeleton, act_img_skeleton, metrics)
            if video_writer: # video mode
                video_writer.write(combined_frames)
            else: # image mode
                cv.imwrite(f'{name}.png', combined_frames)