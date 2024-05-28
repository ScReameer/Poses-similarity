import numpy as np
import torch
import cv2 as cv
from torchvision.utils import draw_keypoints

class Writer:
    def __init__(self, oks_threshold: float) -> None:
        """Class for saving the result of comparing poses as a file (video or image)

        Args:
            `oks_threshold` (`float`): the comparison threshold based on the OKS metric
        """
        self.KEYPOINTS = [
            'nose','left_eye','right_eye',
            'left_ear','right_ear','left_shoulder',
            'right_shoulder','left_elbow','right_elbow',
            'left_wrist','right_wrist','left_hip',
            'right_hip','left_knee', 'right_knee',
            'left_ankle','right_ankle'
        ]
        self.oks_threshold = oks_threshold

    def _get_joint_connections(self, keypoints: list) -> list:
        """Auxiliary function for getting connections between keypoints

        Args:
            `keypoints` (`list`): list of key points

        Returns:
            `limbs`(`list`): list of connections between points by indexes
        """
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
        frame: torch.Tensor,
        all_keypoints: torch.Tensor
    ) -> np.ndarray:
        """Draws a skeleton on a frame

        Args:
            `frame` (`torch.Tensor`): single image tensor with shape `[3, H, W]` and `float` type 
            `all_keypoints` (`torch.Tensor`): keypoints tensor with shape `[B, 17, 3]`

        Returns:
            `result` (`np.ndarray`): frame with a drawn skeleton with shape `[H, W, 3]` and `uint8` type 
        """
        joint_connections = self._get_joint_connections(self.KEYPOINTS)
        keypoints = all_keypoints[..., :-1][0][None, ...] # [1, 17, 2]
        visibility = all_keypoints[..., -1:][0][None, ...] # [1, 17, 1]
        result = draw_keypoints(
            frame,
            keypoints=keypoints,
            visibility=visibility,
            connectivity=joint_connections,
            radius=5,
            width=3,
            colors=(0, 255, 255)
        )
        # [3, H, W]: float -> [H, W, 3]: uint8
        result = (result.permute(1, 2, 0).cpu() * 255).numpy().astype(np.uint8)    
        return result

    def _combine(
        self, 
        reference_frame: np.ndarray,
        actual_frame: np.ndarray,
        metrics: dict
    ) -> np.ndarray:
        """Combines the reference and actual frames in width into one, and also adds a display of the values of the metrics used

        Args:
            `reference_frame` (`np.ndarray`): reference frame with shape `[H, W, 3]` and `uint8` type 
            `actual_frame` (`np.ndarray`): actual frame with shape `[H, W, 3]` and `uint8` type 
            `metrics` (`dict`): dictionary of metrics per batch

        Returns:
            `combined_frame` (`np.ndarray`): combined frame with shape `[H, W, 3]` and `uint8` type 
        """
        # Resize actual frame to the reference frame size
        actual_frame_resized = cv.resize(
            actual_frame,
            dsize=reference_frame.shape[:-1][::-1],
            interpolation=cv.INTER_LINEAR
        )
        # Stack frames by width
        combined_frame = np.hstack((reference_frame[..., ::-1], actual_frame_resized[..., ::-1]))
        if metrics['OKS'] >= self.oks_threshold:
            text_color = (0, 255, 0) # green 
            success_text = 'Good matching'
        else:
            text_color = (0, 0, 255) # red (BGR format for OpenCV)
            success_text = 'Bad matching'
        outline_color = (0, 0, 0) # black
        # 'metric_name: metric_value ...'
        metrics_text = ' '.join([f'{k}={v:.2f}' for k, v in metrics.items()])
        metrics_position = (0, combined_frame.shape[0] - 10) # 10px up from left down corner
        success_position = (0, combined_frame.shape[0] - 50) # 50px up from left down corner
        # Add colored text with outline
        for text_to_add, position in zip(
            [success_text, metrics_text], 
            [success_position, metrics_position]
        ):
            # Add text outline
            cv.putText(
                combined_frame,
                text=text_to_add,
                org=position,
                color=outline_color,
                fontFace=cv.FONT_HERSHEY_COMPLEX,
                fontScale=1,
                thickness=8
            )
            # Add colored text
            cv.putText(
                combined_frame,
                text=text_to_add,
                org=position,
                color=text_color,
                fontFace=cv.FONT_HERSHEY_COMPLEX,
                fontScale=1,
                thickness=2
            )
        return combined_frame
    
    def write(
        self, 
        reference_batch: torch.Tensor,
        actual_batch: torch.Tensor,
        nn_output: dict,
        metrics: dict,
        video_writer: cv.VideoWriter,
        name: str
    ) -> None:
        """Writes the result of comparing two poses to a file (video or image)

        Args:
            `reference_batch` (`torch.Tensor`): single batch from reference dataloader
            `actual_batch` (`torch.Tensor`): single batch from actual dataloader
            `nn_output` (`dict`): output from keypoint model (forward pass) for used reference and actual batch
            `metrics` (`dict`): dictionary of calculated metrics for used reference and actual batch
            `video_writer` (`cv.VideoWriter` or `None`): `None` -> result output file will be image, otherwise video
            `name` (`str`): name of output file
        """
        min_batch_size = min(map(len, [reference_batch, actual_batch]))
        for batch_idx in range(min_batch_size):
            # Take frames from batches
            reference_frame = reference_batch[batch_idx].cpu()
            actual_frame = actual_batch[batch_idx].cpu()
            # Draw skeleton on frames
            reference_frame_skeleton = self._draw_skeleton(
                frame=reference_frame,
                all_keypoints=nn_output['reference_output'][batch_idx]['keypoints'],
            )
            actual_frame_skeleton = self._draw_skeleton(
                frame=actual_frame,
                all_keypoints=nn_output['actual_output'][batch_idx]['keypoints'],
            )
            # Draw metrics and merge 2 frames into one by width
            combined_frames = self._combine(
                reference_frame=reference_frame_skeleton, 
                actual_frame=actual_frame_skeleton,
                metrics=metrics
            )
            if video_writer: # video mode
                video_writer.write(combined_frames)
            else: # image mode
                cv.imwrite(f'{name}.png', combined_frames)