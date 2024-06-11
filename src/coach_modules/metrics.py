import torch
from torch.nn.functional import mse_loss, l1_loss, cosine_similarity

class Metric:
    def __init__(self) -> None:
        """Abstract metrics class"""
        pass
    
    def __call__(self, **nn_output: dict) -> torch.Tensor | None:
        """Combines the call of the `self._prepare_output` and `self._compute` functions

        Returns:
            `torch.Tensor` or `None`: the calculated metric, if the comparison attempt was successful.  
            `None` if the attempt was unsuccessful
        """
        with torch.no_grad():
            prepared_poses = self._prepare_output(**nn_output)
            # Compute metric only if all detections (reference and actual) is not None
            if prepared_poses:
                return self._compute(prepared_poses)
            # Return None
            return prepared_poses
    
    def _drop_visibility(self, reference_poses: torch.Tensor, actual_poses: torch.Tensor) -> tuple:
        """Returns the poses tensors with the visibility dimension removed

        Args:
            `reference_poses` (`torch.Tensor`): reference pose tensor `[B, 17, 3]`
            `actual_poses` (`torch.Tensor`): actual pose tensor `[B, 17, 3]`

        Returns:
            `tuple`: reference and actual poses respectively with deleted visibility dimension `[B, 17, 2]`
        """
        return (
            reference_poses[..., :-1],
            actual_poses[..., :-1]
        )
        
    def _affine_transform(self, reference_poses: torch.Tensor, actual_poses: torch.Tensor) -> torch.Tensor:
        """Computes an affine transformation of the actual pose to the reference pose so that they can be correctly compared

        Args:
            `reference_poses` (`torch.Tensor`): reference pose tensor `[B, 17, 3]`
            `actual_poses` (`torch.Tensor`): actual pose tensor `[B, 17, 3]`

        Returns:
            `transformed_actual_poses`: (`torch.Tensor`): the transformed tensor of actual poses of shape `[B, 17, 3]`
        """
        # Separate [x, y] points and [visibility] vector 
        actual_visibility = actual_poses[..., -1:]
        reference_poses, actual_poses = self._drop_visibility(reference_poses, actual_poses)
        # AX = B, where A - actual pose, X - affine matrix, B - reference pose
        affine_matrix = torch.linalg.lstsq(actual_poses, reference_poses).solution
        transformed_actual_poses = actual_poses @ affine_matrix
        # Return visibility tensor
        transformed_actual_poses = torch.cat([transformed_actual_poses, actual_visibility], dim=-1)
        return transformed_actual_poses
    
    def _prepare_output(self, reference_output: list, actual_output: list) -> dict:
        """Processes the output from keypoint model for the subsequent correct calculation of pose similarity metrics

        Args:
            `reference_output` (`list`): keypoint model output for single batch from reference dataloader
            `actual_output` (`list`): keypoint model output for single batch from actual dataloader

        Returns:
            `dict`: dictionary with prepared output as tensors
        """
        reliable_detections_reference = []
        reliable_detections_reference_boxes = []
        reliable_detections_actual = []
        reliable_kpts_scores = []
        # Iterate over every keypoint model output
        for reference_frame, actual_frame in zip(reference_output, actual_output):
            # Each frame can contain multiple detections,
            # the first one is the most likely, so we will take it for comparison 
            frame_best_idx = [0]
            # Skip iteration if there're at least one frame with no detections 
            if any(map(lambda x: len(x) == 0, [reference_frame['keypoints'], actual_frame['keypoints']])):
                return None
            reference_kpts = reference_frame['keypoints'][frame_best_idx] # [1, 17, 3]
            reference_boxes = reference_frame['boxes'][frame_best_idx] # [1, 4]
            actual_kpts = actual_frame['keypoints'][frame_best_idx] # [1, 17, 3]
            kpts_scores = reference_frame['keypoints_scores'][frame_best_idx] # [1, 17]
            # Accumulate reliable tensors
            reliable_kpts_scores.append(kpts_scores)
            reliable_detections_reference.append(reference_kpts)
            reliable_detections_actual.append(actual_kpts)
            reliable_detections_reference_boxes.append(reference_boxes)
        # Combine accumulated reliable tensors along first dimension (batch)
        reference_output = torch.cat(reliable_detections_reference, dim=0) # [N, 17, 3]
        box_output = torch.cat(reliable_detections_reference_boxes, dim=0) # [N, 4]
        actual_output = torch.cat(reliable_detections_actual, dim=0) # [N, 17, 3]
        confs_output = torch.cat(reliable_kpts_scores, dim=0) # [N, 17]
        # Actual poses -> affine transformed actual poses
        transformed_actual_poses = self._affine_transform(reference_output, actual_output)
        return {
            'reference': reference_output,
            'actual': transformed_actual_poses,
            'boxes': box_output,
            'kpts_scores': confs_output
        }

    def _compute(self, prepared_poses: dict):
        """Abstract metrics function

        Args:
            `prepared_poses` (`dict`): output from `self._prepare_output`
        """
        pass

class ObjectKeypointSimilarity(Metric):
    def __init__(self) -> None:
        """Object keypoints similarity class"""
        super().__init__()
        # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L14
        self.OKS_SIGMA = torch.tensor([
            0.26, 0.25, 0.25, 0.35,
            0.35, 0.79, 0.79, 0.72,
            0.72, 0.62, 0.62, 1.07,
            1.07, 0.87, 0.87, 0.89,
            0.89
        ]) / 10.0
        
    # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L153
    def _compute(self, prepared_poses: dict, eps=1e-7) -> torch.Tensor:
        """Computes OKS between 2 poses

        Args:
            `prepared_poses` (`dict`): output from `self._prepare_output` function
            `eps` (`float`, optional): variable for the stability of calculations. Defaults to `1e-7`.

        Returns:
            `torch.Tensor`: averaged OKS across batch size
        """
        reference_pose, actual_pose, boxes, _ = prepared_poses.values()
        sigma = self.OKS_SIGMA.to(reference_pose.device)
        area = (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])
        d = (reference_pose[:, None, :, 0] - actual_pose[..., 0]).pow(2) + (reference_pose[:, None, :, 1] - actual_pose[..., 1]).pow(2)  # (N, M, 17)
        kpt_mask = reference_pose[..., 2] != 0  # (N, 17)
        e = d / ((2 * sigma).pow(2) * (area[:, None, None] + eps) * 2)  # from cocoeval
        oks = ((-e).exp() * kpt_mask[:, None]).sum(-1) / (kpt_mask.sum(-1)[:, None] + eps)
        # Mean reduction by diagonal, 
        # because previous step returns all oks pairs in batch [BATCH_SIZE, BATCH_SIZE]
        return oks.diag().mean()

class CosineSimilarity(Metric):
    def __init__(self) -> None:
        """Cosine similarity class"""
        super().__init__()

    def _compute(self, prepared_poses: dict) -> torch.Tensor:
        """Computes cosine similarity between 2 poses

        Args:
            `prepared_poses` (`dict`): output from `self._prepare_output` function

        Returns:
            `torch.Tensor`: averaged cosine similarity across batch size
        """
        reference_pose, actual_pose = self._drop_visibility(prepared_poses['reference'], prepared_poses['actual'])
        cossim = cosine_similarity(reference_pose, actual_pose, dim=1)
        return cossim.mean()
        
class RMSE(Metric):
    def __init__(self) -> None:
        """Root mean squared error class"""
        super().__init__()

    def _compute(self, prepared_poses) -> torch.Tensor:
        """Computes RMSE between 2 poses

        Args:
            `prepared_poses` (`dict`): output from `self._prepare_output` function

        Returns:
            `torch.Tensor`: averaged RMSE across batch size
        """
        reference_pose, actual_pose = self._drop_visibility(prepared_poses['reference'], prepared_poses['actual'])
        rmse = torch.sqrt(mse_loss(reference_pose, actual_pose))
        return rmse
    
class MAE(Metric):
    def __init__(self) -> None:
        """Mean absolute error class"""
        super().__init__()

    def _compute(self, prepared_poses) -> torch.Tensor:
        """Computes MAE between 2 poses

        Args:
            `prepared_poses` (`dict`): output from `self._prepare_output` function

        Returns:
            `torch.Tensor`: averaged MAE across batch size
        """
        reference_pose, actual_pose = self._drop_visibility(prepared_poses['reference'], prepared_poses['actual'])
        mae = l1_loss(reference_pose, actual_pose)
        return mae
        
class WeightedDistance(Metric):
    def __init__(self) -> None:
        """Weighted distance class"""
        super().__init__()
     
    def _compute(self, prepared_poses):
        """Computes weighted distance between 2 poses

        Args:
            `prepared_poses` (`dict`): output from `self._prepare_output` function

        Returns:
            `torch.Tensor`: averaged weighted distance across batch size
        """
        # WD(pose1, pose2) = (1 / sum(conf1)) * sum(conf1 * ||pose1 - pose2||) = sum1 * sum2
        reference_pose, actual_pose = self._drop_visibility(prepared_poses['reference'], prepared_poses['actual'])
        confidence = prepared_poses['kpts_scores']
        sum1 = 1 / confidence.sum()
        # [B, 17] -> [B, 17, 2] with same weights for X and Y coordinates
        confidence = torch.cat([
            confidence[..., None],
            confidence[..., None]
        ], dim=-1)
        sum2 = torch.sum(confidence * torch.abs((reference_pose - actual_pose)))
        weighted_dist = sum1 * sum2
        return weighted_dist