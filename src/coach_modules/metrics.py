import torch
from torch.nn.functional import cosine_similarity, mse_loss, l1_loss

# Abstract class
class Metric:
    def __init__(self) -> None:
        pass
    
    def __call__(self, *nn_output, **kwds):
        prepared = self._prepare(*nn_output, **kwds)
        # Compute metric only if all detections (ref and actual) is not None
        if prepared:
            return self._compute(prepared)
        # Return None
        return prepared
    
    def _prepare(self, reference_output: list, actual_output: list) -> dict:
        reliable_detections_ref = []
        reliable_detections_ref_boxes = []
        reliable_detections_actual = []

        with torch.no_grad():
            for ref_frame, actual_frame in zip(reference_output, actual_output):
                frame_best_idx = [0]
                if any(map(lambda x: len(x) == 0, [ref_frame['keypoints'], actual_frame['keypoints']])):
                    return None
                ref_kps = ref_frame['keypoints'][frame_best_idx]
                ref_boxes = ref_frame['boxes'][frame_best_idx]
                act_kps = actual_frame['keypoints'][frame_best_idx]
                reliable_detections_ref.append(ref_kps)
                reliable_detections_actual.append(act_kps)
                reliable_detections_ref_boxes.append(ref_boxes)

            ref_output = torch.cat(reliable_detections_ref, dim=0)
            box_output = torch.cat(reliable_detections_ref_boxes, dim=0)
            act_output = torch.cat(reliable_detections_actual, dim=0)
            affine_output = self._affine_transform(ref_output, act_output)
            return {
                'reference': affine_output[0],
                'actual': affine_output[1],
                'boxes': box_output
            }
            
    def _affine_transform(self, reference_poses: torch.Tensor, actual_poses: torch.Tensor) -> tuple:
        ref_visibility, actual_visibility = reference_poses[..., -1:], actual_poses[..., -1:]
        reference_poses, actual_poses = self._drop_visibility(reference_poses, actual_poses)
        affine_matrix = torch.linalg.lstsq(actual_poses, reference_poses).solution
        transformed_actual_poses = actual_poses @ affine_matrix
        reference_poses = torch.cat([reference_poses, ref_visibility], dim=-1)
        transformed_actual_poses = torch.cat([transformed_actual_poses, actual_visibility], dim=-1)
        return reference_poses, transformed_actual_poses
    
    def _drop_visibility(self, reference_poses: torch.Tensor, actual_poses: torch.Tensor) -> tuple:
        return (
            reference_poses[..., :-1],
            actual_poses[..., :-1]
        )
    
    def _compute(self, prepared_poses: dict):
        pass

class ObjectKeypointSimilarity(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.sigma = (
            torch.tensor(
                [
                    0.26, 0.25, 0.25, 0.35,
                    0.35, 0.79, 0.79, 0.72,
                    0.72, 0.62, 0.62, 1.07,
                    1.07, 0.87, 0.87, 0.89,
                    0.89
                ]
            ) / 10.0
        )
    
    def _compute(self, prepared_poses: dict, eps=1e-7) -> torch.Tensor:
        reference_pose, actual_pose, boxes = prepared_poses.values()
        sigma = self.sigma.to(reference_pose.device)
        area = (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])
        d = (reference_pose[:, None, :, 0] - actual_pose[..., 0]).pow(2) + (reference_pose[:, None, :, 1] - actual_pose[..., 1]).pow(2)  # (N, M, 17)
        kpt_mask = reference_pose[..., 2] != 0  # (N, 17)
        e = d / ((2 * sigma).pow(2) * (area[:, None, None] + eps) * 2)  # from cocoeval
        # e = d / ((area[None, :, None] + eps) * sigma) ** 2 / 2  # from formula
        oks = ((-e).exp() * kpt_mask[:, None]).sum(-1) / (kpt_mask.sum(-1)[:, None] + eps)
        return oks.diag().mean()

class CosineSimilarity(Metric):
    def __init__(self) -> None:
        super().__init__()

    def _compute(self, prepared_poses) -> torch.Tensor:
        reference_pose, actual_pose = self._drop_visibility(prepared_poses['reference'], prepared_poses['actual'])
        cossim = cosine_similarity(reference_pose, actual_pose).mean()
        return cossim
        
class RMSE(Metric):
    def __init__(self) -> None:
        super().__init__()

    def _compute(self, prepared_poses) -> torch.Tensor:
        reference_pose, actual_pose = self._drop_visibility(prepared_poses['reference'], prepared_poses['actual'])
        rmse = torch.sqrt(mse_loss(reference_pose, actual_pose))
        return rmse
    
class MAE(Metric):
    def __init__(self) -> None:
        super().__init__()

    def _compute(self, prepared_poses) -> torch.Tensor:
        reference_pose, actual_pose = self._drop_visibility(prepared_poses['reference'], prepared_poses['actual'])
        mae = l1_loss(reference_pose, actual_pose)
        return mae
        
# # Weighted distance class
# class WeightedDistance(Metric):
#     def __init__(self) -> None:
#         super().__init__()
        
#     @staticmethod
#     def _compute(reference_pose:torch.Tensor, actual_pose:torch.Tensor, confidence:torch.Tensor):
#         sum1 = 1.0 / torch.sum(confidence)
#         sum2 = .0

#         for i in range(len(reference_pose)):
#             conf_ind = torch.floor(i / 2)
#             sum2 = confidence[conf_ind] * abs(reference_pose[i] - actual_pose[i])

#         weighted_dist = sum1 * sum2
#         return weighted_dist