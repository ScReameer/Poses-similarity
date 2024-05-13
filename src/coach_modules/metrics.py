import torch
from torch.nn.functional import cosine_similarity
from torch.nn.functional import mse_loss

# Abstract class
class PoseSimilarity:
    def __init__(self) -> None:
        pass
    
    def __call__(self, reference_output:torch.Tensor, actual_output:torch.Tensor):
        threshold = 0.9
        reliable_detections_ref = []
        reliable_detections_actual = []
        reliable_detections_ref_boxes = []
        with torch.no_grad():
            for ref_frame, actual_frame in zip(reference_output, actual_output):
                # ref_frame_best_idxs = torch.nonzero(ref_frame['scores'] >= threshold).reshape(-1,)
                # act_frame_best_idxs = torch.nonzero(actual_frame['scores'] >= threshold).reshape(-1,)
                ref_frame_best_idxs = [0]
                act_frame_best_idxs = [0]
                if any(map(lambda x: len(x) == 0, [ref_frame['keypoints'], actual_frame['keypoints']])):
                    return None
                ref_kps = ref_frame['keypoints'][ref_frame_best_idxs]
                ref_boxes = ref_frame['boxes'][ref_frame_best_idxs]
                act_kps = actual_frame['keypoints'][act_frame_best_idxs]
                reliable_detections_ref.append(ref_kps)
                reliable_detections_actual.append(act_kps)
                reliable_detections_ref_boxes.append(ref_boxes)

            ref_output = torch.cat(reliable_detections_ref, dim=0)
            act_output = torch.cat(reliable_detections_actual, dim=0)
            box_output = torch.cat(reliable_detections_ref_boxes, dim=0)
            affine_output = self._affine_transform(ref_output, act_output, box_output)
            return self._compute(*affine_output)
            
    def _affine_transform(self, target_poses, source_poses, boxes):
        v_t, v_s = target_poses[..., -1:], source_poses[..., -1:]
        target_poses, source_poses = target_poses[..., :-1], source_poses[..., :-1]
        affine_matrix = torch.linalg.lstsq(source_poses, target_poses).solution
        transformed_poses = source_poses @ affine_matrix
        target_poses = torch.cat([target_poses, v_t], dim=-1)
        transformed_poses = torch.cat([transformed_poses, v_s], dim=-1)
        return target_poses, transformed_poses, boxes
    
    def _compute(self, reference_pose:torch.Tensor, actual_pose:torch.Tensor, boxes):
        pass

# Object keypoint similarity class
class ObjectKeypointSimilarity(PoseSimilarity):
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
    
    def _compute(self, reference_pose:torch.Tensor, actual_pose:torch.Tensor, boxes:torch.Tensor, eps=1e-7):
        with torch.no_grad():
            sigma = self.sigma.to(reference_pose.device)
            # area = torch.tensor([30000.], device=sigma.device)
            area = (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])
            d = (reference_pose[:, None, :, 0] - actual_pose[..., 0]).pow(2) + (reference_pose[:, None, :, 1] - actual_pose[..., 1]).pow(2)  # (N, M, 17)
            kpt_mask = reference_pose[..., 2] != 0  # (N, 17)
            e = d / ((2 * sigma).pow(2) * (area[:, None, None] + eps) * 2)  # from cocoeval
            # e = d / ((area[None, :, None] + eps) * sigma) ** 2 / 2  # from formula
            oks = ((-e).exp() * kpt_mask[:, None]).sum(-1) / (kpt_mask.sum(-1)[:, None] + eps)
            return oks.mean()
        
class CosineSimilarity(PoseSimilarity):
    def __init__(self) -> None:
        super().__init__()

    def _compute(self, reference_pose: torch.Tensor, actual_pose: torch.Tensor, boxes):
        with torch.no_grad():
            reference_pose, actual_pose = reference_pose[..., :-1], actual_pose[..., :-1]
            cossim = cosine_similarity(reference_pose, actual_pose)
            return cossim.mean()
        
class MSE(PoseSimilarity):
    def __init__(self) -> None:
        super().__init__()

    def _compute(self, reference_pose: torch.Tensor, actual_pose: torch.Tensor, boxes):
        with torch.no_grad():
            reference_pose, actual_pose = reference_pose[..., :-1], actual_pose[..., :-1]
            mse = mse_loss(reference_pose, actual_pose)
            return mse.mean()
# # Weighted distance class
# class WeightedDistance(PoseSimilarity):
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