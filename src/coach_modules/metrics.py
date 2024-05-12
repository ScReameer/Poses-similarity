import torch

# Abstract class
class PoseSimilarity:
    def __init__(self) -> None:
        pass
    
    def __call__(self, reference_output:torch.Tensor, actual_output:torch.Tensor):
        # device = device
        threshold = 0.6

        reliable_detections_ref = []
        reliable_detections_actual = []
        for ref_frame, actual_frame in zip(reference_output, actual_output):
            for ref_score_idx, actual_score_idx in zip(range(len(ref_frame['scores'])), range(len(actual_frame['scores']))):
                if ref_frame['scores'][ref_score_idx] >= threshold:
                    reliable_detections_ref.append(ref_frame['keypoints'][ref_score_idx][None, ...])
                if actual_frame['scores'][actual_score_idx] >= threshold:
                    reliable_detections_actual.append(actual_frame['keypoints'][actual_score_idx][None, ...])
        with torch.no_grad():
            reference_pose_batch = torch.cat(reliable_detections_ref)
            actual_pose_batch = torch.cat(reliable_detections_actual)
            affine_output = self._affine_transform(reference_pose_batch, actual_pose_batch)
            return self._compute(*affine_output)
            return self._compute(reference_pose_batch, actual_pose_batch)
            
    
    def _affine_transform(self, reference_pose:torch.Tensor, actual_pose:torch.Tensor):
        padded_reference = self._pad(reference_pose)
        padded_actual = self._pad(actual_pose)
        affine_matrix = torch.linalg.lstsq(padded_actual, padded_reference).solution
        affine_matrix[torch.abs(affine_matrix) < 1e-10] = .0
        transformed_actual_pose = self._unpad(padded_actual @ affine_matrix)
        return (
            reference_pose,
            transformed_actual_pose
        )
    
    @staticmethod
    def _pad(inputs:torch.Tensor):
        return torch.cat([inputs, torch.ones((inputs.shape[0], 1, inputs.shape[-1]), device=inputs.device)], dim=1)
    
    @staticmethod
    def _unpad(inputs:torch.Tensor):
        return inputs[:, :-1, :]
    

    def _compute(self, reference_pose:torch.Tensor, actual_pose:torch.Tensor):
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
    
    def _compute(self, reference_pose:torch.Tensor, actual_pose:torch.Tensor, eps=1e-7):
        with torch.no_grad():
            sigma = self.sigma.to(reference_pose.device)
            area = reference_pose[..., 0].pow(2) + reference_pose[..., 1].pow(2)
            d = (reference_pose[:, None, :, 0] - actual_pose[..., 0]).pow(2) + (reference_pose[:, None, :, 1] - actual_pose[..., 1]).pow(2)  # (N, M, 17)
            kpt_mask = reference_pose[..., 2] != 0  # (N, 17)
            e = d / ((2 * sigma).pow(2) * (area[:, None, None] + eps) * 2)  # from cocoeval
            # e = d / ((area[None, :, None] + eps) * sigma) ** 2 / 2  # from formula
            oks = ((-e).exp() * kpt_mask[:, None]).sum(-1) / (kpt_mask.sum(-1)[:, None] + eps)
            return oks.mean()

# Weighted distance class
class WeightedDistance(PoseSimilarity):
    def __init__(self) -> None:
        super().__init__()
        
    @staticmethod
    def _compute(reference_pose:torch.Tensor, actual_pose:torch.Tensor, confidence:torch.Tensor):
        sum1 = 1.0 / torch.sum(confidence)
        sum2 = .0

        for i in range(len(reference_pose)):
            conf_ind = torch.floor(i / 2)
            sum2 = confidence[conf_ind] * abs(reference_pose[i] - actual_pose[i])

        weighted_dist = sum1 * sum2
        return weighted_dist