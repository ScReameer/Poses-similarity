from .coach_modules.detector import KeypointDetector
from .coach_modules.metrics import  WeightedDistance, ObjectKeypointSimilarity
from .coach_modules.processing import ImagePreprocessor, VideoPreprocessor

class VirtualCoach:
    def __init__(self) -> None:
        self.preprocessors = {
            'image': ImagePreprocessor(),
            'video': VideoPreprocessor()
        }
        self.keypoint_detector = KeypointDetector()
        self.oks = ObjectKeypointSimilarity()
        # self.cosine_similarity = CosineSimilarity()
        # self.weighted_distance = WeightedDistance()
        
        
    def __call__(self, reference_path:str, actual_path:str, mode:str):
        assert mode in ['video', 'image'], 'Wrong mode'
        preprocessor = self.preprocessors[mode]
        # Single paths str -> list
        if not all(map(lambda path: isinstance(path, str), [reference_path, actual_path])):
            print('Path type should be string')
        else:
            reference_dl = preprocessor(reference_path)
            actual_dl = preprocessor(actual_path)
            # nn_output = self._predict(reference_dl, actual_dl)
            metric = self._compare(reference_dl, actual_dl)
            return metric
            
        
    def _predict(self, reference_batch, actual_batch):
        
        return (
            self.keypoint_detector(reference_batch),
            self.keypoint_detector(actual_batch)
        )
        
    def _compare(self, ref_dl, actual_dl):
        total_min_batches = min(map(len, [ref_dl, actual_dl]))
        metric_sum = .0
        for ref_batch, actual_batch in zip(ref_dl, actual_dl):
            nn_output = self._predict(ref_batch, actual_batch)
            oks = self.oks(*nn_output).cpu().item()
            metric_sum += oks
        return metric_sum / total_min_batches