from .coach_modules.processing import ImagePreprocessor, VideoPreprocessor
from .coach_modules.comparator import Comparator
from .coach_modules.metrics import RMSE, ObjectKeypointSimilarity, CosineSimilarity, WeightedDistance

# Used metrics
DEFAULT_METRICS_DICT = {
    'OKS': ObjectKeypointSimilarity(),
    'RMSE': RMSE(),
    'WD': WeightedDistance(),
    'CosSim': CosineSimilarity()
}

class VirtualCoach:
    def __init__(self, oks_threshold=0.7, metrics=DEFAULT_METRICS_DICT) -> None:
        assert 0 < oks_threshold <= 1, 'OKS threshold should be > 0 and <= 1'
        self.oks_threshold = oks_threshold
        # Two base classes for preprocess images and videos
        self.preprocessors = {
            'image': ImagePreprocessor(),
            'video': VideoPreprocessor()
        }
        self.metrics = metrics
        self.comparator = Comparator(oks_threshold=self.oks_threshold, metrics=self.metrics)
    
    def compare_poses(
        self,
        reference_path: str, 
        actual_path: str, 
        mode: str,
        name='result',
        fps=30,
        frame_skip=1, 
        batch_size=1
    ) -> dict:
        assert mode in ['video', 'image'], 'Wrong mode'
        preprocessor = self.preprocessors[mode]
        extension = {
            'video': '.avi',
            'image': '.png'
        }
        if not all(map(lambda path: isinstance(path, str), [reference_path, actual_path])):
            print('Path type should be string')
            
        else:
            try:
                reference_dl = preprocessor(reference_path, frame_skip, batch_size)
                actual_dl = preprocessor(actual_path, frame_skip, batch_size)
            except:
                print('Wrong path')
            else:
                metrics = self.comparator.compare(
                    reference_dl=reference_dl,
                    actual_dl=actual_dl,
                    mode=mode,
                    name=name,
                    fps=fps
                )
                print(f'Success! The results are saved to a file "{name+extension[mode]}"')
                return metrics
            