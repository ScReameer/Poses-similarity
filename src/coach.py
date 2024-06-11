from .coach_modules.processing import ImagePreprocessor, VideoPreprocessor
from .coach_modules.comparator import Comparator
from .coach_modules.metrics import ObjectKeypointSimilarity, Metric

class VirtualCoach:
    def __init__(self, oks_threshold=0.7, aux_metrics=None) -> None:
        """A top-level class for comparing poses

        Args:
            `oks_threshold` (`float`, optional): the comparison threshold based on the OKS metric.  
            If the result is greater than or equal to the threshold, then the comparison is considered successful,  
            if less, then it is not considered successful. `0 < oks_threshold <= 1`. Defaults to `0.7`.
            `aux_metrics` (`dict`, optional): additional metrics for comparing poses.  
            In any case, at least one metric will always be used - Object Keypoint Similarity. Defaults to `None`.
        """
        # Arguments validation
        assert 0 < oks_threshold <= 1, 'OKS threshold should be > 0 and <= 1'
        if aux_metrics is not None:
            assert type(aux_metrics) is dict, 'Aux metrics should be dictionary of metrics {"name": metric}'
            for metric in aux_metrics.values():
                assert issubclass(metric.__class__, Metric), 'Aux metrics should be a subclass of Metric from src.coach_modules.metrics'
        self.oks_threshold = oks_threshold
        # Two base classes for preprocess images and videos
        self.preprocessors = {
            'image': ImagePreprocessor(),
            'video': VideoPreprocessor()
        }
        self.metrics = {'OKS': ObjectKeypointSimilarity()}
        if aux_metrics:
            self.metrics.update(aux_metrics)
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
        """Compares 2 images or 2 videos, saves the result to a file and returns metrics as a dictionary.

        Args:
            `reference_path` (`str`): path to reference image or video
            `actual_path` (`str`): path to actual image or video
            `mode` (`str`): `'image'` or `'video'` depending on what exactly is being used as an input
            `name` (`str`, optional): name for output video or image. Defaults to `result`
            `fps` (`int`, optional): frames per second for `'video'` mode. Defaults to `30`
            `frame_skip` (`int`, optional): affects the speed of the video, for example, 2 means speed 2x.  
            Does not affect the `'image'` mode. Defaults to `1`
            `batch_size` (`int`, optional): batch size for video mode, affects the averaging of metrics among frames.  
            Does not affect the `'image'` mode. Defaults to `1`

        Returns:
            `dict`: calculated metrics
        """
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
                print('Wrong path or file extension')
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
            