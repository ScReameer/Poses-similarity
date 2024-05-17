from .coach_modules.processing import ImagePreprocessor, VideoPreprocessor
from .coach_modules.comparator import Comparator


class VirtualCoach:
    def __init__(self, oks_threshold=0.7) -> None:
        # Two base classes for preprocess images and videos
        self.preprocessors = {
            'image': ImagePreprocessor(),
            'video': VideoPreprocessor()
        }
        self.oks_threshold = oks_threshold
        self.comparator = Comparator(oks_threshold=self.oks_threshold)
    
    def compare_poses(
        self,
        reference_path: str, 
        actual_path: str, 
        mode: str,
        name='result',
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
                metrics = self.comparator.compare(reference_dl, actual_dl, mode, name)
                print(f'Success! The results are saved to a file "{name+extension[mode]}"')
                return metrics
            