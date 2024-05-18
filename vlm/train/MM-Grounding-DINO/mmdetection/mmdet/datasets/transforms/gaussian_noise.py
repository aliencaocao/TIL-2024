from typing import Optional

from .colorspace import ColorTransform
import numpy as np

from mmdet.registry import TRANSFORMS

@TRANSFORMS.register_module()
class GaussianNoise(ColorTransform):
    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.,
                 max_mag: float = 255.) -> None:
        assert 0 <= prob <= 1.0, f'The probability of the transformation ' \
                                 f'should be in range [0,1], got {prob}.'
        assert level is None or isinstance(level, int), \
            f'The level should be None or type int, got {type(level)}.'
        assert level is None or 0 <= level <= _MAX_LEVEL, \
            f'The level should be in range [0,{_MAX_LEVEL}], got {level}.'
        assert isinstance(min_mag, float), \
            f'min_mag should be type float, got {type(min_mag)}.'
        assert isinstance(max_mag, float), \
            f'max_mag should be type float, got {type(max_mag)}.'
        assert min_mag <= max_mag, \
            f'min_mag should smaller than max_mag, ' \
            f'got min_mag={min_mag} and max_mag={max_mag}'
        self.prob = prob
        self.level = level
        self.min_mag = min_mag
        self.max_mag = max_mag

    
    def _transform_img(self, results: dict, mag: float) -> None:
        img = results['img']
        img_dtype = results['img'].dtype
        
        img = img.astype(np.float32) + np.random.normal(0., mag, img.shape, dtype=np.float32)
        results['img'] = np.clip(img, 0., 255.).astype(img_dtype)
    