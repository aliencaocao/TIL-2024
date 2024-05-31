# Co-DETR w/ ViT-Large

## How to use

1. Build backbone

Please move file `vit.py` to `mmdet/models/backbones/vit.py` and import this backbone in `mmdet/models/backbones/__init__.py`.

```
from .csp_darknet import CSPDarknet
from .darknet import Darknet
from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .efficientnet import EfficientNet
from .hourglass import HourglassNet
from .hrnet import HRNet
from .mobilenet_v2 import MobileNetV2
from .pvt import PyramidVisionTransformer, PyramidVisionTransformerV2
from .regnet import RegNet
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .swin import SwinTransformer
from .trident_resnet import TridentResNet
from .vit import ViT

__all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet',
    'MobileNetV2', 'Res2Net', 'HourglassNet', 'DetectoRS_ResNet',
    'DetectoRS_ResNeXt', 'Darknet', 'ResNeSt', 'TridentResNet', 'CSPDarknet',
    'SwinTransformer', 'PyramidVisionTransformer',
    'PyramidVisionTransformerV2', 'EfficientNet', 'ViT'
]
```

2. Build neck

Please move file `sfp.py` to `mmdet/models/necks/sfp.py` and import this neck in `mmdet/models/necks/__init__.py`.

```
from .bfp import BFP
from .channel_mapper import ChannelMapper
from .ct_resnet_neck import CTResNetNeck
from .dilated_encoder import DilatedEncoder
from .dyhead import DyHead
from .fpg import FPG
from .fpn import FPN
from .fpn_carafe import FPN_CARAFE
from .hrfpn import HRFPN
from .nas_fpn import NASFPN
from .nasfcos_fpn import NASFCOS_FPN
from .pafpn import PAFPN
from .rfp import RFP
from .ssd_neck import SSDNeck
from .yolo_neck import YOLOV3Neck
from .yolox_pafpn import YOLOXPAFPN
from .sfp import SFP

__all__ = [
    'FPN', 'BFP', 'ChannelMapper', 'HRFPN', 'NASFPN', 'FPN_CARAFE', 'PAFPN',
    'NASFCOS_FPN', 'RFP', 'YOLOV3Neck', 'FPG', 'DilatedEncoder',
    'CTResNetNeck', 'SSDNeck', 'YOLOXPAFPN', 'DyHead', 'SFP'
]
```

3. Download weights

We provide the inference config `co_dino_5scale_vit_large_lvis.py`. The weights can be downloaded from [this link](https://drive.google.com/drive/folders/1PVGkQljIfc-lq-IujBl3ELJclf5JCiag?usp=share_link). The COCO model is released here: https://drive.google.com/drive/folders/1-vAVIHHJ6Gyw0E6mGdbjdZ1hjkEdo3Rt?usp=sharing

## Cite Co-DETR

If you find this model useful, please use the following BibTeX entry for citation.

```latex
@misc{codetr2022,
      title={DETRs with Collaborative Hybrid Assignments Training},
      author={Zhuofan Zong and Guanglu Song and Yu Liu},
      year={2022},
      eprint={2211.12860},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
