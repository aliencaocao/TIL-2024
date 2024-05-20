import os
import sys

import orjson
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from tqdm import tqdm

sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from codet.config import add_codet_config
from codet.modeling.utils import reset_cls_infer
from codet.modeling.text.text_encoder import build_text_encoder
from detectron2.engine.defaults import DefaultPredictor

from PIL import ImageDraw, Image


class Args:
    def __init__(self, config_file, webcam, cpu, video_input, output, pred_all_class, confidence_threshold, opts):
        self.config_file = config_file
        self.webcam = webcam
        self.cpu = cpu
        self.video_input = video_input
        self.output = output
        self.pred_all_class = pred_all_class
        self.confidence_threshold = confidence_threshold
        self.opts = opts


def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE = "cpu"
    add_centernet_config(cfg)
    add_codet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'  # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


args = Args(
    config_file="configs/CoDet_OVLVIS_EVA_4x.yaml",
    webcam=None,
    cpu=False,
    video_input=None,
    output="output/",
    pred_all_class=False,
    confidence_threshold=0.1,
    opts=['MODEL.WEIGHTS', 'CoDet_OVLVIS_EVA_4x.pth'],
)
cfg = setup_cfg(args)
predictor = DefaultPredictor(cfg)
print(predictor)

text_encoder = build_text_encoder(pretrain=True)
text_encoder.eval()


def get_clip_embeddings(vocabulary, prompt='a '):
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb


with open('../../data/vlm.jsonl', 'r') as f:
    instances = [orjson.loads(line.strip()) for line in f if line.strip() != ""]

result = []

for instance in tqdm(instances):
    image = read_image(os.path.join('../../data/images/', instance['image']), format="BGR")
    captions = set(i['caption'] for i in instance['annotations'])
    annotations = []
    for cap in captions:
        thing_classes = [cap]
        num_classes = len(thing_classes)
        classifier = get_clip_embeddings(thing_classes)
        reset_cls_infer(predictor.model, classifier, num_classes)
        predictions = predictor(image)
        instances = predictions["instances"].to('cpu')
        bbox, score, class_id = instances.pred_boxes.tensor.tolist(), instances.scores.tolist(), instances.pred_classes.tolist()

        # save the highest conf box for each class id only
        bboxes = [(b, s) for b, s in zip(bbox, score)]
        if bboxes:
            highest_conf_pred = max(bboxes, key=lambda x: x[1])
            highest_conf_pred = (*highest_conf_pred, cap)
            annotations.append(highest_conf_pred)

    result.append({'image': instance['image'], 'annotations': [{'bbox': b, 'score': s, 'caption': c} for b, s, c in annotations]})
    # save every image in case of crash
    with open('codet_zeroshot.json', 'wb+') as f:
        f.write(orjson.dumps(result))

    image = image[:, :, ::-1]  # BGR -> RGB for matplotlib
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    for b, s, c in annotations:
        draw.rectangle(b, outline='red')
        draw.text((b[0], b[1]), f'{c}: {s:.2f}', fill='red')
    image.save(os.path.join('output/', instance['image']))
