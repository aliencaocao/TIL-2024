import logging

logging.basicConfig(
  level=logging.WARNING,  # Adjust as needed (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# ------------------- LOAD DATA -------------------

from datasets import Dataset
import orjson
from PIL import Image
from tqdm import tqdm

def format_bbox(bbox, img_width, img_height):
  """
  Accepts bboxes in COCO format and returns them in YOLO format.
  Also clips bboxes such that they lie entirely within the image.
  """
  bbox[0] = max(0, bbox[0])
  bbox[1] = max(0, bbox[1])
  bbox[2] = min(img_width - bbox[0], bbox[2])
  bbox[3] = min(img_height - bbox[1], bbox[3])
  
  bbox[0] = (bbox[0] + bbox[2] / 2) / img_width
  bbox[1] = (bbox[1] + bbox[3] / 2) / img_height
  bbox[2] /= img_width
  bbox[3] /= img_height

  return bbox

def get_split(split):
  ds_list = []
  with open(f"data/{split}.jsonl") as ann_file:
    for i, line in tqdm(enumerate(ann_file)):
      # {"image": "image_0.jpg", "annotations": [{"caption": "grey missile", "bbox": [912, 164, 48, 152]}, {"caption": "red, white, and blue light aircraft", "bbox": [1032, 80, 24, 28]}, {"caption": "green and black missile", "bbox": [704, 508, 76, 64]}, {"caption": "white and red helicopter", "bbox": [524, 116, 112, 48]}]}
      x = orjson.loads(line)
      anns = x["annotations"]
      img = Image.open(f"data/{split}/{x['image']}")
      ds_list.append({
        "image_id": int(x["image"][6:-4]),
        "image": img,
        "width": img.width,
        "height": img.height,
        "objects": {
          "bbox": [format_bbox(ann["bbox"], img.width, img.height) for ann in anns],
          "caption": [ann["caption"] for ann in anns],
        },
      })

  return ds_list

train_ds = Dataset.from_list(get_split("train"))
val_ds = Dataset.from_list(get_split("val"))

# ------------------- TRANSFORMS -------------------

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
# from transformers import Owlv2Processor, Owlv2ForObjectDetection

checkpoint = "google/owlv2-large-patch14-ensemble"

model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint)
processor = AutoProcessor.from_pretrained(checkpoint)

import albumentations

augs = {
  "train": albumentations.Compose(
    transforms=[
      # How noisy will their test images be? var = 2500 is extreme!
      albumentations.GaussNoise(var_limit=2500, p=0.5),
      albumentations.HorizontalFlip(p=0.5),
      albumentations.VerticalFlip(p=0.5),
      albumentations.RandomBrightnessContrast(
        p=0.5,
        brightness_limit=0.3,
        contrast_limit=0.3,
      ),
      # MixUp / Mosaic?
    ],
    bbox_params=albumentations.BboxParams(format="yolo", label_fields=["captions"]),
  ),

  "val": albumentations.Compose(
    transforms=[albumentations.NoOp()],
    bbox_params=albumentations.BboxParams(format="yolo", label_fields=["captions"]),
  )
}

import numpy as np
import torch

def get_transform(split):
  def transform_batch(batch):
    # batch is a Dict[str, list], NOT a list!
    # perform augmentations HERE
    # but perform processing in collate_fn
    images_new = []
    objects_new = []

    for img, obj in zip(batch["image"], batch["objects"]):
      img_augmented = augs[split](
        image=np.asarray(img.convert("RGB")),
        bboxes=obj["bbox"],
        captions=obj["caption"],
      )
      images_new.append(img_augmented["image"])
      objects_new.append([{
        "bbox": bbox,
        "caption": caption,
      } for bbox, caption in zip(img_augmented["bboxes"], img_augmented["captions"])])

    batch["image"] = images_new
    batch["objects"] = objects_new
    return batch

  return transform_batch

def collate_fn(batch_list):
  # add dummy caption to end of samples with most queries
  # because processor forgets the no-object for those samples
  num_max_text_queries = max(len(x["objects"]) for x in batch_list)
  for x in batch_list:
    if len(x["objects"]) == num_max_text_queries:
      x["objects"].append({"bbox": [-1., -1., -1., -1.], "caption": ""})

  batch = processor(
    text=[[obj["caption"] for obj in x["objects"]] for x in batch_list],
    images=[x["image"] for x in batch_list],
    return_tensors="pt",
  )

  batch["labels"] = torch.nn.utils.rnn.pad_sequence(
    sequences=[torch.tensor([obj["bbox"] for obj in x["objects"]]) for x in batch_list],
    batch_first=True,
    padding_value=-1.,
  )

  for k in batch:
    print(f"Key {k}:")
    print(f"Shape:", batch[k].shape)
    print(f"dtype:", batch[k].dtype)

  return batch

train_ds = train_ds.with_transform(get_transform("train"))
val_ds = val_ds.with_transform(get_transform("val"))

# ------------------- CUSTOM LOSS -------------------

# Using Detr-Loss calculation https://github.com/facebookresearch/detr/blob/main/models/matcher.py
# https://www.kaggle.com/code/bibhasmondal96/detr-from-scratch
class BoxUtils(object):
    @staticmethod
    def box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)

    @staticmethod
    def box_xyxy_to_cxcywh(x):
        x0, y0, x1, y1 = x.unbind(-1)
        b = [(x0 + x1) / 2, (y0 + y1) / 2,
             (x1 - x0), (y1 - y0)]
        return torch.stack(b, dim=-1)

    @staticmethod
    def rescale_bboxes(out_bbox, size):
        img_h, img_w = size
        b = BoxUtils.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    @staticmethod
    def box_area(boxes):
        """
        Computes the area of a set of bounding boxes, which are specified by its
        (x1, y1, x2, y2) coordinates.
        Arguments:
            boxes (Tensor[N, 4]): boxes for which the area will be computed. They
                are expected to be in (x1, y1, x2, y2) format
        Returns:
            area (Tensor[N]): area for each box
        """
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
    @staticmethod
    # modified from torchvision to also return the union
    def box_iou(boxes1, boxes2):
        area1 = BoxUtils.box_area(boxes1)
        area2 = BoxUtils.box_area(boxes2)

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        union = area1[:, None] + area2 - inter

        iou = inter / union
        return iou, union

    @staticmethod
    def generalized_box_iou(boxes1, boxes2):
        """
        Generalized IoU from https://giou.stanford.edu/
        The boxes should be in [x0, y0, x1, y1] format
        Returns a [N, M] pairwise matrix, where N = len(boxes1)
        and M = len(boxes2)
        """
        # degenerate boxes gives inf / nan results
        # so do an early check
        assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
        assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
        iou, union = BoxUtils.box_iou(boxes1, boxes2)

        lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        area = wh[:, :, 0] * wh[:, :, 1]

        return iou - (area - union) / area

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        logging.info(f"{outputs.keys()=}")
        bs, num_queries = outputs["logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["class_labels"] for v in targets])
        logging.info(f"forward - {tgt_ids}")
        tgt_ids = tgt_ids.int()
        logging.info(f"forward - {tgt_ids}")

        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -BoxUtils.generalized_box_iou(
            BoxUtils.box_cxcywh_to_xyxy(out_bbox),
            BoxUtils.box_cxcywh_to_xyxy(tgt_bbox)
        )

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, num_non_dummy_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, INCLUDING all no-object categories
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_non_dummy_classes = num_non_dummy_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

        empty_weight = torch.ones(self.num_classes)
        empty_weight[num_non_dummy_classes:] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """

        logging.info(f"loss_labels - {outputs.keys()}")
        assert 'logits' in outputs
        src_logits = outputs['logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["class_labels"][J] for t, (_, J) in zip(targets, indices)]).to(
          device=src_logits.device,
          dtype=torch.int64,
        )
        target_classes = torch.full(src_logits.shape[:2], self.num_classes-1,
                                    dtype=torch.int64, device=src_logits.device).to(torch.int64)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["class_labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(BoxUtils.generalized_box_iou(
            BoxUtils.box_cxcywh_to_xyxy(src_boxes),
            BoxUtils.box_cxcywh_to_xyxy(target_boxes))
        )
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        logging.info(f"{type(outputs)=}")
        logging.info(f"{type(targets)=}")
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Matcher needs logits without no-object
        outputs_without_aux["logits"] = outputs_without_aux["logits"][..., :self.num_non_dummy_classes]
        # Also remove no-object indices from targets
        targets_actual = []
        for t in targets:
          targets_actual.append({
            "boxes": t["boxes"],
            "class_labels": t["class_labels"][:self.num_non_dummy_classes],
          })

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets_actual)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["class_labels"]) for t in targets)

        it = iter(outputs.values())
        next(it)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(it).device)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        return losses

# ------------------- TRAIN -------------------

from transformers import TrainingArguments, Trainer
from collections import OrderedDict

def custom_loss(outputs, labels_boxes):
  # discard dummy bboxes with [-1, -1, -1, -1]
  labels_boxes_actual = labels_boxes[labels_boxes[:, 0] >= 0]

  # need to set num_classes separately for each image
  # because each image has a different number of ground-truth labels  
  num_classes = labels_boxes.shape[0]
  targets = [{
    "boxes": labels_boxes_actual.float(),
    "class_labels": torch.tensor(range(num_classes)),
  }]
  
  matcher = HungarianMatcher(cost_class = 1, cost_bbox = 5, cost_giou = 2)
  weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
  losses = ['labels', 'boxes', 'cardinality']
  criterion = SetCriterion(
    num_classes=num_classes,
    num_non_dummy_classes=labels_boxes_actual.shape[0],
    matcher=matcher,
    weight_dict=weight_dict,
    eos_coef=0.1,
    losses=losses,
  )
  criterion.to(model.device)
  loss = criterion(outputs, targets)
  return loss

class CustomTrainer(Trainer):
  def compute_loss(self, model, inputs, return_outputs=False):
    batch_labels_boxes = inputs.pop("labels")

    # batch_labels_tokens = torch.chunk(inputs["input_ids"], batch_size)
    
    outputs = model(**inputs, return_dict=True)
    # outputs_processed = processor.post_process_object_detection(outputs)

    total_loss = 0.
    for i in range(batch_labels_boxes.shape[0]):
      # compute loss separately for each example in batch
      # since all have different num_classes (= number of text queries)
      try:
        curr_output = OrderedDict([
          (k, torch.unsqueeze(outputs[k][i], 0))
          for k in ("logits", "objectness_logits", "pred_boxes", "text_embeds", "image_embeds", "class_embeds")
        ])

        loss = custom_loss(curr_output, batch_labels_boxes[i])
        total_loss += sum(loss.values())[0]
      except Exception as ex:
        print(outputs[i])
        raise ex

    return (total_loss, outputs) if return_outputs else total_loss

training_args = TrainingArguments(
  output_dir="owlv2-large-patch14-ensemble",
  label_names=["labels"],
  eval_strategy="epoch",
  save_strategy="epoch",
  num_train_epochs=20,
  # Scale optimizer params by batch size? For now we follow the paper.
  optim="adamw_torch_fused",
  learning_rate=2e-5,
  lr_scheduler_type="linear",
  warmup_steps=10,
  per_device_train_batch_size=2,
  per_device_eval_batch_size=6,
  bf16=True,
  bf16_full_eval=True,
  tf32=True,
  torch_compile=True,
  dataloader_num_workers=8,
  gradient_accumulation_steps=16,
  gradient_checkpointing=False,
  remove_unused_columns=False,
)

# ------------------- MEMORY PROFILING -------------------

from transformers import TrainerCallback
import socket
from datetime import datetime

class ProfilerCallback(TrainerCallback):
  def __init__(self, profiler):
    self.profiler = profiler

  def on_step_end(self, args, state, control, **kwargs):
    self.profiler.step()

# Prefix for file names.
host_name = socket.gethostname()
timestamp = datetime.now().strftime("%b_%d_%H_%M_%S")
file_prefix = f"{host_name}_{timestamp}"

def trace_handler(profiler: torch.profiler.profile):
  # Construct the trace file.
  profiler.export_chrome_trace(f"{file_prefix}.json.gz")

  # Construct the memory timeline file.
  profiler.export_memory_timeline(f"{file_prefix}.html", device="cuda:3")

print("Parameter count:", sum(p.numel() for p in model.parameters()))

torch.cuda.memory._record_memory_history(
  max_entries=100000,
)

try:
  with torch.profiler.profile(
      activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
      ],
      schedule=torch.profiler.schedule(wait=0, warmup=2, active=6, repeat=1),
      record_shapes=True,
      profile_memory=True,
      with_stack=True,
      on_trace_ready=trace_handler,
  ) as profiler:
    trainer = CustomTrainer(
      model=model,
      args=training_args,
      train_dataset=train_ds,
      eval_dataset=val_ds,
      data_collator=collate_fn,
      tokenizer=processor,
      callbacks=[ProfilerCallback(profiler)],
    )

    trainer.train()
finally:
  try:
    torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
  except Exception as e:
    print(f"Failed to capture memory snapshot {e}")

  # Stop recording memory snapshot history.
  torch.cuda.memory._record_memory_history(enabled=None)
  
  # Construct the trace file.
  profiler.export_chrome_trace(f"{file_prefix}.json.gz")
  # Construct the memory timeline file.
  profiler.export_memory_timeline(f"{file_prefix}.html", device="cuda:3")
