import copy
import logging
from typing import List, Optional, Union

import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.layers import batched_nms

from . import mapper_utils

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapper_detr_instance"]


class DatasetMapper_detr_instance:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        augmentations_with_crop: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
        dataset_names: tuple = (),
        max_num_phrase: int = 0,
        nms_thresh_phrase: float = 0.0,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.augmentations_with_crop = T.AugmentationList(augmentations_with_crop)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations_with_crop}")

        self.dataset_names = dataset_names

        self.metatada_list = []
        for dataset_name in self.dataset_names:
            metadata = MetadataCatalog.get(dataset_name)
            self.metatada_list.append(metadata)

        self.max_num_phrase = max_num_phrase
        self.nms_thresh_phrase = nms_thresh_phrase

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        raise NotImplementedError(self.__class__.__name__)

    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        # USER: Modify this if you want to keep them for some reason.
        for anno in dataset_dict["annotations"]:
            if not self.use_instance_mask:
                anno.pop("segmentation", None)
            if not self.use_keypoint:
                anno.pop("keypoints", None)

        phrases = [
            obj.get("phrase", "")
            for obj in dataset_dict["annotations"]
            if obj.get("iscrowd", 0) == 0
        ]

        # USER: Implement additional transformations if you have other types of data
        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )

        if sum([len(x) for x in phrases]) > 0:
            instances.phrase_idxs = torch.tensor(range(len(phrases)))

        # After transforms such as cropping are applied, the bounding box may no longer
        # tightly bound the object. As an example, imagine a triangle object
        # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
        # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
        # the intersection of original bounding box and the cropping box.
        if self.recompute_boxes and instances.has("gt_masks"):
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

        if sum([len(x) for x in phrases]) > 0:
            phrases_filtered = []
            for x in dataset_dict["instances"].phrase_idxs.tolist():
                phrases_filtered.append(phrases[x])
            dataset_dict["instances"].phrases = mapper_utils.transform_phrases(
                phrases_filtered, transforms
            )
            dataset_dict["instances"].remove("phrase_idxs")
            # dataset_dict["instances"].gt_classes = torch.tensor(range(len(phrases_filtered)))

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        try:
            image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
            dataset_dict["width"] = image.shape[1]
            dataset_dict["height"] = image.shape[0]
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"read_image fails: {dataset_dict['file_name']}")
            logger.error(f"read_image fails: {e}")
            return None
        utils.check_image_size(dataset_dict, image)

        # ------------------------------------------------------------------------------------
        if (
            self.is_train
            and "annotations" in dataset_dict
            and (
                len(dataset_dict["annotations"]) == 0
                or any(["bbox" not in anno for anno in dataset_dict["annotations"]])
            )
        ):
            if "dataset_id" in dataset_dict:
                dataset_id = dataset_dict["dataset_id"]
            else:
                dataset_id = 0
            metadata = self.metatada_list[dataset_id]
            if "sa1b" in self.dataset_names[dataset_id]:
                metadata = None
            dataset_dict = mapper_utils.maybe_load_annotation_from_file(dataset_dict, meta=metadata)

            for anno in dataset_dict["annotations"]:
                if "bbox" not in anno:
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Box not found: {dataset_dict}")
                    return None
                if "category_id" not in anno:
                    anno["category_id"] = 0
        # ------------------------------------------------------------------------------------

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        # ordinal numbers
        disable_crop = False
        if (
            "annotations" in dataset_dict
            and len(dataset_dict["annotations"]) > 0
            and "phrase" in dataset_dict["annotations"][0]
        ):
            disable_crop = disable_crop or mapper_utils.has_ordinal_num(
                [anno["phrase"] for anno in dataset_dict["annotations"]]
            )
        if "expressions" in dataset_dict:
            disable_crop = disable_crop or mapper_utils.has_ordinal_num(dataset_dict["expressions"])

        if self.augmentations_with_crop is None or disable_crop:
            augmentations = self.augmentations
        else:
            if np.random.rand() > 0.5:
                augmentations = self.augmentations
            else:
                augmentations = self.augmentations_with_crop

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        # transforms = self.augmentations(aug_input)
        transforms = augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if "expressions" in dataset_dict:
            dataset_dict["expressions"] = mapper_utils.transform_expressions(
                dataset_dict["expressions"], transforms
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        if "instances" in dataset_dict and dataset_dict["instances"].has("phrases"):
            num_instances = len(dataset_dict["instances"])

            if self.nms_thresh_phrase > 0:
                boxes = dataset_dict["instances"].gt_boxes.tensor
                scores = torch.rand(num_instances)
                classes = torch.zeros(num_instances)
                keep = batched_nms(boxes, scores, classes, self.nms_thresh_phrase)
            else:
                keep = torch.randperm(num_instances)

            if self.max_num_phrase > 0:
                keep = keep[: self.max_num_phrase]

            phrases = dataset_dict["instances"].phrases
            phrases_filtered = []
            for x in keep:
                phrases_filtered.append(phrases[x])

            dataset_dict["instances"].remove("phrases")
            dataset_dict["instances"] = dataset_dict["instances"][keep]
            dataset_dict["instances"].phrases = phrases_filtered

        return dataset_dict
