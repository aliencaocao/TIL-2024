# OBSS SAHI Tool
# Code written by AnNT, 2023.

import logging
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.utils.cv import get_coco_segmentation_from_bool_mask
from sahi.utils.import_utils import check_requirements


class Yolov8DetectionModel(DetectionModel):
    def check_dependencies(self) -> None:
        check_requirements(["ultralytics"])

    def load_model(self, **kwargs):
        """
        Detection model is initialized and set to self.model.
        """

        from ultralytics import YOLO

        self.is_trt = '.engine' in self.model_path
        self.cfg = kwargs.get('cfg', None)  # For TensorRT only
        if self.is_trt and self.cfg is None:
            raise ValueError("cfg is required for TensorRT model")

        try:
            if not self.is_trt:
                model = YOLO(self.model_path)
                model.to(self.device)
            else:
                model = YOLO(self.model_path, task=self.cfg['task'])
                if self.cfg.get('standard_pred_image_size', None) is not None and self.cfg.get('standard_pred_model_path', None) is None:
                    print('WARNING: you provided standard pred image size but not a path to the TRT engine. Unless your TRT engine is build with dynamic inputs, you should provide another TRT engine with the correct input shape.')
                self.standard_pred_model = YOLO(self.cfg.get('standard_pred_model_path', self.model_path), task=self.cfg['task'])  # for full res
            self.set_model(model)
        except Exception as e:
            raise TypeError("model_path is not a valid yolov8 model path: ", e)

    def set_model(self, model: Any):
        """
        Sets the underlying YOLOv8 model.
        Args:
            model: Any
                A YOLOv8 model
        """

        self.model = model
        # set category_mapping
        if not self.category_mapping:
            if not self.is_trt:
                category_mapping = {str(ind): category_name for ind, category_name in enumerate(self.category_names)}
                self.category_mapping = category_mapping
            else:
                self.category_mapping = self.cfg["names"]

    def perform_inference(self, image: np.ndarray, num_batch: int = 1):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        If predictions have masks, each prediction is a tuple like (boxes, masks).
        Args:
            image: np.ndarray or list of np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.

        """

        from ultralytics.engine.results import Masks

        # Confirm model is loaded
        if self.model is None:
            raise ValueError("Model is not loaded, load it by calling .load_model()")

        kwargs = {"cfg": self.config_path, "verbose": False, "conf": self.confidence_threshold, "device": self.device}

        if self.is_trt:
            kwargs = {"imgsz": self.cfg["imgsz"], "half": self.cfg['half'], **kwargs}
        else:
            if self.standard_pred_image_size is not None and num_batch == 1:
                kwargs = {"imgsz": self.standard_pred_image_size, **kwargs}
            elif self.image_size is not None:
                kwargs = {"imgsz": self.image_size, **kwargs}

        if num_batch == 1:
            if self.is_trt:
                kwargs["imgsz"] = self.cfg["standard_pred_image_size"]
                prediction_result = self.standard_pred_model(image[:, :, ::-1], **kwargs)  # Special TRT engine for full res shape input which is always bs=1
            else:
                prediction_result = self.model(image[:, :, ::-1], **kwargs)  # YOLOv8 expects numpy arrays to have BGR
        else:
            prediction_result = self.model([img[:, :, ::-1] for img in image], **kwargs)  # YOLOv8 expects numpy arrays to have BGR


        if self.has_mask:
            if not prediction_result[0].masks:
                prediction_result[0].masks = Masks(
                    torch.tensor([], device=self.model.device), prediction_result[0].boxes.orig_shape
                )

            # We do not filter results again as confidence threshold is already applied above
            prediction_result = [
                (
                    result.boxes.data,
                    result.masks.data,
                )
                for result in prediction_result
            ]

        else:  # If model doesn't do segmentation then no need to check masks
            # We do not filter results again as confidence threshold is already applied above
            prediction_result = [result.boxes.data for result in prediction_result]

        self._original_predictions = prediction_result
        self._original_shape = image[0].shape if isinstance(image, list) else image.shape

    @property
    def category_names(self):
        return self.model.names.values() if not self.is_trt else self.cfg["names"]

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        return len(self.model.names) if not self.is_trt else len(self.cfg["names"])

    @property
    def has_mask(self):
        """
        Returns if model output contains segmentation mask
        """
        return self.model.overrides["task"] == "segment"

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions

        # compatilibty for sahi v0.8.15
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)
        # handle all predictions
        object_prediction_list_per_image = []
        for image_ind, image_predictions in enumerate(original_predictions):
            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[0]
            object_prediction_list = []
            if self.has_mask:
                image_predictions_in_xyxy_format = image_predictions[0]
                image_predictions_masks = image_predictions[1]
                for prediction, bool_mask in zip(
                    image_predictions_in_xyxy_format.cpu().detach().numpy(),
                    image_predictions_masks.cpu().detach().numpy(),
                ):
                    x1 = prediction[0]
                    y1 = prediction[1]
                    x2 = prediction[2]
                    y2 = prediction[3]
                    bbox = [x1, y1, x2, y2]
                    score = prediction[4]
                    category_id = int(prediction[5])
                    category_name = self.category_mapping[str(category_id)]

                    orig_width = self._original_shape[1]
                    orig_height = self._original_shape[0]
                    bool_mask = cv2.resize(bool_mask.astype(np.uint8), (orig_width, orig_height))
                    segmentation = get_coco_segmentation_from_bool_mask(bool_mask)
                    if len(segmentation) == 0:
                        continue
                    # fix negative box coords
                    bbox[0] = max(0, bbox[0])
                    bbox[1] = max(0, bbox[1])
                    bbox[2] = max(0, bbox[2])
                    bbox[3] = max(0, bbox[3])

                    # fix out of image box coords
                    if full_shape is not None:
                        bbox[0] = min(full_shape[1], bbox[0])
                        bbox[1] = min(full_shape[0], bbox[1])
                        bbox[2] = min(full_shape[1], bbox[2])
                        bbox[3] = min(full_shape[0], bbox[3])

                    # ignore invalid predictions
                    if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                        logger.warning(f"ignoring invalid prediction with bbox: {bbox}")
                        continue
                    object_prediction = ObjectPrediction(
                        bbox=bbox,
                        category_id=category_id,
                        score=score,
                        segmentation=segmentation,
                        category_name=category_name,
                        shift_amount=shift_amount,
                        full_shape=full_shape,
                    )
                    object_prediction_list.append(object_prediction)
                object_prediction_list_per_image.append(object_prediction_list)
            else:  # Only bounding boxes
                # process predictions
                for prediction in image_predictions.data.cpu().detach().numpy():
                    x1 = prediction[0]
                    y1 = prediction[1]
                    x2 = prediction[2]
                    y2 = prediction[3]
                    bbox = [x1, y1, x2, y2]
                    score = prediction[4]
                    category_id = int(prediction[5])
                    category_name = self.category_mapping[str(category_id)]

                    # fix negative box coords
                    bbox[0] = max(0, bbox[0])
                    bbox[1] = max(0, bbox[1])
                    bbox[2] = max(0, bbox[2])
                    bbox[3] = max(0, bbox[3])

                    # fix out of image box coords
                    if full_shape is not None:
                        bbox[0] = min(full_shape[1], bbox[0])
                        bbox[1] = min(full_shape[0], bbox[1])
                        bbox[2] = min(full_shape[1], bbox[2])
                        bbox[3] = min(full_shape[0], bbox[3])

                    # ignore invalid predictions
                    if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                        logger.warning(f"ignoring invalid prediction with bbox: {bbox}")
                        continue

                    object_prediction = ObjectPrediction(
                        bbox=bbox,
                        category_id=category_id,
                        score=score,
                        segmentation=None,
                        category_name=category_name,
                        shift_amount=shift_amount,
                        full_shape=full_shape,
                    )
                    object_prediction_list.append(object_prediction)
                object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image
