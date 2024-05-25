# Copyright 2024 The Scenic Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=line-too-long
r"""Default configs for COCO detection using CenterNet.

"""
# pylint: enable=line-too-long

import ml_collections


def get_config():
  """get config."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'centernet2_CXT_LSJ_4x'

  # Dataset.
  config.dataset_name = 'coco_centernet_detection'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.prefetch_to_device = 2
  config.dataset_configs.shuffle_buffer_size = 10_000
  config.dataset_configs.max_boxes = 100
  config.dataset_configs.scale_range = (0.1, 2.0)
  config.dataset_configs.crop_size = 1024
  config.dataset_configs.size_divisibility = 32
  config.dataset_configs.remove_crowd = True
  config.data_dtype_str = 'float32'

  config.rng_seed = 0

  # Model.
  config.model = ml_collections.ConfigDict()
  config.model.model_dtype_str = 'float32'
  config.model.model_name = 'centernet2'
  config.model.backbone_name = 'convnext'
  config.model.num_classes = -1  # classification head for the proposal network
  config.model.strides = (8, 16, 32, 64, 128)
  config.model.pixel_mean = (103.530, 116.280, 123.675)
  config.model.pixel_std = (57.375, 57.120, 58.395)
  config.model.backbone_args = ml_collections.ConfigDict()
  config.model.backbone_args.size = 'T'
  config.model.backbone_args.drop_path_rate = 0.3
  config.model.freeze_model_state = False

  # CenterNet2 parameters
  config.model.hm_weight = 0.5
  config.model.reg_weight = 1.0
  config.model.score_thresh = 0.05
  config.model.pre_nms_topk_train = 2000
  config.model.post_nms_topk_train = 1000
  config.model.pre_nms_topk_test = 1000
  config.model.post_nms_topk_test = 256
  config.model.iou_thresh = 0.9
  config.model.roi_matching_threshold = (0.6, 0.7, 0.8)
  config.model.roi_nms_threshold = 0.7

  # optimizer
  config.optimizer = ml_collections.ConfigDict()
  config.optimizer.optimizer = 'adamw'
  config.optimizer.weight_decay = 0.05
  config.optimizer.skip_scale_and_bias_regularization = True

  # Training.
  config.batch_size = 64
  config.num_training_steps = 90000
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.steps_per_cycle = config.num_training_steps
  config.lr_configs.warmup_steps = 500
  config.lr_configs.base_learning_rate = 0.0002

  # Pretrained_backbone.
  config.weights = '/path/to/convnext_tiny_in22k/'
  config.load_prefix = 'backbone/bottom_up/'
  config.checkpoint_steps = 5000
  config.log_eval_steps = 2500

  # Logging.
  config.write_summary = True
  config.xprof = True  # Profile using xprof.
  config.log_summary_steps = 20  # train summary steps
  config.log_large_summary_steps = 1000  # Expensive summary operations freq
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.

  return config
