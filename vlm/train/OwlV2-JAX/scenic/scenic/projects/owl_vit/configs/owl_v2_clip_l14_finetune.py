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
r"""OWL v2 CLIP L/14 config."""
import ml_collections

CHECKPOINTS = {
    # https://arxiv.org/abs/2306.09683 Table 1 row 12:
    'owl2-l14-1008-st-ngrams': 'gs://scenic-bucket/owl_vit/checkpoints/owl2-l14-1008-st-ngrams_0881fd6',
    # https://arxiv.org/abs/2306.09683 Table 1 row 15:
    'owl2-l14-1008-st-ngrams-ft-lvisbase': 'gs://scenic-bucket/owl_vit/checkpoints/owl2-l14-1008-st-ngrams-ft-lvisbase_8ca674c',
    # https://arxiv.org/abs/2306.09683 Figure A1 weight ensemble:
    'owl2-l14-1008-st-ngrams-ft-lvisbase-ens-cold-weight-04': 'gs://scenic-bucket/owl_vit/checkpoints/owl2-l14-1008-st-ngrams-ft-lvisbase-ens-cold-weight-04_8ca674c',
}

CHECKPOINTS['canonical_checkpoint'] = CHECKPOINTS[
    'owl2-l14-1008-st-ngrams-ft-lvisbase-ens-cold-weight-04'
]

DETECTION_FEATURES = ('boxes', 'crowd', 'image', 'instance_labels',
                      'instance_text_labels', 'negative_labels',
                      'negative_text_labels', '_seed', 'seed')


def get_train_preproc_spec(
    *,
    input_size: int,
    num_instances: int,
    max_queries: int,
    max_query_length: int = 16,
    min_area_fraction: float = 0.0,
    iou_threshold: float = 0.9):
  """Constructs training preprocess string."""
  ops = (
      f'keep({DETECTION_FEATURES})'
      f'|random_flip_left_right'
      f'|random_crop(min_area_fraction={min_area_fraction})'
      f'|resize_with_pad(size={input_size})'
      f'|add_random_negative_labels(total_num_negatives=50)'
      f'|canonicalize_text_labels'
      f'|crop_or_pad({input_size}, {num_instances})'
      f'|crop_or_pad_meta_data({num_instances}, {num_instances})'
      f'|add_random_prompts'
      f'|remove_promptability_marker'
      f'|single_to_multi_label(max_num_labels={num_instances})'
      f'|merge_overlapping_instances(iou_threshold={iou_threshold})'
      f'|add_query_set(lower=True, max_queries={max_queries},'
      f' include_negatives=True)'
      f'|clip_tokenize_queries(max_token_len={max_query_length})')
  return ops


def get_eval_preproc_spec(
    *,
    input_size: int,
    num_instances: int,
    max_queries: int,
    max_query_length: int = 16,
    ):
  """Constructs training preprocess string."""
  return (
      f'resize_with_pad(size={input_size})'
      f'|canonicalize_text_labels'
      f'|crop_or_pad({input_size}, {num_instances})'
      f'|crop_or_pad_meta_data({num_instances}, {num_instances})'
      f'|single_to_multi_label(max_num_labels={num_instances})'
      f'|add_query_set(lower=True, max_queries={max_queries},'
      f' include_negatives=True)'
      f'|clip_tokenize_queries(max_token_len={max_query_length})')


def get_config(init_mode='canonical_checkpoint'):
  """Returns the configuration for text-query-based detection using OWL-ViT."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'til-owlv2-clip-l14'

  config.count_flops = False

  # Dataset.
  config.dataset_name = 'owl_vit'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.input_size = 1008
  config.dataset_configs.input_range = None
  config.dataset_configs.num_instances = 100
  config.dataset_configs.max_queries = 100
  config.dataset_configs.max_query_length = 16
  config.dataset_configs.min_area_fraction = 0.6
  config.dataset_configs.iou_threshold = 0.9
  config.dataset_configs.add_random_negatives = True
  config.dataset_configs.total_num_negatives = 50
  config.dataset_configs.prefetch_to_device = 2

  # For best performance, the shuffle buffer should be large, e.g. 10_000, but
  # this will require >50GB RAM.
  config.dataset_configs.shuffle_buffer_size = 10_000

  config.dataset_configs.train = ml_collections.ConfigDict()
  config.dataset_configs.train.tfds_names = ['til']
  config.dataset_configs.train.splits = ['train']
  config.dataset_configs.train.dataset_probs = [1.0]
  config.dataset_configs.train.decoder_kwarg_list = ({},)
  config.dataset_configs.train.preproc_spec = get_train_preproc_spec(
      input_size=config.dataset_configs.input_size,
      num_instances=config.dataset_configs.num_instances,
      max_queries=config.dataset_configs.num_instances,
      max_query_length=config.dataset_configs.max_query_length,
      min_area_fraction=config.dataset_configs.min_area_fraction)
  # When using mosaics, use an input_size that is divisible by all mosaic_sizes.
  config.dataset_configs.train.mosaic_sizes = (1, 2, 3)
  config.dataset_configs.train.mosaic_probs = (.4, .3, .3)

  config.dataset_configs.eval = ml_collections.ConfigDict()
  config.dataset_configs.eval.tfds_names = ['til']
  config.dataset_configs.eval.splits = ['validation']
  config.dataset_configs.eval.dataset_probs = [1.0]
  config.dataset_configs.eval.decoder_kwarg_list = ({},)
  config.dataset_configs.eval.preproc_spec = get_eval_preproc_spec(
      input_size=config.dataset_configs.input_size,
      num_instances=config.dataset_configs.num_instances,
      max_queries=config.dataset_configs.num_instances,
      max_query_length=config.dataset_configs.max_query_length)

  config.eval_top_k_preds = 512  # Only return the top-k predictions.
  config.data_dtype_str = 'float32'

  # Model.
  config.model_name = 'text_zero_shot_detection'

  config.matcher = 'hungarian_cover_tpu'

  config.model = ml_collections.ConfigDict()
  config.model.normalize = True

  config.model.body = ml_collections.ConfigDict()
  config.model.body.type = 'clip'
  config.model.body.variant = 'vit_l14'
  config.model.body.merge_class_token = 'mul-ln'
  config.model.box_bias = 'both'

  # CLIP stochastic depth.
  config.model.body.text_stochastic_droplayer_rate = 0.1
  config.model.body.vision_stochastic_droplayer_rate = 0.2

  # Loss.
  config.bbox_loss_coef = 1.0
  config.giou_loss_coef = 1.0
  config.class_loss_coef = 1.0
  config.focal_loss = True
  config.focal_gamma = 2.0
  config.focal_alpha = 0.3
  config.prior_prob = 0.01  # Prior prob of predicting not padding.
  config.normalization = 'per_example'  # 'per_example' or 'global'.

  # Training.
  config.num_training_steps = 10_000
  config.batch_size = 8
  config.rng_seed = 0

  # Image backbone + head training configuration.
  sched = ml_collections.ConfigDict()
  sched.re = '(?!backbone/clip/text/.*)(.*)'  # Negative lookahead.
  sched.lr_configs = ml_collections.ConfigDict({  # Learning rate.
      'learning_rate_schedule': 'compound',
      'factors': 'constant*rsqrt_decay',
      'steps_per_cycle': config.get_ref('num_training_steps'),
      'total_steps': config.get_ref('num_training_steps'),
      'warmup_steps': 0,  # Necessary for higher LR and large batch size.
      'base_learning_rate': 2e-5,
  })

  # Text backbone training configuration.
  sched_txt = ml_collections.ConfigDict()
  sched_txt.re = '(backbone/clip/text/.*)'
  sched_txt.lr_configs = ml_collections.ConfigDict({
      'learning_rate_schedule': 'compound',
      'factors': 'constant*rsqrt_decay',
      'steps_per_cycle': config.get_ref('num_training_steps'),
      'total_steps': config.get_ref('num_training_steps'),
      'warmup_steps': 0,  # Necessary for higher LR and large batch size.
      'base_learning_rate': 2e-6,
  })

  # Configure both learning rate schedules.
  config.schedule = ml_collections.ConfigDict({
      'img_heads': sched,
      'txt': sched_txt,
  })

  # *Single* optimizer.
  optim = ml_collections.ConfigDict()
  optim.optax_name = 'adafactor'
  optim.optax_configs = ml_collections.ConfigDict({  # Optimizer settings.
      # 'b1': 0.9,
      # 'b2': 0.999,
  })

  # Gradient clipping.
  optim.max_grad_norm = 1.0
  optim.per_example_clipping = True
  optim.optax_grad_pmean = True  # For per-example gradients Optax calls pmean.

  # Explicit WD (not via an optimizer).
  optim.weight_decay = 0.0
  optim.weight_decay_decouple = True

  config.optimizer = optim

  assert (optim.per_example_clipping or config.normalization != 'per_example'
          'Per example clipping only makes sense with local normalization')

  # Objectness head.
  # config.model.objectness_head = None
  config.model.objectness_head = ml_collections.ConfigDict()
  config.model.objectness_head.stop_gradient = True

  # Init.
  config.init_from = ml_collections.ConfigDict()
  checkpoint_path = CHECKPOINTS.get(init_mode, None)
  if checkpoint_path is None:
    raise ValueError('Unknown init_mode: {}'.format(init_mode))
  config.init_from.checkpoint_path = checkpoint_path

  # Logging.
  config.xprof = True  # Profile using xprof.
  config.log_summary_steps = 1  # Train summary steps.
  config.checkpoint = True  # Do checkpointing.
  config.checkpoint_steps = 500
  config.debug_train = True  # Debug mode during training.
  config.debug_eval = True  # Debug mode during eval.
  config.log_eval_steps = 500

  return config
