"""til dataset."""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from pathlib import Path
import orjson

def format_bbox(bbox, img_width, img_height):
    return np.array([
        max(0., bbox[1] / img_height),
        max(0., bbox[0] / img_width),
        min(1., (bbox[1] + bbox[3]) / img_height),
        min(1., (bbox[0] + bbox[2]) / img_width),
    ], dtype=np.float32)

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for til dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(til): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(encoding_format='jpeg'),
            'width': np.int64,
            'height': np.int64,
            'regions': tfds.features.Sequence({
                'region_id': np.int64,
                'image_id': np.int64,
                'phrase': tfds.features.Text(),
                'bbox': tfds.features.BBoxFeature(),
            }),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=None,  # Set to `None` to disable
        homepage='https://dataset-homepage/',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(til): Downloads the data and defines the splits
    # path = dl_manager.download_and_extract('https://todo-data-url')

    # TODO(til): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        tfds.Split.TRAIN: self._generate_examples(Path('train')),
        tfds.Split.VALIDATION: self._generate_examples(Path('val'))
    }

  def _generate_examples(self, path: Path):
    """Yields examples."""
    # TODO(til): Yields (key, example) tuples from the dataset

    annotations = {}
    with open("../vlm.jsonl") as f:
        for line in f:
            x = orjson.loads(line)
            image_id = int(x['image'][6:-4])
            annotations[x['image']] = [{
                'region_id': np.int64(image_id * 100 + i),
                'image_id': np.int64(image_id),
                'phrase': ann['caption'],
                'bbox': format_bbox(ann['bbox'], img_width=1520, img_height=870),
            } for i, ann in enumerate(x['annotations'])]

    for f in path.glob('*.jpg'):
        image_id = int(str(f.name)[6:-4])
        yield image_id, {
            'image': f,
            'width': np.int64(1520),
            'height': np.int64(870),
            'regions': annotations[f.name],
        }
