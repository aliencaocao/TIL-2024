"""til dataset."""
import orjson
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for til dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(shape=(384, 384, 3), dtype=np.uint8),
                'labels': tfds.features.Text(),
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('image', 'labels'),  # Set to `None` to disable
            homepage='https://dataset-homepage/',
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        return {
            tfds.Split.TRAIN: self._generate_examples('train'),
            tfds.Split.VALIDATION: self._generate_examples('val')
        }

    def _generate_examples(self, split: str):
        """Yields examples."""
        ds = orjson.loads(Path('til_siglip_ds.json').read_text())
        path_to_image = Path('../../../til_siglip_ds')

        train_index = 4085  # 0 to image_4085 is train, rest is val
        train_ds = [sample for sample in ds if int(sample['image'].split('_')[1]) <= train_index]
        val_ds = [sample for sample in ds if int(sample['image'].split('_')[1]) > train_index]

        ds = train_ds if split == 'train' else val_ds

        for sample in ds:
            yield sample['image'], {
                'image': np.asarray(Image.open(path_to_image / sample['image']).resize((384, 384)), dtype=np.uint8),
                'labels': sample['label'],
            }
