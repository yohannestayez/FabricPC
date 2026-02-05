import numpy as np
from fabricpc.utils.data.data_utils import one_hot, split_np_seed


class MnistLoader:
    """JAX-compatible data loader using TensorFlow Datasets.

    Provides the same interface as PyTorch DataLoader but uses tfds
    data parallelism based on C++ that bypasses GIL and does not inherit GPU state.
    Avoids os.fork warnings with JAX.

    Args:
        split: Dataset split to load. Use 'train' for training data or
               'test' for test data. Also supports slicing syntax like
               'train[:80%]' or 'train[80%:]' for custom splits.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data each epoch.
        seed: Random seed for reproducibility. When set, ensures deterministic
              shuffling across runs and machines. If None, shuffling is random.
        normalize_mean: Mean for normalization (default: MNIST mean).
        normalize_std: Std for normalization (default: MNIST std).
    """

    def __init__(
        self,
        split: str,
        batch_size: int,
        shuffle: bool = True,
        seed: int = None,
        tensor_format: str = "NHWC",  # image tensor 'flat' or 'NHWC' batch-height-width-channels
        normalize_mean: float = 0.1307,
        normalize_std: float = 0.3081,
    ):
        import tensorflow_datasets as tfds
        import tensorflow as tf

        # Disable GPU for TensorFlow (we only use it for data loading)
        tf.config.set_visible_devices([], "GPU")

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.tensor_format = tensor_format
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

        # Split seed into two independent seeds for file and buffer shuffling
        file_seed, buffer_seed = split_np_seed(seed, n=2)

        # Configure read options for reproducibility
        read_config = tfds.ReadConfig(
            shuffle_seed=file_seed,
            interleave_cycle_length=1,  # Sequential reading for determinism
        )

        # Load dataset with pinned version for cross-machine reproducibility
        ds, info = tfds.load(
            "mnist:3.0.1",
            split=split,
            with_info=True,
            as_supervised=True,
            read_config=read_config,
            shuffle_files=shuffle and seed is not None,
        )
        self.num_examples = info.splits[split].num_examples
        self._num_batches = (self.num_examples + batch_size - 1) // batch_size

        # Build pipeline
        if shuffle:
            ds = ds.shuffle(
                buffer_size=self.num_examples, seed=buffer_seed
            )  # mnist fits in memory (~60MB) so the buffer is the full dataset
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        self.ds = ds

    def __iter__(self):
        for images, labels in self.ds:
            # Convert to numpy, normalize, and flatten
            images = images.numpy().astype(np.float32) / 255.0
            images = (images - self.normalize_mean) / self.normalize_std

            # images shape is (Batch, 28, 28, 1)
            if self.tensor_format == "flat":
                images = images.reshape(images.shape[0], -1)  # Flatten to (Batch, 784)

            # One-hot encode labels
            labels = one_hot(labels.numpy(), num_classes=10)

            yield images, labels

    def __len__(self):
        return self._num_batches
