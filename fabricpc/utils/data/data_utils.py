import jax.numpy as jnp
import numpy as np


class OneHotWrapper:
    """Wrap DataLoader to provide one-hot labels."""

    def __init__(self, loader):
        self.loader = loader
        self.dataset = loader.dataset

    def __iter__(self):
        for x_data, y_label in self.loader:
            y_onehot = one_hot(y_label.numpy(), num_classes=10)
            yield x_data, y_onehot

    def __len__(self):
        return len(self.loader)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def one_hot(labels, num_classes=10):
    """Convert labels to one-hot encoding."""
    return jnp.eye(num_classes)[labels]


def split_np_seed(seed, n=2):
    """Split a seed into n independent seeds using SeedSequence.

    Uses NumPy's SeedSequence.spawn() for cryptographically proper
    seed splitting, similar to JAX's jax.random.split().

    Args:
        seed: Initial seed (int) or None for random seeds.
        n: Number of independent seeds to generate.

    Returns:
        List of n integer seeds, or [None]*n if seed is None.
    """
    if seed is None:
        return [None] * n
    children = np.random.SeedSequence(seed).spawn(n)
    return [child.generate_state(1)[0] for child in children]
