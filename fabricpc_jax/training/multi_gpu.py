"""
Multi-GPU training utilities using JAX pmap.

This module provides data-parallel training across multiple GPUs using pmap,
which replicates the computation across devices and averages gradients.
"""

from typing import Dict, Tuple, Any
import jax
import jax.numpy as jnp
import optax
from functools import partial

from fabricpc_jax.core.types import GraphParams, GraphState, GraphStructure
from fabricpc_jax.core.inference import run_inference
from fabricpc_jax.models.graph_net import initialize_state


def replicate_params(params: GraphParams, n_devices: int) -> GraphParams:
    """
    Replicate parameters across devices for pmap.

    Args:
        params: Single-device parameters
        n_devices: Number of devices

    Returns:
        Replicated parameters with leading device axis
    """
    return jax.tree_util.tree_map(lambda x: jnp.stack([x] * n_devices), params)


def replicate_opt_state(opt_state: optax.OptState, n_devices: int) -> optax.OptState:
    """
    Replicate optimizer state across devices.

    Args:
        opt_state: Single-device optimizer state
        n_devices: Number of devices

    Returns:
        Replicated optimizer state with leading device axis
    """
    return jax.tree_util.tree_map(lambda x: jnp.stack([x] * n_devices), opt_state)


def shard_batch(batch: Dict[str, jnp.ndarray], n_devices: int) -> Dict[str, jnp.ndarray]:
    """
    Shard a batch across devices for pmap.

    Reshapes batch from (total_batch_size, ...) to (n_devices, batch_per_device, ...).

    Args:
        batch: Batch with shape (total_batch_size, ...)
        n_devices: Number of devices

    Returns:
        Sharded batch with shape (n_devices, batch_per_device, ...)

    Example:
        >>> batch = {'x': jnp.zeros((128, 784)), 'y': jnp.zeros((128, 10))}
        >>> sharded = shard_batch(batch, n_devices=4)
        >>> sharded['x'].shape
        (4, 32, 784)
    """
    def shard_array(x):
        batch_size = x.shape[0]
        if batch_size % n_devices != 0:
            raise ValueError(
                f"Batch size {batch_size} must be divisible by number of devices {n_devices}"
            )
        batch_per_device = batch_size // n_devices
        return x.reshape(n_devices, batch_per_device, *x.shape[1:])

    return jax.tree_util.tree_map(shard_array, batch)


def unshard_losses(losses: jnp.ndarray) -> float:
    """
    Average losses from all devices.

    Args:
        losses: Losses from each device, shape (n_devices,)

    Returns:
        Average loss across devices
    """
    return float(jnp.mean(losses))


def train_step_pmap(
    params: GraphParams,
    opt_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    optimizer: optax.GradientTransformation,
    T_infer: int,
    eta_infer: float = 0.1,
) -> Tuple[GraphParams, optax.OptState, jnp.ndarray, GraphState]:
    """
    Training step parallelized across devices using pmap.

    This is the core pmap'ed training step that will be replicated across devices.
    Gradients are averaged using pmean before the parameter update.

    Args:
        params: Replicated parameters (has device axis)
        opt_state: Replicated optimizer state (has device axis)
        batch: Sharded batch (has device axis)
        structure: Graph structure
        optimizer: Optax optimizer
        T_infer: Number of inference steps
        eta_infer: Inference learning rate

    Returns:
        Tuple of (updated_params, updated_opt_state, loss_per_device, final_state)
    """

    def loss_fn(params: GraphParams) -> Tuple[float, GraphState]:
        """Compute energy loss for the local batch shard."""
        # Get batch size for this device shard
        batch_size = next(iter(batch.values())).shape[0]

        # Map batch to clamps
        clamps = {}
        for task_name, task_value in batch.items():
            if task_name in structure.task_map:
                node_name = structure.task_map[task_name]
                clamps[node_name] = task_value

        # Initialize state
        state = initialize_state(
            structure, batch_size, clamps=clamps, init_method="feedforward", params=params
        )

        # Run inference
        final_state = run_inference(
            params, state, clamps, structure, T_infer, eta_infer
        )

        # Compute energy
        energy = 0.0
        for node_name, node_info in structure.nodes.items():
            if node_info.in_degree > 0:
                energy += jnp.sum(final_state.error[node_name] ** 2)

        energy = energy / batch_size

        return energy, final_state

    # Compute loss and gradients on this device
    (loss, final_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    # Average gradients across all devices (this is the key for data parallelism!)
    grads = jax.lax.pmean(grads, axis_name="devices")

    # Update parameters (each device now has same gradients)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss, final_state


# Create pmap version of train_step
# Note: We can't use static_broadcasted_argnums because GraphStructure contains dicts
# Instead, we'll create a closure that captures the static arguments
def create_pmap_train_step(
    structure: GraphStructure,
    optimizer: optax.GradientTransformation,
    T_infer: int,
    eta_infer: float,
):
    """
    Create a pmap'ed training step with static arguments captured in closure.

    Args:
        structure: Graph structure (static)
        optimizer: Optimizer (static)
        T_infer: Number of inference steps (static)
        eta_infer: Inference learning rate (static)

    Returns:
        Pmap'ed training step function
    """

    def step_fn(params, opt_state, batch):
        return train_step_pmap(
            params, opt_state, batch, structure, optimizer, T_infer, eta_infer
        )

    return jax.pmap(step_fn, axis_name="devices")


def train_pcn_multi_gpu(
    params: GraphParams,
    structure: GraphStructure,
    train_loader: Any,
    config: dict,
    verbose: bool = True,
) -> GraphParams:
    """
    Train PCN using all available GPUs with data parallelism.

    Args:
        params: Initial parameters (single device)
        structure: Graph structure
        train_loader: Data loader
        config: Training configuration
        verbose: Whether to print progress

    Returns:
        Trained parameters (single device)

    Example:
        >>> params, structure = create_pc_graph(config, jax.random.PRNGKey(0))
        >>> trained = train_pcn_multi_gpu(params, structure, train_loader, config)
    """
    from fabricpc_jax.training.optimizers import create_optimizer

    # Get available devices
    n_devices = jax.device_count()
    if verbose:
        print(f"Training on {n_devices} device(s): {jax.devices()}")

    if n_devices == 1:
        if verbose:
            print("Only 1 device available, falling back to single-GPU training")
        from fabricpc_jax.training import train_pcn
        return train_pcn(params, structure, train_loader, config, verbose)

    # Create optimizer
    optimizer = create_optimizer(config.get("optimizer", {"type": "adam", "lr": 1e-3}))
    opt_state = optimizer.init(params)

    # Replicate params and optimizer state across devices
    params = replicate_params(params, n_devices)
    opt_state = replicate_opt_state(opt_state, n_devices)

    # Get training hyperparameters
    T_infer = config.get("T_infer", 20)
    eta_infer = config.get("eta_infer", 0.1)
    num_epochs = config.get("num_epochs", 10)

    # Create pmap'ed training step
    pmap_train_step = create_pmap_train_step(structure, optimizer, T_infer, eta_infer)

    # Training loop
    for epoch in range(num_epochs):
        epoch_losses = []

        for batch_idx, batch_data in enumerate(train_loader):
            # Convert batch to JAX format
            if isinstance(batch_data, (list, tuple)):
                batch = {
                    "x": jnp.array(batch_data[0]),
                    "y": jnp.array(batch_data[1]),
                }
            elif isinstance(batch_data, dict):
                batch = {k: jnp.array(v) for k, v in batch_data.items()}
            else:
                raise ValueError(f"Unsupported batch format: {type(batch_data)}")

            # Shard batch across devices
            try:
                batch_sharded = shard_batch(batch, n_devices)
            except ValueError as e:
                if verbose and batch_idx == 0:
                    print(f"Warning: Skipping batch (size not divisible by {n_devices}): {e}")
                continue

            # Training step (parallelized across devices)
            params, opt_state, losses, _ = pmap_train_step(
                params, opt_state, batch_sharded
            )

            # Average losses from all devices
            avg_loss = unshard_losses(losses)
            epoch_losses.append(avg_loss)

        # Compute average loss for epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0

        if verbose:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Extract params from first device (all devices have same params due to pmean)
    params = jax.tree_util.tree_map(lambda x: x[0], params)

    return params


def evaluate_pcn_multi_gpu(
    params: GraphParams,
    structure: GraphStructure,
    test_loader: Any,
    config: dict,
) -> Dict[str, float]:
    """
    Evaluate PCN using all available GPUs.

    Args:
        params: Trained parameters (single device)
        structure: Graph structure
        test_loader: Test data loader
        config: Evaluation configuration

    Returns:
        Dictionary of metrics
    """
    n_devices = jax.device_count()

    if n_devices == 1:
        from fabricpc_jax.training import evaluate_pcn
        return evaluate_pcn(params, structure, test_loader, config)

    # Replicate params across devices
    params = replicate_params(params, n_devices)

    T_infer = config.get("T_infer", 20)
    eta_infer = config.get("eta_infer", 0.1)

    # Create pmap'ed inference function
    def inference_fn(params, batch):
        batch_size = next(iter(batch.values())).shape[0]
        clamps = {}
        for task_name, task_value in batch.items():
            if task_name in structure.task_map and task_name == "x":
                node_name = structure.task_map[task_name]
                clamps[node_name] = task_value

        state = initialize_state(
            structure, batch_size, clamps=clamps, init_method="feedforward", params=params
        )
        final_state = run_inference(params, state, clamps, structure, T_infer, eta_infer)
        return final_state

    pmap_inference = jax.pmap(inference_fn, axis_name="devices")

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_data in test_loader:
        # Convert batch
        if isinstance(batch_data, (list, tuple)):
            batch = {"x": jnp.array(batch_data[0]), "y": jnp.array(batch_data[1])}
        else:
            batch = {k: jnp.array(v) for k, v in batch_data.items()}

        batch_size = next(iter(batch.values())).shape[0]

        # Skip if batch size not divisible by n_devices
        if batch_size % n_devices != 0:
            continue

        # Shard batch
        batch_sharded = shard_batch(batch, n_devices)

        # Run inference
        final_states = pmap_inference(params, batch_sharded)

        # Gather results from all devices
        # (We need to reshape back from (n_devices, batch_per_device, ...) to (batch_size, ...))
        if "y" in structure.task_map:
            y_node = structure.task_map["y"]
            predictions = final_states.z_latent[y_node]  # (n_devices, batch_per_device, dim)
            predictions = predictions.reshape(batch_size, -1)
            targets = batch["y"]

            pred_labels = jnp.argmax(predictions, axis=1)
            true_labels = jnp.argmax(targets, axis=1)
            correct = jnp.sum(pred_labels == true_labels)

            total_correct += int(correct)
            total_samples += batch_size

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    return {"accuracy": accuracy}
