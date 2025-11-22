# add comment block describing the experiment
"""This script visualizes the training and evaluation of a Predictive Coding Network (PCN) on a specified dataset (MNIST)."""
import torch
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
from plotly import colors


# Plot
def plot_energy_history_interactive(
    energy_history, title="Batch-averaged Energy Trajectories"
):
    """
    Interactive energy trajectories with Plotly.
    - energy_history: list of epochs; each epoch is list of batch energy lists.
    - infer_steps: number of inference steps.
    Hover to see Epoch, Batch, Step, Energy.
    """
    num_epochs = len(energy_history)
    # Sample one color per epoch from Viridis
    epoch_colors = colors.sample_colorscale(
        colors.sequential.Viridis,
        [i / (num_epochs - 1) if num_epochs > 1 else 0 for i in range(num_epochs)],
    )

    fig = go.Figure()
    for epoch_idx, epoch_energies in enumerate(energy_history):
        color = epoch_colors[epoch_idx]
        for batch_idx, batch_vals in enumerate(epoch_energies):
            steps = list(range(len(batch_vals)))
            # Attach epoch and batch metadata for hover
            customdata = [[epoch_idx + 1, batch_idx + 1] for _ in steps]
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=np.log10(batch_vals),
                    mode="lines",
                    line=dict(color=color, width=1),
                    hovertemplate=(
                        "Epoch %{customdata[0]}<br>"
                        "Batch %{customdata[1]}<br>"
                        "Step %{x}<br>"
                        "Energy %{y:.4f}<extra></extra>"
                    ),
                    customdata=customdata,
                    showlegend=False,
                )
            )

    fig.update_layout(
        title=title, xaxis_title="Inference Step t", yaxis_title="Log10 Energy"
    )

    # Optional: log-scale y-axis if needed
    # fig.update_yaxes(type='log')

    fig.show()


# %%
def plot_epoch_avg_interactive(
    energy_history, title="Batch-Averaged Energy Trajectories (Mean ± 1-std)"
):
    """
    Interactive per-epoch mean ±1-std energy trajectories using Plotly.
    - energy_history: list of epochs; each epoch is list of batch energy trajectories.
    - infer_steps: number of inference steps.
    Hover to see Epoch, Step, Mean Energy.
    """
    num_epochs = len(energy_history)
    epoch_colors = colors.sample_colorscale(
        colors.sequential.Viridis,
        [i / (num_epochs - 1) if num_epochs > 1 else 0 for i in range(num_epochs)],
        colortype="hex",  # Makes each color something like '#440154'
    )

    epoch_colors = [
        (
            c
            if isinstance(c, str)
            else "#{0:02x}{1:02x}{2:02x}".format(*(int(round(255 * v)) for v in c))
        )
        for c in epoch_colors
    ]

    fig = go.Figure()

    for epoch_idx, epoch_batches in enumerate(energy_history):
        arr = np.array(epoch_batches)  # shape (n_batches, total_steps+1)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        steps = list(range(len(mean)))
        color = epoch_colors[epoch_idx]
        # Upper bound (mean + std)
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=mean + std,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Lower bound (mean - std) with fill
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=mean - std,
                mode="lines",
                fill="tonexty",
                # Convert hex '#rrggbb' → rgba(r,g,b,0.2)
                fillcolor=f"rgba({int(color[1:3],16)},"
                f"{int(color[3:5],16)},"
                f"{int(color[5:7],16)},0.2)",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Mean line
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=mean,
                mode="lines",
                line=dict(color=color, width=2),
                name=f"Epoch {epoch_idx+1}",
                hovertemplate=(
                    f"Epoch {epoch_idx+1}<br>"
                    "Step %{x}<br>"
                    "Energy %{y:.4f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(title=title, xaxis_title="Inference Step t", yaxis_title="Energy")
    # Optional: log-scale y-axis if needed
    # fig.update_yaxes(type='log')
    fig.show()


def plot_train_val_metric(
    train_result: list, val_result: list, yaxis_title="Metric Name", logy=False
):
    """
    Plot training and validation metrics over epochs using Plotly.
    """
    train_color = "#440154"
    val_color = "#053061"
    if len(train_result) != len(val_result):
        raise ValueError("train_result and val_result must have the same length.")
    fig = go.Figure()
    # Add line plots to the figure
    fig.add_trace(
        go.Scatter(
            x=list(range(len(train_result))),
            y=train_result,
            mode="lines",
            line=dict(color=train_color, width=1),
            name="Train",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(val_result))),
            y=val_result,
            mode="lines",
            line=dict(color=val_color, width=2),
            name="Val",
        )
    )
    fig.update_layout(
        title="Training and Validation Metric Over Epochs",
        xaxis_title="Epoch",
        yaxis_title=yaxis_title,
        legend=dict(
            title="Split",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
        ),
    )
    # Optional: log-scale y-axis if needed
    if logy:
        fig.update_yaxes(type="log")

    fig.show()


def unflatten_images(images, image_size=[3, 32, 32], normalize=True):
    """Unflatten images. Tile them vertically."""
    img = images.view(-1, *image_size).cpu().numpy()  # First dimension is batch size
    # pixel_std = img[img > 0].std()
    # pixel_mean = img[img > 0].mean()
    # # clamp at 2 stddev above mean
    # img = np.clip(img, 0, pixel_mean + 2 * pixel_std)
    if normalize:
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = (255 * img).astype(np.uint8)  # Scale and convert to uint8
    if image_size[0] == 1:
        # Replicate image 3 times in dimension 1
        img = np.repeat(img, 3, axis=1)  # change from monochrome to grayscale
    img = np.concatenate(
        [img[i, :, :, :] for i in range(img.shape[0])], axis=1
    )  # Tile vertically
    img = np.transpose(img, (1, 2, 0))  # Change to HWC for plotly
    return img


def plot_generated_images(
    model, x_batch, device=torch.device("cuda"), image_size=(1, 28, 28)
):
    # Store a copy of the latents
    # check if the model has attribute z_latent
    if hasattr(model, "z_latent"):
        latents_given_image = [x.detach().clone() for x in model.z_latent]
    else:
        latents_given_image = None

    B = x_batch.size(0)

    # Input image
    reference_image = unflatten_images(x_batch[0:10, :], image_size=image_size)
    fig = go.Figure(go.Image(z=reference_image))
    fig.update_layout(title="Input")
    fig.show()

    # Generated an image from label
    y_dim = model.get_dim_for_key(model.task_map["y"])
    y_batch = F.one_hot(
        torch.tensor([i for i in range(10)], device=device), num_classes=y_dim
    ).float()
    # Clamp layers {node:data}
    clamps_gen_image = {model.task_map["y"]: y_batch}  # classification vector

    # Initialize latents
    model.init_latents(
        clamp_dict=clamps_gen_image, batch_size=y_batch.shape[0], device=device
    )
    # Inference
    model.infer(clamps_gen_image)

    # Output image
    gen_img = model.get_task_result("x")
    img = unflatten_images(gen_img, image_size=image_size)
    fig = go.Figure(go.Image(z=img))
    fig.update_layout(title="Generated Images")
    fig.show()

    # for i in range(min(10, B)):
    #     # plot the histogram of pixel values
    #     fig = go.Figure(data=[go.Histogram(x=gen_img[i, :].cpu().numpy())])
    #     fig.update_layout(title=f"Histogram of pixel values for image {i}")
    #     fig.show()

    if latents_given_image is None:
        return
    # Class re-styled Image (assuming a sequential PC_MLP model)
    # Change the class of an image (infer latents given an image; clamp class latents to desired class; infer again to generate image)
    class_type = 3
    target_class = F.one_hot(torch.tensor([class_type] * B), num_classes=y_dim).float()
    # Generate image inferring on modified latents.
    # Copy latent state from example images to initialize
    clamps_gen_image = {
        model.task_map["y"]: target_class.to(device),  # classification vector
        1: latents_given_image[1].detach().clone(),
    }
    # Initialize latents
    model.init_latents(
        clamp_dict=clamps_gen_image, batch_size=B, device=device, std=0.1
    )
    # Clamp layers for inference
    clamps_gen_image = {
        model.task_map["y"]: target_class.to(device)
    }  # classification vector
    # Inference
    model.infer(clamps_gen_image)

    # Output image
    gen_img = model.get_task_result("x")
    img = unflatten_images(gen_img[0:10, :], image_size=image_size)
    fig = go.Figure(go.Image(z=img))
    fig.update_layout(title="Class re-styled Images")
    fig.show()
