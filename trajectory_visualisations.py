import json
import os
import pickle
import sys
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


def _load_rollout_file(dataset: str):
    """Load the first rollout pickle file in a directory."""
    trajectory_dir = f'data/{dataset}/train'
    rollout_files = [os.path.join(trajectory_dir, file) for file in os.listdir(trajectory_dir) if file.endswith(".pkl")]
    if not rollout_files:
        raise FileNotFoundError(f"No valid .pkl files found in {trajectory_dir}.\n"
                                f"Ground truth data is located in '.data/.../train' directories.")
    with open(rollout_files[0], "rb") as f:
        return pickle.load(f)


def _load_container_bounds(dataset: str):
    """Load the container bounds from the metadata.json file."""
    metadata_path = f'data/{dataset}/metadata.json'
    with open(metadata_path, "rb") as f:
        metadata = json.load(f)
    return metadata["bounds"]


def _add_container(bounds: list, ax: plt.Axes, margin: float = 0.01):
    """Add container bounds to the plot."""
    x_min, x_max = bounds[0][0] - margin, bounds[0][1] + margin
    y_min, y_max = bounds[1][0] - margin, bounds[1][1] + margin

    # plot the container
    ax.set_xlim(x_min - 0.01, x_max + 0.01)
    ax.set_ylim(y_min - 0.01, y_max + 0.01)
    ax.set_aspect(1.0)
    ax.axis("off")
    ax.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], color="black", linewidth=2)
    return ax


def render_animation_gif(dataset: str, step_stride: int = 1):
    """
    Renders an animation of particle trajectories with container bounds.

    Args:
        dataset (str): The dataset name.
        step_stride (int): The stride for frame updates.
    """
    rollout_data = _load_rollout_file(dataset)
    trajectory = rollout_data["position"]
    bounds = _load_container_bounds(dataset)
    num_steps, _, _ = trajectory.shape
    fig, ax = plt.subplots(figsize=(10, 10))

    ax = _add_container(bounds, ax)
    scatter = ax.scatter([], [], s=2, color="blue")

    def update(frame):
        scatter.set_offsets(trajectory[frame])
        return scatter,

    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, num_steps, step_stride), interval=50)

    output_path = f'./animations/{dataset}.gif'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ani.save(output_path, fps=30, writer="pillow")
    print(f"GIF animation saved at: {output_path}")


def render_animation_mp4(dataset: str, step_stride: int = 1):
    """
    Renders an animation of particle trajectories with container bounds and saves it as an MP4.

    Args:
        dataset (str): The dataset name.
        step_stride (int): The stride for frame updates.
    """
    rollout_data = _load_rollout_file(dataset)
    trajectory = rollout_data["position"]
    bounds = _load_container_bounds(dataset)

    num_steps, _, _ = trajectory.shape
    fig, ax = plt.subplots(figsize=(10, 10))

    ax = _add_container(bounds, ax)  # Add container bounds (assuming _add_container is implemented)
    scatter = ax.scatter([], [], s=2, color="blue")

    def update(frame):
        scatter.set_offsets(trajectory[frame])
        return scatter,

    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, num_steps, step_stride), interval=50)

    output_path = f'./animations/{dataset}.mp4'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the animation as MP4
    ani.save(output_path, fps=30, writer="ffmpeg")
    print(f"MP4 animation saved at: {output_path}")


def plot_trajectories(dataset: str, particles: int = 3):
    """
    Plot static trajectories of selected particles with container bounds.

    Args:
        dataset (str): The dataset name.
        particles (int): The number of particles to plot.
    """
    rollout_data = _load_rollout_file(dataset)
    trajectory = rollout_data["position"]
    bounds = _load_container_bounds(dataset)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax = _add_container(bounds, ax)
    particle_indices = np.random.choice(trajectory.shape[1], particles, replace=False)
    for idx in particle_indices:
        ax.plot(trajectory[:, idx, 0], trajectory[:, idx, 1], linewidth=1)
        # add a point at the start of the trajectory
        ax.scatter(trajectory[0, idx, 0], trajectory[0, idx, 1])
    ax.set_title(f"Trajectories of {particles} Particles ({dataset})", fontsize=12)
    plot_path = f"./static_plots/{dataset}_trajectories.png"
    plt.tight_layout(pad=0.1)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, bbox_inches="tight")
    print(f"Static plot saved at: {plot_path}")


def plot_frames(dataset: str, frames: list = None):
    """
    Plot static frames of particle positions with container bounds.

    Args:
        dataset (str): The dataset name.
        frames (list): The frame indices to plot.
    """
    rollout_data = _load_rollout_file(dataset)
    trajectory = rollout_data["position"]
    bounds = _load_container_bounds(dataset)

    if not frames:
        frames = [0, trajectory.shape[0] // 2, trajectory.shape[0] - 1]

    fig, axes = plt.subplots(1, len(frames), figsize=(12, 6))

    for ax, frame in zip(axes, frames):
        ax = _add_container(bounds, ax)
        ax.scatter(trajectory[frame, :, 0], trajectory[frame, :, 1], s=5, color="blue")
        ax.set_title(f'{dataset} time step {frame + 1}/{trajectory.shape[0]}', fontsize=12)
    # Tighten layout
    plt.tight_layout(pad=0.1)
    plot_path = f"./static_plots/{dataset}_frames.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, bbox_inches="tight")
    print(f"Static plot saved at: {plot_path}")


def arg_parse():
    parser = argparse.ArgumentParser(description='Visualisation parameters')
    parser.add_argument('--dataset', help='Dataset to visualise', type=str, default='WaterDrop')
    parser.add_argument('--particles', type=int, help='Number of particles to plot for the trajectory',
                        default=10)
    return parser.parse_args()

def main():
    args = arg_parse()
    print(f'Plotting mp4 animation for {args.dataset} dataset')
    render_animation_mp4(args.dataset)
    print(f'Plotting trajectories for {args.dataset} dataset')
    plot_trajectories(args.dataset, args.particles)
    print(f'Plotting static frames for {args.dataset} dataset')
    plot_frames(args.dataset)


if __name__ == "__main__":
    main()
