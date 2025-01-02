# Simulating Complex Physics Using Graph Neural Simulators

This repository contains a PyTorch implementation of the “Graph Network-based Simulators” (GNS) model ([Learning to simulate complex physics with graph networks](https://arxiv.org/abs/2002.09405), ICML 2020) from DeepMind for simulating particle-based dynamics using graph networks.
This repo uses purely PyTorch's native APIs, and is in an adaptation of code [GNS-PyTorch](https://github.
com/zhouxian/GNS-PyTorch/tree/main).

## Table of Contents
1. [Setup](#setup)
2. [Download Dataset](#download-dataset)
3. [Ground Truth Trajectory Visualisations (Optional)](#ground-truth-trajectory-visualisations-optional)
4. [Usage](#usage)
    - [Training](#training)
    - [Evaluation](#evaluation)

## Setup

This project was developed using Python 3.12. To set up the environment, follow the steps below:

1. Create a venv:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
## Download Dataset

To download the dataset and convert it to `pickle` format, run the following commands:

```bash
python -m data_downloader.py {dataset_name}
```

where, for this project, `{dataset_name}` is `WaterDropSample`.
In addition to this dataset, the following datasets are available

- `Water`
- `Sand`
- `Goop`
- `MultiMaterial`
- `RandomFloor`
- `WaterRamps`
- `SandRamps`
- `FluidShake`
- `FluidShakeBox`
- `Continuous`
- `WaterDrop-XL`
- `Water-3D`
- `Sand-3D`
- `Goop-3D`


The datasets are downloaded from the following links:

* [https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/{dataset_name}/{version}.tfrecord](https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/{dataset_name}/{version}.tfrecord)
  for the `tfrecord` files containing the trajectories. {version} can be `train`, `valid`, or `test`.

* [https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/{dataset_name}/metadata.json](https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/{dataset_name}/metadata.json) for 
  the metadata file containing the dataset statistics Metadata file with dataset information (*sequence length, dimensionality, box bounds, default connectivity radius, 
statistics for normalization, etc*).

where `{version}` is one of:
- train
- valid
- test

This creates a directory called 'data' in the root of the project containing the dataset files.

## Ground Truth Trajectory Visualisations (Optional)
The trajectories can be visualised using the following command:

```bash
python -m trajectory_visualisations --dataset {dataset} --particles {no_of_particles}
```

where `{dataset_name}` is the name of a dataset that has been downloaded or simulated and `{no_of_particles}` is the number of particles to include in the
static trajectories.

Running this command generates 3 files:
- `static_plots/{dataset_name}_trajectories.png` - a static plot of the trajectories
- `static_plots/{dataset_name}_frames.png` - a set of 3 plots showing the particle configurations at the start, middle and end of the trajectory
- `animations/{dataset_name}.mp4` - an `mp4` video of the trajectory

## Usage
### Training
Once the data has been downloaded and converted, the model can be trained using the following command:

```bash 
python -m train
```

This only works for the `WaterDropSample` dataset. To train on a different dataset, a new `config.py` file must be created with the appropriate parameters.
Training the model creates a directory called `ckpts` in the root of the project containing the model checkpoints.
The model checkpoints are snapshots of the trained model at different stages of training.

### Evaluation
Once the model has been trained and the checkpoints have been saved, the model can be evaluated using the following command:

```bash
python -m eval --ckpt ckpts/WaterDropSample/{checkpoint}
```

where checkpoint is the name of the checkpoint file to evaluate. 
These can be found in the `ckpts/WaterDropSample` directory.
Running this command generates a new directory under `animations` called `WaterDropSample/{checkpoint}`
where it stores `gifs` of the model predictions along side the ground truth trajectories.
