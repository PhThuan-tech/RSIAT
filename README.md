# RSIAT Reproduction

This repository is a reproduced implementation of **Representation-Steered Incremental Adapter-Tuning for Class-Incremental Learning with Pre-Trained Models**.

The original project is not mine; this copy was rebuilt to reproduce the training/evaluation pipeline and includes a small amount of local reproduction support, such as CUB preparation and checkpoint resume.

![Overall pipeline of RSIAT](images/framework.png)

## Environment

The code was reproduced with the following baseline environment:

- Ubuntu 20.04 LTS
- Python 3.10
- CUDA 11.8
- NVIDIA GPU with CUDA support

The repository can also be inspected on Windows, but the training scripts are intended to run in a Linux/CUDA Python environment.

## Setup

Create and activate a fresh environment:

```bash
conda create -n rsiat python=3.10 -y
conda activate rsiat
```

Install PyTorch first. For CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The repository uses `timm>=1.0` APIs and pins a Python 3.12-compatible timm
release so it can run on current Google Colab runtimes. Older environments that
still have `timm==0.6.12` must upgrade it before importing the model modules.

Optional GPU check:

```bash
python testGPU.py
```

If the GPU is available, the script prints the CUDA device used by a matrix multiplication.

## Repository Structure

```text
RSIAT/
|-- main.py                         # Entry point
|-- trainer.py                      # Training loop, logging, checkpoint resume
|-- prepare_cub.py                  # Converts raw CUB_200_2011 into ImageFolder layout
|-- args.sh                         # Example commands for all supported configs
|-- requirements.txt
|-- exps/                           # JSON experiment configs
|-- data/
|   |-- data.py                     # Dataset definitions
|   |-- data_manager.py             # Incremental task/data manager
|   `-- datasets/                   # Local datasets, ignored by git
|-- models/                         # RSIAT model implementation
|-- network/                        # ViT adapter/classifier modules
|-- utils/                          # Shared helpers/losses/net factory
|-- logs/                           # Training logs
|-- ckpt/                           # Checkpoints, ignored by git
`-- images/
```

## Data Preparation

All datasets are expected under:

```text
data/datasets/
```

The expected directory layout is:

```text
data/datasets/
|-- cifar-100-python/               # Auto-downloaded by torchvision when running CIFAR100
|-- cub/
|   |-- train/
|   `-- test/
|-- imagenet-a/
|   |-- train/
|   `-- test/
|-- imagenet-r/
|   |-- train/
|   `-- test/
|-- omnibenchmark/
|   |-- train/
|   `-- test/
`-- vtab/
    |-- train/
    `-- test/
```

For image-folder datasets, each `train/` and `test/` directory must contain one subdirectory per class, as required by `torchvision.datasets.ImageFolder`.

### CUB-200-2011

Put the extracted original CUB dataset here:

```text
data/datasets/CUB_200_2011/
```

The folder must contain files such as `images.txt`, `train_test_split.txt`, and the `images/` directory.

Then run:

```bash
python prepare_cub.py
```

This creates:

```text
data/datasets/cub/train/
data/datasets/cub/test/
```

### CIFAR-100

CIFAR-100 is loaded through `torchvision.datasets.CIFAR100` and will be downloaded automatically into:

```text
data/datasets/
```

when running the CIFAR config.

## Running Experiments

The main command format is:

```bash
python main.py --config ./exps/<config-file>.json
```

Available configs:

```bash
python main.py --config ./exps/adapter_cifar224.json
python main.py --config ./exps/adapter_cub.json
python main.py --config ./exps/adapter_imageneta.json
python main.py --config ./exps/adapter_imagenetr.json
python main.py --config ./exps/adapter_omnibench.json
python main.py --config ./exps/adapter_vtab.json
```

These commands are also listed in `args.sh`.

For a quick smoke test on CIFAR-100, use:

```bash
python main.py --config ./exps/adapter_cifar224_smoke.json
```

## UMT-RSIAT Research Variant

The repository also includes an experimental variant with a weakly nonlinear
low-rank projector, class-wise mean/diagonal-variance alignment, adaptive
top-K separation, and projector-based transport of old class statistics.

Run its lightweight integration check first:

```bash
python -m unittest tests.test_research_components
python main.py --config ./exps/umt_adapter_cifar224_smoke.json
```

Run the CIFAR-100 experiment for seed 1993 with automatic task-level resume:

```bash
python main.py --config ./exps/umt_adapter_cifar224.json
```

After preparing ImageNet-R in the layout documented above, run:

```bash
python main.py --config ./exps/umt_adapter_imagenetr.json
```

Research checkpoints are isolated by seed under
`ckpt/umt_weak_moment_topk/cifar224/10_10/seed_<seed>/`. Set `resume` to
`true` only when continuing an experiment with the same method and seed.
The supplied research configs keep only the latest task checkpoint per seed to
avoid filling Google Drive. Set `keep_last_checkpoint` to `false` if every
task checkpoint is needed for offline diagnostics.

The full-covariance checkpoint format is retained for baseline compatibility.
When `statistics_transport` is `diagonal_mc`, old Gaussian statistics are
transported by Monte Carlo samples and stored as diagonal covariance matrices.
Research configs save only the diagonal through `compact_diagonal_checkpoint`
to substantially reduce checkpoint size.
A ready-to-run Google Colab workflow is available in
`RSIAT_UMT_Colab.ipynb`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PhThuan-tech/RSIAT/blob/main/RSIAT_UMT_Colab.ipynb)

The optional `output_root` config key redirects `logs/` and `ckpt/` to another
location; the Colab notebook uses it to persist outputs on Google Drive while
training from Colab's faster local disk.
The notebook also links `data/datasets` to
`MyDrive/RSIAT_data/datasets`, so CIFAR-100 is downloaded only once and reused
after Colab runtime resets.

The smoke config uses fewer epochs and a smaller batch size, so it is useful for checking that the environment, CUDA, model loading, and data pipeline work before running the full reproduction.

## GPU Selection

Each JSON config contains a `device` field, for example:

```json
"device": ["0"]
```

Change it to the CUDA GPU id you want to use. For example, use `["1"]` for `cuda:1`.

## Resume, Logs, and Checkpoints

Some configs include:

```json
"resume": true
```

When resume is enabled, the trainer searches for the latest checkpoint in:

```text
ckpt/<prefix>/<dataset>/<init_cls>_<increment>/task_N.pkl
```

For example, CUB with `prefix=all`, `init_cls=10`, and `increment=10` saves checkpoints under:

```text
ckpt/all/cub/10_10/
```

Training logs are saved under:

```text
logs/<model_name>/<dataset>/<init_cls>/<increment>/
```

Example:

```text
logs/adapter/cub/0/10/all_1993_pretrained_vit_b16_224_in21k_adapter.log
```

To force a run to start from task 0, either remove/rename the corresponding checkpoint folder or set `"resume": false` in the selected config.

## Citation

If you use this reproduction in research, please cite the original work:

```text
@inproceedings{zhao2026representation,
  title={Representation-Steered Incremental Adapter-Tuning for Class-Incremental Learning with Pre-Trained Models},
  author={Zhao, Jiarui and Huang, Libo and Li, Xiangqi and An, Zhulin and Yang, Chuanguang and Wang, Yu and Diao, Boyu and Xu, Yongjun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18010--18020},
  year={2026}
}
```

## Acknowledgement

This repository is based on ideas and code structure from [PILOT](https://github.com/LAMDA-CL/LAMDA-PILOT) and [SSIAT](https://github.com/HAIV-Lab/SSIAT).
