# hybrid-vit-performer 

Hybrid Vision Transformer project comparing standard self-attention, Performer attention, and hybrid architectures. This repository explores mixing **standard self-attention** with **Performer attention** (via `performer-pytorch`) using configurable attention placement patterns. Includes training, benchmarking, plotting scripts, and notebooks to reproduce experiments on **MNIST**, **CIFAR-10**, and a **synthetic ImageNet-style** input setting. 

**Developed for:** IEOR6617: Machine Learning & High-Dimensional Data Mining at Columbia University (Fall 2025) 

**Instructor:** Prof. Krzysztof Choromanski, who introduced Performer attention 

📄 **[Read the full report](report.pdf)**


## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hybrid-vit-performer.git
cd hybrid-vit-performer

# Option A: pip
pip install -r requirements.txt

# Option B: conda
conda env create -f environment.yml
conda activate hybrid-vit-performer
```

## What's in this repo

- **Core package:** `src/hybrid_vit`
  - **Model:** `HybridViT` (patch embedding, class token, positional embeddings, transformer blocks)
  - **Attention modules:**
    - `StandardAttention`
    - `PerformerAttention` using `performer_pytorch.FastAttention`
      - `kernel_type=relu`: generalized attention with ReLU kernel
      - `kernel_type=softmax`: FAVOR+ softmax approximation
  - **Architecture modes:**
    - `all_standard`
    - `all_performer`
    - `intertwined`
    - `performer_first`
    - `standard_first`
  - **Device selection:** auto-resolves to `cuda`, `mps`, or `cpu`

- **Datasets:**
  - `MNIST` and `CIFAR10` via `torchvision.datasets` (downloaded into `./data`)
  - `IMAGENET_SYNTH`: synthetic dataset generating random images/labels (configurable sample counts)

- **Configs (YAML):**
  - `configs/mnist.yaml`
  - `configs/cifar10.yaml`
  - `configs/imagenet_synth.yaml`

- **Training & evaluation:**
  - `train.py` contains `fit`, `train_one_epoch`, and `evaluate`

- **Benchmarking:**
  - `benchmark.py` measures inference time per batch (default: 50 batches)

- **Outputs (CSV):**
  - Written to `results/<dataset>/`
  - Baseline:
    - `summary_baseline.csv`
    - `history_baseline.csv`
  - Non-baseline:
    - `summary_arch-<arch>_kernel-<kernel>_m-<m>.csv`
    - `history_arch-<arch>_kernel-<kernel>_m-<m>.csv`

## Running experiments

### Run one experiment (CLI overrides)

`scripts/run_one.py` runs a single configuration with required CLI arguments:

**Required:**
- `--dataset`
- `--architecture`
- `--kernel_type`
- `--m_features`

**Optional:**
- `--seed`
- `--device` (auto / cpu / cuda / mps)

**Example:**
```bash
python scripts/run_one.py \
  --dataset MNIST \
  --architecture performer_first \
  --kernel_type relu \
  --m_features 128 \
  --seed 42
```

### Run the full sweep grid (MNIST or CIFAR-10)

`scripts/full_grid.py` runs the baseline plus a grid sweep.

**Options:**
- `--dataset`
- `--base_config`
- `--save_dir`
- `--device`

**Sweeps:**
- `architecture` (excluding baseline)
- `kernel_type` in {relu, softmax}
- `m_features` in [16, 32, 64, 128, 256]

**Example:**
```bash
python scripts/full_grid.py --dataset CIFAR10
```
