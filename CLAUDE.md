# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rules

### Fixed Hyperparameters (MUST follow everywhere)
These values are fixed for all experiments and must be consistent across ALL files — model code, training scripts, callbacks, and shell scripts:
- `initial_kd_weight = 0.01`
- `router_temp_start = 1.0`
- `router_ema_start = 0.999`

When editing any MoE-related file, verify these values. Flag any deviations immediately.

### Code Consistency
- All backbone MoE files (`llava_phi_moe.py`, `llava_qwen_moe.py`, `llava_stablelm_moe.py`) must use the same default hyperparameters.
- `RouterArguments` defaults in `train.py` must match the fixed hyperparameters above.
- When making a change to one backbone's MoE file, check if the same change is needed in other backbones.

### Training Scripts
- Training scripts live in `scripts/v1/<backbone>/`
- Eval scripts live in `scripts/v1/eval/moe_llava/`
- Use `deepspeed` launcher for both training and eval
- Phi2 conv_mode: `phi`, Qwen conv_mode: `qwen`, StableLM conv_mode: `stablelm`

### Eval Data
- Eval data is stored in `moellava/eval/pope/`, `moellava/eval/textvqa/`, etc. (NOT a top-level `eval/` directory)
- Eval scripts must point to `moellava/eval/<benchmark>/` paths

### General
- Conda environment: `moellava_mine`
- Do not modify the original MoE-LLaVA-main reference repo at `../MoE-LLaVA-main/`
- Checkpoints go under `checkpoints_<backbone>_student/` (e.g., `checkpoints_phi_student/`, `checkpoints_qwen_student/`)
- Always check GPU availability before launching training/eval jobs
- Use `--include localhost:<gpu_ids>` to control GPU assignment with deepspeed

## Project Overview

MoE-LLaVA is a Mixture-of-Experts Vision-Language Model that achieves LLaVA-1.5-7B performance with ~3B sparsely-activated parameters. The key innovation is converting dense FFN layers in a pretrained LLaVA model into sparse MoE layers using K-means-initialized, knowledge-distilled router gates.

## Setup

```bash
conda create -n moellava python=3.10 -y
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

Key dependencies: `torch==2.0.1`, `transformers==4.37.0`, `deepspeed==0.9.5`.

## Training

Training follows three stages (docs in `docs/TRAIN.md`):

1. **Pretraining** — align vision projector with frozen LLM
2. **Fine-tuning** — full instruction tuning (dense model)
3. **MoE tuning** — convert FFN layers to MoE, train routers + MLP experts

Launch training via DeepSpeed scripts in `scripts/v1/`:
```bash
# Example: Phi2 MoE fine-tuning
bash scripts/v1/phi2/finetune_moe.sh
```

Key MoE training args:
- `--moe_enable True --num_experts 4 --top_k_experts 2 --moe_mode sparse`
- `--router_centroids_path <path>` — K-means centroids for router initialization
- `--router_init_mode` — one of `random`, `student_warm`, `teacher_kd`, `no_teacher`
- `--train_modules fc1 fc2 wg` — which weights are trainable
- `--router_aux_loss_coef` — load balancing loss coefficient

## Evaluation

See `docs/EVAL.md`. Eval scripts are in `scripts/v1/eval/moe_llava/`:
```bash
bash scripts/v1/eval/moe_llava/pope_phi_student.sh
```

Supported benchmarks: VQAv2, GQA, TextVQA, ScienceQA, MMBench, POPE, MME, MM-Vet, SEED-Bench.

## Environment & Hardware

- Conda env: moellava_mine
- 8x NVIDIA A100-SXM4-40GB GPUs (40GB each)
- CUDA 12.8, Python 3.10
- Project path: /scratch/prafull/MoE-LLaVA_mine
- No sudo access

## Architecture

### Model Components

```
Image → Vision Encoder (CLIP/SigLIP) → Multimodal Projector (MLP/QFormer) → LLM with MoE
```

- **`moellava/model/llava_arch.py`** — base classes `LlavaMetaModel` and `LlavaMetaForCausalLM`; handles vision module init, image/video encoding, multimodal projection
- **`moellava/model/builder.py`** — `load_pretrained_model()`, handles all model variants, LoRA loading, MoE conversion, quantization (4/8-bit)
- **`moellava/model/language_model/`** — per-backbone implementations (`llava_phi.py`, `llava_phi_moe.py`, `llava_qwen.py`, `llava_qwen_moe.py`, etc.). MoE variants use `deepspeed.moe.layer.MoE`
- **`moellava/model/kd_gate.py`** — Knowledge Distillation gate with input/weight normalization, T²-scaled KL divergence, and EMA teacher update

### Training Infrastructure

- **`moellava/train/train.py`** — main training script; defines `ModelArguments`, `DataArguments`, `TrainingArguments`, `MoEArguments`; handles MoE layer initialization from centroids
- **`moellava/train/llava_trainer.py`** — custom HuggingFace Trainer with MoE-aware loss aggregation
- **`moellava/train/router_callback.py`** — `DynamicHyperparamCallback`: schedules temperature, KD weight, and EMA decay during Stage 3
- **`moellava/train/replace_gate.py`** — utilities to swap dense FFN gates with KD router gates post-initialization

### Router / Gate Variants

- `moellava/model/language_model/normalized_router_flexible.py` — primary KD router with normalization and EMA
- The `router_init_mode` arg controls whether teacher KD is active and how the student is initialized

### Supported Backbones

Phi2 (2.7B), Qwen (1.8B), StableLM (1.6B), QWen1.5, Mistral, MiniCPM, Llama — each with a dense and `_moe` variant in `language_model/`.

### DeepSpeed Config

`scripts/zero2.json` (ZeRO Stage 2) and `scripts/zero3.json` (ZeRO Stage 3) are used for multi-GPU training. Stage 2 is standard for MoE tuning.