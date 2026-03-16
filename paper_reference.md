# MoE-LLaVA Paper Reference

## Architecture

| Model | Experts | Top-k | MoE Layers | Activated Params | Total Params |
|-------|---------|-------|------------|-----------------|--------------|
| StableLM-1.6B×4-Top2 | 4 | 2 | 12 (of 24) | 2.0B | 2.9B |
| Qwen-1.8B×4-Top2 | 4 | 2 | 12 (of 24) | 2.2B | 3.1B |
| Phi2-2.7B×4-Top2 | 4 | 2 | 16 (of 32) | 3.6B | 5.3B |

- MoE layers are placed in **alternating** (interval) fashion — half of layers are MoE
- Vision encoder: CLIP-Large/336, image resolution: 336×336
- MLP projector: 2 linear layers with GeLU

## MoE-Tuning (3-Stage Training)

| Config | Stage I | Stage II | Stage III |
|--------|---------|----------|-----------|
| **Purpose** | Align vision to LLM | Multi-modal understanding | Sparsify LVLM |
| **Data** | LLaVA-PT (558k) | Hybrid-FT (964k) | LLaVA-FT (665k) |
| **Trainable** | MLP only | All except VE | MoE layers only (fc1, fc2, wg) |
| **DeepSpeed** | Zero2 | Zero2 | Zero2 offload |
| **LR** | 1e-3 | 2e-5 | 2e-5 |
| **Batch/GPU** | 32 | 16 | 16 |
| **Epochs** | 1 | 1 | 1 |
| **Precision** | BF16 | BF16 | BF16 |

- Stage III: FFN weights are replicated to initialize all experts
- Auxiliary load balancing loss coefficient α = 0.01
- Capacity factor: 1.5 (BPR routing strategy)

## Author's Benchmark Results

### Table 3: Image Understanding (main results)

| Method | LLM | Act. | VQAv2 | GQA | VisWiz | SQA^I | VQA^T | POPE | MME | MMB | LLaVA^W | MM-Vet |
|--------|-----|------|-------|-----|--------|-------|-------|------|-----|-----|---------|--------|
| LLaVA-1.5 | V-7B | 6.7B | 78.5* | 62.0* | 50.0 | 66.8 | 58.2 | 85.9 | 1510.7 | 64.3 | 63.4 | 30.5 |
| LLaVA-1.5 | V-13B | 13B | 80.0* | 63.3* | 53.6 | 71.6 | 61.3 | 85.9 | 1531.3 | 67.7 | 70.7 | 35.4 |
| LLaVA-Phi | P-2.7B | 2.7B | 71.4* | - | 35.9 | 68.4 | 48.6 | 85.0 | 1335.1 | 59.8 | - | 28.9 |
| **MoE-StableLM-1.6B×4** | S-1.6B | 2.0B | 76.7* | 60.3* | 36.2 | 62.6 | 50.1 | 85.7 | 1318.2 | 60.2 | 86.8 | 26.9 |
| **MoE-Qwen-1.8B×4** | Q-1.8B | 2.2B | 76.2* | 61.5* | 32.6 | 63.1 | 48.0 | 87.0 | 1291.6 | 59.7 | 88.7 | 25.3 |
| **MoE-Phi2-2.7B×4** | P-2.7B | 3.6B | 77.6* | 61.4* | 43.9 | 68.5 | 51.4 | 86.3 | 1423.0 | 65.2 | 94.1 | 34.3 |

\* = overlap in training data with benchmark

### Table 4: POPE Hallucination (detailed)

| Method | Adversarial ||| Popular ||| Random |||
|--------|-----|------|-----|-----|------|-----|-----|------|-----|
| | Acc | F1 | Yes | Acc | F1 | Yes | Acc | F1 | Yes |
| LLaVA-1.5-13B | 85.5 | 84.4 | 43.3 | 87.4 | 86.2 | 41.3 | 88.0 | 87.1 | 41.7 |
| **MoE-1.6B×4 (StableLM)** | 86.9 | 85.7 | 41.7 | 85.3 | 84.2 | 43.5 | 88.0 | 87.1 | 41.6 |
| **MoE-1.8B×4 (Qwen)** | 86.1 | 85.4 | 44.9 | 88.6 | 87.7 | 42.5 | 88.7 | 88.0 | 43.0 |
| **MoE-2.7B×4 (Phi2)** | 85.9 | 84.9 | 43.2 | 87.5 | 86.4 | 41.8 | 88.5 | 87.7 | 41.8 |

### Table 7: Dense vs MoE comparison

| Model | MoE | VQAv2 | SQA^I | VQA^T | MMB | LLaVA^W |
|-------|-----|-------|-------|-------|-----|---------|
| StableLM | No | 74.5 | 62.0 | 48.8 | 58.2 | 83.2 |
| StableLM | Yes | 76.7 | 62.6 | 50.1 | 60.2 | 86.8 |
| Qwen | No | 74.9 | 60.2 | 48.3 | 60.6 | 86.3 |
| Qwen | Yes | 76.2 | 63.1 | 48.0 | 59.7 | 88.7 |
| Phi-2 | No | 75.6 | 67.8 | 50.0 | 65.0 | 91.3 |
| Phi-2 | Yes | 77.6 | 68.5 | 51.4 | 65.2 | 94.1 |
| OpenChat-7B | No | 77.9 | 69.0 | 54.7 | 66.9 | 89.7 |
| OpenChat-7B | Yes | 78.9 | 62.8 | 52.5 | 65.9 | 86.3 |

Note: OpenChat-7B MoE **degrades** — insufficient multimodal data for 10B+ sparse learning.

## Ablation Studies Summary

### Training Strategy (Table 6)
- **3-stage is critical**: Skipping Stage II and directly doing MoE (variant a) is worst
- Best: Stage II with Hybrid-FT → Stage III with LLaVA-FT (variant b)
- Dense model with more data (variant c) still worse than MoE (variant b)

### Tuning Parameters (Table 5a)
- Training only FFN layers (fc1, fc2, wg) ≈ full-parameter tuning, but 75% training time

### Number of Experts (Table 5b)
- 2 experts with top-2 (actually 4 experts, same activated) > 1 expert (dense)
- Improvement: +1.1% POPE, +0.6% SQA

### Top-k (Table 5c)
- Top-2 >> Top-1 across all benchmarks
- VQAv2: 76.2 vs 74.5, POPE: 88.7 vs 85.7

### MoE Architecture (Table 5d)
- **Interval** (alternating) is best overall
- First-Half ≈ Second-Half
- All layers as MoE → worse + 60% more training time

### Capacity Factor (Table 11)
- 1.5 consistently better than 1.0 across all model sizes

## Key Findings from Routing Analysis

1. **Expert specialization**: Experts learn distinct patterns — e.g., expert 3 dominates layers 17-27 in Phi2
2. **No modality preference**: Each expert handles both text and image tokens similarly — no hard modality split
3. **Token pathways**: Top-2 pathways concentrate on specific experts in deeper layers
4. **Balanced OpenChat issue**: 7B model routing stays too balanced (like initialization) → suggests insufficient data for sparse learning at that scale

## Full List of Benchmarks Used

1. **VQAv2** — visual question answering (eval server submission required)
2. **GQA** — compositional visual reasoning (local eval)
3. **VisWiz** — VQA from blind users (eval server submission)
4. **ScienceQA-IMG (SQA^I)** — science questions with images (local eval)
5. **TextVQA (VQA^T)** — text-rich image understanding (local eval)
6. **POPE** — object hallucination, 3 splits: random/popular/adversarial (local eval)
7. **MME** — perception + cognition benchmark (local eval)
8. **MMBench (MMB)** — multi-modal understanding (eval server submission)
9. **LLaVA-Bench (LLaVA^W)** — in-the-wild evaluation (GPT-based eval)
10. **MM-Vet** — integrated capabilities (GPT-based eval)

### Eval requirements:
- **Local eval possible**: GQA, ScienceQA, TextVQA, POPE, MME, SEED-Bench
- **Server submission needed**: VQAv2, MMBench, VisWiz
- **GPT API needed**: LLaVA-Bench, MM-Vet

## Training Objectives

L_total = L_regressive + α · L_aux

- **L_regressive**: Standard auto-regressive cross-entropy loss on generated text tokens
- **L_aux**: Load balancing loss (Fedus et al., 2022) — ensures balanced token distribution across experts
- **α = 0.01** (balancing coefficient)

## Comparison Points for Our Experiments

Our experiments test 3 router initialization schemes (vs paper's random FFN copy):
1. **random (author)** — original paper method: replicate FFN, random router init
2. **student-only (no_teacher)** — K-means initialized router, no KD
3. **teacher-student (teacher_kd)** — K-means initialized router + KD from dense teacher

Key question: Does KD-initialized routing improve over random routing?
