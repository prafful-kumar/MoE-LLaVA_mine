# Diagnostic Plots — Tracking Sheet

All figures live in `diagnostics/figures/`.  
All data files live in `diagnostics/data/`.  
Scripts live in `diagnostics/`.

---

## Version Naming Convention

Files use the pattern `{figure_name}_{dataset}{n_samples}_{variant}.{ext}`:

| Tag | Meaning |
|-----|---------|
| `sqa500` | ScienceQA test set, 500 samples |
| `sqa1000` | ScienceQA test set, 1000 samples |
| `gqa1000` | GQA testdev-balanced, 1000 samples |
| `new_entropy` | Model B = `checkpoints_qwen_entropy` (topk_entropy_loss, **no** L_var) |
| `topk_var` | Model B = `checkpoints_qwen_entropy_topk_var` (topk_entropy + L_var) — **current** |

---

## Panel A — Routing Entropy & Feature Norms by Layer

**Question:** Does the routing entropy collapse with depth? Do hidden-state norms grow?

**Plot script:** `diagnostics/plot_layer_stats.py`  
**Collection script:** `diagnostics/collect_layer_stats.py`

### Sub-figures produced per run
| Filename pattern | Content |
|-----------------|---------|
| `layer_entropy_and_norm_{tag}.png` | **Main figure** — dual y-axis: routing entropy H (left) + L2 norm (right) across MoE layers |
| `routing_entropy_by_layer_{tag}.png` | Routing entropy only, single axis |
| `feature_norms_{tag}.png` | Feature norms only, single axis |

### Existing figures
| File | Models | Dataset | N | Status |
|------|--------|---------|---|--------|
| `layer_entropy_and_norm_sqa500_new_entropy.png` | author vs new_entropy | SQA | 500 | Superseded — wrong variant |
| `layer_entropy_and_norm_sqa1000_topk_var.png` | author vs entropy_topk_var | SQA | 1000 | **Current** |
| `layer_entropy_and_norm_gqa1000_topk_var.png` | author vs entropy_topk_var | GQA | 1000 | **Current** |

### Existing data files
| File | Model | Dataset | N |
|------|-------|---------|---|
| `layer_stats_A_sqa500_new_entropy.json` | author | SQA | 500 |
| `layer_stats_B_sqa500_new_entropy.json` | new_entropy | SQA | 500 |
| `layer_stats_A_sqa1000.json` | author | SQA | 1000 |
| `layer_stats_B_sqa1000.json` | entropy_topk_var | SQA | 1000 |
| `layer_stats_A_gqa1000.json` | author | GQA | 1000 |
| `layer_stats_B_gqa1000.json` | entropy_topk_var | GQA | 1000 |

### Key findings (SQA 1000 / GQA 1000)
- **Author:** H ∈ [1.23, 1.38] SQA / [1.17, 1.38] GQA — stable, 0 collapsed layers
- **Topk-entropy + L_var:** H ∈ [0.04, 0.50] SQA / [0.04, 0.35] GQA — **11–12/12 layers collapsed**
- Feature norms are nearly identical between models (both 14 → 41) — norms are not a differentiator
- Finding is consistent across both datasets

### How to regenerate
```bash
# SQA 1000
python diagnostics/plot_layer_stats.py \
    --data_suffix _sqa1000 --out_tag sqa1000_topk_var

# GQA 1000
python diagnostics/plot_layer_stats.py \
    --data_suffix _gqa1000 --out_tag gqa1000_topk_var
```

---

## Panel B — Within-k Routing Balance (Split Ratio / Secondary Weight CDF)

**Question:** Does the second expert in top-2 routing receive meaningful weight, or is it a dead expert?

**Plot script:** `diagnostics/plot_split_ratios.py`  
**Collection script:** `diagnostics/collect_split_ratios.py`

**Definitions:**
- `split_ratio` = max(p_top1, p_top2) / (p_top1 + p_top2) ∈ [0.5, 1.0]. Near 1.0 = one expert dominates.
- `p̃(2)` = 1 − split_ratio = secondary expert weight ∈ [0, 0.5]. Near 0 = dead second expert.

### Sub-figures produced per run
| Filename pattern | Content |
|-----------------|---------|
| `split_ratio_histogram_{tag}.png` | Histogram of split_ratio across all tokens × layers |
| `secondary_weight_cdf_{tag}.png` | **Main figure** — CDF of p̃(2); annotates % dead second expert |

### Existing figures
| File | Models | Dataset | N | Status |
|------|--------|---------|---|--------|
| `split_ratio_histogram_sqa500.png` | author, new_entropy, entropy_w01 | SQA | 500 | Superseded — wrong variant |
| `secondary_weight_cdf_sqa500.png` | author, new_entropy, entropy_w01 | SQA | 500 | Superseded — wrong variant |
| `split_ratio_histogram_sqa1000_topk_var.png` | author vs entropy_topk_var | SQA | 1000 | **Current** |
| `secondary_weight_cdf_sqa1000_topk_var.png` | author vs entropy_topk_var | SQA | 1000 | **Current** |
| `split_ratio_histogram_gqa1000_topk_var.png` | author vs entropy_topk_var | GQA | 1000 | **Current** |
| `secondary_weight_cdf_gqa1000_topk_var.png` | author vs entropy_topk_var | GQA | 1000 | **Current** |

### Existing data files
| File | Model | Dataset | N |
|------|-------|---------|---|
| `split_ratios_author_sqa500.npz` | author | SQA | 500 |
| `split_ratios_new_entropy_sqa500.npz` | new_entropy | SQA | 500 |
| `split_ratios_entropy_w01_sqa500.npz` | entropy_w01 | SQA | 500 |
| `split_ratios_author_sqa1000.npz` | author | SQA | 1000 |
| `split_ratios_entropy_topk_var_sqa1000.npz` | entropy_topk_var | SQA | 1000 |
| `split_ratios_author_gqa1000.npz` | author | GQA | 1000 |
| `split_ratios_entropy_topk_var_gqa1000.npz` | entropy_topk_var | GQA | 1000 |

### Key findings (SQA 1000 / GQA 1000)
| Metric | Author | Topk-entropy + L_var |
|--------|--------|----------------------|
| % dead 2nd expert (p̃(2) < 0.05) | 0% | **79–80%** |
| % near-balanced (p̃(2) > 0.4) | 67–70% | 3% |
| Median split ratio | 0.559 | **1.000** |

- Finding is consistent across SQA and GQA — the topk_entropy + L_var loss causes severe within-k collapse

### How to regenerate
```bash
# SQA 1000
python diagnostics/plot_split_ratios.py \
    --labels author entropy_topk_var \
    --data_tag sqa1000 --out_tag sqa1000_topk_var

# GQA 1000
python diagnostics/plot_split_ratios.py \
    --labels author entropy_topk_var \
    --data_tag gqa1000 --out_tag gqa1000_topk_var
```

---

## Panel C — Training Loss Curves

**Question:** Which initialization leads to lower step-1 loss / faster convergence?

**Plot script:** `diagnostics/plot_loss.py`  
**Data source:** `trainer_state.json` from each checkpoint's HPC directory (not in `diagnostics/data/`)

### Existing figures
| File | Models | Status |
|------|--------|--------|
| `training_loss.png` | author, student, TS, new_entropy (qwen variants) | Current |

### Models compared
| Legend | Checkpoint | Init |
|--------|-----------|------|
| Random init (author) | `checkpoints_qwen_author` | Random, no KD |
| K-means init (student) | `checkpoints_qwen_student` | K-means, no KD |
| Teacher-Student (TS) | `checkpoints_qwen_TS` | K-means + KD |
| New topk-entropy loss | `checkpoints_qwen_entropy` | K-means + KD + topk_entropy |

### How to regenerate
```bash
python diagnostics/plot_loss.py
```

---

## Panel D — Expert Utilization Heatmap

**Question:** Which experts receive the most token assignments across layers? Is utilization balanced?

**Plot script:** `diagnostics/plot_utilization.py`  
**Collection script:** `diagnostics/collect_utilization.py`

### Existing figures
| File | Models | Dataset | N | Status |
|------|--------|---------|---|--------|
| `expert_utilization_heatmap_sqa500_new_entropy.png` | author vs entropy_topk_var | SQA | 500 | Current (uses topk_var data despite the filename tag) |

### Existing data files
| File | Model | Dataset | N |
|------|-------|---------|---|
| `utilization_author_sqa500.npz` | author | SQA | 500 |
| `utilization_entropy_topk_var_sqa500.npz` | entropy_topk_var | SQA | 500 |

### How to regenerate
```bash
python diagnostics/plot_utilization.py \
    --label_a author_sqa500 --label_b entropy_topk_var_sqa500 \
    --data_dir diagnostics/data --output_dir diagnostics
```

> **TODO:** Re-collect utilization with n=1000 and GQA to match Panel A/B versioning.

---

## Figure E — Routing Stability Over Training

**Question:** Do routing assignments stabilise during training, and does student (K-means) init stabilise earlier than TS?

**Plot script:** `diagnostics/plot_routing_stability.py`  
**Collection script:** `diagnostics/collect_routing_stability.py`

### Existing figures
| File | Models | Status |
|------|--------|--------|
| `routing_stability.png` | student, TS | Current |

### Existing data files
| File | Variant | Checkpoints used |
|------|---------|-----------------|
| `stability_student.json` | student | checkpoint-1 … checkpoint-1000 (11 checkpoints) |
| `stability_TS.json` | TS | checkpoint-1 … checkpoint-1000 (11 checkpoints) |

### How to regenerate
```bash
python diagnostics/plot_routing_stability.py \
    --variants student TS --data_dir diagnostics/data --output_dir diagnostics
```

---

## Convergence (legacy)

**Plot script:** `diagnostics/plot_convergence.py`  
**Data:** `diagnostics/data/convergence_qwen.csv`

| File | Content | Note |
|------|---------|------|
| `convergence_qwen.png` | Step-1 ScienceQA accuracy across variants | Step-1 accuracy is identical across all variants (54.8%) — not informative. Replaced by Panel C (training loss). |

---

## Dataset Reference

| Tag | Full name | Question file | Image folder | N available |
|-----|-----------|--------------|-------------|-------------|
| `sqa` | ScienceQA test | `moellava/eval/scienceqa/llava_test_CQM-A.json` | `moellava/eval/scienceqa/images/test` | ~4241 |
| `gqa` | GQA testdev-balanced | `moellava/eval/gqa/llava_gqa_testdev_balanced.jsonl` | `moellava/eval/gqa/data/images` | 12578 |
| `pope` | POPE adversarial | `moellava/eval/pope/llava_pope_test.jsonl` | `moellava/eval/pope/val2014` | 8910 |

---

## Model Variant Reference

| Label | Checkpoint (HPC) | Router type | Loss | Init |
|-------|-----------------|------------|------|------|
| `author` | `checkpoints_qwen_author` | Standard dot-product gate | Aux loss only | Random |
| `student` | `checkpoints_qwen_student` | Cosine-normalized (KD) | Aux loss only | K-means |
| `TS` | `checkpoints_qwen_TS` | Cosine-normalized (KD) | Aux + KD | K-means |
| `new_entropy` | `checkpoints_qwen_entropy` | Cosine-normalized (KD) | Aux + topk_entropy (L_leak + L_imbal) | K-means |
| `entropy_topk_var` | `checkpoints_qwen_entropy_topk_var` | Cosine-normalized (KD) | Aux + topk_entropy + **L_var** | K-means |
| `entropy_w01` | `checkpoints_qwen_entropy_w01` | Cosine-normalized (KD) | Aux + raw-H (w=0.1) | K-means |
