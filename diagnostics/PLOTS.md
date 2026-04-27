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

## Panel F — Training-Distribution Routing Specialization

**Question:** Does the router differentiate by data source (COCO, GQA, OCR-VQA, TextVQA, VG, NLP) on the actual training distribution, or does it route all sources uniformly?

Source labels come from the data itself (image subfolder name / NLP for text-only) — not from a hand-crafted probe set, eliminating the cherry-picking objection.

**Probe script:** `moellava/eval/model_routing_probe_train_dist.py`  
**Vis script:** `moellava/vis/vis_train_dist_routing.py`  
**Shell driver:** `scripts/v1/stablelm/viz_train_dist_routing.sh`  
**Output root:** `train_dist_analysis/<ckpt_folder>/`

### Sub-figures produced per checkpoint
| Filename | Content |
|----------|---------|
| `specialization_heatmap.png` | **Fig A** — `[n_sources × n_layers]` grid; cell color = dominant expert, opacity = routing strength. Side-by-side: image tokens vs text tokens. |
| `specialization_score.png` | **Fig B** — Mean pairwise symmetric KL divergence between source routing distributions, per layer. Single-number comparison metric between methods. |
| `best_layer_detail.png` | **Fig C** — Grouped bar chart for top-3 most specialized layers; verifies heatmap signal. |
| `dual_detail/dual_analysis_layer_L_category.png` | Per-layer t-SNE + bar charts from `vis_dual_routing_v2.py` (color_by=source). |

### Data source labels (derived automatically from image paths)
| Label | Source | Rough N in training data |
|-------|--------|--------------------------|
| `coco` | COCO (LLaVA-Instruct) | ~500k |
| `gqa` | GQA scene graphs | ~72k |
| `ocr_vqa` | OCR-VQA | ~207k |
| `textvqa` | TextVQA | ~21k |
| `vg` | Visual Genome | ~86k |
| `nlp` | Text-only (nlp_tune.json) | ~41k |

### Existing data files
| File | Checkpoint | Samples/source | Status |
|------|-----------|----------------|--------|
| *(none yet — run the shell script)* | — | — | Pending |

### How to run
```bash
# Full pipeline: probe + 3 summary plots + dual detail for selected layers
bash scripts/v1/stablelm/viz_train_dist_routing.sh
# GPU 3, ~80 samples/source, ~480 total samples per checkpoint
```

To compare two checkpoints (method vs baseline), add both paths to `CKPT_FOLDERS` in the shell script and overlay their `specialization_score.png` plots.

### Comparing with existing vis infrastructure
The `.pt` output is format-compatible with `vis_dual_routing_v2.py` (uses `category=source_label`), so any per-layer detail plot from that script works on the new data without changes.

---

---

## NeurIPS Figure 1 — Routing Confidence Distribution (Qwen backbone)

**Folder:** `neurips/fig2_routing_confidence/`  
**Purpose:** Show that author (random init) keeps routing near-uniform (split_ratio ≈ 0.5) whereas our entropy variants collapse to committed routing (split_ratio ≈ 1.0). Motivates the entropy loss as a way to drive specialisation.

**Metric:** `split_ratio` = max(p_top1, p_top2) / (p_top1 + p_top2) ∈ [0.5, 1.0]  
- 0.5 → both selected experts get equal weight (uncertain/sharing)  
- 1.0 → one expert gets everything (committed/collapsed)

**Scripts:**
- `neurips/fig2_routing_confidence/collect.py` — hooks `gate.wg`, runs N SQA questions, saves per-layer split_ratios to `.npz`
- `neurips/fig2_routing_confidence/plot.py` — produces `v1_side_by_side`, `v2_overlay`, `v3_annotated` variants

### Collected data files (`neurips/fig2_routing_confidence/data/`)
| File | Model | Dataset | N | Key finding |
|------|-------|---------|---|-------------|
| `author.npz` | checkpoints_qwen_author | SQA | 1000 | median=0.535, 62% sharing, 0% committed |
| `adaptive_entropy.npz` | checkpoints_qwen_entropy | SQA | 1000 | median=0.999, 2.3% sharing, 78% committed |
| `power_adaptive.npz` | checkpoints_qwen_power_adaptive_v2 | SQA | 1000 | median=0.999, 2.3% sharing, 78% committed — identical to adaptive_entropy |

**Finding:** No bimodal distribution found in any Qwen variant. Author clusters near 0.5 (uncertain); entropy variants collapse to a spike at 1.0 (over-committed). The paper narrative is: author = uncertain, ours = commits. True bimodal target not yet achieved.

### How to run
```bash
python neurips/fig2_routing_confidence/collect.py \
    --model_path <ckpt_path> --label <label> --n_samples 1000 --gpu 6

python neurips/fig2_routing_confidence/plot.py \
    --files neurips/fig2_routing_confidence/data/author.npz \
            neurips/fig2_routing_confidence/data/adaptive_entropy.npz \
    --labels "Author (random)" "Ours (adaptive entropy)"
```

---

## NeurIPS Figure 2 — Routing Confidence Distribution (StableLM backbone)

**Folder:** `neurips/fig3_stablelm_routing/`  
**Purpose:** Same split_ratio analysis for StableLM backbone, comparing author (random), TS (K-means), and adaptive_entropy variants. Reveals that StableLM TS does NOT collapse (unlike Qwen entropy variants) while adaptive_entropy does.

**Scripts:**
- `neurips/fig3_stablelm_routing/collect.py` — same as fig2 but with two critical fixes:
  1. **DeepSpeed bypass:** StableLM checkpoints using `SimplifiedNormalizedGate` don't save `wg.weight`, causing meta-tensor crash when `deepspeed.init_inference` tries to move them to GPU. Fixed by monkey-patching `deepspeed.init_inference` to return a stub (`_DSEngineStub`) before any imports.
  2. **Port isolation:** `MASTER_PORT` derived from `CUDA_VISIBLE_DEVICES` (`12400 + gpu_id`) to prevent `Address already in use` when multiple jobs run in parallel.
- `neurips/fig3_stablelm_routing/plot.py` — identical to fig2/plot.py with updated paths

### Collected data files (`neurips/fig3_stablelm_routing/data/`)
| File | Model | Dataset | N | Key finding |
|------|-------|---------|---|-------------|
| `stablelm_author.npz` | hpc/random_no_KD_0.01_aux | SQA | 1000 | median=0.535, 61.9% sharing, 0.0% committed — uncertain |
| `stablelm_TS.npz` | checkpoints_stablelm_TS | SQA | 1000 | median=0.555, 46.8% sharing, 0.2% committed — also near-uncertain |
| `stablelm_adaptive_entropy.npz` | checkpoints_stablelm_adaptive_entropy | SQA | 1000 | median=0.995, 4.3% sharing, 69.4% committed — collapsed |

**Finding:** StableLM TS initialization alone does not cause routing collapse (unlike Qwen entropy variants). Only adaptive_entropy collapses. This shows the entropy loss is the driving factor, not K-means init alone.

### How to run
```bash
python neurips/fig3_stablelm_routing/collect.py \
    --model_path <ckpt_path> --label <label> \
    --n_samples 1000 --conv_mode stablelm --gpu 6

python neurips/fig3_stablelm_routing/plot.py \
    --files neurips/fig3_stablelm_routing/data/stablelm_author.npz \
            neurips/fig3_stablelm_routing/data/stablelm_adaptive_entropy.npz \
    --labels "Author (random)" "Adaptive entropy"
```

**Note:** Always apply the DeepSpeed bypass patch before running any StableLM collect script. See the top of `collect.py` for the pattern.

---

## NeurIPS Figure 3 — Routing Entropy Over Training Steps

**Folder:** `neurips/fig_entropy_trajectory/`  
**Purpose:** Demonstrate that Fisher initialization (K-means centroids) drives routing entropy to near-zero **from step 1**, before any gradient updates. The author baseline (random init) stays near maximum entropy (ln 4 ≈ 1.386) throughout all 9240 training steps. This is the core evidence that initialization quality determines routing structure.

**Metric:** Mean routing entropy H = -Σ p_j log(p_j) over all 4 experts, averaged across all 12 MoE layers and all tokens in N samples.  
Reference lines:
- `ln(4) ≈ 1.386` — uniform over all 4 experts (maximum entropy, router is completely undecided)
- `ln(2) ≈ 0.693` — ideal top-2: both selected experts get exactly equal weight

**Scripts:**
- `neurips/fig_entropy_trajectory/collect.py` — discovers all `checkpoint-N` subdirs under `--model_path`, loads each, hooks `gate.wg`, computes per-layer H across N questions, saves incrementally to JSON. Frees GPU between checkpoints with `del model; torch.cuda.empty_cache()`.
- `neurips/fig_entropy_trajectory/plot.py` — four variants: v1 (linear x), v2 (heatmap first vs last), v3 (per-layer small multiples), v4 (log x-axis — best for the 1…1000…9240 spacing)

**Output JSON format:**
```json
{
  "label": "qwen_author",
  "steps": [1, 100, 200, ..., 9240],
  "mean_H": [1.346, 1.344, ...],
  "layer_H": { "0": [h_step1, h_step2, ...], "4": [...], ... }
}
```

### Collected data files (`neurips/fig_entropy_trajectory/data/`)
| File | Model | Checkpoints | N/ckpt | Key finding |
|------|-------|------------|--------|-------------|
| `qwen_author.json` | checkpoints_qwen_author | 12 (steps 1–9240) | 200 | H ≈ 1.34 at ALL steps — flat, never converges |
| `qwen_student.json` | checkpoints_qwen_student | 12 (steps 1–9240) | 200 | H = 0.18 at step 1 — already 7.5× lower than author, stays flat |

### Generated figures (`neurips/fig_entropy_trajectory/figures/`)
| File | Description | Recommended for |
|------|-------------|----------------|
| `v1_mean_entropy.pdf/.png` | Mean H vs step, linear x-axis | Quick reference |
| `v2_first_last_heatmap.pdf/.png` | Per-layer heatmap: step 1 vs final checkpoint | Supplementary |
| `v3_per_layer.pdf/.png` | Small multiples: one subplot per MoE layer | Supplementary |
| `v4_log_scale.pdf/.png` | Mean H vs step, **log x-axis** | **Paper figure** — best for irregular 1…1000…9240 spacing |

### Key findings
| Metric | Author (random init) | Student (Fisher init) |
|--------|---------------------|----------------------|
| H at step 1 | **1.346** (≈ max entropy ln 4) | **0.180** (7.5× lower) |
| H at step 1000 | 1.344 (unchanged) | 0.186 (unchanged) |
| H at step 9240 (final) | 1.344 (unchanged) | 0.188 (unchanged) |
| Trajectory | Completely flat — training does not reduce entropy | Flat from step 1 — structure already encoded by K-means init |

**Interpretation:** Routing structure is determined by initialization, not by gradient updates. Fisher/K-means centroids provide semantically meaningful router init that immediately (before training) encodes low-entropy, decisive routing. Random init never recovers.

### How to run
```bash
# Collect entropy at each checkpoint (runs ~20 min — loads 12 checkpoints sequentially)
python neurips/fig_entropy_trajectory/collect.py \
    --model_path /home/prafull/scratch/hpc/checkpoints_qwen_author/llavaqwen-1.8b-finetune-moe \
    --label qwen_author --n_samples 200 --conv_mode qwen --gpu 2

python neurips/fig_entropy_trajectory/collect.py \
    --model_path /home/prafull/scratch/hpc/checkpoints_qwen_student/llavaqwen-1.8b-finetune-moe \
    --label qwen_student --n_samples 200 --conv_mode qwen --gpu 3

# Generate all figure variants
python neurips/fig_entropy_trajectory/plot.py \
    --files neurips/fig_entropy_trajectory/data/qwen_author.json \
            neurips/fig_entropy_trajectory/data/qwen_student.json \
    --labels "Author (random init, aux loss)" "Student (Fisher init, KD)"
```

**Note:** `collect.py` includes the same DeepSpeed port-isolation fix as fig3 (`MASTER_PORT = 12400 + gpu_id`). Use different `--gpu` values when running both jobs in parallel.

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
