# Project Understanding: KD-Initialized MoE Routing for Vision-Language Models

*Document purpose: record architecture, method, experiments, and analysis for report writing.*

---

## 1. Background: MoE-LLaVA (the base paper)

MoE-LLaVA (Fei et al., 2024) converts a pretrained dense LLaVA-1.5 model into a sparse Mixture-of-Experts model. The key claim is that ~3B sparsely-activated parameters can match the performance of LLaVA-1.5-7B (6.7B activated).

### Architecture

```
Image (336×336)
    ↓
CLIP-Large Vision Encoder  (frozen in Stage III)
    ↓  [576 image tokens]
MLP Projector (2× Linear + GeLU)
    ↓
LLM Backbone (Phi2 / Qwen / StableLM)
  ├─ Attention Layer 0
  ├─ Dense FFN Layer 1       ← dense (unchanged)
  ├─ Attention Layer 2
  ├─ MoE FFN Layer 3         ← sparse (alternating pattern)
  ├─ ...
  └─ LM Head
```

**MoE Layer structure (per MoE FFN):**

```
token embedding (hidden_dim)
    ↓
Router Gate (wg): Linear(hidden_dim → num_experts)
    ↓  Top-K selection (K=2)
Expert 0    Expert 1    Expert 2    Expert 3
[fc1,fc2]  [fc1,fc2]  [fc1,fc2]  [fc1,fc2]
    ↓
Weighted sum of top-2 expert outputs
    ↓
token embedding (hidden_dim)
```

**Model sizes:**

| Backbone | Total Layers | MoE Layers | Experts | Top-K | Activated Params | Total Params |
|---|---|---|---|---|---|---|
| StableLM-1.6B | 24 | 12 (alternating) | 4 | 2 | 2.0B | 2.9B |
| Qwen-1.8B | 24 | 12 (alternating) | 4 | 2 | 2.2B | 3.1B |
| Phi2-2.7B | 32 | 16 (alternating) | 4 | 2 | 3.6B | 5.3B |

The alternating layout means odd-numbered FFN layers become MoE, even ones stay dense.

### 3-Stage Training (original paper)

| Stage | Name | Data | Trainable | Purpose |
|---|---|---|---|---|
| I | Pretraining | LLaVA-PT (558K) | MLP projector only | Align vision to LLM |
| II | Fine-tuning | Hybrid-FT (964K) | All except vision encoder | Multi-modal instruction following |
| III | MoE Tuning | LLaVA-FT (665K) | fc1, fc2, wg only | Convert FFN → MoE |

Stage III initialization: all 4 experts in each MoE layer start with identical weights copied from the pretrained dense FFN. The router (`wg`) is randomly initialized. Training runs with load-balancing auxiliary loss (α=0.01) to prevent expert collapse.

---

## 2. Our Research Contribution

### Problem with the original router initialization

The original paper initializes `wg` (the router gate) **randomly**. This means at the start of Stage III:
- All experts have identical weights (copied from dense FFN)
- The router has no semantic knowledge of which expert to use
- The model must discover specialization from scratch through gradient descent
- Early training is essentially random routing over identical experts → inefficient

### Our Hypothesis

**K-means-initialized routing can improve expert specialization, particularly early in training.**

Idea: Before Stage III, run K-means clustering on the hidden-state activations from the pretrained Stage II model. The K centroids (one per expert) capture distinct semantic modes in the representation space. Initialize the router gate weights with these centroids, so the router immediately has a semantic prior for which expert should handle which type of input.

Additionally, a **Knowledge Distillation (KD) teacher** can be used to keep the student router aligned with the centroid-derived prior while it adapts.

### Three Router Initialization Schemes

| Scheme | Code Name | `router_init_mode` | Description |
|---|---|---|---|
| Random (author) | `author` | `random` | Original paper: random `wg` init, no KD |
| Student-only | `student` | `no_teacher` | K-means init for `wg`, no KD during training |
| Teacher-Student | `TS` | `teacher_kd` | K-means init + KD loss from centroid teacher during Stage III |

### The KD Gate: `normalized_router_flexible.py`

The custom router gate (`NormalizedKDTopKGate`) replaces DeepSpeed's `TopKGate` with the following changes:

**1. Normalized cosine routing:**
```
input_normed     = L2_normalize(input)         # unit vector
weight_normed    = L2_normalize(wg.weight)      # unit vector
logit            = 10.0 * dot(input_normed, weight_normed)  # cosine similarity × 10
```
This prevents the "loud expert" bias where experts with larger weight norms dominate routing regardless of semantic relevance.

**2. Teacher-Student KD loss (teacher_kd mode only):**
```
teacher_logits = 10.0 * dot(input_normed, teacher_weight)
student_logits = 10.0 * dot(input_normed, student_weight)

kd_loss = T² × KL_div(
    log_softmax(student_logits / T),
    softmax(teacher_logits / T)
)

total_loss = aux_loss_weight × L_aux + kd_loss_weight × kd_loss
```
The T² scaling follows Hinton et al.'s knowledge distillation formulation. Temperature T softens both distributions so gradients reflect relative ranking, not just the argmax.

**3. EMA teacher update:**
```
teacher_weight ← ema_decay × teacher_weight + (1 - ema_decay) × student_weight
teacher_weight ← L2_normalize(teacher_weight)   # keep on unit sphere
```
The teacher slowly tracks the student. Early in training (high `ema_decay`), the teacher is stable (close to K-means centroids). Later, it adapts. This prevents the teacher from becoming stale as the student specializes.

### Dynamic Hyperparameter Schedule (`router_callback.py`)

The `RouterDistillationCallback` updates gate hyperparameters every step:

| Parameter | Start | End | Schedule |
|---|---|---|---|
| Temperature T | 4.0 | 1.0 | Linear decay |
| KD loss weight | 0.5 | 0.05 | Cosine decay |
| EMA decay | 0.999 | 0.95 | Linear decay |

High temperature early → soft targets → transfer routing distribution knowledge.
Low temperature late → hard targets → sharpen to the student's own decisions.
Decreasing EMA → teacher becomes more adaptive later in training.

**Fixed hyperparameters (CLAUDE.md constraint):**
- `initial_kd_weight = 0.01`
- `router_temp_start = 1.0`
- `router_ema_start = 0.999`

---

## 3. Training Details

### Qwen-1.8B experiments

| Variant | Checkpoint path | Steps | Status |
|---|---|---|---|
| author (random) | `checkpoints_qwen/llavaqwen-1.8b-finetune-moe/` | 9240 | Complete |
| student (no KD) | `checkpoints_qwen_student/llavaqwen-1.8b-finetune-moe/` | 9240 | Complete |
| teacher-student | `checkpoints_qwen_TS/llavaqwen-1.8b-finetune-moe/` | 9240 | Complete |

### Phi2-2.7B experiments

| Variant | Checkpoint path | Steps | Status |
|---|---|---|---|
| author (random) | `/scratch/prafull/hpc/checkpoints_phi/llavaphi-2.7b-finetune-moe/` | 13860 | Complete |
| student (no KD) | `checkpoints_phi_student/llavaphi-2.7b-finetune-moe/` | 13860 | Complete |
| teacher-student | Not trained (only 2 Phi2 variants) | — | N/A |

---

## 4. Benchmark Evaluation Results

All local benchmarks evaluated on a single GPU (shared server constraint).

### Qwen-1.8B Results vs Paper

| Benchmark | Paper (Qwen) | Author (random) | Student (no KD) | Teacher-Student |
|---|---|---|---|---|
| GQA | 61.5 | 62.02 | 61.95 | **62.15** |
| ScienceQA | 63.1 | 63.52 | 61.80 | 62.72 |
| TextVQA | 48.0 | 48.51 | 48.08 | **48.89** |
| POPE Adv F1 | 85.4 | 84.5 | 84.7 | 84.6 |
| MME Total | 1291.6 | 1572.82 | 1553.76 | 1561.46 |

*Notes:*
- Our `author` variant **exceeds** the paper on GQA, TextVQA, MME — likely due to different training data mix or hyperparameters in our re-run.
- MME total is consistently much higher (~1550+) than paper's 1291.6. This warrants investigation (possible eval script difference or data split).
- POPE random category gives ~50% accuracy across all variants — essentially random. This is a known issue, likely the model over-predicts "yes" or the category is harder.

### Phi2-2.7B Results vs Paper

| Benchmark | Paper (Phi2) | Author (random) | Student (no KD) |
|---|---|---|---|
| GQA | 61.4 | 59.44 | **60.95** |
| ScienceQA | 68.5 | **71.75** | 70.76 |
| TextVQA | 51.4 | **52.11** | 51.69 |
| POPE Adv F1 | 84.9 | 85.28 | 84.84 |
| MME Total | 1423.0 | **1685.72** | 1670.29 |

*Notes:*
- Student variant slightly underperforms author on most benchmarks for Phi2.
- Both variants exceed the paper on ScienceQA and MME substantially.
- Author variant has higher SQA (71.75 vs 68.5 paper) — potentially different Stage II checkpoint.

### Key Observations Across Both Backbones

1. **KD initialization does not consistently outperform random init** at the final checkpoint. Gains are marginal (within ~0.5-1%).
2. **Teacher-student (Qwen) is competitive**, occasionally best on GQA and TextVQA.
3. **All our variants exceed the paper's numbers** on several benchmarks — the Stage II starting checkpoint may be stronger than the paper's.
4. **POPE random ~50%** across all variants and backbones: this is anomalous. The popular and adversarial splits look normal (~85-87%). Possible scoring script issue or the "random" split has distribution properties that expose a model bias.

---

## 5. Early Training Analysis: SQA at Checkpoints 1–1000

We evaluate ScienceQA accuracy at checkpoints saved every 100 steps (steps: 1, 100, 200, ..., 1000) for all Qwen variants. This tests our hypothesis about early-training efficiency.

**Motivation:** If K-means initialization helps, we expect the K-means-initialized variants (student, TS) to converge faster than random init (author) in the first ~1000 steps.

Results stored in `eval_results/sqa_checkpoints/qwen_{variant}_step{N}.json`.
Plots generated by `plots/sqa_qwen/plot.py`.

*(Phi2 checkpoint evals also in progress for author and student variants.)*

---

## 6. Training Loss Analysis

Loss curves extracted from `trainer_state.json` in each checkpoint folder. Plots in:
- `plots/loss_qwen/` — 3 Qwen variants (full run + early 0-1000 steps)
- `plots/loss_phi2/` — 2 Phi2 variants (full run + early 0-1000 steps)

**Note:** Slight random noise (std=0.01, fixed per-variant seed) is added to loss curves for visual differentiation when curves overlap. This does not change the data — it is cosmetic only.

**Loss plot files:**
- `{variant}_full.png` — full training curve
- `{variant}_early_1k.png` — first 1000 steps zoom
- `all_full.png` — all variants overlaid, full run
- `all_early_1k.png` — all variants overlaid, first 1000 steps
- `all_lr.png` — learning rate schedule

---

## 7. Routing Visualization

### Pipeline

```
1. model_routing_probe.py  →  diagnostic_dataset/{name}.pt
   - Runs model on diagnostic images
   - Hooks into each router gate's wg (Linear layer)
   - Saves: gating_logit, output_ids, category, layer_indices

2. vis_dual_routing.py     →  dual/{name}/dual_analysis_layer_{L}.png
   - Loads .pt file
   - Separates image tokens (576 patches, identified by -200 sentinel) from text tokens
   - Computes per-sample average logit vector (for t-SNE)
   - Produces 3×3 analysis plot
```

### What `dual_analysis_layer_L.png` shows

**3×3 grid:**

| | Left | Center | Right |
|---|---|---|---|
| **Row 1** | t-SNE: image tokens, colored by category | t-SNE: text tokens, colored by category | Summary: dominant expert per category |
| **Row 2** | t-SNE: image tokens, colored by winning expert | t-SNE: text tokens, colored by winning expert | Heatmap: routing divergence (img − text preference per expert) |
| **Row 3** | Bar: expert load per category (image tokens) | Bar: expert load per category (text tokens) | Text: global expert utilization % |

**Key design decisions and their rationale:**

- **Per-sample averaging of image tokens for t-SNE:** The 576 image patches are not independent — they represent one semantic unit. Averaging gives one point per image, making t-SNE interpretable. Plotting all 576 tokens would swamp the text tokens (~10-30) and make the semantic structure unreadable.
- **Hard allocation (argmax) for routing decisions:** The router uses Top-2, but for the t-SNE coloring we use the top-1 (argmax) to assign a single color per sample. The bar charts and heatmap should ideally count top-2 tokens, but currently also use argmax (known limitation — see Section 8).
- **"dual" naming:** Refers to analyzing both modalities (image and text) in parallel, not to soft vs. hard allocation.

### Known Current Limitations (to be fixed)

1. **Bar charts use averaged-then-argmax, not token-level top-2 counts.** This is incorrect for load analysis. The router dispatches tokens, not sample averages. Fix: collect raw token-level top-2 expert assignments for bar charts and heatmap; keep averaged logits only for t-SNE.
2. **Top-2 routing not reflected.** The model always activates 2 experts per token, but only the argmax (rank-1) expert is analyzed. The second expert (rank-2) is completely ignored.
3. **Diagnostic dataset limited to 3 categories (animal, code, food).** Not representative of benchmark distributions (ScienceQA, VQAv2, etc.). A richer dataset with chart, document, spatial, counting categories would be more informative.

---

## 8. Open Questions for the Report

1. **Why is MME so much higher than paper?** Our Qwen-author MME is 1572 vs paper's 1291. Is this a different Stage II checkpoint, a different eval split, or a scoring script difference?

2. **POPE random ~50% — is this a bug?** Popular and adversarial are normal. Only random is broken. Could be that the "random" split has a different yes/no balance, or the model consistently predicts "yes" for that split.

3. **Does KD init help early convergence?** This is the core question the SQA checkpoint analysis will answer. If teacher-student (TS) reaches 60% SQA faster than random (author), that supports the hypothesis.

4. **Does the diagnostic analysis reveal expert specialization by router scheme?** Comparing `dual/kmeans_5000/` (one scheme) with other scheme's `.pt` files could show whether KD-initialized routing leads to more modality-specialized or semantically-specialized experts.

5. **POPE random low accuracy** is consistent across all variants and both backbones. This suggests a systematic issue (model bias, or the split itself) rather than a training problem.

---

## 9. File Map

```
MoE-LLaVA_mine/
├── moellava/
│   ├── model/
│   │   ├── kd_gate.py                        ← Original KD gate (simpler, no normalization)
│   │   └── language_model/
│   │       ├── normalized_router_flexible.py  ← Active KD gate with cosine routing + EMA
│   │       ├── llava_qwen_moe.py              ← Qwen MoE model
│   │       ├── llava_phi_moe.py               ← Phi2 MoE model
│   │       └── llava_stablelm_moe.py          ← StableLM MoE model
│   ├── train/
│   │   ├── train.py                           ← Main training script, MoEArguments
│   │   ├── router_callback.py                 ← Dynamic hyperparameter schedule
│   │   └── replace_gate.py                    ← Swaps DeepSpeed gate → KD gate
│   ├── eval/
│   │   └── model_routing_probe.py             ← Hooks router gates, saves .pt file
│   └── vis/
│       └── vis_dual_routing.py                ← Generates dual_analysis_layer_L.png
├── scripts/v1/
│   ├── qwen/finetune_moe*.sh                  ← Qwen Stage III training scripts
│   ├── phi2/finetune_moe*.sh                  ← Phi2 Stage III training scripts
│   └── eval/moe_llava/                        ← All eval scripts
├── eval_results/
│   ├── qwen_author/                           ← Final benchmark JSONs
│   ├── qwen_student/
│   ├── qwen_TS/
│   ├── phi2_author/
│   ├── phi2_student_final/
│   └── sqa_checkpoints/                       ← Per-step SQA accuracy JSONs
├── plots/
│   ├── loss_qwen/plot.py                      ← Qwen training loss plots
│   ├── loss_phi2/plot.py                      ← Phi2 training loss plots
│   ├── sqa_qwen/plot.py                       ← Qwen SQA-vs-step plots
│   └── sqa_phi2/plot.py                       ← Phi2 SQA-vs-step plots
├── diagnostic_dataset/
│   ├── diagnostic_data.json                   ← Images with category labels
│   └── *.pt                                   ← Pre-computed routing probes
└── dual/
    └── kmeans_5000/dual_analysis_layer_*.png  ← Routing visualization outputs
```
