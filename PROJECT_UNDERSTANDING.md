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

### Four Router Initialization / Regularization Schemes

| Scheme | Code Name | `router_init_mode` | `entropy_loss_weight` | Description |
|---|---|---|---|---|
| Random (author) | `author` | `random` | 0 | Original paper: random `wg` init, no KD |
| Student-only | `student` | `no_teacher` | 0 | K-means init for `wg`, no KD during training |
| Teacher-Student (TS) | `TS` | `teacher_kd` | 0 | K-means init + KD loss from centroid teacher during Stage III |
| Entropy | `entropy` | `no_teacher` | 0.01–0.1 | K-means init + entropy minimization to force peaky routing |

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

### Entropy Regularization (`SimplifiedNormalizedGate`)

Added to the `no_teacher` gate to force router specialization without a teacher:

```
logits = logit_scale * dot(normalize(input), normalize(wg.weight))
probs  = softmax(logits)
H      = -(probs * log(probs + 1e-8)).sum(-1).mean()    # per-token entropy, averaged
loss   = aux_loss + entropy_loss_weight * H
```

Adding `+H` to the loss causes gradient descent to minimize entropy → pushes router toward peaky (confident) distributions like [0.99, 0.01, 0.0, 0.0]. This is the opposite of the standard "maximize entropy for diversity" trick — here we want each token to strongly prefer one expert.

**Key implementation details:**
- Logits computed **outside** `torch.no_grad()` so gradients flow: `H → probs → logits → wg.weight`
- `last_entropy_loss` stores raw H (unweighted) for logging, consistent with `last_moe_loss` pattern
- Only present in `SimplifiedNormalizedGate`, NOT in `NormalizedKDTopKGate` (TS)

### Entropy Warmup (`EntropyWarmupCallback`)

Linear warmup of `entropy_loss_weight` from 0 to the configured target over the first 10% of training steps, then constant. This prevents the entropy penalty from disrupting routing before the LM loss has established basic token-expert assignments.

```
if step < 0.1 * total_steps:
    entropy_weight = target_weight * (step / (0.1 * total_steps))
else:
    entropy_weight = target_weight
```

**Status:** Code ready in `router_callback.py`, automatically activated when `router_init_mode=no_teacher` and `entropy_loss_weight > 0`. Not yet used in a completed training run (the currently-running `qwen_entropy_w01` uses the pre-warmup code).

### Dynamic Hyperparameter Schedule (`RouterDistillationCallback`)

**⚠️ IMPORTANT: This callback is currently COMMENTED OUT in `train.py` (lines 1604–1612). All current TS experiments use FIXED hyperparameters — no dynamic scheduling.**

The callback was designed to update gate hyperparameters every step:

| Parameter | Start | End | Schedule |
|---|---|---|---|
| Temperature T | 4.0 | 1.0 | Linear decay |
| KD loss weight | 0.5 | 0.05 | Cosine decay |
| EMA decay | 0.999 | 0.95 | Linear decay |

But since it's disabled, current TS models train with:
- Temperature: **1.0 fixed** (from `router_temp_start`)
- KD weight: **0.01 fixed** (from `initial_kd_weight`)
- EMA decay: **0.999 fixed** (from `router_ema_start`)

**Implications:**
1. KD signal is very weak: 0.01 × KL_div ≈ 0.001–0.01 vs LM loss of ~2.5 → KD is <1% of total loss
2. Teacher drifts completely: at ema=0.999, K-means anchor is ~95% gone by step 3000 (0.999^3000 ≈ 0.05)
3. No temperature annealing: T=1.0 means no distribution softening (hard targets from the start)

**Recommended changes (not yet implemented):**
- Re-enable callback with: `temp_start=2.0, temp_end=1.0, weight_start=0.05, weight_end=0.01, ema_start=0.999, ema_end=0.999` (freeze teacher to preserve K-means anchor)

**Fixed hyperparameters (CLAUDE.md constraint):**
- `initial_kd_weight = 0.01`
- `router_temp_start = 1.0`
- `router_ema_start = 0.999`

---

## 3. Training Details

### Qwen-1.8B experiments (9240 steps each, 3 GPUs, batch=2, grad_accum=12)

| Variant | Checkpoint path (HPC) | entropy_weight | Status |
|---|---|---|---|
| author (random) | `checkpoints_qwen_author/llavaqwen-1.8b-finetune-moe/` | — | Complete |
| student (no KD) | `checkpoints_qwen_student/llavaqwen-1.8b-finetune-moe/` | — | Complete |
| teacher-student | `checkpoints_qwen_TS/llavaqwen-1.8b-finetune-moe/` | — | Complete |
| entropy (w=0.01) | `checkpoints_qwen_entropy/llavaqwen-1.8b-finetune-moe/` | 0.01 | Complete |
| entropy (w=0.1) | `checkpoints_qwen_entropy_w01/llavaqwen-1.8b-finetune-moe/` | 0.1 | Training (~76%) |

### Phi2-2.7B experiments (13860 steps each, 4 GPUs, batch=1, grad_accum=12)

| Variant | Checkpoint path (HPC) | entropy_weight | Status |
|---|---|---|---|
| author (random) | `checkpoints_phi/llavaphi-2.7b-finetune-moe/` | — | Complete |
| student (no KD) | `checkpoints_phi_student/llavaphi-2.7b-finetune-moe/` | — | Complete |
| teacher-student | `checkpoints_phi_TS/llavaphi-2.7b-finetune-moe/` | — | Complete |
| entropy (w=0.01) | `checkpoints_phi_entropy/llavaphi-2.7b-finetune-moe/` | 0.01 | Complete |

### StableLM-1.6B experiments (3 GPUs, batch=2, grad_accum=12)

| Variant | Checkpoint path (HPC) | entropy_weight | Status |
|---|---|---|---|
| author (random) | — | — | Not trained |
| student (no KD) | `checkpoints_stablelm_student/llava-stablelm-1.6b-finetune-moe/` | — | Complete |
| teacher-student | `checkpoints_stablelm_TS/llava-stablelm-1.6b-finetune-moe/` | — | Complete |
| entropy (w=0.01) | `checkpoints_stablelm_entropy/llava-stablelm-1.6b-finetune-moe/` | 0.01 | Complete |

---

## 4. Benchmark Evaluation Results

All local benchmarks evaluated on a single GPU (shared server constraint).

### MME Scoring Methodology

**IMPORTANT**: MME evaluations report **two component scores** that must be **summed together** to get the total MME score:

- **MME-P (Perception)**: Evaluation of object recognition, color identification, positional understanding, etc.
- **MME-C (Cognition)**: Evaluation of reasoning, counting, scene understanding, etc.
- **MME Total** = MME-P + MME-C

Example:
```
Qwen entropy (w=0.01): MME-P=1301.3 + MME-C=233.2 = MME Total=1534.5
```

**Do not report only the perception score or only the cognition score as "MME".** Always report the sum.

### POPE Evaluation Bug (CRITICAL FIX)

**⚠️ BUG IDENTIFIED AND FIXED (March 23, 2026):**

The original `eval_pope.py` script had a critical bug that caused completely wrong accuracy calculations for Popular and Random splits.

**The Problem:**
- The script matched model answers to annotation labels **by position** (line 1, line 2, ...) instead of by `question_id`
- Question IDs were completely mismatched across files:
  - Adversarial: IDs 1-3000 in both files (worked by coincidence)
  - Popular: IDs 1-3000 in annotations, but 20000001-20003000 in questions (WRONG)
  - Random: IDs 1-3000 in annotations, but 10000001-10002910 in questions (WRONG)

**Example of the bug:**
```
Annotation file line 1:  question_id=1, label="yes"
Question file adversarial:  question_id=1 → Matches ✓
Question file popular:  question_id=20000001 → WRONG match!
Question file random:  question_id=10000001 → WRONG match!
```

**The Fix:**
Modified `eval_pope()` function to:
1. Accept a `label_dict` (dict keyed by question_id) instead of reading lines in order
2. Match answers to labels using `question_id` lookup
3. Skip answers with missing question_ids and report warnings

**Before Fix (WRONG):**
- Adversarial: 85.9% (accidentally correct, but for wrong reasons)
- Popular: 87.9% (COMPLETELY WRONG due to ID mismatch)
- Random: 50.5% (COMPLETELY WRONG due to ID mismatch)

**After Fix (CORRECT):**
- Popular: 73.23% ← Model avoids hallucinating common objects
- Adversarial: 58.47% ← Moderate hallucination on co-occurring objects
- Random: 52.16% ← Baseline, barely above random guessing

**Comparison with POPE Paper LLaVA Baseline:**
Paper shows LLaVA struggling with hallucinations:
- Random: 54.43%, Popular: 52.43%, Adversarial: 50.77%

Our model performs much better (higher accuracy = less hallucination), suggesting better training or initialization.

**Root Causes Fixed:**
1. **Question ID mismatch**: Annotation files used IDs 1-3000, but question files used IDs 1-3000 (adversarial), 20000001+ (popular), 10000001+ (random)
   - Solution: Match by question **text** instead of ID
2. **Text typos**: 246 questions had "imange" instead of "image"
   - Solution: Normalize text before matching
3. **Position-based matching bug**: Original code matched by line order instead of content
   - Solution: Build dict keyed by normalized question text

**Files Modified:**
- `moellava/eval/eval_pope.py` — Fixed to match by normalized question text

### Qwen-1.8B Results

| Benchmark | Paper | Author | Student | TS | Entropy (w=0.01) |
|---|---|---|---|---|---|
| GQA | 61.5 | 62.02 | 61.95 | **62.15** | 61.73 |
| ScienceQA | 63.1 | **63.52** | 61.80 | 62.72 | 61.80 |
| TextVQA | 48.0 | 48.51 | 48.08 | **48.89** | 48.07 |
| POPE Pop Acc | 88.6 | **88.6%** | — | — | 87.9% |
| POPE Adv Acc | 86.1 | 86.1% | — | — | 85.9% |
| MME-P | — | — | — | — | 1301.3 |
| MME-C | — | — | — | — | 233.2 |
| MME Total | 1291.6 | **1572.8** | 1553.8 | 1561.5 | 1534.5 |

### Phi2-2.7B Results

| Benchmark | Paper | Author | Student | TS | Entropy (w=0.01) |
|---|---|---|---|---|---|
| GQA | 61.4 | 59.44 | **60.95** | — | 60.49 |
| ScienceQA | 68.5 | **71.75** | 70.76 | — | 70.08 |
| TextVQA | 51.4 | **52.11** | 51.69 | — | 51.50 |
| POPE Pop Acc | 87.5 | **87.5%** | — | — | 87.0% |
| POPE Adv Acc | 85.9 | **85.9%** | — | — | 85.7% |
| MME-P | — | — | — | — | 1337.1 |
| MME-C | — | — | — | — | 272.5 |
| MME Total | 1423.0 | **1685.7** | 1670.3 | — | 1609.6 |

### StableLM-1.6B Results

| Benchmark | Paper | Student | TS | Entropy (w=0.01) |
|---|---|---|---|---|
| GQA | 60.3 | **62.12** | 61.55 | 62.01 |
| ScienceQA | 62.6 | 60.55 | 60.58 | 59.75 |
| TextVQA | 50.1 | 49.94 | 50.06 | **50.18** |
| POPE Pop Acc | 85.3 | 74.3% | 74.5% | 74.4% |
| MME-P | — | 1362.2 | 1358.4 | **1369.4** |
| MME-C | — | 244.6 | 230.0 | **256.4** |
| MME Total | 1318.2 | 1606.9 | 1588.4 | **1625.8** |

*(Note: StableLM has no author/random baseline trained yet.)*

### Key Observations

1. **KD initialization does not consistently outperform random init** at the final checkpoint. Gains are marginal (within ~0.5-1%).
2. **Teacher-student (Qwen) is competitive**, occasionally best on GQA and TextVQA.
3. **All our variants exceed the paper's numbers** on several benchmarks — the Stage II starting checkpoint may be stronger than the paper's.
4. **POPE random ~50%** across all variants and backbones: this is anomalous. The popular and adversarial splits look normal (~85-87%). Possible scoring script issue or the "random" split has distribution properties that expose a model bias.
5. **Entropy (w=0.01) does NOT outperform student baseline** on Qwen or Phi2. Results are near-identical or slightly worse. On StableLM, entropy shows a slight edge in MME (+19 over student) and TextVQA (+0.24%), but loses on ScienceQA (-0.8%) and GQA (-0.11%).
6. **MME is consistently much higher than paper** across all backbones (~200-300 points above). This needs investigation.
7. **StableLM POPE scores (~74%) are much lower than paper's (85%)** — likely a different POPE eval methodology or the checkpoint itself is weaker on this benchmark.
8. **TS has no clear advantage over student** across any backbone — the KD signal at weight=0.01, T=1.0 with no dynamic scheduling may be too weak to matter.

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

### Existing Questions

1. **Why is MME so much higher than paper?** Our Qwen-author MME is 1572 vs paper's 1291. StableLM is 1606+ vs paper's 1318. Phi2 is 1609+ vs paper's 1423. This 200-300 point gap is consistent across all backbones. Hypotheses: (a) different Stage II checkpoint, (b) different eval split, (c) scoring script difference, (d) different image preprocessing.

2. **POPE random ~50% — is this a bug?** Popular and adversarial splits are normal (~85-87%). Only the "random" split gives ~50% accuracy. Consistent across all variants and backbones — suggests a systematic issue with eval methodology rather than model quality. Need to check: is our POPE random split the correct file? Are the answer labels correct?

3. **Does KD init help early convergence?** This is the core question the SQA checkpoint analysis should answer. If teacher-student (TS) reaches 60% SQA faster than random (author), that supports the hypothesis. Results stored in `eval_results/sqa_checkpoints/`. *(Analysis partially done for Qwen, need to revisit and complete.)*

4. **Does the diagnostic analysis reveal expert specialization by router scheme?** Comparing `dual/kmeans_5000/` with other scheme's `.pt` files could show whether KD-initialized routing leads to more specialized experts.

### New Questions (from recent experiments)

5. **Entropy regularization (w=0.01) shows no clear benefit.** On Qwen: GQA 61.73 vs Student 61.95, SQA 61.80 = Student, TextVQA 48.07 vs 48.08. On Phi2: all metrics slightly below student. On StableLM: slight edge in MME but below on SQA. **Is w=0.01 too weak?** The entropy loss at w=0.01 contributes roughly 0.01 × H to total loss, where H ≈ 1.0–1.4 for 4 experts, so the contribution is ~0.01–0.014 vs LM loss of ~2.5. That's <1% — similar to the KD weight problem. The w=0.1 experiment (qwen_entropy_w01, in progress) will test whether stronger entropy actually helps.

6. **RouterDistillationCallback is disabled — were TS results compromised?** All TS experiments used fixed T=1.0, kd_weight=0.01, ema=0.999 with no dynamic scheduling. The KD loss at these values is <1% of total loss. If re-enabled with stronger values (T=2.0, kd_weight=0.05), TS results might significantly improve. This is the biggest unexplored lever.

7. **K-means anchor erodes during TS training.** At ema=0.999, teacher_weight^{N=3000} ≈ 0.05 × original K-means init. By step 3000 (1/3 of Qwen training), the teacher has drifted almost entirely to track the student. The supposed anchor role of K-means is gone. Options: freeze teacher (no EMA), or use ema=0.9999 (preserves 41% at 9000 steps).

8. **StableLM POPE scores (~74%) are far below paper (85%).** Our POPE popular accuracy for StableLM is 74.3-74.5% vs paper's 85.3%. This is a ~11% gap, much larger than for Qwen or Phi2. Could indicate: (a) the StableLM Stage II checkpoint is different, (b) the eval images are different, or (c) the checkpoint itself was less well-trained. Need to verify the Stage II model used.

9. **Does entropy warmup help?** New `EntropyWarmupCallback` is coded and ready. It ramps entropy weight from 0 to target over first 10% of training. Hypothesis: letting LM loss settle before applying entropy pressure should lead to better-organized routing. Needs a training run to validate.

10. **Should we train a StableLM author (random) baseline?** Currently StableLM only has student, TS, and entropy. Without the random baseline, we can't determine if K-means init actually helps for StableLM. This is a gap in our comparison.

---

## 9. File Map

```
MoE-LLaVA_mine/
├── moellava/
│   ├── model/
│   │   ├── kd_gate.py                        ← Original KD gate (simpler, no normalization)
│   │   └── language_model/
│   │       ├── normalized_router_flexible.py  ← KDTopKGate, NormalizedKDTopKGate, SimplifiedNormalizedGate
│   │       ├── llava_qwen_moe.py              ← Qwen MoE model
│   │       ├── llava_phi_moe.py               ← Phi2 MoE model
│   │       └── llava_stablelm_moe.py          ← StableLM MoE model
│   ├── train/
│   │   ├── train.py                           ← Main training script, ModelArguments, entropy_loss_weight
│   │   ├── router_callback.py                 ← RouterDistillationCallback (DISABLED), EntropyWarmupCallback (active)
│   │   └── replace_gate.py                    ← Swaps DeepSpeed gate → KD gate
│   ├── eval/
│   │   ├── eval_gqa.py                        ← GQA scoring (NOT in gqa/ subfolder!)
│   │   └── model_routing_probe.py             ← Hooks router gates, saves .pt file
│   └── vis/
│       └── vis_dual_routing.py                ← Generates dual_analysis_layer_L.png
├── scripts/v1/
│   ├── qwen/
│   │   ├── finetune_moe.sh                    ← Qwen author (random)
│   │   ├── finetune_moe_entropy.sh            ← Qwen entropy w=0.01
│   │   └── finetune_moe_entropy_w01.sh        ← Qwen entropy w=0.1
│   ├── phi2/
│   │   ├── finetune_moe.sh                    ← Phi2 author
│   │   └── finetune_moe_entropy.sh            ← Phi2 entropy w=0.01
│   ├── stablelm/
│   │   ├── finetune_moe_student.sh            ← StableLM student (no_teacher)
│   │   └── finetune_moe_TS.sh                 ← StableLM teacher-student
│   └── eval/moe_llava/
│       ├── stablelm_all.sh                    ← StableLM: all 5 benchmarks
│       ├── stablelm_mme_gqa.sh                ← StableLM: MME + GQA only
│       ├── phi_entropy_all.sh                 ← Phi entropy: all 5 benchmarks
│       └── qwen_entropy_all.sh                ← Qwen entropy: all 5 benchmarks
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
├── plan.md                                    ← Project plan and TODO (KEEP UPDATED)
├── PROJECT_UNDERSTANDING.md                   ← This file
├── paper_reference.md                         ← Author's benchmark numbers for comparison
├── diagnostic_dataset/
│   ├── diagnostic_data.json                   ← Images with category labels
│   └── *.pt                                   ← Pre-computed routing probes
└── dual/
    └── kmeans_5000/dual_analysis_layer_*.png  ← Routing visualization outputs
```
