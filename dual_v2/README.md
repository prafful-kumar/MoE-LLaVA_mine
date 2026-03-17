# Routing Analysis Plots — dual_v2/

## What is this directory?

This directory contains routing analysis plots for the **Qwen 1.8B MoE** model across 3 training variants:
- `qwen_author/` — Random router initialization (paper's original method)
- `qwen_student/` — K-means initialized router, no teacher KD
- `qwen_TS/` — K-means initialized router + teacher-student knowledge distillation

Each variant has **24 plots**: 12 MoE layers × 2 color modes.

---

## How was the data collected?

A diagnostic dataset of **250 images** (50 per category: animal, food, chart, scene, screenshot) was
fed through each trained model. At each MoE layer, forward hooks captured the **router logits**
(raw scores before softmax) — a [num_tokens, 4] tensor for each input sample.

Each sample also has a **prompt type** label:
- `attribute` — ask about a property (color, size, etc.)
- `count` — count objects or people
- `describe` — describe the overall scene or content
- `reason` — why/how reasoning question
- `yesno` — yes/no question

The 12 Qwen MoE layers are at even transformer block indices: 0, 2, 4, ..., 22.

---

## File naming

```
dual_analysis_layer_{N}_{color_mode}.png
```

- `N` = transformer block index (0, 2, 4, ..., 22)
- `color_mode` = `category` or `prompt_type`

**The only difference between `_category` and `_prompt_type` is the color grouping.**
The underlying routing data is identical — only what the colors represent changes.

- `_category`: colors = image content type (animal, food, chart, scene, screenshot)
  → Answers: "Does the router treat visually different content differently?"

- `_prompt_type`: colors = question type (attribute, count, describe, reason, yesno)
  → Answers: "Does the router treat different question types differently?"

---

## Plot layout — 3×3 grid

### Row 1: Semantic Space (t-SNE of routing logits)

| Left | Right |
|------|-------|
| IMAGE tokens | TEXT tokens |

- Each dot = **one sample** (one image or one text sequence), represented as the **average** of all its tokens' routing logits
- t-SNE reduces the 4-dimensional logit vector to 2D for visualization
- Colors = groups (category or prompt_type)

**What to look for:**
- If same-color dots cluster together → the router is sensitive to that grouping
- If colors are mixed randomly → the router ignores that grouping
- If image and text plots look similar → modality doesn't change routing behavior
- If image clusters but text doesn't → router specializes based on visual content, not question type

### Row 2: Routing Decisions (t-SNE colored by expert assignment)

| Left | Right | Far right |
|------|-------|-----------|
| IMAGE tokens | TEXT tokens | Routing Divergence heatmap |

- Same t-SNE positions as Row 1, but **re-colored by which expert won** (argmax of logit)
- Colors = Expert 0, 1, 2, 3

**Routing Divergence heatmap (top-right):**
- X-axis = category (or prompt type), Y-axis = expert index
- Each cell = difference between image routing preference and text routing preference for that (expert, group) pair
- Blue = image tokens go to this expert more than text tokens
- Red = text tokens go to this expert more than image tokens
- Near-zero (white) = no modality preference for this expert/group combination

**What to look for:**
- If expert assignments (Row 2) align with semantic clusters (Row 1) → routing has learned content-based specialization
- If Row 2 shows one dominant color everywhere → routing is collapsed (one expert wins almost all tokens)
- If the divergence heatmap is all near-zero → router treats image and text tokens the same (no modality split)

### Row 3: Expert Preference Bar Charts + Utilization

| Left | Right | Far right |
|------|-------|-----------|
| IMAGE token distribution | TEXT token distribution | Global utilization summary |

- Bar charts: for each group (animal/food/etc.), what fraction of tokens went to each expert (top-2 routing, so fractions sum to ~2.0 / 4 = 0.5 each)
- **This uses raw token counts, not averaged per sample** — so it reflects actual expert load

**Global utilization (bottom-right):**
- Shows overall % of all tokens assigned to each expert across the full dataset
- Ideal balanced routing = 25% each
- Also reports "Routing Entropy" for image and text separately

**What to look for:**
- If bars are similar across all groups → router is content-agnostic (expert load is uniform)
- If one group strongly prefers a specific expert → routing has specialized for that content type
- If image bars differ significantly from text bars → modality-aware specialization
- If one expert bar is much taller than others in Global Utilization → load imbalance

---

## How to read a layer progression

Look at the same plot (e.g., `_category`) across layers 0, 4, 8, 12, 16, 20, 22:

- **Early layers (0–6)**: routing tends to be less specialized, clusters may be diffuse
- **Middle layers (8–16)**: specialization often emerges here
- **Late layers (18–22)**: routing may be very concentrated on specific experts

Comparing across `qwen_author` / `qwen_student` / `qwen_TS` at the same layer reveals
whether K-means initialization or KD produces different routing patterns.

---

## Key questions this analysis tries to answer

1. **Does the router specialize by visual content?** (category plots)
2. **Does the router specialize by question type?** (prompt_type plots)
3. **Is there a modality split?** (image vs text divergence heatmap)
4. **Is expert load balanced?** (global utilization panel)
5. **Does KD training change routing behavior vs random init?** (compare variants)
6. **At which layer does specialization emerge?** (layer progression)
