# MoE-LLaVA Scripts Inventory

This file lists all Python and Bash scripts in the repository, organized by directory structure with brief descriptions.

---

## Root Level Scripts

```
/
├── compare_all_architectures.py       - Compare performance across different model architectures
├── convergence_speed.py               - Analyze and visualize training convergence rates
├── convergence_speed_author.py        - Author's reference implementation for convergence analysis
├── diagnostic_probe_0.py              - Probe diagnostic data from trained models
├── diagnostic_dataset_claude_1.py     - Generate diagnostic dataset variant 1 (Claude)
├── diagnostic_dataset_claude_2.py     - Generate diagnostic dataset variant 2 (Claude)
├── diagnostic_dataset_claude_vtab.py  - Generate diagnostic dataset for VTAB (Visual Task Adaptation Benchmark)
├── diagnostic_dataset_gemini_2.py     - Generate diagnostic dataset variant 2 (Gemini)
├── inspect_pt.py                      - Inspect PyTorch model checkpoints and weights
├── plot_loss.py                       - Plot training loss curves
└── predict.py                         - Run inference and generate predictions
```

---

## `/diagnostic_dataset/` - Diagnostic Dataset Generation

```
diagnostic_dataset/
└── build_diagnostic_data.py           - Build diagnostic dataset from raw data
```

---

## `/eval_results/` & `/excel_results/` - Results Processing

```
eval_results/
└── generate_excel.py                  - Generate Excel spreadsheet from eval results

excel_results/
└── generate_excel.py                  - Generate Excel spreadsheet (variant)

eval_results_mine/
└── generate_tables.py                 - Generate formatted evaluation tables
```

---

## `/get_kmeans_centroids/` - K-Means Centroid Computation

```
get_kmeans_centroids/
├── compute_fisher_directions.py       - Compute Fisher information directions for weight importance
├── compute_fisher_directions_phi.py   - Compute Fisher directions for Phi2 backbone
├── compute_fisher_directions_qwen.py  - Compute Fisher directions for Qwen backbone
├── get_centroids.py                   - Extract k-means centroids from FFN activations
├── get_centroids_adv.py               - Advanced k-means centroid extraction with options
├── verify_centroids.py                - Validate computed k-means centroids
├── run.sh                             - Script to run centroid computation pipeline
└── run_adv.sh                         - Script to run advanced centroid computation
```

---

## `/moellava/eval/` - Evaluation Scripts

```
moellava/eval/
├── eval_gpt_mmvet.py                  - Evaluate using GPT-4 review on MM-Vet benchmark
├── eval_gpt_review.py                 - GPT review-based evaluation for VQA tasks
├── eval_gpt_review_bench.py           - GPT review for benchmark datasets
├── eval_gpt_review_visual.py          - GPT review for visual question answering
├── eval_gqa.py                        - Evaluate on GQA (Grounded QA) benchmark
├── eval_mmlu.py                       - Evaluate on MMLU benchmark
├── eval_pope.py                       - Evaluate on POPE (Object Hallucination) benchmark
├── eval_science_qa.py                 - Evaluate on ScienceQA benchmark
├── eval_science_qa_gpt4.py            - ScienceQA evaluation with GPT-4 review
├── eval_science_qa_gpt4_requery.py    - ScienceQA with GPT-4 re-querying logic
├── eval_textvqa.py                    - Evaluate on TextVQA benchmark
├── generate_webpage_data_from_table.py- Convert evaluation results to webpage format
├── m4c_evaluator.py                   - Evaluator for multi-modal content understanding
├── model_qa.py                        - Generic VQA model evaluation wrapper
├── model_routing_probe.py              - Analyze expert routing decisions during inference
├── model_routing_probe_v2.py           - Improved routing probe with enhanced logging
├── model_vqa.py                       - Main VQA model evaluation script
├── model_vqa_loader.py                - Helper to load models for VQA evaluation
├── model_vqa_mmbench.py               - Evaluate on MMBench benchmark
├── model_vqa_qbench.py                - Evaluate on Q-Bench quality assessment benchmark
├── model_vqa_science.py               - Evaluate on science-related benchmarks
├── qa_baseline_gpt35.py               - GPT-3.5 baseline for QA tasks
├── run_llava.py                       - Run LLaVA model for evaluation
└── summarize_gpt_review.py            - Summarize GPT review results
```

### `/moellava/eval/MME/` - MME Benchmark Specific

```
moellava/eval/MME/
├── convert_answer_to_mme.py           - Convert model answers to MME format
└── eval_tool/
    └── calculation.py                 - MME metric calculation
```

### `/moellava/eval/mm-vet/` - MM-Vet Benchmark Specific

```
moellava/eval/mm-vet/
└── convert_answers.py                 - Convert answers to MM-Vet format
```

### `/moellava/eval/seed_bench/` - SEED Benchmark Specific

```
moellava/eval/seed_bench/
└── extract_video_frames.py            - Extract frames from videos for SEED benchmark
```

---

## `/moellava/model/` - Core Model Architecture

```
moellava/model/
├── apply_delta.py                     - Apply weight deltas to base models
├── builder.py                         - Build/load models from config (handles MoE conversion)
├── consolidate.py                     - Consolidate model weights
├── kd_gate.py                         - Knowledge Distillation gate implementation
├── llava_arch.py                      - Base LLaVA architecture classes
├── make_delta.py                      - Create weight deltas from model differences
├── utils.py                           - Model utility functions
```

### `/moellava/model/language_model/` - Backbone Models (Dense & MoE Variants)

```
moellava/model/language_model/
├── llava_llama.py                     - Llama base model
├── llava_llama_moe.py                 - Llama with MoE layers
├── llava_minicpm.py                   - MiniCPM base model
├── llava_minicpm_moe.py               - MiniCPM with MoE layers
├── llava_mistral.py                   - Mistral base model
├── llava_mistral_moe.py               - Mistral with MoE layers
├── llava_mpt.py                       - MPT base model
├── llava_phi.py                       - Phi base model
├── llava_phi_moe.py                   - Phi with MoE layers ⭐ [FIXED HYPERPARAMS]
├── llava_qwen.py                      - Qwen base model
├── llava_qwen_moe.py                  - Qwen with MoE layers ⭐ [FIXED HYPERPARAMS]
├── llava_qwen1_5.py                   - Qwen1.5 base model
├── llava_qwen1_5_moe.py               - Qwen1.5 with MoE layers
├── llava_stablelm.py                  - StableLM base model
├── llava_stablelm_moe.py              - StableLM with MoE layers ⭐ [FIXED HYPERPARAMS]
├── normalized_router_flexible.py      - Primary KD router with EMA teacher & normalization
├── normalized_router_init_only.py     - Router with init-only mode
```

### `/moellava/model/language_model/phi/` - Phi Backbone Config

```
moellava/model/language_model/phi/
├── configuration_phi.py               - Phi model configuration
└── modeling_phi.py                    - Phi model implementation
```

### `/moellava/model/language_model/qwen/` - Qwen Backbone Config

```
moellava/model/language_model/qwen/
├── configuration_qwen.py              - Qwen model configuration
├── cpp_kernels.py                     - C++ kernel wrappers for Qwen
├── modeling_qwen.py                   - Qwen model implementation
├── qwen_generation_utils.py           - Qwen-specific generation utilities
└── tokenization_qwen.py               - Qwen tokenizer
```

### `/moellava/model/language_model/stablelm/` - StableLM Backbone Config

```
moellava/model/language_model/stablelm/
├── configuration_stablelm_epoch.py    - StableLM configuration
├── modeling_stablelm_epoch.py         - StableLM model implementation
└── tokenization_arcade100k.py         - StableLM tokenizer
```

### `/moellava/model/language_model/minicpm/` - MiniCPM Backbone Config

```
moellava/model/language_model/minicpm/
├── configuration_minicpm.py           - MiniCPM configuration
└── modeling_minicpm.py                - MiniCPM model implementation
```

### `/moellava/model/language_model/mpt/` - MPT Backbone Config

```
moellava/model/language_model/mpt/
├── adapt_tokenizer.py                 - Adapt tokenizer for MPT
├── attention.py                       - MPT attention mechanism
├── blocks.py                          - MPT transformer blocks
├── configuration_mpt.py               - MPT configuration
├── custom_embedding.py                - Custom embedding layer
├── flash_attn_triton.py               - Flash attention Triton implementation
├── hf_prefixlm_converter.py           - Convert to HuggingFace PrefixLM format
├── meta_init_context.py               - Meta initialization context
├── modeling_mpt.py                    - MPT model implementation
├── norm.py                            - Normalization layers
└── param_init_fns.py                  - Parameter initialization functions
```

### `/moellava/model/multimodal_encoder/` - Vision Encoder

```
moellava/model/multimodal_encoder/
├── builder.py                         - Build vision encoders (CLIP/SigLIP)
├── clip_encoder.py                    - CLIP vision encoder
└── siglip_encoder.py                  - SigLIP vision encoder
```

### `/moellava/model/multimodal_projector/` - Multimodal Projector

```
moellava/model/multimodal_projector/
├── builder.py                         - Build projector (Linear/QFormer)
├── pool_block.py                      - Pooling projection block
├── qformer.py                         - QFormer projector
└── simple_block.py                    - Simple linear projector
```

---

## `/moellava/train/` - Training Infrastructure

```
moellava/train/
├── llama_flash_attn_monkey_patch.py  - Monkey patch for Llama flash attention
├── llama_xformers_attn_monkey_patch.py - Monkey patch for Llama xformers attention
├── llava_trainer.py                  - Custom HuggingFace Trainer with MoE support
├── replace_gate.py                   - Replace dense gates with KD router gates
├── router_callback.py                - Dynamic callback for scheduling temperature/KD weight ⭐ [FIXED HYPERPARAMS]
├── train.py                          - Main training script with MoE arguments ⭐ [FIXED HYPERPARAMS]
├── train_mem.py                      - Memory-optimized training variant
└── train_xformers.py                 - Training with xformers optimizations
```

---

## `/moellava/serve/` - Model Serving

```
moellava/serve/
├── cli.py                             - Command-line interface for inference
├── cli_multi.py                       - Multi-instance CLI
├── controller.py                      - Distributed inference controller
├── gradio_utils.py                    - Gradio UI utilities
├── gradio_web_server.py               - Gradio web UI server
├── model_worker.py                    - Model worker for distributed serving
├── register_worker.py                 - Register workers with controller
├── test_message.py                    - Test message protocol
└── utils.py                           - Serving utilities
```

---

## `/moellava/vis/` - Visualization & Analysis

```
moellava/vis/
├── vis1.py                            - Visualization variant 1
├── vis2.py                            - Visualization variant 2
├── vis3.py                            - Visualization variant 3
├── vis_dual_routing.py                - Visualize expert routing decisions
├── vis_dual_routing_v2.py             - Improved routing visualization
└── vis_tsne.py                        - t-SNE visualization of embeddings
```

---

## `/plots/` - Plot Generation

```
plots/
├── loss_phi2/
│   └── plot.py                        - Plot Phi2 training loss curves
├── loss_qwen/
│   └── plot.py                        - Plot Qwen training loss curves
├── mme_qwen/
│   └── plot.py                        - Plot Qwen MME benchmark results
├── sqa_phi2/
│   └── plot.py                        - Plot Phi2 ScienceQA results
└── sqa_qwen/
    └── plot.py                        - Plot Qwen ScienceQA results

plots_mine/
└── plot_qwen_checkpoints.py           - Plot Qwen checkpoints across training steps
```

---

## `/scripts/` - Training & Evaluation Scripts

### `/scripts/` Root - Utility Scripts

```
scripts/
├── convert_gqa_for_eval.py            - Convert GQA dataset to eval format
├── convert_mmbench_for_submission.py  - Convert MMBench results for submission
├── convert_mmvet_for_eval.py          - Convert MM-Vet data for evaluation
├── convert_seed_for_submission.py     - Convert SEED benchmark results
├── convert_sqa_to_llava.py            - Convert ScienceQA to LLaVA format
├── convert_sqa_to_llava_base_prompt.py - Convert ScienceQA with base prompts
├── convert_vizwiz_for_submission.py   - Convert VizWiz results
├── convert_vqav2_for_submission.py    - Convert VQAv2 results
├── extract_mm_projector.py            - Extract multimodal projector weights
├── finetune.sh                        - Base fine-tuning script
├── finetune_full_schedule.sh          - Fine-tuning with full schedule
├── finetune_lora.sh                   - LoRA fine-tuning
├── finetune_qlora.sh                  - QLoRA fine-tuning
├── finetune_sqa.sh                    - Fine-tuning on ScienceQA
├── merge_lora_weights.py              - Merge LoRA weights into base model
├── merge_moe_lora_weights.py          - Merge LoRA weights for MoE models
├── pretrain.sh                        - Vision projector pretraining
├── pretrain_xformers.sh               - Pretraining with xformers
├── sqa_eval_batch.sh                  - Batch ScienceQA evaluation
├── sqa_eval_gather.sh                 - Gather ScienceQA results
```

### `/scripts/v1/eval/llava/` - Dense Model Evaluation

```
scripts/v1/eval/llava/
├── gqa.sh                             - Evaluate dense model on GQA
├── llavabench.sh                      - Evaluate on LLaVA-Bench
├── mmbench.sh                         - Evaluate on MMBench
├── mmbench_cn.sh                      - Evaluate on MMBench-CN
├── mme.sh                             - Evaluate on MME benchmark
├── mmvet.sh                           - Evaluate on MM-Vet
├── pope.sh                            - Evaluate on POPE
├── seed.sh                            - Evaluate on SEED-Bench
├── sqa.sh                             - Evaluate on ScienceQA
├── textvqa.sh                         - Evaluate on TextVQA
├── vizwiz.sh                          - Evaluate on VizWiz
└── vqav2.sh                           - Evaluate on VQAv2
```

### `/scripts/v1/eval/moe_llava/` - MoE Model Evaluation ⭐ [Uses --include localhost:<gpu_ids>]

```
scripts/v1/eval/moe_llava/
├── gqa.sh                             - Evaluate MoE model on GQA (single model)
├── gqa_all.sh                         - Evaluate all MoE checkpoints on GQA
├── gqa_checkpoints_qwen.sh            - Evaluate Qwen MoE checkpoints on GQA
├── llavabench.sh                      - Evaluate MoE on LLaVA-Bench
├── mmbench.sh                         - Evaluate MoE on MMBench
├── mmbench_all.sh                     - Evaluate all MoE checkpoints on MMBench
├── mmbench_cn.sh                      - Evaluate MoE on MMBench-CN
├── mme.sh                             - Evaluate MoE on MME
├── mme_all.sh                         - Evaluate all MoE checkpoints on MME
├── mme_checkpoints_qwen.sh            - Evaluate Qwen MoE checkpoints on MME
├── mmvet.sh                           - Evaluate MoE on MM-Vet
├── phi2_student_final_all.sh          - Evaluate Phi2 student final checkpoints
├── phi_entropy_all.sh                 - Evaluate Phi2 entropy mode on all benchmarks
├── pope.sh                            - Evaluate MoE on POPE
├── pope_phi_student.sh                - Evaluate Phi2 student on POPE
├── pope_qwen_author.sh                - Evaluate Qwen author baseline on POPE
├── pope_qwen_student.sh               - Evaluate Qwen student on POPE
├── pope_qwen_TS.sh                    - Evaluate Qwen TS (Teacher-Student) on POPE
├── seed.sh                            - Evaluate MoE on SEED-Bench
├── seed_all.sh                        - Evaluate all MoE checkpoints on SEED
├── sqa.sh                             - Evaluate MoE on ScienceQA
├── sqa_all.sh                         - Evaluate all MoE checkpoints on ScienceQA
├── sqa_checkpoints_phi2.sh            - Evaluate Phi2 MoE checkpoints on ScienceQA
├── sqa_checkpoints_qwen.sh            - Evaluate Qwen MoE checkpoints on ScienceQA
├── stablelm_all.sh                    - Evaluate all StableLM MoE checkpoints
├── stablelm_mme_gqa.sh                - Evaluate StableLM on MME and GQA
├── textvqa.sh                         - Evaluate MoE on TextVQA
├── textvqa_all.sh                     - Evaluate all MoE checkpoints on TextVQA
├── vizwiz.sh                          - Evaluate MoE on VizWiz
├── vizwiz_all.sh                      - Evaluate all MoE checkpoints on VizWiz
├── vqav2.sh                           - Evaluate MoE on VQAv2
└── vqav2_all.sh                       - Evaluate all MoE checkpoints on VQAv2
```

### `/scripts/v1/phi2/` - Phi2 Backbone Training Scripts

```
scripts/v1/phi2/
├── finetune.sh                        - Fine-tune dense Phi2
├── finetune_moe.sh                    - Fine-tune Phi2 MoE
├── finetune_moe_entropy.sh            - Phi2 MoE with entropy loss
├── pretrain.sh                        - Pretrain Phi2 vision projector
```

### `/scripts/v1/qwen/` - Qwen Backbone Training Scripts

```
scripts/v1/qwen/
├── finetune.sh                        - Fine-tune dense Qwen
├── finetune_moe.sh                    - Fine-tune Qwen MoE
├── finetune_moe_entropy.sh            - Qwen MoE with entropy loss
├── finetune_moe_entropy_w01.sh        - Qwen MoE entropy with w=0.1
├── pretrain.sh                        - Pretrain Qwen vision projector
├── viz_routing_old_3cls.sh            - Visualize routing (old 3-class variant)
└── viz_routing_qwen_v2.sh             - Visualize routing (v2 improved)
```

### `/scripts/v1/stablelm/` - StableLM Backbone Training Scripts

```
scripts/v1/stablelm/
├── finetune.sh                        - Fine-tune dense StableLM
├── finetune_moe.sh                    - Fine-tune StableLM MoE
├── finetune_moe_entropy.sh            - StableLM MoE with entropy loss
├── finetune_moe_student.sh            - StableLM MoE student mode
├── finetune_moe_TS.sh                 - StableLM MoE Teacher-Student mode
├── pretrain.sh                        - Pretrain StableLM vision projector
├── router_hyp_dyn.sh                  - Dynamic hyperparameter sweep for routers
├── run_experiments.sh                 - Run multiple experiments sequentially
├── run_experiments_v2.sh              - Run experiments (v2 variant)
├── run_multiple.sh                    - Run multiple training jobs
├── run_prioritized_experiments.sh     - Run prioritized experiment queue
├── viz_routing_dist.sh                - Visualize routing distribution
└── viz_routing_dist_v2.sh             - Visualize routing distribution (v2)
```

### `/scripts/v1/openchat/` - OpenChat Backbone Training Scripts

```
scripts/v1/openchat/
├── finetune.sh                        - Fine-tune dense OpenChat
├── finetune_moe.sh                    - Fine-tune OpenChat MoE
└── pretrain.sh                        - Pretrain OpenChat vision projector
```

---

## Core Library Files

```
moellava/
├── __init__.py                        - Package initialization
├── constants.py                       - Constants (conv modes, special tokens)
├── conversation.py                    - Conversation template definitions
├── mm_utils.py                        - Multimodal utility functions
└── utils.py                           - General utility functions
```

---

## Key Notes

⭐ **Files with FIXED HYPERPARAMETERS** (from CLAUDE.md):
- `moellava/model/language_model/llava_phi_moe.py`
- `moellava/model/language_model/llava_qwen_moe.py`
- `moellava/model/language_model/llava_stablelm_moe.py`
- `moellava/train/train.py`
- `moellava/train/router_callback.py`

Fixed values to verify:
```python
initial_kd_weight = 0.01
router_temp_start = 1.0
router_ema_start = 0.999
```

⭐ **Eval scripts must use**:
- DeepSpeed launcher: `deepspeed --include localhost:<gpu_ids> ...`
- Single GPU for shared server: `--include localhost:0`
- Data paths: `moellava/eval/<benchmark>/` (NOT root-level `eval/`)

⭐ **Training scripts location**:
- Training: `scripts/v1/<backbone>/finetune_*.sh`
- Evaluation: `scripts/v1/eval/moe_llava/*.sh`

⭐ **Checkpoint naming**:
- Format: `checkpoints_<backbone>_<mode>/`
- Examples: `checkpoints_phi_student/`, `checkpoints_qwen_entropy/`

---

Last updated: 2026-03-22
