import torch
import torch.nn as nn
import torch.nn.functional as F
from deepspeed.moe.sharded_moe import TopKGate
import os
from typing import Dict, List
import numpy as np


class KDTopKGate(TopKGate):
    """
    Knowledge Distillation Gate.
    - Teacher: Initialized with K-means centroids.
    - Student: Randomly initialized.
    - Features: T^2 Loss Scaling, EMA Weights, Dynamic Hyperparameters.
    """
    def __init__(self, model_dim, num_experts, k=1, centroids=None, 
                 temperature=2.0, kd_loss_weight=0.01, aux_loss_weight=0.01, ema_decay=0.999, 
                 **kwargs):
        super().__init__(model_dim, num_experts, k, **kwargs)
        
        # Hyperparameters (Stored as attributes so they can be updated)
        self.temperature = temperature
        self.kd_loss_weight = kd_loss_weight
        self.aux_loss_weight = aux_loss_weight
        self.ema_decay = ema_decay
        
        # Teacher Initialization
        if centroids is not None:
            # Validate and register buffer
            # Ensure input is a tensor
            if not isinstance(centroids, torch.Tensor):
                centroids = torch.from_numpy(centroids)
            
            assert centroids.shape == (num_experts, model_dim), f"Shape mismatch: {centroids.shape}"
            
            teacher_init = centroids.float()
            self.register_buffer('teacher_weight', teacher_init, persistent=False)
            self.has_teacher = True
        else:
            # Fallback
            self.register_buffer('teacher_weight', self.wg.weight.data.clone(), persistent=False)
            self.has_teacher = True
        
        # Logging placeholders
        self.last_kd_loss = 0.0
        self.last_moe_loss = 0.0

    def update_hyperparameters(self, temperature=None, kd_loss_weight=None, ema_decay=None):
        """
        External hook for the Trainer Callback to update schedules.
        """
        if temperature is not None:
            self.temperature = temperature
        if kd_loss_weight is not None:
            self.kd_loss_weight = kd_loss_weight
        if ema_decay is not None:
            self.ema_decay = ema_decay

    def forward(self, input, used_token=None, use_tutel=False):
        input_fp32 = input.float()
        
        # --- STEP 1: CALCULATE SEMANTIC (NORMALIZED) LOGITS ---
        # Normalize input to handle Feature Norm Growth (Layer 16 issue)
        input_normed = F.normalize(input_fp32, p=2, dim=-1)
        
        # Normalize weights to remove "Loud Expert" bias (Norm 67 vs 13)
        # We use a scale of 10.0 to ensure the Softmax isn't too flat
        student_w_normed = F.normalize(self.wg.weight.float(), p=2, dim=-1)
        student_logits = 10.0 * F.linear(input_normed, student_w_normed)

        # --- STEP 2: HIJACK THE ROUTING DECISION ---
        # Temporarily swap the weights so super().forward uses our normalized direction
        # This ensures the Top-K selection is based on the semantic angle.
        original_weight = self.wg.weight.data
        self.wg.weight.data = student_w_normed * 10.0
        
        # Now DeepSpeed/TopKGate uses the 'fair' normalized weights to pick experts
        gate_output = super().forward(input_normed, used_token, use_tutel)
        
        # Restore the original weights so gradients flow correctly back to them
        self.wg.weight.data = original_weight

        # --- STEP 3: HANDLE LOSSES ---
        raw_aux_loss = gate_output[0]
        weighted_aux_loss = self.aux_loss_weight * raw_aux_loss
        self.last_moe_loss = raw_aux_loss.item() if isinstance(raw_aux_loss, torch.Tensor) else 0.0
        
        # 4. Knowledge Distillation (Training Only)
        if self.training and self.has_teacher:
            with torch.no_grad():
                # Teacher is already normalized in __init__
                teacher_w = self.teacher_weight.to(device=input_fp32.device, dtype=input_fp32.dtype)
                teacher_logits = 10.0 * F.linear(input_normed, teacher_w)
            
            T = self.temperature
            kd_loss = F.kl_div(
                F.log_softmax(student_logits / T, dim=-1),
                F.softmax(teacher_logits / T, dim=-1),
                reduction='batchmean'
            ) * (T ** 2)
            
            self.last_kd_loss = kd_loss.item()
            
            # EMA Update: Keep Teacher on the unit sphere
            with torch.no_grad():
                self.teacher_weight.mul_(self.ema_decay).add_(student_w_normed, alpha=1.0 - self.ema_decay)
                # Re-normalize teacher to prevent norm drift
                self.teacher_weight.copy_(F.normalize(self.teacher_weight, p=2, dim=-1))
            
            total_loss = weighted_aux_loss + (self.kd_loss_weight * kd_loss)
        else:
            total_loss = weighted_aux_loss

        # 5. Return Combined Loss
        gate_output = (total_loss,) + gate_output[1:]
        return gate_output
    
    def get_loss_dict(self):
        return {
            'moe_loss': self.last_moe_loss,
            'kd_loss': self.last_kd_loss,
            'total_aux_loss': self.last_moe_loss + self.kd_loss_weight * self.last_kd_loss,
            'temperature': self.temperature,
            'ema_decay': self.ema_decay
        }
    
    def disable_teacher(self):
        if self.has_teacher:
            self.teacher_weight = None
            self.has_teacher = False
            torch.cuda.empty_cache()
import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalizedKDTopKGate(TopKGate):
    """
    Flexible Router with configurable normalization strategy.
    Aligned with DeepSpeed TopKGate style and return signatures.
    """
    def __init__(self, model_dim, num_experts, k=1, centroids=None, 
                 temperature=2.0, kd_loss_weight=0.01, aux_loss_weight=0.01, 
                 ema_decay=0.999, logit_scale=10.0, 
                 normalize_weights='training', 
                 normalize_input=True,
                 **kwargs):
        super().__init__(model_dim, num_experts, k, **kwargs)
        
        self.temperature = temperature
        self.kd_loss_weight = kd_loss_weight
        self.aux_loss_weight = aux_loss_weight
        self.ema_decay = ema_decay
        self.logit_scale = logit_scale
        self.normalize_weights = normalize_weights
        self.normalize_input = normalize_input
        
        # --- DeepSpeed Style Init ---
        # Force gate weights to float32 for routing precision
        self.wg.weight.data = self.wg.weight.data.float()
        
        if normalize_weights in ['init', 'training']:
            with torch.no_grad():
                self.wg.weight.data = F.normalize(self.wg.weight.data, p=2, dim=-1)
        
        # --- Teacher Initialization (Anchor) ---
        if centroids is not None:
            if not isinstance(centroids, torch.Tensor):
                centroids = torch.from_numpy(centroids)
            teacher_init = F.normalize(centroids.float(), p=2, dim=-1)
            self.register_buffer('teacher_weight', teacher_init, persistent=False)
            self.has_teacher = True
        else:
            teacher_init = F.normalize(self.wg.weight.data.clone().float(), p=2, dim=-1)
            self.register_buffer('teacher_weight', teacher_init, persistent=False)
            self.has_teacher = True
            
        self.last_kd_loss = 0.0
        self.last_moe_loss = 0.0

    def forward(self, input, used_token=None, use_tutel=False):
        # 1. DeepSpeed Logic: Ensure gating happens in float32
        if self.wg.weight.dtype != torch.float32:
            self.wg = self.wg.float()
        
        input_fp32 = input.float()
        
        # 2. Normalization Strategy
        if self.normalize_input:
            input_normed = F.normalize(input_fp32, p=2, dim=-1)
        else:
            input_normed = input_fp32
        
        # 3. Handle weight normalization based on strategy
        if self.normalize_weights == 'training':
            # Runtime normalization to enforce Cosine Similarity
            w_normed = F.normalize(self.wg.weight.float(), p=2, dim=-1)
            
            # Temporary Weight Swap for super().forward()
            original_weight = self.wg.weight
            # We scale here so the dot product inside TopKGate becomes scaled cosine sim
            self.wg.weight = nn.Parameter(w_normed * self.logit_scale)
            try:
                # Returns (l_aux, combine_weights, dispatch_mask, exp_counts)
                gate_output = super().forward(input_normed, used_token, use_tutel)
            finally:
                self.wg.weight = original_weight
            
            # Calculate logits for KD loss
            student_logits = self.logit_scale * F.linear(input_normed, w_normed)
        else:
            # Initialization-only or None
            gate_output = super().forward(input_normed, used_token, use_tutel)
            student_logits = F.linear(input_normed, self.wg.weight.float())

        # 4. Extract and Weight MoE Loss
        raw_aux_loss = gate_output[0]
        weighted_aux_loss = self.aux_loss_weight * raw_aux_loss
        self.last_moe_loss = raw_aux_loss.item() if isinstance(raw_aux_loss, torch.Tensor) else 0.0
        
        # 5. --- FIX: KNOWLEDGE DISTILLATION (DType Safe) ---
        if self.training and self.has_teacher:
            with torch.no_grad():
                # CRITICAL FIX: Explicitly cast teacher to float32 to match input_normed
                # This prevents the "Float but found BFloat16" error
                teacher_w = self.teacher_weight.float().to(input_normed.device)
                teacher_logits = self.logit_scale * F.linear(input_normed.float(), teacher_w)
            
            T = self.temperature
            # Perform KL divergence in float32 for stability
            kd_loss = F.kl_div(
                F.log_softmax(student_logits.float() / T, dim=-1),
                F.softmax(teacher_logits.float() / T, dim=-1),
                reduction='batchmean'
            ) * (T ** 2)
            
            self.last_kd_loss = kd_loss.item()
            
            # EMA Update in float32
            with torch.no_grad():
                student_w_norm = F.normalize(self.wg.weight.data.float(), p=2, dim=-1).to(teacher_w.device)
                self.teacher_weight.mul_(self.ema_decay).add_(student_w_norm, alpha=1.0 - self.ema_decay)
                self.teacher_weight.copy_(F.normalize(self.teacher_weight, p=2, dim=-1))
            
            total_loss = weighted_aux_loss + (self.kd_loss_weight * kd_loss)
        else:
            total_loss = weighted_aux_loss

        # 6. Return standard 4-tuple signature: (loss, weights, mask, counts)
        return (total_loss,) + gate_output[1:]

    def get_loss_dict(self):
        return {
            'moe_loss': self.last_moe_loss,
            'kd_loss': self.last_kd_loss,
            'total_aux_loss': (self.aux_loss_weight * self.last_moe_loss) + (self.kd_loss_weight * self.last_kd_loss),
            'temperature': self.temperature,
            'ema_decay': self.ema_decay
        }
    
    def disable_teacher(self):
        if self.has_teacher:
            self.teacher_weight = None
            self.has_teacher = False
            torch.cuda.empty_cache()
            
    def update_hyperparameters(self, temperature=None, kd_loss_weight=None, ema_decay=None):
        if temperature is not None: self.temperature = temperature
        if kd_loss_weight is not None: self.kd_loss_weight = kd_loss_weight
        if ema_decay is not None: self.ema_decay = ema_decay

# class SimplifiedNormalizedGate(TopKGate):
#     """
#     Normalized routing with Fisher initialization.
#     No teacher, no KD - just good init + normalization.
#     """
#     def __init__(self, model_dim, num_experts, k=1, 
#                  fisher_directions=None,
#                  logit_scale=10.0,
#                  aux_loss_weight=0.01,
#                  normalize_input=True,
#                  **kwargs):
#         super().__init__(model_dim, num_experts, k, **kwargs)
        
#         self.logit_scale = logit_scale
#         self.aux_loss_weight = aux_loss_weight
#         self.normalize_input = normalize_input
        
#         # Initialize with Fisher directions
#         if fisher_directions is not None:
#             with torch.no_grad():
#                 fisher_norm = F.normalize(
#                     torch.from_numpy(fisher_directions).float(), 
#                     p=2, dim=-1
#                 )
#                 self.wg.weight.data = fisher_norm * self.logit_scale
#         else:
#             # Fallback: normalize random init
#             with torch.no_grad():
#                 self.wg.weight.data = F.normalize(
#                     self.wg.weight.data, p=2, dim=-1
#                 ) * self.logit_scale
        
#         # No teacher buffer!
#         # No KD loss!
        
#     def forward(self, input, used_token=None, use_tutel=False):
#         input_fp32 = input.float()
        
#         # Normalize input
#         if self.normalize_input:
#             input_normed = F.normalize(input_fp32, p=2, dim=-1)
#         else:
#             input_normed = input_fp32
        
#         # Normalize weights for routing
#         w_normed = F.normalize(self.wg.weight.float(), p=2, dim=-1)
        
#         # Compute routing with normalized weights
#         # Temporarily swap (this works for routing decision)
#         original_weight = self.wg.weight.data
#         self.wg.weight.data = w_normed * self.logit_scale
        
#         gate_output = super().forward(input_normed, used_token, use_tutel)
        
#         self.wg.weight.data = original_weight
        
#         # Just return MoE aux loss (no KD)
#         aux_loss = self.aux_loss_weight * gate_output[0]
#         return (aux_loss,) + gate_output[1:]

class SimplifiedNormalizedGate(TopKGate):
    """
    Normalized routing with Fisher initialization.
    No teacher, no KD - just good init + normalization.
    Works for both standard (fp32) and Qwen-style (bf16/fp16) models.
    """
    def __init__(self, model_dim, num_experts, k=1,
                 fisher_directions=None,
                 logit_scale=10.0,
                 aux_loss_weight=0.01,
                 entropy_loss_weight=0.0,
                 normalize_input=True,
                 **kwargs):
        super().__init__(model_dim, num_experts, k, **kwargs)

        self.logit_scale = logit_scale
        self.aux_loss_weight = aux_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.normalize_input = normalize_input

        self.last_moe_loss = 0.0
        self.last_kd_loss = 0.0
        self.last_entropy_loss = 0.0

        with torch.no_grad():
            if fisher_directions is not None:
                # Handle numpy or tensor, always move to correct device
                if not isinstance(fisher_directions, torch.Tensor):
                    fisher_directions = torch.tensor(
                        fisher_directions, 
                        device=self.wg.weight.device
                    )
                else:
                    fisher_directions = fisher_directions.to(self.wg.weight.device)

                # Normalize in fp32 for numerical stability, cast back to module dtype
                fisher_norm = F.normalize(fisher_directions.float(), p=2, dim=-1)
                self.wg.weight.data = (fisher_norm * self.logit_scale).to(
                    self.wg.weight.dtype
                )
            else:
                # Same: normalize in fp32, cast back
                w_norm = F.normalize(self.wg.weight.float(), p=2, dim=-1)
                self.wg.weight.data = (w_norm * self.logit_scale).to(
                    self.wg.weight.dtype
                )

    def forward(self, input, used_token=None, use_tutel=False):
        # Preserve original dtype (critical for bf16/fp16 models like Qwen)
        original_dtype = input.dtype

        # fp32 for numerically stable normalization
        input_fp32 = input.float()

        if self.normalize_input:
            input_normed = F.normalize(input_fp32, p=2, dim=-1)
        else:
            input_normed = input_fp32

        # Normalize weights in fp32, then cast back to module's dtype
        w_normed = F.normalize(self.wg.weight.float(), p=2, dim=-1)

        original_weight_data = self.wg.weight.data
        self.wg.weight.data = (w_normed * self.logit_scale).to(original_dtype)

        try:
            # Cast input back to original dtype before DeepSpeed/parent processes it
            gate_output = super().forward(
                input_normed.to(original_dtype), used_token, use_tutel
            )
        finally:
            self.wg.weight.data = original_weight_data

        raw_aux_loss = gate_output[0]
        aux_loss = self.aux_loss_weight * raw_aux_loss
        self.last_moe_loss = (
            raw_aux_loss.item() if isinstance(raw_aux_loss, torch.Tensor) else 0.0
        )

        # Entropy regularization — gradients MUST flow through wg.weight
        if self.entropy_loss_weight > 0.0:
            w_normed_fp32 = F.normalize(self.wg.weight.float(), p=2, dim=-1)
            logits = F.linear(input_normed, w_normed_fp32) * self.logit_scale
            probs = F.softmax(logits.float(), dim=-1)
            H = -(probs * torch.log(probs + 1e-8)).sum(-1).mean()
            entropy_loss = self.entropy_loss_weight * H
            self.last_entropy_loss = H.item()
            total_loss = aux_loss + entropy_loss
        else:
            self.last_entropy_loss = 0.0
            total_loss = aux_loss

        return (total_loss,) + gate_output[1:]

    def get_loss_dict(self):
        return {
            'moe_loss': self.last_moe_loss,
            'kd_loss': 0.0,
            'entropy_loss': self.last_entropy_loss,
            'total_aux_loss': (self.aux_loss_weight * self.last_moe_loss) + (self.entropy_loss_weight * self.last_entropy_loss),
            'temperature': None,
            'ema_decay': None,
        }

    def update_hyperparameters(self, temperature=None, kd_loss_weight=None, ema_decay=None):
        pass