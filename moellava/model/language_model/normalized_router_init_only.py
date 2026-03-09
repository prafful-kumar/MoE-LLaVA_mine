import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure you have the correct import for the parent class
try:
    from deepspeed.moe.sharded_moe import TopKGate
except ImportError:
    class TopKGate(nn.Module): pass # Placeholder if testing locally

class NormalizedKDTopKGate(TopKGate):
    """
    Router with normalized initialization.
    Weights start on the unit sphere but can grow freely (Volume Knob is unlocked).
    """
    def __init__(self, model_dim, num_experts, k=1, centroids=None, 
                 temperature=2.0, kd_loss_weight=0.01, aux_loss_weight=0.01, 
                 ema_decay=0.999, logit_scale=10.0, **kwargs):
        super().__init__(model_dim, num_experts, k, **kwargs)
        
        self.temperature = temperature
        self.kd_loss_weight = kd_loss_weight
        self.aux_loss_weight = aux_loss_weight
        self.ema_decay = ema_decay
        self.logit_scale = logit_scale
        
        # --- 1. NORMALIZE & SCALE STUDENT AT INIT ---
        with torch.no_grad():
            # A. Normalize to silence "Loud Experts" (Expert 0 vs 1)
            self.wg.weight.data = F.normalize(self.wg.weight.data, p=2, dim=-1)
            
            # B. Scale up to ensure Peakiness
            # Without this, logits are [-1, 1] and routing is random.
            # We want logits ~[-10, 10] so the router makes actual choices.
            self.wg.weight.data *= self.logit_scale
        
        # --- 2. TEACHER INITIALIZATION ---
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
        # 1. Input Normalization (Always Keep This!)
        # This prevents Layer 16 feature growth from exploding the logits
        input_fp32 = input.float()
        input_normed = F.normalize(input_fp32, p=2, dim=-1)
        
        # 2. Standard Forward (Student weights grow freely)
        gate_output = super().forward(input_normed, used_token, use_tutel)

        # 3. Extract MoE Loss
        raw_aux_loss = gate_output[0]
        weighted_aux_loss = self.aux_loss_weight * raw_aux_loss
        
        if isinstance(raw_aux_loss, torch.Tensor):
            self.last_moe_loss = raw_aux_loss.item()
        else:
            self.last_moe_loss = 0.0
        
        # 4. Knowledge Distillation
        if self.training and self.has_teacher:
            # Student Logits (Unnormalized weights)
            student_logits = F.linear(input_normed, self.wg.weight.float())
            
            with torch.no_grad():
                teacher_w = self.teacher_weight.to(device=input_fp32.device, dtype=input_fp32.dtype)
                
                # FIX: Scale Teacher Logits
                # Teacher must be "confident" to provide a useful target.
                # Unit Norm Inputs @ Unit Norm Teacher = Range [-1, 1]
                # Multiplied by logit_scale = Range [-10, 10]
                teacher_logits = self.logit_scale * F.linear(input_normed, teacher_w)
            
            T = self.temperature
            kd_loss = F.kl_div(
                F.log_softmax(student_logits / T, dim=-1),
                F.softmax(teacher_logits / T, dim=-1),
                reduction='batchmean'
            ) * (T ** 2)
            
            self.last_kd_loss = kd_loss.item()
            
            # EMA Update
            with torch.no_grad():
                # Teacher tracks the *normalized* direction of the student
                student_w_norm = F.normalize(self.wg.weight.data, p=2, dim=-1).to(teacher_w.device)
                self.teacher_weight.mul_(self.ema_decay).add_(
                    student_w_norm, alpha=1.0 - self.ema_decay
                )
                self.teacher_weight.copy_(F.normalize(self.teacher_weight, p=2, dim=-1))
            
            total_loss = weighted_aux_loss + (self.kd_loss_weight * kd_loss)
        else:
            total_loss = weighted_aux_loss

        return (total_loss,) + gate_output[1:]
    
    # ... Helper functions (get_loss_dict, etc.) remain the same ...
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
    def update_hyperparameters(self, temperature=None, kd_loss_weight=None, ema_decay=None):
        if temperature is not None: self.temperature = temperature
        if kd_loss_weight is not None: self.kd_loss_weight = kd_loss_weight
        if ema_decay is not None: self.ema_decay = ema_decay