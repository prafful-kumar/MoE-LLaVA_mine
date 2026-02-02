import torch
import torch.nn.functional as F
from deepspeed.moe.sharded_moe import TopKGate
import logging

logger = logging.getLogger(__name__)

class KDTopKGate(TopKGate):
    """
    Knowledge Distillation Gate for MoE-LLaVA.
    - Teacher: Initialized from K-Means Centroids (Linear Layer).
    - Student: Randomly initialized (Standard Linear Layer).
    - Update: EMA on Weights (Fast & Stable).
    """
    def __init__(self, model_dim, num_experts, k=1, 
                 centroids=None, 
                 temperature=2.0, kd_loss_weight=0.1, ema_decay=0.999,
                 **kwargs):
        # Initialize standard DeepSpeed Gate
        super().__init__(model_dim, num_experts, k, **kwargs)

        # KD Hyperparameters
        self.temperature = temperature
        self.kd_loss_weight = kd_loss_weight
        self.ema_decay = ema_decay
        self.has_teacher = False

        # Initialize Teacher from Centroids
        if centroids is not None:
            # Convert to tensor if numpy
            if not isinstance(centroids, torch.Tensor):
                centroids = torch.tensor(centroids)
            
            # Validation
            if centroids.shape != (num_experts, model_dim):
                raise ValueError(f"Centroid shape {centroids.shape} mismatch. Expected ({num_experts}, {model_dim})")

            # Create Teacher Weights
            # We treat centroids as Linear Weights for dot-product compatibility
            teacher_init = centroids.float()
            
            # Register as buffer (persistent=False avoids DeepSpeed checkpoint issues)
            self.register_buffer('teacher_weight', teacher_init, persistent=False)
            self.has_teacher = True
            
            logger.info("KDTopKGate: Teacher initialized from Centroids.")

    def update_hyperparameters(self, temperature=None, kd_loss_weight=None, ema_decay=None):
        """Called by Callback to update schedule"""
        if temperature: self.temperature = temperature
        if kd_loss_weight: self.kd_loss_weight = kd_loss_weight
        if ema_decay: self.ema_decay = ema_decay

    def forward(self, input, used_token=None, use_tutel=False):
        # 1. Compute Student Logits (Needed for Loss)
        # We must cast to float32 for stability, consistent with MoE-LLaVA
        input_fp32 = input.float()
        
        # Ensure student weights are float32 for this op
        if self.wg.weight.dtype != torch.float32:
            self.wg = self.wg.float()
        student_logits = self.wg(input_fp32)

        # 2. Run Standard DeepSpeed Routing
        # Returns: (l_aux, combined_output, ...)
        gate_output = super().forward(input, used_token, use_tutel)

        # 3. Knowledge Distillation (Training Only)
        if self.training and self.has_teacher:
            with torch.no_grad():
                # Teacher Forward (Linear Projection)
                # This aligns the math: Student and Teacher both do Dot Product
                teacher_w = self.teacher_weight.to(input_fp32.device).float()
                teacher_logits = F.linear(input_fp32, teacher_w)
                
                # EMA Update: Teacher weights slowly track Student weights
                # Schedule: High Decay (Stable) -> Lower Decay (Adaptive)
                student_w = self.wg.weight.data.to(input_fp32.device).float()
                self.teacher_weight.mul_(self.ema_decay).add_(
                    student_w, alpha=(1.0 - self.ema_decay)
                )

            # 4. Calculate Soft KD Loss
            # T^2 Scaling is required (Hinton et al.)
            T = self.temperature
            kd_loss = F.kl_div(
                F.log_softmax(student_logits / T, dim=-1),
                F.softmax(teacher_logits / T, dim=-1),
                reduction='batchmean'
            ) * (T ** 2)

            # 5. Inject Loss into Return Tuple
            # DeepSpeed TopKGate usually returns (l_aux, ...)
            if isinstance(gate_output, tuple):
                # Add weighted KD loss to the auxiliary load-balancing loss
                total_aux_loss = gate_output[0] + (self.kd_loss_weight * kd_loss)
                gate_output = (total_aux_loss,) + gate_output[1:]
        
        return gate_output