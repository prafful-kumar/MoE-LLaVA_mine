import math
import logging
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

logger = logging.getLogger(__name__)


class RouterDistillationCallback(TrainerCallback):
    """
    Dynamically updates Router KD hyperparameters during training.
    
    Schedules:
        Temperature:  4.0  → 1.0   (linear decay)
        KD Weight:    0.5  → 0.05  (cosine decay)
        EMA Decay:    0.999 → 0.95  (linear decay, stable → adaptive)
    
    Design:
        - total_steps resolved in on_train_begin (model not available here)
        - Gates cached lazily on first on_step_begin (model IS available here)
    """
    
    def __init__(self, 
                 total_steps=None,
                 temp_start=4.0, temp_end=1.0,
                 weight_start=0.5, weight_end=0.05,
                 ema_start=0.999, ema_end=0.95,
                 log_interval=100):
        
        self.total_steps_override = total_steps
        self.total_steps = None                 # Resolved in on_train_begin
        
        self.temp_range = (temp_start, temp_end)
        self.weight_range = (weight_start, weight_end)
        self.ema_range = (ema_start, ema_end)
        
        self.log_interval = log_interval
        self.last_log_step = -1
        
        # Gate cache (populated lazily on first step)
        self.gate_cache = []
        self._gates_cached = False

    def on_train_begin(self, args: TrainingArguments, state: TrainerState,
                       control: TrainerControl, **kwargs):
        """
        Resolve total_steps ONCE here.
        
        NOTE: 'model' is NOT passed to on_train_begin by HuggingFace Trainer.
        Do NOT try to access or cache gates here.
        """
        # Resolve total_steps
        # Priority: user override > state.max_steps > error
        if self.total_steps_override is not None and self.total_steps_override > 0:
            self.total_steps = self.total_steps_override
        elif state.max_steps is not None and state.max_steps > 0:
            self.total_steps = state.max_steps
        else:
            raise ValueError(
                "RouterDistillationCallback: Could not determine total_steps.\n"
                "  1. Set training_args.max_steps explicitly, or\n"
                "  2. Pass total_steps=<value> to RouterDistillationCallback"
            )
        
        logger.info("=" * 60)
        logger.info("RouterDistillationCallback: total_steps resolved")
        logger.info(f"  Total steps:  {self.total_steps}")
        logger.info(f"  Temperature:  {self.temp_range[0]} → {self.temp_range[1]}")
        logger.info(f"  KD Weight:    {self.weight_range[0]} → {self.weight_range[1]}")
        logger.info(f"  EMA Decay:    {self.ema_range[0]} → {self.ema_range[1]}")
        logger.info(f"  Gates:        will be cached on first step")
        logger.info("=" * 60)

    def on_step_begin(self, args: TrainingArguments, state: TrainerState,
                      control: TrainerControl, model=None, **kwargs):
        
        # --- Lazy Gate Caching (runs ONCE on first step) ---
        # model IS guaranteed here by HuggingFace Trainer
        if not self._gates_cached:
            if model is None:
                return  # Safety, should not happen in on_step_begin
            
            # Unwrap DDP / DeepSpeed
            actual_model = model.module if hasattr(model, 'module') else model
            
            # ROBUST FIX: Check for the method directly, ignore class type
            self.gate_cache = []
            for m in actual_model.modules():
                if hasattr(m, 'update_hyperparameters'):
                    self.gate_cache.append(m)
            
            if len(self.gate_cache) > 0:
                logger.info(f"[RouterKD] ✅ Cached {len(self.gate_cache)} KD Gates successfully.")
            else:
                logger.warning("[RouterKD] ⚠️ No gates found with 'update_hyperparameters' method!")
            
            self._gates_cached = True
            
            # Log resume info here too (first step is the right time)
            if state.global_step > 0:
                progress = state.global_step / self.total_steps
                logger.info(
                    f"  Resuming schedule from step {state.global_step} "
                    f"({progress*100:.1f}%)"
                )

        # --- If no gates found, do nothing ---
        if not self.gate_cache:
            return

        # --- Schedule Calculations ---
        current_step = state.global_step
        progress = min(current_step / self.total_steps, 1.0)

        # Temperature: Linear Decay (4.0 → 1.0)
        curr_temp = (
            self.temp_range[0] 
            + (self.temp_range[1] - self.temp_range[0]) * progress
        )

        # KD Weight: Cosine Decay (0.5 → 0.05)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        curr_weight = (
            self.weight_range[1] 
            + (self.weight_range[0] - self.weight_range[1]) * cosine_decay
        )

        # EMA Decay: Linear Decay (0.999 → 0.95)
        # Stable early, adaptive late
        curr_ema = (
            self.ema_range[0] 
            + (self.ema_range[1] - self.ema_range[0]) * progress
        )

        # --- Update Cached Gates ---
        for gate in self.gate_cache:
            gate.update_hyperparameters(
                temperature=curr_temp,
                kd_loss_weight=curr_weight,
                ema_decay=curr_ema
            )

        # --- Logging ---
        if current_step - self.last_log_step >= self.log_interval:
            logger.info(
                f"[Router] Step {current_step}/{self.total_steps} "
                f"({progress*100:.1f}%) | "
                f"T={curr_temp:.3f} | "
                f"KD={curr_weight:.4f} | "
                f"EMA={curr_ema:.4f}"
            )
            self.last_log_step = current_step