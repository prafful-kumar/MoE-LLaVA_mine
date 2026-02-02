import torch
import joblib
import logging
from deepspeed.moe.sharded_moe import TopKGate
from moellava.model.kd_gate import KDTopKGate

logger = logging.getLogger(__name__)

def replace_gates_with_kd(model, centroids_path, initial_temp=4.0):
    """
    Replaces existing TopKGate instances with KDTopKGate initialized with centroids.
    """
    logger.info(f"Loading K-Means Centroids from {centroids_path}...")
    # Expecting dict: {layer_index: centroids_array}
    centroids_dict = joblib.load(centroids_path)
    
    replaced_count = 0
    
    # Iterate through all modules
    for name, module in model.named_modules():
        # Check if the module has a 'gate' attribute that is a TopKGate
        if hasattr(module, 'gate') and isinstance(module.gate, TopKGate):
            
            # Extract layer index from name (e.g., 'model.layers.14.mlp.deepspeed_moe')
            # Adjust regex based on exact StableLM structure
            import re
            match = re.search(r'layers\.(\d+)', name)
            if not match:
                continue
            
            layer_idx = int(match.group(1))
            
            # Check if we have centroids for this layer
            if layer_idx in centroids_dict:
                old_gate = module.gate
                
                # Create New KD Gate
                # Copy configuration from the old gate
                new_gate = KDTopKGate(
                    model_dim=old_gate.wg.in_features,
                    num_experts=old_gate.num_experts,
                    k=old_gate.k,
                    centroids=centroids_dict[layer_idx],
                    temperature=initial_temp,
                    # Pass through DeepSpeed specific args if they exist
                    min_capacity=getattr(old_gate, 'min_capacity', 0),
                    noisy_gate_policy=getattr(old_gate, 'noisy_gate_policy', None)
                )
                
                # IMPORTANT: Keep Student Random! 
                # We do NOT copy weights from old_gate.wg to new_gate.wg
                # because we want to distill from scratch.
                
                # Move to correct device/dtype
                new_gate.to(old_gate.wg.weight.device)
                if old_gate.wg.weight.dtype == torch.bfloat16:
                     new_gate = new_gate.bfloat16()

                # Perform the swap
                module.gate = new_gate
                replaced_count += 1
                
    logger.info(f"✅ Successfully replaced {replaced_count} gates with KD Gates.")