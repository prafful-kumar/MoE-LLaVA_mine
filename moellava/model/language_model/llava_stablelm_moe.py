#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM
from .stablelm.configuration_stablelm_epoch import StableLMEpochConfig
from .stablelm.modeling_stablelm_epoch import StableLMEpochModel, StableLMEpochForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

from deepspeed.moe.layer import MoE
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from torch.nn import CrossEntropyLoss
from transformers.models.llama.modeling_llama import logger
from transformers.utils import ModelOutput

import joblib
from deepspeed.moe.sharded_moe import TopKGate
import os
from typing import Dict, List
import numpy as np
from deepspeed.moe.experts import Experts

local_rank = None

class KDTopKGate(TopKGate):
    """
    Knowledge Distillation Gate with Teacher initialized from centroids.
    
    Teacher: Initialized from K-means centroids (stable, good initialization)
    Student: Random initialization (learns optimal routing)
    """
    
    def __init__(self, model_dim, num_experts, k=1, centroids=None, 
                 kd_loss_weight=0.01, aux_loss_weight=0.01, ema_decay=0.999, **kwargs):
        """
        Args:
            model_dim: Model hidden dimension
            num_experts: Number of experts
            k: Top-k experts to route to
            centroids: K-means centroids [num_experts, model_dim] - for teacher init
            kd_loss_weight: Weight for KD loss term
            ema_decay: EMA decay rate for teacher updates (0.999 = very slow)
            **kwargs: Other TopKGate parameters
        """
        super().__init__(model_dim, num_experts, k, **kwargs)
        
        self.kd_loss_weight = kd_loss_weight
        self.ema_decay = ema_decay
        self.aux_loss_weight = aux_loss_weight
        
        # Student weights: Already randomly initialized by parent's __init__
        # (self.wg.weight is created in parent with default PyTorch init)
        
        # Teacher weights: Initialize from centroids
        if centroids is not None:
            # Validate shape
            assert centroids.shape == (num_experts, model_dim), \
                f"Centroid shape {centroids.shape} != expected ({num_experts}, {model_dim})"
            
            # Convert to tensor with proper device/dtype
            # Don't specify device here - will be moved when gate is moved to device
            teacher_init = torch.from_numpy(centroids).float()
            
            # Register as buffer (persistent=False means not saved in checkpoint)
            self.register_buffer('teacher_weight', teacher_init, persistent=False)
            self.has_teacher = True
        else:
            # No centroids provided - initialize teacher as copy of random student
            # This is fallback, but not recommended
            self.register_buffer('teacher_weight', 
                               self.wg.weight.data.clone().detach(), 
                               persistent=False)
            self.has_teacher = True
        
        # For monitoring/logging
        self.last_kd_loss = 0.0
        self.last_moe_loss = 0.0
    
    def forward(self, input, used_token=None, use_tutel=False):
        """
        Forward pass with student-teacher distillation.
        
        Flow:
        1. Get student logits (need to compute separately for KD)
        2. Call parent's forward for standard gating
        3. Get teacher logits (from stable teacher weights)
        4. Compute KD loss
        5. Update teacher via EMA
        6. Combine losses
        """
        
        # Ensure input is float32 for gate computation
        input_fp32 = input.float()
        
        # --- [FIX 2] Safer Student Logits Calculation ---
        # Do NOT convert self.wg to float() permanently. 
        # Just cast weights for this specific operation to avoid breaking mixed precision.
        student_logits = F.linear(input_fp32, self.wg.weight.to(dtype=torch.float32))


        # Call parent's forward for standard gating logic
        # This handles: top-k selection, capacity constraints, tutel, etc.
        gate_output = super().forward(input, used_token, use_tutel)

        raw_aux_loss = gate_output[0]
        weighted_aux_loss = self.aux_loss_weight * raw_aux_loss
        
        # Store for logging purposes (handle case where loss might be None or scalar)
        if isinstance(raw_aux_loss, torch.Tensor):
             self.last_moe_loss = raw_aux_loss.item()
        else:
             self.last_moe_loss = 0.0
        
        # Knowledge Distillation (only during training)
        if self.training and self.has_teacher:
            # Compute teacher logits using stable teacher weights
            with torch.no_grad():
                # Ensure teacher is on same device/dtype as input
                teacher_weight = self.teacher_weight.to(
                    device=input_fp32.device,
                    dtype=input_fp32.dtype
                )
                teacher_logits = F.linear(input_fp32, teacher_weight)
            
            # KL Divergence Loss: Student learns from Teacher
            # Student (log probabilities) vs Teacher (probabilities)
            kd_loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(teacher_logits, dim=-1),  # Teacher is detached (no grad)
                reduction='batchmean'
            )
            
            self.last_kd_loss = kd_loss.item()
            
            # EMA Update: Teacher slowly follows Student
            # This makes teacher adapt to student's learning while staying stable
            with torch.no_grad():
                # Ensure same device/dtype for EMA
                teacher_device = self.teacher_weight.device
                teacher_dtype = self.teacher_weight.dtype
                student_weight = self.wg.weight.data.to(
                    device=teacher_device,
                    dtype=teacher_dtype
                )
                
                # EMA: teacher = decay * teacher + (1-decay) * student
                self.teacher_weight.mul_(self.ema_decay).add_(
                    student_weight,
                    alpha=1.0 - self.ema_decay
                )
            
            weighted_kd_loss = self.kd_loss_weight * kd_loss
            
            # Combine
            total_loss = weighted_aux_loss + weighted_kd_loss
        else:
            total_loss = weighted_aux_loss

        gate_output = (total_loss,) + gate_output[1:]
        
        return gate_output
    
    def get_loss_dict(self):
        """Return dictionary of losses for logging"""
        return {
            'moe_loss': self.last_moe_loss,
            'kd_loss': self.last_kd_loss,
            'total_aux_loss': self.last_moe_loss + self.kd_loss_weight * self.last_kd_loss
        }
    
    def disable_teacher(self):
        """
        Disable teacher for inference or before saving checkpoint.
        Call this after training is complete.
        """
        if self.has_teacher:
            self.teacher_weight = None
            self.has_teacher = False
            torch.cuda.empty_cache()

def copy_mlp_to_experts(original_mlp: nn.Module, 
                       moe_layer,
                       num_experts: int,
                       verbose: bool = True) -> Dict[str, bool]:
    """
    Copy weights from original MLP to all experts in MoE layer.
    
    Args:
        original_mlp: The original MLP module (before MoE conversion)
        moe_layer: The CustomMoE layer (after conversion)
        num_experts: Number of experts
        verbose: Print detailed information
    
    Returns:
        Dictionary with verification results
    """
    
    print(f"\n{'='*60}")
    print(f"Copying MLP weights to {num_experts} experts")
    print(f"{'='*60}\n")
    
    # Get the experts module
    experts = moe_layer.deepspeed_moe.experts
    
    # Get original MLP state dict
    original_state_dict = original_mlp.state_dict()
    
    if verbose:
        print("Original MLP parameters:")
        for name, param in original_state_dict.items():
            print(f"  {name}: {param.shape}")
        print()
    
    # Copy to each expert
    for expert_idx in range(num_experts):
        expert = experts.deepspeed_experts[expert_idx]
        
        if verbose:
            print(f"Copying to Expert {expert_idx}...")
        
        # Load the same weights into each expert
        expert.load_state_dict(original_state_dict, strict=True)
        
        if verbose:
            print(f"  ✓ Expert {expert_idx} weights copied successfully")
    
    print(f"\n✓ All {num_experts} experts initialized with MLP weights\n")
    
    # Verify the copying
    verification_results = verify_weight_copying(
        original_mlp, 
        moe_layer, 
        num_experts,
        verbose=verbose
    )
    
    return verification_results


def verify_weight_copying(original_mlp: nn.Module,
                         moe_layer,
                         num_experts: int,
                         atol: float = 1e-6,
                         verbose: bool = True) -> Dict[str, bool]:
    """
    Verify that all experts have the same weights as original MLP.
    
    Args:
        original_mlp: Original MLP module
        moe_layer: CustomMoE layer
        num_experts: Number of experts
        atol: Absolute tolerance for comparison
        verbose: Print detailed verification
    
    Returns:
        Dictionary with verification status for each check
    """
    
    print(f"{'='*60}")
    print(f"VERIFICATION: Checking weight equality")
    print(f"{'='*60}\n")
    
    results = {
        'all_experts_match_mlp': True,
        'experts_match_each_other': True,
        'parameter_details': {}
    }
    
    # Get original MLP state dict
    original_state_dict = original_mlp.state_dict()
    experts = moe_layer.deepspeed_moe.experts
    
    # Check each parameter
    for param_name, original_param in original_state_dict.items():
        if verbose:
            print(f"Checking parameter: {param_name}")
            print(f"  Shape: {original_param.shape}")
        
        param_matches = True
        expert_params = []
        
        # Compare each expert with original MLP
        for expert_idx in range(num_experts):
            expert = experts.deepspeed_experts[expert_idx]
            expert_state_dict = expert.state_dict()
            
            if param_name not in expert_state_dict:
                print(f"  ✗ Expert {expert_idx}: Parameter '{param_name}' not found!")
                param_matches = False
                results['all_experts_match_mlp'] = False
                continue
            
            expert_param = expert_state_dict[param_name]
            expert_params.append(expert_param)
            
            # Check if expert parameter matches original
            if not torch.allclose(expert_param, original_param, atol=atol):
                max_diff = torch.max(torch.abs(expert_param - original_param)).item()
                print(f"  ✗ Expert {expert_idx}: MISMATCH! Max diff: {max_diff:.2e}")
                param_matches = False
                results['all_experts_match_mlp'] = False
            else:
                if verbose:
                    print(f"  ✓ Expert {expert_idx}: Match (diff < {atol})")
        
        # Check if all experts match each other
        if len(expert_params) > 1:
            for i in range(1, len(expert_params)):
                if not torch.allclose(expert_params[0], expert_params[i], atol=atol):
                    max_diff = torch.max(torch.abs(expert_params[0] - expert_params[i])).item()
                    print(f"  ✗ Expert 0 vs Expert {i}: MISMATCH! Max diff: {max_diff:.2e}")
                    results['experts_match_each_other'] = False
        
        results['parameter_details'][param_name] = param_matches
        
        if verbose:
            print()
    
    # Print summary
    print(f"{'='*60}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"✓ All experts match MLP: {results['all_experts_match_mlp']}")
    print(f"✓ All experts match each other: {results['experts_match_each_other']}")
    
    if results['all_experts_match_mlp'] and results['experts_match_each_other']:
        print("\n🎉 SUCCESS: All weight copying verified correctly!")
    else:
        print("\n⚠️  WARNING: Weight copying verification failed!")
    
    print(f"{'='*60}\n")
    
    return results


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


class MoELLaVAStablelmConfig(StableLMEpochConfig):
    model_type = "moe_llava_stablelm"

    def __init__(self,
                 moe_enable=True,
                 moe_mode='sparse',
                 moe_layers_idx=None,
                 ep_size=1,
                 top_k_experts=2,
                 capacity_factor=1.,
                 eval_capacity_factor=1.,
                 min_capacity=4,
                 use_residual=False,
                 router_aux_loss_coef=0.01,
                 **kwargs):
        self.moe = dict(
            moe_enable=moe_enable,
            moe_mode=moe_mode,
            moe_layers_idx=moe_layers_idx,
            ep_size=ep_size,
            top_k_experts=top_k_experts,
            capacity_factor=capacity_factor,
            eval_capacity_factor=eval_capacity_factor,
            min_capacity=min_capacity,
            use_residual=use_residual,
            router_aux_loss_coef=router_aux_loss_coef,
            train_modules=[
                # 'up_proj', 'down_proj', 'gate_proj', 'wg',
                # 'embed_tokens', 'lm_head'
            ]
        )

        super(MoELLaVAStablelmConfig, self).__init__(**kwargs)


class MoELLaVAStablelmModel(LlavaMetaModel, StableLMEpochModel):
    config_class = MoELLaVAStablelmConfig

    def __init__(self, config: StableLMEpochConfig):
        super(MoELLaVAStablelmModel, self).__init__(config)


@dataclass
class MoEBaseModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    moe_loss_list: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class MoECausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    moe_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    moe_loss_list: Optional[Tuple[torch.FloatTensor]] = None


def MoEStablelmDecoderLayer_forward(self):
    def forward(
            # self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            # padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        # import ipdb
        # ipdb.set_trace()
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            # padding_mask=padding_mask,  # unuseful but conflict to flashattn
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # import ipdb
        # ipdb.set_trace()
        moe_losses = []
        if len(hidden_states) == 3:
            moe_losses.append(hidden_states[1])
            hidden_states = hidden_states[0]
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        outputs += (moe_losses,)

        return outputs

    return forward


def MoEStablelmModel_forward(self):
    def forward(
            # self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            output_moe_loss: Optional[bool] = True,
    ) -> Union[Tuple, MoEBaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # Embed positions
        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            if attention_mask is None:
                attention_mask = torch.ones(
                    (batch_size, seq_length_with_past),
                    dtype=torch.bool,
                    device=inputs_embeds.device,
                )
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        all_moe_loss = [] if output_moe_loss else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer), hidden_states, attention_mask, position_ids
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_moe_loss:
                all_moe_loss.extend(layer_outputs[-1])

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_moe_loss] if
                v is not None)
        return MoEBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            moe_loss_list=all_moe_loss,
        )

    return forward


class MoELLaVAStablelmForCausalLM(StableLMEpochForCausalLM, LlavaMetaForCausalLM):
    config_class = MoELLaVAStablelmConfig

    def __init__(self, config):
        super(StableLMEpochForCausalLM, self).__init__(config)
        self.model = MoELLaVAStablelmModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MoECausalLMOutputWithPast]:
        # print('before prepare_inputs_labels_for_multimodal')
        # import ipdb
        # ipdb.set_trace()
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )
        # import ipdb
        # ipdb.set_trace()
        # print('after prepare_inputs_labels_for_multimodal')
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # import ipdb
        # ipdb.set_trace()
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        moe_loss, moe_losses = None, []
        if len(outputs[-1]) > 0:
            moe_loss_list = outputs[-1]
            # import ipdb
            # ipdb.set_trace()
            for moe_loss in moe_loss_list:
                if moe_loss is not None:
                    moe_losses.append(moe_loss)
            moe_loss = self.router_aux_loss_coef * sum(moe_losses)
            if labels is not None:
                # print(loss, sum(moe_losses), loss + moe_loss)
                loss += moe_loss
        # import ipdb
        # ipdb.set_trace()
        if not return_dict:
            output = (logits,) + outputs[1:]
            output = (moe_loss,) + output if moe_loss is not None else output
            return (loss,) + output if loss is not None else output

        return MoECausalLMOutputWithPast(
            loss=loss,
            moe_loss=moe_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            moe_loss_list=outputs.moe_loss_list,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs

    def initialize_moe_modules(self, model_args):


        self.config.moe['moe_enable'] = model_args.moe_enable
        self.config.moe['train_modules'] = model_args.train_modules
        self.config.moe['moe_mode'] = model_args.moe_mode
        self.config.moe['moe_layers_idx'] = model_args.moe_layers_idx
        self.config.moe['ep_size']= model_args.ep_size
        self.config.moe['top_k_experts'] = model_args.top_k_experts
        self.config.moe['capacity_factor'] = model_args.capacity_factor
        self.config.moe['eval_capacity_factor'] = model_args.eval_capacity_factor
        self.config.moe['min_capacity'] = model_args.min_capacity
        self.config.moe['use_residual'] = model_args.use_residual
        self.config.moe['router_aux_loss_coef'] = self.router_aux_loss_coef = model_args.router_aux_loss_coef
        # self.config.moe['train_modules'] = [
        #         # 'mlp.w1', 'mlp.w2', 'mlp.c_proj', 'wg',
        #         # 'wte', 'lm_head'
        #     ]
        if self.config.moe['train_modules'] is not None and len(self.config.moe['train_modules']) > 0:
            for n, p in self.named_parameters():
                if any(name in n for name in self.config.moe['train_modules']):
                    continue
                else:
                    p.requires_grad = False


        user_aux_weight = getattr(model_args, 'router_aux_loss_coef', 0.01)
        # Force the Model's global multiplier to 1.0
        # This ensures "loss += 1.0 * (weighted_aux + weighted_kd)"
        if hasattr(self, 'router_aux_loss_coef'):
            print(f"⚠️ Overriding global router_aux_loss_coef from {self.router_aux_loss_coef} to 1.0")
            print(f"   (Handling weighting internally: Aux={user_aux_weight}, KD={model_args.kd_loss_weight})")
            self.router_aux_loss_coef = 1.0
            self.config.router_aux_loss_coef = 1.0

        num_layers = self.config.num_hidden_layers

        moe_layers_idx = model_args.moe_layers_idx
        if model_args.moe_layers_idx is not None:
            model_args.moe_mode = 'custom'
            assert len(model_args.moe_layers_idx) <= num_layers
            assert max(model_args.moe_layers_idx) < num_layers
            assert min(model_args.moe_layers_idx) >= 0
        else:
            if model_args.moe_mode == "first_half":
                moe_layers_idx = list(range(0, num_layers // 2))
            elif model_args.moe_mode == "second_half":
                moe_layers_idx = list(range(num_layers // 2, num_layers))
            elif model_args.moe_mode == "sparse":
                moe_layers_idx = list(range(num_layers))[::2]
            elif model_args.moe_mode == "dense":
                moe_layers_idx = list(range(num_layers))
            else:
                raise NotImplementedError(
                    f'Only support ["first_half", "second_half", "sparse", "dense"], but found {model_args.moe_mode}')

        self.config.moe['moe_layers_idx'] = moe_layers_idx
        if len(model_args.num_experts) == 1:
            self.config.moe['num_experts'] = model_args.num_experts * len(moe_layers_idx)
        assert len(self.config.moe['num_experts']) == len(moe_layers_idx)

        # Load centroids if provided
        centroid_file = getattr(model_args, 'router_centroids_path', None)
        all_centroids = None
        if centroid_file:
            print(f"Loading router centroids from: {centroid_file}")
            all_centroids = joblib.load(centroid_file)
            print(f"Loaded centroids for layers: {list(all_centroids.keys())}")
        else:
            print("⚠️  Warning: No centroids provided. Teacher will use random init.")

        if True: #student_teacher == 
            print(f"\n{'='*70}")
            print("Initializing MoE Modules with Student-Teacher Routing")
            print(f"{'='*70}\n")

        for num_experts, layer_num in zip(self.config.moe['num_experts'], moe_layers_idx):

            # 1. Save original MLP state for expert initialization
            original_mlp = self.model.layers[layer_num].mlp
            pretrained_state_dict = original_mlp.state_dict()
            
            print(f"Original MLP parameters: {sum(p.numel() for p in original_mlp.parameters()):,}")
            
            self.model.layers[layer_num].mlp = MoE(
                self.config.hidden_size,
                expert=original_mlp,
                num_experts=num_experts,
                ep_size=model_args.ep_size,
                k=model_args.top_k_experts,
                capacity_factor=model_args.capacity_factor,
                eval_capacity_factor=model_args.eval_capacity_factor,
                min_capacity=model_args.min_capacity,
                use_residual=model_args.use_residual,
            )
            for e in self.model.layers[layer_num].mlp.deepspeed_moe.experts.deepspeed_experts:  # check weight
                loaded_state_dict = e.state_dict()
                assert all([torch.allclose(pretrained_state_dict[k], v) for k, v in loaded_state_dict.items()])
                assert all([torch.allclose(loaded_state_dict[k], v) for k, v in pretrained_state_dict.items()])


            # 3. Copy pretrained MLP weights to all experts
            print("Copying MLP weights to experts...")
            copy_mlp_to_experts(
                original_mlp=original_mlp,
                moe_layer=self.model.layers[layer_num].mlp,
                num_experts=num_experts,
                verbose=False
            )
            
            # Verify copying
            verification = verify_weight_copying(
                original_mlp=original_mlp,
                moe_layer=self.model.layers[layer_num].mlp,
                num_experts=num_experts,
                verbose=False
            )
            
            if not verification['all_experts_match_mlp']:
                raise RuntimeError(f"Expert weight copying failed for layer {layer_num}")
            
            print("✓ Expert weights initialized correctly")

            # router = self.model.layers[layer_num].mlp.deepspeed_moe.gate.wg

            # # load centroid file for this layer
            # path = f"/home/prafull/scratch/MoE-LLaVA-main/router_inits/centroids_model_layers_{layer_num}_post_attention_layernorm.pt" 
            # ckpt = torch.load(path, map_location="cpu")
            # centroids = ckpt["centroids"]   # [num_experts, hidden_size]

            # # assign to router
            # with torch.no_grad():
            #     router.weight.copy_(centroids)
            # print(f"[Init] Router at layer {layer_num} initialized from {path}")
            # 4. Inject K-Means Centroids into Router
            if all_centroids is not None and layer_num in all_centroids:
                print("Apply kmeans initialization")
                router = self.model.layers[layer_num].mlp.deepspeed_moe.gate.wg
                
                # Convert numpy array from pkl to torch tensor
                # Your pkl shape is [num_experts, hidden_size]
                centroids = torch.from_numpy(all_centroids[layer_num]).to(torch.float32)

                with torch.no_grad():
                    # Ensure device and dtype match (important for bfloat16/fp16 training)
                    router.weight.copy_(centroids.to(device=router.weight.device, dtype=router.weight.dtype))
                
                print(f"[Init] Router at layer {layer_num} successfully initialized from centroids.")
            else:
                print(f"[Init] Router at layer {layer_num} initialized randomly (no centroids found).")


            # 4. Get centroids for this layer
            layer_centroids = None
            if all_centroids and layer_num in all_centroids:
                layer_centroids = all_centroids[layer_num]
                print(f"✓ Using K-means centroids for teacher initialization")
                print(f"  Centroid shape: {layer_centroids.shape}")
            else:
                print(f"⚠️  No centroids for layer {layer_num}, using random teacher init")
            
            # 5. Create KD Gate
            kd_gate = KDTopKGate(
                model_dim=self.config.hidden_size,
                num_experts=num_experts,
                k=model_args.top_k_experts,
                centroids=layer_centroids,
                kd_loss_weight=getattr(model_args, 'kd_loss_weight', 0.1),
                aux_loss_weight=user_aux_weight,
                ema_decay=getattr(model_args, 'ema_decay', 0.999),
                capacity_factor=model_args.capacity_factor,
                eval_capacity_factor=model_args.eval_capacity_factor,
                min_capacity=model_args.min_capacity,
                noisy_gate_policy=getattr(model_args, 'noisy_gate_policy', None),
                drop_tokens=getattr(model_args, 'drop_tokens', True),
                use_rts=getattr(model_args, 'use_rts', True),
            )
            
            # Move gate to correct device
            kd_gate = kd_gate.to(device=self.device, dtype=self.dtype)
            
            print(f"✓ KD Gate created:")
            print(f"  Student weights: Random (PyTorch default)")
            print(f"  Teacher weights: {'Centroids' if layer_centroids is not None else 'Random'}")
            print(f"  KD loss weight: {kd_gate.kd_loss_weight}")
            print(f"  EMA decay: {kd_gate.ema_decay}")
            
            # 6. Swap the gate in MOELayer
            self.model.layers[layer_num].mlp.deepspeed_moe.gate = kd_gate
            
            # Verify student is random (not equal to teacher)
            if layer_centroids is not None:
                student_w = kd_gate.wg.weight.data
                teacher_w = kd_gate.teacher_weight
                max_diff = torch.max(torch.abs(student_w - teacher_w)).item()
                print(f"\n✓ Student-Teacher difference: {max_diff:.6f}")
                if max_diff < 1e-4:
                    print("  ⚠️  WARNING: Student and teacher are too similar!")
                    print("     Student should be random, not copied from teacher.")
            
            print(f"\n✓ Layer {layer_num} MoE initialization complete")

        print(f"\n{'='*70}")
        print("✅ All MoE modules initialized with Student-Teacher routing")
        print(f"{'='*70}\n")

        # ipdb.set_trace()
        rank0_print(f"LLM num_layers: {num_layers}, MoE num_layers: {len(moe_layers_idx)}, where\n",
                    *[f'layer-{layer_num} has {num_experts} experts\n' for num_experts, layer_num in
                      zip(self.config.moe['num_experts'], moe_layers_idx)])

        for m in self.model.layers:
            m.forward = MoEStablelmDecoderLayer_forward(m)
        rank0_print(f'replace StablelmDecoderLayer.forward to MoEStablelmDecoderLayer.forward')
        self.model.forward = MoEStablelmModel_forward(self.model)
        rank0_print(f'replace StablelmModel.forward to MoEStablelmModel.forward')
        # ipdb.set_trace()


def remove_teachers_before_save(model):
    """
    Remove all teacher components before saving checkpoint.
    
    Call this after training is complete, before model.save_pretrained()
    """
    print("\n" + "="*70)
    print("Removing Teacher Components")
    print("="*70 + "\n")
    
    removed_count = 0
    
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, 'deepspeed_moe'):
            gate = layer.mlp.deepspeed_moe.gate
            if hasattr(gate, 'disable_teacher'):
                print(f"Layer {layer_idx}: Removing teacher...")
                gate.disable_teacher()
                removed_count += 1
    
    print(f"\n✓ Removed {removed_count} teacher components")
    print("✓ Model now contains only student routers")
    print("="*70 + "\n")

class EvalMoELLaVAStablelmForCausalLM(MoELLaVAStablelmForCausalLM):
    config_class = MoELLaVAStablelmConfig

    def __init__(self, config):
        super(EvalMoELLaVAStablelmForCausalLM, self).__init__(config)

        self.router_aux_loss_coef = self.config.moe['router_aux_loss_coef']
        num_layers = self.config.num_hidden_layers
        moe_layers_idx = self.config.moe['moe_layers_idx']

        for num_experts, layer_num in zip(self.config.moe['num_experts'], moe_layers_idx):
            self.model.layers[layer_num].mlp = MoE(
                self.config.hidden_size,
                expert=self.model.layers[layer_num].mlp,
                num_experts=num_experts,
                ep_size=self.config.moe['ep_size'],
                k=self.config.moe['top_k_experts'],
                capacity_factor=self.config.moe['capacity_factor'],
                eval_capacity_factor=self.config.moe['eval_capacity_factor'],
                min_capacity=self.config.moe['min_capacity'],
                use_residual=self.config.moe['use_residual'],
            )
        rank0_print(f"LLM num_layers: {num_layers}, MoE num_layers: {len(moe_layers_idx)}, where\n",
                    *[f'layer-{layer_num} has {num_experts} experts\n' for num_experts, layer_num in
                      zip(self.config.moe['num_experts'], moe_layers_idx)])

        for m in self.model.layers:
            m.forward = MoEStablelmDecoderLayer_forward(m)
        rank0_print(f'replace StablelmDecoderLayer.forward to MoEStablelmDecoderLayer.forward')
        self.model.forward = MoEStablelmModel_forward(self.model)
        rank0_print(f'replace StablelmModel.forward to MoEStablelmModel.forward')



AutoConfig.register("moe_llava_stablelm", MoELLaVAStablelmConfig)
AutoModelForCausalLM.register(MoELLaVAStablelmConfig, MoELLaVAStablelmForCausalLM)

AutoModelForCausalLM.register(MoELLaVAStablelmConfig, EvalMoELLaVAStablelmForCausalLM)
