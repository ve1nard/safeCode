import os
import torch
from typing import Optional, Tuple, Union, List
from transformers import AutoTokenizer, AutoConfig, logging
from transformers.modeling_outputs import CausalLMOutputWithPast, CausalLMOutputWithCrossAttentions
from sven.hf import CodeGenForCausalLM, XGLMForCausalLM, GPT2LMHeadCustomModel, GPT2CustomConfig, Qwen2ForCausalLM
from sven.hf import DeepseekV3ForCausalLM 
from transformers import PreTrainedModel, PretrainedConfig
from torch import nn
from transformers import AutoTokenizer
#from transformers import LlamaTokenizer
from transformers import AutoConfig
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

class DeepseekPrefixCausalLM(DeepseekV3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        self.n_embed_per_head = config.hidden_size // config.num_attention_heads
        self.prefix_params = torch.nn.ParameterList()

        for _ in range(config.n_control):
            for _ in range(config.num_hidden_layers):
                for _ in range(2):  # key and value
                    param_size = (config.num_attention_heads, config.n_prefix_token, self.n_embed_per_head)
                    param = torch.nn.Parameter(torch.zeros(param_size, requires_grad=True))
                    self.prefix_params.append(param)

        self.dropout = torch.nn.Dropout(config.prefix_dropout)

    def get_past_from_prefix(self, control_ids):
        past = list()
        for i in range(self.config.num_hidden_layers):
            past.append([])
            key_stack, val_stack = [], []
            for control_id in control_ids:
                key_idx = control_id * self.config.num_hidden_layers * 2 + i * 2
                val_idx = key_idx + 1
                key = self.dropout(self.prefix_params[key_idx])
                val = self.dropout(self.prefix_params[val_idx])
                key_stack.append(key)
                val_stack.append(val)
            past[i].append(torch.stack(key_stack))  # [batch, n_heads, n_prefix, dim]
            past[i].append(torch.stack(val_stack))
        return past

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
        else:
            control_ids = [kwargs['control_id']] * input_ids.shape[0]
            past = self.get_past_from_prefix(control_ids)

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": None,
            "attention_mask": kwargs.get("attention_mask", None),
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        control_id=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

class CodeQwenPrefixCausalLM(Qwen2ForCausalLM):

    def __init__(self, config):
        super().__init__(config)

        # ───── GQA geometry ─────────────────────────────
        # total heads H, heads per group h, so groups = H // h
        self.heads_per_group = config.num_key_value_heads
        self.num_groups      = config.num_attention_heads // self.heads_per_group

        # prefix length and head dims
        self.n_prefix_token   = config.n_prefix_token
        self.n_embed_per_head = config.hidden_size // config.num_attention_heads

        # ───── Prefix parameters ────────────────────────
        self.prefix_params = torch.nn.ParameterList()
        for _ in range(config.n_control):
            for _ in range(config.num_hidden_layers):
                # each key/value block shaped [groups, heads_per_group, prefix_len, head_dim]
                key = torch.nn.Parameter(torch.zeros(
                    self.num_groups,
                    self.heads_per_group,
                    self.n_prefix_token,
                    self.n_embed_per_head
                ))
                val = torch.nn.Parameter(torch.zeros_like(key))
                self.prefix_params.extend([key, val])

        self.dropout = torch.nn.Dropout(config.prefix_dropout)
        self.post_init()  # HF‐style weight initialization

    #def get_input_embeddings(self):
     #   return self.embed_tokens
    
    #def set_input_embeddings(self, new_embeddings):
     #   self.embed_tokens = new_embeddings
        

    def get_past_from_prefix(self, control_ids):
        past = []
        params_per_layer = 2 * self.config.num_hidden_layers
        for layer_idx in range(self.config.num_hidden_layers):
            key_stack, val_stack = [], []
            for cid in control_ids:
                base = cid*params_per_layer + layer_idx*2
                key = self.dropout(self.prefix_params[base])
                val = self.dropout(self.prefix_params[base+1])
                key_stack.append(key)
                val_stack.append(val)
            # each stack: [batch, num_groups, heads_per_group, prefix_len, head_dim]
            past.append([torch.stack(key_stack), torch.stack(val_stack)])
        return past

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
        else:
            control_ids = [kwargs['control_id']] * input_ids.shape[0]
            past = self.get_past_from_prefix(control_ids)
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": None,
            "attention_mask": None,
            "token_type_ids": token_type_ids,
        }

    def _flatten_past(self, raw_past):
        """
        Collapse an extra 'group' level if present:
          raw_past: List of [key_tensor, val_tensor], where each tensor can be
                    either 4-D ([batch, heads, seq, head_dim]) or 5-D
                    ([batch, groups, heads_per_group, seq, head_dim]).
        Returns a HF-style tuple of (key, val) pairs, each 4-D.
        """
        flat = []
        for layer in raw_past:
            key_states, val_states = layer  # layer is [key_tensor, val_tensor]
            # if there's a group dim, merge it into the head dim
            if key_states.ndim == 5:
                b, groups, heads_pg, seq, hd = key_states.shape
                key_states = key_states.reshape(b, groups * heads_pg, seq, hd)
                val_states = val_states.reshape(b, groups * heads_pg, seq, hd)
            flat.append((key_states, val_states))
        return tuple(flat)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        control_id: Optional[int] = None,           # NEW argument
        past_key_values=None,                       # we’ll ignore this if control_id is set
        **kwargs
    ):
        # 1) If trainer passed control_id (chat‑style or prefix‑style), build & flatten past:
        if control_id is not None:
            raw_past = self.get_past_from_prefix([control_id])
            flat_past = []
            for key_states, val_states in raw_past:
                # 5‑D → 4‑D: merge group dim into head dim if needed
                if key_states.ndim == 5:
                    b, groups, heads_pg, seqlen, hd = key_states.shape
                    key_states = key_states.view(b, groups * heads_pg, seqlen, hd)
                    val_states = val_states.view(b, groups * heads_pg, seqlen, hd)
                flat_past.append((key_states, val_states))
            past_key_values = tuple(flat_past)

        # 2) Delegate to HF’s forward, feeding in our flattened past_key_values
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

class CodeGenPrefixCausalLM(CodeGenForCausalLM): # the use of CodeGenForCasualLM in the argument of the class makes it the child of the parent class "CodeGenCasualLM"
    def __init__(self, config):
        # Initialize the parent class (CodeGenForCausalLM) with the given configuration
        super().__init__(config)
        
        # Calculate the number of embeddings per attention head
        self.n_embed_per_head = config.n_embd // config.n_head
        
        # Create a list of prefix parameters that will be used as learnable prefix tokens for each layer
        self.prefix_params = torch.nn.ParameterList()
        for _ in range(config.n_control):  # Iterate over the number of control prefixes
            for _ in range(config.n_layer):  # Iterate over the number of transformer layers
                for _ in range(2):  # Each layer has two prefix parameters (key and value)
                    # Define the parameter size as (number of heads, number of prefix tokens, embedding per head)
                    param_size = (config.n_head, config.n_prefix_token, self.n_embed_per_head)
                    # Initialize the parameter with zeros and set it as trainable
                    param = torch.nn.Parameter(torch.zeros(param_size, requires_grad=True))
                    self.prefix_params.append(param)
        
        # Apply dropout to the prefix tokens during training to prevent overfitting
        self.dropout = torch.nn.Dropout(config.prefix_dropout)

    def get_past_from_prefix(self, control_ids):
        # Generate the "past" key-value pairs from the prefix parameters based on the control IDs
        past = []
        for i in range(self.config.num_hidden_layers):  # Iterate over transformer layers
            past.append([])
            key_stack, val_stack = [], []
            for control_id in control_ids:  # For each control prefix
                # Compute indices for key and value in the prefix_param list
                key_idx = control_id * self.config.num_hidden_layers * 2 + i * 2
                val_idx = key_idx + 1

                # Retrieve and apply dropout to key and value
                key = self.dropout(self.prefix_params[key_idx])
                val = self.dropout(self.prefix_params[val_idx])

                key_stack.append(key)
                val_stack.append(val)

            # Stack across batch/control dimension and append to past
            past[i].append(torch.stack(key_stack))  # Shape: [batch, num_heads, prefix_len, embed]
            past[i].append(torch.stack(val_stack))
        return past


    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        # Prepare the inputs for text generation, including handling past key-value states and control IDs
        token_type_ids = kwargs.get("token_type_ids", None)
        if past:
            # If past key-value states are provided, use only the last token as input
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
        else:
            # If no past is provided, generate the "past" from prefix using control IDs
            control_ids = [kwargs['control_id']] * input_ids.shape[0]  # Use the same control ID for all input samples
            past = self.get_past_from_prefix(control_ids)

        # Return the prepared input dictionary for generation
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": None,
            "attention_mask": None,
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        control_id = None,  # Placeholder for passing checks of huggingface, actually unused in this function
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # Call the forward method of the parent class (CodeGenForCausalLM) with the provided inputs
        return super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class IncoderPrefixLM(XGLMForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        self.n_embed_per_head = config.d_model // config.attention_heads
        self.prefix_params = torch.nn.ParameterList()
        for _ in range(config.n_control):
            for _ in range(config.num_layers):
                for _ in range(2):
                    param_size = (config.attention_heads, config.n_prefix_token, self.n_embed_per_head)
                    param = torch.nn.Parameter(torch.zeros(param_size, requires_grad=True))
                    self.prefix_params.append(param)
        self.dropout = torch.nn.Dropout(config.prefix_dropout)

    def get_past_from_prefix(self, control_ids):
        past = list()
        for i in range(self.config.num_layers):
            past.append(list())
            key_stack, val_stack = [], []
            for control_id in control_ids:
                key_idx = control_id * self.config.num_layers * 2 + i * 2
                val_idx = key_idx + 1
                key = self.dropout(self.prefix_params[key_idx])
                val = self.dropout(self.prefix_params[val_idx])
                key_stack.append(key)
                val_stack.append(val)
            past[i].append(torch.stack(key_stack))
            past[i].append(torch.stack(val_stack))
        return past

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, use_cache=None, **kwargs):
        if past:
            input_ids = input_ids[:, -1:]
        else:
            control_ids = [kwargs['control_id']] * input_ids.shape[0]
            past = self.get_past_from_prefix(control_ids)
        # first step, decoder_cached_states are empty
        return {
            "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
            "attention_mask": None,
            "past_key_values": past,
            "use_cache": use_cache,
        }

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        control_id = None, # placeholder for passing checks of huggingface, actually unused in this function
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        return super().forward(
            input_ids,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            head_mask,
            cross_attn_head_mask,
            past_key_values,
            inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

class SantaPrefixLM(GPT2LMHeadCustomModel):
    def __init__(self, config):
        super().__init__(config)

        self.n_embed_per_head = config.n_embd // config.n_head
        self.prefix_params = torch.nn.ParameterList()
        for _ in range(config.n_control):
            for _ in range(config.n_layer):
                # mha
                for _ in range(2):
                    param_size = (config.n_head, config.n_prefix_token, self.n_embed_per_head)
                    param = torch.nn.Parameter(torch.zeros(param_size, requires_grad=True))
                    self.prefix_params.append(param)
        self.dropout = torch.nn.Dropout(config.prefix_dropout)

    def get_past_from_prefix(self, control_ids):
        past = list()
        for i in range(self.config.n_layer):
            past.append(list())
            key_stack, val_stack = [], []
            for control_id in control_ids:
                key_idx = control_id * self.config.n_layer * 2 + i * 2
                val_idx = key_idx + 1
                key = self.dropout(self.prefix_params[key_idx])
                val = self.dropout(self.prefix_params[val_idx])
                key_stack.append(key)
                val_stack.append(val)
            past[i].append(torch.stack(key_stack))
            past[i].append(torch.stack(val_stack))
        return past

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
        else:
            control_ids = [kwargs['control_id']] * input_ids.shape[0]
            past = self.get_past_from_prefix(control_ids)

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": None,
            "attention_mask": None,
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        control_id = None, # placeholder for passing checks of huggingface, actually unused in this function
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        return super().forward(
            input_ids,
            past_key_values,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

class OpenCodeInterpreterPrefixCausalLM(PreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

        # Define model's basic components from the base model (OpenCodeInterpreter-DS-33B)
        self.model = AutoModelForCausalLM.from_pretrained('m-a-p/OpenCodeInterpreter-DS-33B')
        self.n_embed_per_head = config.hidden_size // config.num_attention_heads
        
        # Initialize prefix parameters to control generation
        self.prefix_params = nn.ParameterList()
        for _ in range(config.num_control_prefixes):
            for _ in range(config.num_hidden_layers):
                for _ in range(2):  # Two prefix parameters for each layer (key and value)
                    param_size = (config.num_attention_heads, config.num_prefix_tokens, self.n_embed_per_head)
                    param = nn.Parameter(torch.zeros(param_size, requires_grad=True))  # Initialize parameter as zeros
                    self.prefix_params.append(param)
        
        # Dropout layer to apply on prefix parameters
        self.dropout = nn.Dropout(config.prefix_dropout_rate)
        self.post_init()

    def get_past_from_prefix(self, control_ids):
        past = list()
        for i in range(self.config.num_hidden_layers):
            past.append(list())
            key_stack, val_stack = [], []
            for control_id in control_ids:
                # Calculate the indices for the key and value prefix parameters
                key_idx = control_id * self.config.num_hidden_layers * 2 + i * 2
                val_idx = key_idx + 1
                # Apply dropout to the prefix parameters
                key = self.dropout(self.prefix_params[key_idx])
                val = self.dropout(self.prefix_params[val_idx])
                key_stack.append(key)
                val_stack.append(val)
            # Stack the key and value tensors for the current layer
            past[i].append(torch.stack(key_stack))
            past[i].append(torch.stack(val_stack))
        return past

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        if past_key_values:
            # If past key values are provided, use only the last token for generation
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
        else:
            # If no past key values, generate them from the prefix parameters using control IDs
            control_ids = [kwargs['control_id']] * input_ids.shape[0]
            past_key_values = self.get_past_from_prefix(control_ids)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "position_ids": None,
            "attention_mask": kwargs.get("attention_mask", None),
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        control_id: Optional[int] = None,  # Placeholder for passing control signal
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # Pass the prepared inputs into the base model and get output
        return self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

def model_from_pretrained(lm_path, model_type, config):
    # pick the right class
    if lm_path == "Qwen/Qwen2.5-Coder-1.5B-Instruct":
        model_class = Qwen2ForCausalLM if model_type == 'lm' else CodeQwenPrefixCausalLM

    elif lm_path.startswith('Salesforce/codegen-'):
        model_class = CodeGenForCausalLM if model_type == 'lm' else CodeGenPrefixCausalLM

    elif lm_path.startswith('facebook/incoder-'):
        # zero out dropouts on Incoder
        if config is not None:
            config.attention_dropout = 0.0
            config.dropout = 0.0
        model_class = XGLMForCausalLM if model_type == 'lm' else IncoderPrefixLM

    elif lm_path == 'bigcode/santacoder':
        # use our custom GPT2 config for SantaCoder
        if config is not None:
            config.attn_pdrop = config.embd_pdrop = config.resid_pdrop = 0.0
        model_class = GPT2LMHeadCustomModel if model_type == 'lm' else SantaPrefixLM

    elif lm_path.startswith('m-a-p/OpenCodeInterpreter-DS-33B'):
        model_class = OpenCodeInterpreterForCausalLM if model_type == 'lm' else OpenCodeInterpreterPrefixCausalLM

    elif lm_path.startswith("deepseek-ai/deepseek-coder"):
        model_class = DeepseekV3ForCausalLM if model_type == 'lm' else DeepseekPrefixCausalLM

    else:
        raise ValueError(f"No model mapping for {lm_path} / {model_type}")

    # load with or without custom config
    if config is None:
        return model_class.from_pretrained(lm_path)
    return model_class.from_pretrained(lm_path, config=config)


def config_from_pretrained(lm_path, path):
    if lm_path == "Qwen/Qwen2.5-Coder-1.5B-Instruct":
        return AutoConfig.from_pretrained(path)
    if lm_path == 'bigcode/santacoder':
        return GPT2CustomConfig.from_pretrained(path, revision='mha')
    if lm_path.startswith("deepseek-ai/deepseek-coder"):
        return DeepseekV3Config.from_pretrained(path)
    return AutoConfig.from_pretrained(path)


def config_from_pretrained(lm_path, path):
    if lm_path == "Qwen/Qwen2.5-Coder-1.5B-Instruct":
        return AutoConfig.from_pretrained(path)
    elif lm_path == 'bigcode/santacoder':
        return GPT2CustomConfig.from_pretrained(path, revision='mha')
    elif lm_path == 'deepseek-ai/deepseek-coder-1.3b-base':
        return DeepseekV3Config.from_pretrained(path)
    else:
        return AutoConfig.from_pretrained(path)


def save_model(model, path, args):
    if isinstance(model, (
        CodeGenPrefixCausalLM,
        IncoderPrefixLM,
        SantaPrefixLM,
        OpenCodeInterpreterPrefixCausalLM,
        CodeQwenPrefixCausalLM,
        DeepseekPrefixCausalLM
    )):
        assert args.pretrain_dir.startswith('Salesforce/codegen-') \
            or args.pretrain_dir.startswith('facebook/incoder-') \
            or args.pretrain_dir == 'bigcode/santacoder' \
            or args.pretrain_dir.startswith('m-a-p/OpenCodeInterpreter-DS-33B') \
            or args.pretrain_dir == "Qwen/Qwen2.5-Coder-1.5B-Instruct" \
            or args.pretrain_dir.startswith("deepseek-ai/deepseek-coder")

        config_file = os.path.join(path)
        model.config.save_pretrained(config_file)
        prefix_file = os.path.join(path, 'pytorch_model.bin')
        state_dict = model.prefix_params.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = v.cpu()
        torch.save(state_dict, prefix_file)
        lm_path_file = os.path.join(path, 'lm.txt')
        with open(lm_path_file, 'w') as f:
            f.write(args.pretrain_dir)
    else:
        model.save_pretrained(path)


def load_model(model_type, path, is_training, args):
    logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.eos_token_id = tokenizer.eos_token_id or tokenizer.bos_token_id
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    if model_type == 'lm':
        cfg = config_from_pretrained(path, path)
        model = model_from_pretrained(path, 'lm', cfg)

    elif model_type == 'prefix':
        lm_path = path if is_training else open(os.path.join(path, 'lm.txt')).read().strip()
        cfg = config_from_pretrained(lm_path, path if not is_training else lm_path)
        # set prefix params on cfg
        cfg.n_prefix_token = args.n_prefix_token
        cfg.prefix_dropout = args.dropout
        cfg.n_control = 2

        model = model_from_pretrained(lm_path, 'prefix', cfg)
        if not is_training:
            # load the saved prefix weights
            prefix_file = os.path.join(path, 'pytorch_model.bin')
            model.prefix_params.load_state_dict(torch.load(prefix_file, map_location='cpu'))

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.resize_token_embeddings(len(tokenizer))
    device = parallelize_model(model, args)
    return tokenizer, model, device



def parallelize_model(model, args):
    if args.n_gpu > 1:
        model.parallelize()
        input_device = model.first_device
    else:
        model.to(args.device)
        input_device = args.device
    return input_device

    
