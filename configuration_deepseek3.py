"""                 self-written DeepSeek V3 model in PyTorch, trainable with FSDP+EP                """
""" modified from https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/configuration_deepseek.py """
"""          tested with transformers==4.46.3, torch==2.4.1, cuda==12.4, flash-attn==2.6.3           """

from typing import Dict
import torch
import torch.distributed as dist

from transformers.configuration_utils import PretrainedConfig

class DeepseekV3Config(PretrainedConfig):
    model_type = "deepseek_v3"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 129280, # v2 102400, v3 129280
        hidden_size: int = 7168, # Dimension of the hidden representations.
        intermediate_size: int = 18432, # Dimension of the MLP representations.
        moe_intermediate_size: int = 2048, # Dimension of the MoE representations.
        num_hidden_layers: int = 61, # Number of hidden layers in the Transformer decoder. num_hidden_layers=8 is runnable with FSDP and EP=8 on 32gpus
        num_nextn_predict_layers: int = 1, # Number of MTP layers
        num_attention_heads: int = 128, # Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads: int= 128, # Number of key_value heads for each attention layer in the Transformer decoder.
        n_shared_experts: int = 1, # Number of shared experts, None means dense model.
        n_routed_experts: int = 256, # Number of routed experts, None means dense model.
        routed_scaling_factor: float = 2.5, # Scaling factor or routed experts.
        kv_lora_rank: int = 512,
        q_lora_rank: int = 1536,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        qk_nope_head_dim: int = 128,
        topk_method: str = 'noaux_tc', # Topk method used in routed gate.
        n_group: int = 8, # Number of groups for routed experts.
        topk_group: int = 4, # Number of selected groups for each token(for each token, ensuring the selected experts is only within `topk_group` groups).
        num_experts_per_tok: int = 8, # Number of selected experts, None means dense model.
        moe_layer_freq: int = 1, # The frequency of the MoE layer: one expert layer for every `moe_layer_freq - 1` dense layers.
        first_k_dense_replace: int = 3, # Number of dense layers in shallow layers.
        norm_topk_prob: bool = True, # Whether to normalize the weights of the routed experts.
        scoring_func: str = 'sigmoid', # Method of computing expert weights.
        hidden_act: str = "silu", # The non-linear activation function (function or string) in the decoder.
        max_position_embeddings: int = 4096,
        initializer_range: float = 0.02, # The standard deviation for initializing all weights.
        rms_norm_eps: float = 1e-6, # The epsilon used by the rms normalization layers.
        use_cache: bool = False,
        pad_token_id: int = 0,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        tie_word_embeddings: bool = False, # Whether the model's input and output word embeddings should be tied.
        rope_theta: float = 10000.0, # The base period of the RoPE embeddings.
        rope_scaling: Dict = None,
        attention_bias: bool = False, # Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout: float = 0.0, # The dropout ratio for the attention probabilities.
        mlp_bias: bool = False, # Whether to use bias in MLP (both FFN and GroupedFFN).
        mlp_layer_fusion: bool = False, # Whether to use layer fusion in MLP (both FFN and GroupedFFN).
        # MoE load balancing loss
        moe_loss_type: str = "default", # ("none", "default", "seq_aux")
        moe_loss_alpha: float = 0.0001, # Auxiliary moe loss coefficient.
        # Others
        ep_group: dist.ProcessGroup = None, # Expert parallel process group
        device = "meta",
        dtype: torch.dtype = torch.bfloat16,
        multiple_of: int = 1,
        checkpoint: float = 1, # Layerwise activation checkpoint
        attn_implementation: str = "flash_attention_2",
        return_dict: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.mlp_layer_fusion = mlp_layer_fusion
        self.moe_loss_type = moe_loss_type
        self.moe_loss_alpha = moe_loss_alpha
        self.ep_group = ep_group
        self.device = device
        self.dtype = dtype
        self.multiple_of = multiple_of
        self.checkpoint = checkpoint
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            attn_implementation=attn_implementation,
            return_dict=return_dict,
            **kwargs,
        )
