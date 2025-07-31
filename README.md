# Efficient DeepSeekV3 HuggingFace Modeling

For study purpose, we refined original DeepSeekV3 HuggingFace modeling (https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py) to make it capable of training with `FSDP+EP` at scale (Minimum requirement of 256x A100/H100 GPUs is enough for full-size DeepSeek-V3 671B), and might implement the missing part of the original modeling:

1. Multi Token Prediction; 
2. Auxiliary Free Load Balancing; 
3. Grouped GEMM for Experts;
4. Expert Parallelism;

based on the details of DeepSeek-V3 Technical Report (https://arxiv.org/abs/2412.19437) and open-sourced projects (some code snippet just adapted from https://github.com/InternLM/InternEvo).

Our implementation is particularly useful for fast prototyping or integration into RL systems (such as VeRL/OpenRLHF), and keeps good balance between usability and efficiency.

## Convert HF checkpoint to DCP checkpoint

``` bash
python convert_ckpt_hf2dcp.py --input input_hf_ckpt_path --output output_dcp_ckpt_path
```

## Train with FSDP and Expert Parallelism

### Pre-requisites

Since some assertions in the FSDP source code might be too strict, we need to comment out two assertions:

- https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_init_utils.py#L680-L683
``` python
#if _get_module_fsdp_state(module):
#    # TODO: We may relax this by taking the FSDP instance's wrapped
#    # module to provide more flexibility to the user.
#    raise ValueError("`ignored_modules` should not include FSDP modules")
```

- https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_wrap_utils.py#L43-L45
```python
## TODO: We may relax this no-nested-wrapping constraint to support manual
## wrapping followed by auto wrapping.
#_check_nested_wrapping(root_module)
```

We have discussed the details with FSDP developer, and the accuracy is guaranteed.

### How to wrap Expert modules and non-Expert modules with separate process group?

Assume `expert_data_process_group` is the process group where you want to shard Expert modules, and `data_process_group` is the process group where you want to shard non-Expert modules.

``` python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy, BackwardPrefetch

ignored_mod = []
for layer_id, layer in enumerate(model.layers):
    if layer_id >= config.first_k_dense_replace:
        layer.mlp.experts = FSDP(
            layer.mlp.experts, 
            process_group=expert_data_process_group,
            sharding_strategy=ShardingStrategy.FULL_SHARD, 
            forward_prefetch=True,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            limit_all_gathers=True,
            use_orig_params=True,
        )
        ignored_mod.append(layer.mlp.experts)
model = FSDP(
    module=model,
    process_group=data_process_group,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    auto_wrap_policy=ModuleWrapPolicy(wrap_cls),
    forward_prefetch=True,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    limit_all_gathers=True,
    use_orig_params=True,
    ignored_modules=ignored_mod,
)
```

### How to load HuggingFace pretrained weights?

After FSDP wrap finished, you might use below code snippet to load converted DCP checkpoint.

``` python
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict

state_dict = get_model_state_dict(model=model)
state_dict = {key: state_dict[key].clone().detach() for key in state_dict}
dcp.load(state_dict=state_dict, checkpoint_id=output_dcp_ckpt_path)
set_model_state_dict(model=model, model_state_dict=state_dict)
del state_dict
torch.cuda.empty_cache()
```

For more details, please refer to https://github.com/pytorch/pytorch/issues/149396

## Developers

[@zigzagcai](https://github.com/zigzagcai)
[@rui23](https://github.com/rui23)
