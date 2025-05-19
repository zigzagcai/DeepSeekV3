# Efficient DeepSeekV3 HuggingFace Modeling

For study purpose, we refined original DeepSeekV3 HuggingFace modeling (https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py) to make it capable of training with `FSDP+EP` at scale, and might implement the missing part of the original modeling:

1. Multi Token Prediction; 
2. Auxiliary Free Load Balancing; 
3. Grouped GEMM for Experts;
4. Expert Parallelism;

based on the details of DeepSeek-V3 Technical Report (https://arxiv.org/abs/2412.19437) and other open-sourced projects such as vllm/TensorRT-LLM.

## Convert HF checkpoint to DCP checkpoint
```
python convert_ckpt_hf2dcp.py --input input_hf_ckpt_path --output output_dcp_ckpt_path
```

## Train with FSDP and Expert Parallelism

Please reference the details in https://github.com/pytorch/pytorch/issues/149396
