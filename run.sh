#!/bin/bash
export MKL_THREADING_LAYER=GNU

# Original test (정상 동작 검증 테스트)

# CUDA_VISIBLE_DEVICES=0 \
# python -u simple_test.py \
#   --batch_size 1 \
#   --prefill_bsz 1 \
#   --gen_len 16 \
#   --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
#   --attn_type RetroInfer \
#   --dtype bf16 \
#   --retrieval_budget 0.018 \
#   --estimation_budget 0.232 \
#   --cache_ratio 0.05

# Trace test
# export RETRO_TRACE=1
# export RETRO_TRACE_DIR="$PWD/traces/fixed_120k"
# export RETRO_TRACE_RUN_ID="simple_120k_b1_g256"
# export RETRO_TRACE_TASK="simple_test"

# CUDA_VISIBLE_DEVICES=0 \
# python -u simple_test.py \
#   --batch_size 1 \
#   --prefill_bsz 1 \
#   --gen_len 257 \
#   --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
#   --attn_type RetroInfer \
#   --dtype bf16 \
#   --retrieval_budget 0.018 \
#   --estimation_budget 0.232 \
#   --cache_ratio 0.05 \
#   --data_path simple_test_data.json

# RULER trace
# cd benchmark/ruler

# for length in 32768 65536 131072 262144 524288
#   do
#   for task in qa_1 vt fwe #fwe vt niah_multikey_3 niah_multiquery qa_2 
#   do
#       export RETRO_TRACE=1
#       export RETRO_TRACE_DIR="$PWD/../../traces/ruler_context_length/${length}/${task}"
#       export RETRO_TRACE_RUN_ID="ruler_128k_${task}"
#       export RETRO_TRACE_TASK="${task}"

#       NUM_SAMPLES=8 \
#       CACHE_RATIO=0.05 \
#       CUDA_VISIBLE_DEVICES=0 \
#       bash ruler_run.sh \
#         llama-3-8b-1048k \
#         full \
#         RetroInfer \
#         "${length}" \
#         "${task}" \
#         bf16 \
#         0.018 \
#         0.232
#   done
# done

# for cache_ratio in 0.01 0.025 0.05 0.10
#   do
#   for task in qa_1 vt fwe #fwe vt niah_multikey_3 niah_multiquery qa_2 
#   do
#       export RETRO_TRACE=1
#       export RETRO_TRACE_DIR="$PWD/../../traces/ruler_cache_ratio/${cache_ratio}/${task}"
#       export RETRO_TRACE_RUN_ID="ruler_128k_${task}"
#       export RETRO_TRACE_TASK="${task}"

#       NUM_SAMPLES=8 \
#       CACHE_RATIO="${cache_ratio}" \
#       CUDA_VISIBLE_DEVICES=0 \
#       bash ruler_run.sh \
#         llama-3-8b-1048k \
#         full \
#         RetroInfer \
#         131072 \
#         "${task}" \
#         bf16 \
#         0.018 \
#         0.232
#   done
# done

# Test 1

# CUDA_VISIBLE_DEVICES=0 \
# python -u simple_test.py \
#   --data_path experiment_inputs/converted/vt_131072_one.json \
#   --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
#   --device cuda:0 \
#   --batch_size 1 \
#   --prefill_bsz 1 \
#   --gen_len 64 \
#   --attn_type RetroInfer \
#   --dtype bf16 \
#   --cache_ratio 0.05 \
#   --retrieval_budget 0.018 \
#   --estimation_budget 0.232 \
#   2>&1 | tee experiment_logs/vt_131072_b1.log

# Test 1 + Nsight Systems
CUDA_VISIBLE_DEVICES=0 nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  -o profiles/retroinfer_vt_128k_b1_sparse \
  python -u simple_test.py \
    --data_path experiment_inputs/converted/vt_131072_one.json \
    --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
    --device cuda:0 \
    --batch_size 1 \
    --prefill_bsz 1 \
    --gen_len 64 \
    --attn_type RetroInfer \
    --dtype bf16 \
    --cache_ratio 0.05 \
    --retrieval_budget 0.018 \
    --estimation_budget 0.232
