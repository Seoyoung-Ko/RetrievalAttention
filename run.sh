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
export RETRO_TRACE=1
export RETRO_TRACE_DIR="$PWD/traces/fixed_120k"
export RETRO_TRACE_RUN_ID="simple_120k_b1_g256"
export RETRO_TRACE_TASK="simple_test"

CUDA_VISIBLE_DEVICES=0 \
python -u simple_test.py \
  --batch_size 1 \
  --prefill_bsz 1 \
  --gen_len 257 \
  --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
  --attn_type RetroInfer \
  --dtype bf16 \
  --retrieval_budget 0.018 \
  --estimation_budget 0.232 \
  --cache_ratio 0.05 \
  --data_path simple_test_data.json