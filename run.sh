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
# CUDA_VISIBLE_DEVICES=0 nsys profile \
#   --trace=cuda,nvtx,osrt \
#   --sample=none \
#   --capture-range=cudaProfilerApi \
#   --capture-range-end=stop \
#   --force-overwrite=true \
#   -o profiles/retroinfer_qa1_128k_b1_sparse_cr05 \
#   python -u simple_test.py \
#     --data_path experiment_inputs/converted/qa1_131072_one.json \
#     --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
#     --device cuda:0 \
#     --batch_size 1 \
#     --prefill_bsz 1 \
#     --gen_len 128 \
#     --attn_type RetroInfer \
#     --dtype bf16 \
#     --cache_ratio 0.05 \
#     --retrieval_budget 0.018 \
#     --estimation_budget 0.232

# Test 2 + profoiling
# RETRO_TRACE=1 \
#   RETRO_TRACE_DIR="$PWD/traces/exp2/qa1_cr01" \
#   RETRO_TRACE_RUN_ID="qa1_128k_cr01" \
#   RETRO_TRACE_TASK="QA1" \
#   RETRO_TRACE_SAMPLE_ID="0" \
#   RETRO_TRACE_WARMUP_STEPS=8 \
#   CUDA_VISIBLE_DEVICES=0 \
#   python -u simple_test.py \
#     --data_path experiment_inputs/converted/qa1_131072_one.json \
#     --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
#     --device cuda:0 \
#     --batch_size 1 \
#     --prefill_bsz 1 \
#     --gen_len 129 \
#     --attn_type RetroInfer \
#     --dtype bf16 \
#     --cache_ratio 0.1 \
#     --retrieval_budget 0.018 \
#     --estimation_budget 0.232

# Actual trace -> analysis -> heatmap generation
# RETRO_TRACE=1 \
#   RETRO_TRACE_DIR="$PWD/traces/exp2/vt_cr005" \
#   RETRO_TRACE_RUN_ID="vt_128k_cr005" \
#   RETRO_TRACE_TASK="VT" \
#   RETRO_TRACE_SAMPLE_ID="0" \
#   RETRO_TRACE_WARMUP_STEPS=8 \
#   CUDA_VISIBLE_DEVICES=0 \
#   python -u simple_test.py \
#     --data_path experiment_inputs/converted/vt_131072_one.json \
#     --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
#     --device cuda:0 \
#     --batch_size 1 \
#     --prefill_bsz 1 \
#     --gen_len 129 \
#     --attn_type RetroInfer \
#     --dtype bf16 \
#     --cache_ratio 0.05 \
#     --retrieval_budget 0.018 \
#     --estimation_budget 0.232

# python tool/analyze_retrieval_trace.py \
#   --input traces/exp2/vt_cr005 \
#   --output-dir analysis/exp2/vt_cr005 \
#   --windows 1,2,4,8 \
#   --cache-state steady

# for w in 1 2 4 8; do
#   python tool/window_heatmap.py \
#     --input analysis/exp2/vt_cr005/lookback_coverage_samples.csv \
#     --output figures/exp2/vt_cr005/heatmap_window${w}.png \
#     --window-size ${w} \
#     --task VT
# done

# Cache hit ratio violin plot 그리는 코드
# python tool/analyze_retrieval_trace.py \
#     --input traces/exp2/qa1_cr001 traces/exp2/qa1_cr0025 traces/exp2/qa1_cr005 traces/exp2/qa1_cr01 \
#     --output-dir analysis/exp2/qa1_cache_sweep \
#     --context-bin 4096

# MPLCONFIGDIR=/tmp/matplotlib \
#   python tool/plot_experiment2.py \
#     --input-dir analysis/exp2/qa1_cache_sweep \
#     --output-dir figures/exp2/qa1_cache_sweep

# Retention heatmap 코드

# python tool/analyze_retrieval_trace.py \
#   --input traces/exp2/qa1_cr005 \
#   --output-dir analysis/exp2/qa1_cr005 \
#   --windows 1,2,4,8 \
#   --cache-state steady

# for window in 1 2 4 8
# do
#   python tool/window_heatmap.py \
#     --input analysis/exp2/fwe_cr005/lookback_coverage_samples.csv \
#     --output figures/exp2/fwe_cr005/heatmap_window${window}.png \
#     --window-size ${window} \
#     --task FWE
# done

# Heatmap (a)
# RETRO_TRACE=1 \
# RETRO_WEIGHT_TRACE=1 \
# RETRO_WEIGHT_TRACE_MAX_GROUPS=1 \
# RETRO_WEIGHT_TRACE_MAX_STEPS=128 \
# RETRO_WEIGHT_TRACE_MAX_KV_POS=131072 \
# RETRO_WEIGHT_TRACE_BUCKET_SIZE=256 \
# RETRO_WEIGHT_TRACE_SELECTED_ONLY=1 \
# RETRO_WEIGHT_TRACE_HEAD_STRIDE=4 \
# RETRO_WEIGHT_TRACE_LAYERS=0 \
# RETRO_TRACE_DIR="$PWD/traces/exp2/vt_cr005" \
# RETRO_TRACE_RUN_ID="vt_128k_cr005" \
# RETRO_TRACE_TASK="VT" \
# RETRO_TRACE_SAMPLE_ID="0" \
# RETRO_TRACE_WARMUP_STEPS=8 \
# CUDA_VISIBLE_DEVICES=0 \
# python -u simple_test.py \
#   --data_path experiment_inputs/converted/vt_131072_one.json \
#   --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
#   --device cuda:0 \
#   --batch_size 1 \
#   --prefill_bsz 1 \
#   --gen_len 129 \
#   --attn_type RetroInfer \
#   --dtype bf16 \
#   --cache_ratio 0.05 \
#   --retrieval_budget 0.018 \
#   --estimation_budget 0.232
  



RETRO_TRACE=1 \
RETRO_TOPK_TRACE=1 \
RETRO_TOPK_TRACE_K=1024 \
RETRO_WEIGHT_TRACE=1 \
RETRO_WEIGHT_TRACE_MAX_GROUPS=1 \
RETRO_WEIGHT_TRACE_MAX_STEPS=128 \
RETRO_WEIGHT_TRACE_MAX_KV_POS=131072 \
RETRO_WEIGHT_TRACE_BUCKET_SIZE=256 \
RETRO_WEIGHT_TRACE_SELECTED_ONLY=1 \
RETRO_WEIGHT_TRACE_HEAD_STRIDE=4 \
RETRO_WEIGHT_TRACE_LAYERS=0 \
RETRO_TRACE_DIR="$PWD/traces/exp2/fwe_cr005" \
RETRO_TRACE_RUN_ID="fwe_128k_cr005" \
RETRO_TRACE_TASK="FWE" \
RETRO_TRACE_SAMPLE_ID="0" \
RETRO_TRACE_WARMUP_STEPS=8 \
CUDA_VISIBLE_DEVICES=0 \
python -u simple_test.py \
  --data_path experiment_inputs/converted/fwe_131072_one.json \
  --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
  --device cuda:0 \
  --batch_size 1 \
  --prefill_bsz 1 \
  --gen_len 129 \
  --attn_type RetroInfer \
  --dtype bf16 \
  --cache_ratio 0.05 \
  --retrieval_budget 0.018 \
  --estimation_budget 0.232

# Heatmap (a)
python tool/plot_attention_heatmap.py \
  --input traces/exp2/fwe_cr005 \
  --output figures/exp2/fwe_cr005/attention_heatmap_b256.png \
  --layer 0 \
  --head 0 \
  --task FWE \
  --max-kv-pos 131072 \
  --bucket-size 256

# Heatmap (b)
python tool/plot_topk_kv_heatmap.py \
  --input traces/exp2/fwe_cr005 \
  --output figures/exp2/fwe_cr005/topk_kv_heatmap_k1024.png \
  --layer 0 \
  --head 0 \
  --task FWE \
  --top-k 1024 \
  --max-kv-pos 131072 \
  --bucket-size 64

# Heatmap (c)
python tool/plot_topk_overlap_heatmap.py \
--input traces/exp2/fwe_cr005 \
--output figures/exp2/fwe_cr005/topk_overlap_heatmap_k1024.png \
--layer 0 \
--head 0 \
--task FWE \
--top-k 1024 \
--deltas 1,2,4,8,16,32,64,128

# Heatmap (d) -> Discard
# python tool/plot_topk_hbm_hit_heatmap.py \
#   --input traces/exp2/fwe_cr005 \
#   --output figures/exp2/fwe_cr005/topk_hbm_hit_heatmap_k100.png \
#   --layer 0 \
#   --head 0 \
#   --task FWE \
#   --top-k 100 \
#   --cache-sizes-gb 64,32,16,8,4,2,1,0.5 \
#   --num-layers 32 \
#   --num-kv-heads 8 \
#   --head-dim 128 \
#   --dtype-bytes 2

# Heatmap (e)
python tool/plot_topk_jaccard_heatmap.py \
  --input traces/exp2/fwe_cr005 \
  --output figures/exp2/fwe_cr005/topk_jaccard_heatmap_k1024.png \
  --layer 0 \
  --head 0 \
  --task FWE \
  --top-k 1024

# Test 3, GPU benchmark
# python tool/benchmark_sparse_primitives.py \
#     --output benchmark_results/gpu_vt_trace_points.csv \
#     --device cuda:0 \
#     --batch 1 \
#     --kv-heads 8 \
#     --query-heads 32 \
#     --head-dim 128 \
#     --dtype bf16 \
#     --layer-events analysis/exp2/vt_cr005/layer_events.csv \
#     --selected-vectors 512,1024,2048,4096 \
#     --hit-ratios 0.1,0.25,0.5,0.75,1.0 \
#     --layout contiguous \
#     --warmup 100 \
#     --iterations 500 \
#     --runs 5