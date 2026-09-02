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
#   -o profiles/retroinfer_vt_128k_b1_sparse_cr90 \
#   python -u simple_test.py \
#     --data_path experiment_inputs/converted/vt_131072_one.json \
#     --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
#     --device cuda:0 \
#     --batch_size 1 \
#     --prefill_bsz 1 \
#     --gen_len 64 \
#     --attn_type RetroInfer \
#     --dtype bf16 \
#     --cache_ratio 0.90 \
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

# Test 3, GPU benchmark
python tool/benchmark_sparse_primitives.py \
    --output benchmark_results/gather_random.csv \
    --device cuda:0 \
    --batch 1 \
    --kv-heads 8 \
    --query-heads 32 \
    --head-dim 128 \
    --dtype bf16 \
    --layer-events analysis/exp2/vt_cr005/layer_events.csv \
    --selected-vectors 512,1024,2048,4096 \
    --hit-ratios 0.1,0.25,0.5,0.75,1.0 \
    --layout random \
    --warmup 100 \
    --iterations 500 \
    --runs 5