#!/usr/bin/env bash
# This engine vs vLLM on Qwen3-0.6B, same GPU, same client (vllm bench serve,
# random 256 in / 256 out, EOS ignored, greedy), same session. LOG.md has the
# pre-registration. Arms: ours (CUDA graphs), vllm, ours (graphs), vllm, then
# ours without graphs once. Needs the Inference Lab checked out next to this
# repo (its .venv for the benchmark client, scripts/docker-vllm.sh for vLLM).
#   bash scripts/compare.sh
set -uo pipefail
cd "$(dirname "$0")/.."
LAB=${LAB:-../inference-lab}
BENCH="$LAB/.venv/bin/vllm bench serve"
MODEL=${MODEL_DIR:-$HOME/models/Qwen3-0.6B}
O=results/compare
TL=logs/timeline.txt
mkdir -p "$O" logs
LEVELS="1:16 4:32 16:64 64:128"      # concurrency:prompts
mark() { echo "$(date -u +%FT%TZ) $*" | tee -a "$TL"; }

sweep() {  # $1 label, $2 base url
  local i=0
  for pair in $LEVELS; do
    c=${pair%%:*}; n=${pair##*:}; i=$((i + 1))
    $BENCH --backend vllm --base-url "$2" --model qwen3-0.6b --tokenizer "$MODEL" \
      --dataset-name random --random-input-len 256 --random-output-len 256 --ignore-eos \
      --num-prompts "$n" --max-concurrency "$c" --temperature 0 --seed $((500 + 10 * i)) \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
      --save-result --result-dir "$O/$1" --result-filename "c$c.json" > "logs/bench-$1-c$c.log" 2>&1
    grep -E 'Failed requests|Output token throughput|Median TPOT|Median TTFT' "logs/bench-$1-c$c.log" \
      | sed "s/^/  $1 c=$c /" | tee -a "$TL"
  done
}

ours() {  # $1 label, $2 extra server flags
  mark "$1 start"
  .venv/bin/python server.py --port 8001 $2 > "logs/server-$1.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 180); do curl -sf http://127.0.0.1:8001/health >/dev/null && break; sleep 1; done
  mark "$1 ready"
  $BENCH --backend vllm --base-url http://127.0.0.1:8001 --model qwen3-0.6b --tokenizer "$MODEL" \
    --dataset-name random --random-input-len 256 --random-output-len 256 --ignore-eos \
    --num-prompts 8 --max-concurrency 4 --temperature 0 --seed 400 > "logs/warmup-$1.log" 2>&1
  sweep "$1" http://127.0.0.1:8001
  kill "$pid"; wait "$pid" 2>/dev/null
  mark "$1 stopped"
}

vllm() {  # $1 label. vLLM 0.30.0 Docker, its V2 default runner with pinned memory on (the
          # WSL2 setting that matters, see the Inference Lab), prefix caching off like ours.
  mark "$1 start"
  (cd "$LAB" && MODEL_DIR=$MODEL SERVED_NAME=qwen3-0.6b DOCKER_ENV="VLLM_WSL2_ENABLE_PIN_MEMORY=1" \
    bash scripts/docker-vllm.sh start "engine-cmp-$1" bind --no-enable-prefix-caching) | tee -a "$TL"
  $BENCH --backend vllm --base-url http://127.0.0.1:8000 --model qwen3-0.6b --tokenizer "$MODEL" \
    --dataset-name random --random-input-len 256 --random-output-len 256 --ignore-eos \
    --num-prompts 8 --max-concurrency 4 --temperature 0 --seed 400 > "logs/warmup-$1.log" 2>&1
  sweep "$1" http://127.0.0.1:8000
  (cd "$LAB" && bash scripts/docker-vllm.sh stop "engine-cmp-$1") | tee -a "$TL"
}

mark "=== compare start"
ours graphs-1 --cuda-graphs
vllm vllm-2
ours graphs-3 --cuda-graphs
vllm vllm-4
ours eager-5 ""
mark "=== compare done"
