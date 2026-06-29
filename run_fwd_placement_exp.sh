#!/bin/bash
# FT Forward Placement Experiment — A/B/C/D on 2×A100-80GB
# Conditions:
#   A  baseline  No FT forward
#   B  prefill   FT fwd synchronous on main stream during prefill
#   C  decode    FT fwd on low-priority stream during decode
#   D  bubble    FT fwd scheduled into TP all_reduce rank-imbalance bubbles

set -euo pipefail

VENV="/mnt/nfs/home/ramya/vllm/.venv"
export PATH="$VENV/bin:$PATH"

MODEL="/mnt/nfs/home/ramya/models/Qwen/Qwen3-30B-A3B"
PORT=8000
RESULTS_DIR="/mnt/nfs/home/ramya/vllm/fwd_placement_results"
SCRIPT_DIR="/mnt/nfs/home/ramya/vllm"
T_IN=8192
REPS=20

mkdir -p "$RESULTS_DIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RESULTS_DIR/experiment.log"; }

wait_for_server() {
    local timeout=${1:-240}
    log "  waiting for server on port $PORT..."
    for i in $(seq 1 $timeout); do
        if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
            log "  server ready (${i}s)"
            return 0
        fi
        sleep 1
    done
    log "  ERROR: server did not start within ${timeout}s"
    return 1
}

kill_server() {
    if [ -n "${SERVER_PID:-}" ]; then
        log "  stopping server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
        sleep 8  # allow CUDA context cleanup
    fi
}

trap 'kill_server' EXIT

SERVER_PID=""

log "=== FT Forward Placement Experiment ==="
log "Model:   $MODEL"
log "T_in:    $T_IN tokens"
log "Reps:    $REPS"
log "Results: $RESULTS_DIR"
echo ""

# ─── Condition A: baseline (no FT forward) ────────────────────────────────────
log "=== CONDITION A: baseline (no FT forward) ==="
VLLM_FT_FWD_MODE=off \
"$VENV/bin/vllm" serve "$MODEL" \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --enable-ep-weight-filter \
    --moe-backend triton \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 32768 \
    --trust-remote-code \
    --host 0.0.0.0 --port "$PORT" \
    --enforce-eager \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    > "$RESULTS_DIR/A_server.log" 2>&1 &
SERVER_PID=$!
log "  server PID $SERVER_PID"
wait_for_server

log "  measuring condition A..."
"$VENV/bin/python" "$SCRIPT_DIR/ft_fwd_placement_exp.py" \
    --condition A --port "$PORT" \
    --t-in "$T_IN" --reps "$REPS" \
    --measure-tpot \
    --save "$RESULTS_DIR/A.json" \
    2>&1 | tee "$RESULTS_DIR/A_measure.log"
kill_server
log "  condition A done"
echo ""

# ─── Condition B: prefill (FT fwd sync on main stream) ────────────────────────
log "=== CONDITION B: prefill (FT fwd sync on main stream during prefill) ==="
VLLM_FT_FWD_MODE=prefill VLLM_FT_FWD_DEMO=1 VLLM_FT_FWD_FILLS=9 \
"$VENV/bin/vllm" serve "$MODEL" \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --enable-ep-weight-filter \
    --moe-backend triton \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 32768 \
    --trust-remote-code \
    --host 0.0.0.0 --port "$PORT" \
    --enforce-eager \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    > "$RESULTS_DIR/B_server.log" 2>&1 &
SERVER_PID=$!
log "  server PID $SERVER_PID"
wait_for_server

log "  measuring condition B..."
"$VENV/bin/python" "$SCRIPT_DIR/ft_fwd_placement_exp.py" \
    --condition B --port "$PORT" \
    --t-in "$T_IN" --reps "$REPS" \
    --measure-tpot \
    --baseline-json "$RESULTS_DIR/A.json" \
    --save "$RESULTS_DIR/B.json" \
    2>&1 | tee "$RESULTS_DIR/B_measure.log"
kill_server
log "  condition B done"
echo ""

# ─── Condition C: decode (FT fwd on low-priority stream during decode) ─────────
log "=== CONDITION C: decode (FT fwd async low-pri stream during decode) ==="
VLLM_FT_FWD_MODE=decode VLLM_FT_FWD_DEMO=1 VLLM_FT_FWD_FILLS=9 \
"$VENV/bin/vllm" serve "$MODEL" \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --enable-ep-weight-filter \
    --moe-backend triton \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 32768 \
    --trust-remote-code \
    --host 0.0.0.0 --port "$PORT" \
    --enforce-eager \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    > "$RESULTS_DIR/C_server.log" 2>&1 &
SERVER_PID=$!
log "  server PID $SERVER_PID"
wait_for_server

log "  measuring condition C..."
"$VENV/bin/python" "$SCRIPT_DIR/ft_fwd_placement_exp.py" \
    --condition C --port "$PORT" \
    --t-in "$T_IN" --reps "$REPS" \
    --measure-tpot \
    --baseline-json "$RESULTS_DIR/A.json" \
    --save "$RESULTS_DIR/C.json" \
    2>&1 | tee "$RESULTS_DIR/C_measure.log"
kill_server
log "  condition C done"
echo ""

# ─── Condition D: bubble (FT fwd in TP all_reduce rank-imbalance gaps) ─────────
log "=== CONDITION D: bubble (FT fwd in TP all_reduce bubbles) ==="
VLLM_FT_FWD_MODE=bubble VLLM_FT_FWD_DEMO=1 VLLM_FT_FWD_FILLS=1 \
"$VENV/bin/vllm" serve "$MODEL" \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --enable-ep-weight-filter \
    --moe-backend triton \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 32768 \
    --trust-remote-code \
    --host 0.0.0.0 --port "$PORT" \
    --enforce-eager \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    > "$RESULTS_DIR/D_server.log" 2>&1 &
SERVER_PID=$!
log "  server PID $SERVER_PID"
wait_for_server

log "  measuring condition D..."
"$VENV/bin/python" "$SCRIPT_DIR/ft_fwd_placement_exp.py" \
    --condition D --port "$PORT" \
    --t-in "$T_IN" --reps "$REPS" \
    --measure-tpot \
    --baseline-json "$RESULTS_DIR/A.json" \
    --save "$RESULTS_DIR/D.json" \
    2>&1 | tee "$RESULTS_DIR/D_measure.log"
kill_server
log "  condition D done"
echo ""

log "=== ALL CONDITIONS COMPLETE ==="
log "Results in $RESULTS_DIR:"
for cond in A B C D; do
    if [ -f "$RESULTS_DIR/${cond}.json" ]; then
        log "  ${cond}.json — $(wc -c < "$RESULTS_DIR/${cond}.json") bytes"
    fi
done
