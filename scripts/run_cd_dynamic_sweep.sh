#!/usr/bin/env bash
# Full QPS sweep for the new C+D_dynamic chunking-collocation policy, plus
# the static-window control that isolates "bigger window helps" from
# "adaptive window helps" (see the plan in
# /home/prabhu/.claude/plans/logical-wondering-papert.md, Verification §2-3).
#
# Runs sequentially (single 2-GPU TP=2 server, can't parallelize across
# combos on the same GPUs). Continues past individual run failures so one
# bad config doesn't kill the whole sweep; failures are recorded in the
# summary index at the end.

set -uo pipefail  # NOT -e: one failed run must not abort the sweep
cd "$(dirname "$0")/.."

QPS_POINTS=(0.4 0.6 1 2 4)
NUM_PROMPTS=200
STAMP="0713"

declare -A MODEL_PATH=(
    [qwen15moe]="/mnt/nfs/home/ramya/models/Qwen/Qwen1.5-MoE-A2.7B"
    [qwen3]="/mnt/nfs/home/ramya/models/Qwen/Qwen3-30B-A3B"
)
declare -A LORA_PATH=(
    [qwen15moe]="/mnt/nfs/home/ramya/scratch/qwen15-moe-toy-lora"
    [qwen3]="/mnt/nfs/home/ramya/slora-plus/S-LoRA/test/qwen3/adapters/qwen3-toy-lora"
)
declare -A MAX_LEN_FOR=(
    [qwen15moe]="8192"
    [qwen3]="16384"
)

SUMMARY="compare_results/cd_dynamic_sweep_${STAMP}_summary.log"
: > "$SUMMARY"

run_one() {
    local tag="$1" model="$2" qps="$3" result_dir="$4"
    shift 4
    echo "=== [$(date +%H:%M:%S)] $tag qps=$qps -> $result_dir ===" | tee -a "$SUMMARY"
    if MODEL="${MODEL_PATH[$model]}" LORA="${LORA_PATH[$model]}" \
       MAX_LEN="${MAX_LEN_FOR[$model]}" NUM_PROMPTS="$NUM_PROMPTS" QPS="$qps" \
       RESULT_DIR="$result_dir" \
       bash scripts/bench_arxiv.sh --system bubble_tea "$@" \
       >> "${result_dir}.driverlog" 2>&1; then
        local steps note
        if [ -f "${result_dir}/bubble_tea.json" ]; then
            steps=$("$PWD/.venv-bubble/bin/python" -c "
import json
d = json.load(open('${result_dir}/bubble_tea.json'))
tr = d.get('training', {})
inf = d.get('inference', {})
print(f\"steps={tr.get('training_steps')} peft_s={tr.get('peft_samples_s')} \"
      f\"tpot={inf.get('mean_tpot_ms')} ttft_p50={inf.get('p50_ttft_ms')} \"
      f\"ok={inf.get('successful_requests')}\")
" 2>&1)
        else
            steps="NO_JSON_OUTPUT"
        fi
        echo "  OK: $steps" | tee -a "$SUMMARY"
    else
        echo "  FAILED (see ${result_dir}.driverlog)" | tee -a "$SUMMARY"
    fi
}

echo "Sweep started $(date)" | tee -a "$SUMMARY"

# ── Part 1: static-window control (isolates bigger-window-helps from
# adaptive-window-helps) — C+D_batch at fixed t_ft=128 (existing baseline)
# and t_ft=512 (max, no solver), bwd-mode held at "both". ──────────────────
for model in qwen15moe qwen3; do
    for qps in "${QPS_POINTS[@]}"; do
        for tft in 128 512; do
            rd="compare_results/${model}_arxiv_${NUM_PROMPTS}_${qps}qps_cdbatch_tft${tft}_${STAMP}"
            run_one "${model} cdbatch t_ft=${tft}" "$model" "$qps" "$rd" \
                --ft-mode C+D_batch --t-ft "$tft" --bwd-mode both
        done
    done
done

# ── Part 2: C+D_dynamic full sweep, crossed with the backward-dispatch
# axis (both / slo_budget_both) to answer the EP-fairness question
# empirically (plan Verification §5). ───────────────────────────────────────
for model in qwen15moe qwen3; do
    for qps in "${QPS_POINTS[@]}"; do
        for bwd in both slo_budget_both; do
            rd="compare_results/${model}_arxiv_${NUM_PROMPTS}_${qps}qps_cddynamic_${bwd}_${STAMP}"
            run_one "${model} cddynamic bwd=${bwd}" "$model" "$qps" "$rd" \
                --ft-mode C+D_dynamic --t-ft-max 512 \
                --window-overhead-frac 0.15 --window-probe-cycle 10 \
                --bwd-mode "$bwd"
        done
    done
done

echo "Sweep finished $(date)" | tee -a "$SUMMARY"
