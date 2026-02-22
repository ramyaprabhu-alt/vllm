#!/usr/bin/env bash

MODEL="openai/gpt-oss-120b"
DATASET="random"
INPUT_LEN=1024
OUTPUT_LEN=128

for NUM_PROMPTS in 8 16 32 64 128
do
  echo "Running benchmark with num_prompts=${NUM_PROMPTS}"

  vllm bench serve \
    --model "$MODEL" \
    --dataset-name "$DATASET" \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --ignore-eos \
    --num_prompts "$NUM_PROMPTS"

  echo "Completed run for num_prompts=${NUM_PROMPTS}"
  echo "------------------------------------------"
done
