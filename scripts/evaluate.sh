#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

CONTAINER="sacnli.sif"
INSTANCE="sacnli"
HF_TOKEN=""
MODELS=(meta-llama/Llama-3.2-3B-Instruct Qwen/Qwen2.5-3B-Instruct deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B microsoft/Phi-4-mini-reasoning)
PROMPTS=(QS CoT REACT scrit)
USE_TYPE=(Base LoRa)

start_container() { apptainer instance start --nv "$CONTAINER" "$INSTANCE"; sleep 5; apptainer exec instance://$INSTANCE bash -c "huggingface-cli login --token '$HF_TOKEN'"; }
stop_container() { apptainer instance stop "$INSTANCE"; }

evaluate_nli4ct() {
  for type in "${USE_TYPE[@]}"; do
    for model in "${MODELS[@]}"; do
      mshort=$(basename "$model")
      for prompt in "${PROMPTS[@]}"; do
        dir="./results_re/$type/$mshort"
        file="$dir/${prompt}_predictions.jsonl"
        [ -f "$file" ] || continue
        mkdir -p "$dir"
        echo "[$(date)] Eval NLI4CT $type | $mshort | $prompt"
        apptainer exec instance://$INSTANCE python3 src/evaluate_nli4ct_valid.py \
          --prompt_template "$prompt" --output_dir "$dir" --results_file "$file" --model_type "$type"
      done
    done
  done
}

evaluate_mednli() {
  for type in "${USE_TYPE[@]}"; do
    for model in "${MODELS[@]}"; do
      mshort=$(basename "$model")
      for prompt in "${PROMPTS[@]}"; do
        dir="./results_re/$type/$mshort"
        file="$dir/${prompt}_MedNLI_predictions.jsonl"
        [ -f "$file" ] || continue
        echo "[$(date)] Eval MedNLI $type | $mshort | $prompt"
        apptainer exec instance://$INSTANCE python3 src/evaluate_mednli.py \
          --prompt_template "$prompt" --output_dir "$dir" --results_file "$file"
      done
    done
  done
}

evaluate_trec() {
  for type in "${USE_TYPE[@]}"; do
    for model in "${MODELS[@]}"; do
      mshort=$(basename "$model")
      for prompt in "${PROMPTS[@]}"; do
        dir="./results_re/$type/$mshort"
        file="$dir/${prompt}_TREC_predictions.jsonl"
        [ -f "$file" ] || continue
        echo "[$(date)] Eval TREC $type | $mshort | $prompt"
        apptainer exec instance://$INSTANCE python3 src/evaluate_trec.py \
          --prompt_template "$prompt" --output_dir "$dir" --results_file "$file"
      done
    done
  done
}

# Main
start_container
evaluate_nli4ct
evaluate_mednli
evaluate_trec
stop_container