#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# --- Configuration ---
CONTAINER="sacnli.sif"
INSTANCE="sacnli"
HF_TOKEN=""
MODELS=(
  "meta-llama/Llama-3.2-3B-Instruct"
  "Qwen/Qwen2.5-3B-Instruct"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  "microsoft/Phi-4-mini-reasoning"
)
PROMPTS=(QS CoT REACT scrit)
USE_LORA=(false true)

# Map dataset keys to args: dataset path, prompt subdir, batch size
declare -A DATASETS=(
  [NLI4CT]="--dataset Mael7307/NLI4CT --prompt-dir prompts/NLI4CT --batch 4 --script src/nli4ct_inference.py"
  [MedNLI]="--dataset ./MedNLI --prompt-dir prompts/MedNLI --batch 16 --script src/mednli_inference.py"
  [TREC]  "--dataset ./TREC    --prompt-dir prompts/TREC    --batch 8  --script src/trec_inference.py"
)

# --- Functions ---
start_container() {
  apptainer instance start --nv "$CONTAINER" "$INSTANCE"
  sleep 5
  apptainer exec instance://$INSTANCE bash -c "huggingface-cli login --token '$HF_TOKEN'"
}

stop_container() {
  apptainer instance stop "$INSTANCE"
}

run_model() {
  local ds_key=$1 opts model prompt lora_flag out_dir cmd
  opts="${DATASETS[$ds_key]}"
  for model in "${MODELS[@]}"; do
    mshort=$(basename "$model")
    for prompt in "${PROMPTS[@]}"; do
      for lora_flag in "${USE_LORA[@]}"; do
        out_dir="./results/$( $lora_flag && echo LoRa || echo Base )/$mshort"
        mkdir -p "$out_dir"
        echo "[$(date)] $ds_key | $mshort | prompt=$prompt | LoRA=$lora_flag"
        cmd=( python3 ${opts//--dataset/--dataset} )
        # Build command array
        cmd=( python3 $(echo "$opts" | awk '{print $NF}') \
          $(echo "$opts" | sed 's/.*\( --dataset [^ ]*\).*/\1/') \
          --output_dir "$out_dir" \
          --model "$model" \
          --prompt_template "$(echo "$opts" | awk '{print $3}')/$prompt.txt" \
          --prompt "$prompt" )
        $lora_flag && cmd+=( --lora_path "./lora_outputs/${mshort}_${prompt}/lora_adapters" )
        cmd+=( --device cuda --batch_size $(echo "$opts" | awk '{print $5}') )
        apptainer exec instance://$INSTANCE "${cmd[@]}"
      done
    done
  done
}

# --- Main ---
start_container
for ds in "${!DATASETS[@]}"; do
  run_model "$ds"
done
stop_container

