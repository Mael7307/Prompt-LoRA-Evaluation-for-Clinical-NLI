#!/usr/bin/env bash
#SBATCH --job-name=all-lora-train
#SBATCH --output=logs/lora_train_%j.out
#SBATCH --error=logs/lora_train_%j.err
#SBATCH --partition gpu --gres gpu:h100:1 --mem 80G --cpus-per-task 4 --time 12:00:00 --account rational
set -euo pipefail

CONTAINER="sacnli.sif"
INSTANCE="sacnli"
HF_TOKEN=""
MODELS=(meta-llama/Llama-3.2-3B-Instruct Qwen/Qwen2.5-3B-Instruct deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B microsoft/Phi-4-mini-reasoning)
PROMPTS=(QS CoT REACT scrit baseline)
# hyperparams
MAX_LEN=4096; BSZ_T=1; BSZ_E=1; LR=2e-5; EPOCHS=1; WARMUP=10; LOG=10; SAVE=500; MAX_STEPS=500; SEED=42

start_container() { apptainer instance start --nv "$CONTAINER" "$INSTANCE"; sleep 5; apptainer exec instance://$INSTANCE bash -c "huggingface-cli login --token '$HF_TOKEN'"; }
stop_container() { apptainer instance stop "$INSTANCE"; }

start_container
for model in "${MODELS[@]}"; do
  mshort=$(basename "$model")
  for prompt in "${PROMPTS[@]}"; do
    out_dir="./lora_outputs/${mshort}_${prompt}"
    mkdir -p "$out_dir"
    ds="Mael7307/NLI4CT_${prompt}_demo"
    script="src/train${model##*Phi*}_phi.py"
    echo "[$(date)] Training $mshort + $prompt"
    apptainer exec instance://$INSTANCE python3 "$script" --model_name "$model" --dataset_name "$ds" \
      --output_dir "$out_dir" --max_length $MAX_LEN --train_batch_size $BSZ_T --eval_batch_size $BSZ_E \
      --learning_rate $LR --num_train_epochs $EPOCHS --warmup_steps $WARMUP \
      --logging_steps $LOG --save_steps $SAVE --max_steps $MAX_STEPS --seed $SEED
  done
done
stop_container
