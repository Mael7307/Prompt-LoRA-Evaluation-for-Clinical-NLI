# 🧠 ClinicalReasonBench: Prompt & LoRA Evaluation for Clinical NLI

This repository implements the experimental framework from our AAAI paper:

> **Dissecting Clinical Reasoning in Language Models: A Comparative Study of Prompts and Model Adaptation Strategies**  
> [[[paper link]](https://arxiv.org/abs/2507.04142)] | BibTeX: *(see below)*

We provide a unified pipeline for evaluating structured prompting strategies and parameter-efficient fine-tuning (LoRA) across multiple compact language models on clinical Natural Language Inference (NLI) tasks, including NLI4CT, MedNLI, and TREC.

---

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Inference](#inference)
  - [Training (LoRA)](#training-lora)
  - [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)

---

## ✨ Features

- ✅ Modular inference, training, and evaluation bash pipelines
- ✅ Support for 4 reasoning-focused prompting strategies: NLR (CoT), ISRR, TAR (ReAct), SSR (QuaSAR)
- ✅ LoRA-based fine-tuning for efficient adaptation of 1.5–3.8B models
- ✅ Support for MedNLI and TREC for generalization evaluation
- ✅ Fully automated Apptainer-based container setup

---

## 🛠️ Installation

### 1. Clone this repo
```bash
git clone https://github.com/Mael7307/Prompt-LoRA-Evaluation-for-Clinical-NLI.git
cd Prompt-LoRA-Evaluation-for-Clinical-NLI
````

### 2. Build the Apptainer container

```bash
sbatch build_apptainer.sh
```

This will produce `SACNLI.sif`, used by all scripts.

---
🔑 Dataset Access
📌 MedNLI (not redistributed)
The MedNLI dataset cannot be distributed with this repository.
To access it, please request it directly from the authors via PhysioNet:

→ https://physionet.org/content/mednli/1.0.0/

Once downloaded, place the MedNLI dataset in a local directory (e.g., ./MedNLI/) as expected by the scripts.

## 🚀 Usage

### 🔍 Inference

Run predictions across:

* All models (`llama`, `qwen`, `phi`, `deepseek`)
* All prompt types (QS, CoT, REACT, scrit)
* All datasets (NLI4CT, MedNLI, TREC)
* With and without LoRA

```bash
bash inference.sh
```

---

### 🏋️ Training (LoRA)

Fine-tunes models on NLI4CT subsets using LoRA adapters.

```bash
sbatch train.sh
```

LoRA weights are saved in `lora_outputs/`.

---

### 📊 Evaluation

Evaluates predictions across all datasets and saves metrics.

```bash
bash evaluate.sh
```

Results go to `results/`.

---

## 📁 Project Structure

```
.
├── prompts/                 # Prompt templates
├── src/                     # Core Python scripts
│   ├── mednli_inference.py
│   ├── nli4ct_inference.py
│   ├── evaluate_mednli.py
│   ├── evaluate_trec.py
│   └── train.py
├── results/                 # Results
├── lora_outputs/            # Trained LoRA adapters
├── inference.sh             # Inference pipeline
├── train.sh                 # Training pipeline (SLURM)
├── evaluate.sh              # Evaluation script
└── build_apptainer.sh       # Apptainer container build
```

---

## 🔁 Reproducibility

To reproduce experiments:

```bash
sbatch build_apptainer.sh     # 1. Build container
sbatch train.sh               # 2. Train LoRA adapters
bash inference.sh             # 3. Run inference
bash evaluate.sh              # 4. Evaluate outputs
```

---

## 📄 Citation

If you use this code or dataset, please cite our AAAI paper:

> **Dissecting Clinical Reasoning in Language Models: A Comparative Study of Prompts and Model Adaptation Strategies**

```bibtex
@article{jullien2025dissecting,
  title={Dissecting Clinical Reasoning in Language Models: A Comparative Study of Prompts and Model Adaptation Strategies},
  author={Jullien, Mael and Valentino, Marco and Ranaldi, Leonardo and Freitas, Andre},
  journal={arXiv preprint arXiv:2507.04142},
  year={2025}
}
```
