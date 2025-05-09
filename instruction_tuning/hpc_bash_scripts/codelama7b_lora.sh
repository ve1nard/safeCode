#!/bin/bash

# SLURM Resource requests
#SBATCH -p nvidia               # Partition to submit to (for GPU)
#SBATCH --gres=gpu:a100:1       #request 1 gpus
#SBATCH --mem=64G              # Memory request
#SBATCH -t 1-00:00:00           # Time limit: 1 day
#SBATCH -o codellama7b_lora_23.out      # Output file
#SBATCH -e codellama7b_lora_23.err      # Error file

cd /scratch/dn2206/Capstone2/SafeCoder/scripts

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

#Training
#python train.py --pretrain_name codellama-7b --output_name codellama-7b-safecoder --datasets evol sec-desc sec-new-desc

# Security Evaluation
#python sec_eval.py --output_name codellama7b-safecoder --model_name codellama7b-safecoder --eval_type trained
#python sec_eval.py --output_name codellama7b-safecoder --model_name codellama7b-safecoder --eval_type trained-new

PYTHONNOUSERSITE=1 python sec_eval.py --output_name codellama-7b-lora-safecoder --model_name codellama-7b-lora-safecoder --eval_type trained --model_dir /scratch/dn2206/Capstone2/SafeCoder
PYTHONNOUSERSITE=1 python sec_eval.py --output_name codellama-7b-lora-safecoder --model_name codellama-7b-lora-safecoder --eval_type trained-new --model_dir /scratch/dn2206/Capstone2/SafeCoder


# Print Security Evaluation Results
#python print_results.py --eval_name codellama7b-safecoder --eval_type trained-joint --detail

PYTHONNOUSERSITE=1 python print_results.py --eval_name codellama-7b-lora-safecoder --eval_type trained-joint --detail
