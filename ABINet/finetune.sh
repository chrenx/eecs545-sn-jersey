#!/bin/bash
#SBATCH --job-name=single_job_test    # Job name
#SBATCH --mail-type=END       # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=yuningc@umich.edu     # Where to send mail	
#SBATCH --nodes=1                    # Run on a single CPU
#SBATCH --cpus-per-task=14
#SBATCH --mem-per-cpu=300m
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00               # Time limit hrs:min:sec
#SBATCH --output=finetune.log
#SBATCH --account=eecs545w24_class
#SBATCH --get-user-env



pwd; hostname; date

module load python3.10-anaconda/2023.03

module load cuda/11.8.0

source ~/.bashrc

conda activate DBC

CUDA_VISIBLE_DEVICES=0 python main.py --config=configs/train_abinet.yaml --checkpoint /home/yuningc/ABINet/workdir/best-train-abinet.pth