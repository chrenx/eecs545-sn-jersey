#!/bin/bash
#SBATCH --job-name=single_job_test    # Job name
#SBATCH --mail-type=END          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=yuningc@umich.edu     # Where to send mail	
#SBATCH --nodes=1                    # Run on a single CPU
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=1gb                     # Job memory request
#SBATCH --time=00:05:00               # Time limit hrs:min:sec
#SBATCH --output=single_test_%j.log   # Standard output and error log
#SBATCH --get-user-env
pwd; hostname; date

module load python3.10-anaconda/2023.03

module load cuda/11.8.0

source ~/.bashrc

echo "Running plot script on a single CPU core"

nvidia-smi

conda activate DBC

python setup.py build_ext --inplace
echo $CUDA_HOME

date




