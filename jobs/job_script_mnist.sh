#!/bin/bash
# Train interaction diffusion on the BEHAVE datase

# The interpreter used to execute the script

#"#SBATCH" directives that convey submission options:

#SBATCH --job-name=double_digit
#SBATCH --mail-user=chrenx@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --time=00-00:08:00
#SBATCH --ntasks-per-node=1
#SBATCH --mem=5GB
#SBATCH --account=eecs545w24_class
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/chrenx/eecs545-sn-jersey/mnist/ckpt/training_1.log


eval "$(conda shell.bash hook)"
conda init bash
conda activate soccernet
cd /home/chrenx/eecs545-sn-jersey/mnist

python -m main # --save_last_ckpt