#!/bin/bash
# Train interaction diffusion on the BEHAVE datase

# The interpreter used to execute the script

#"#SBATCH" directives that convey submission options:

#SBATCH --job-name=obb-train
#SBATCH --mail-user=chrenx@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --time=00-08:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --mem=45GB
#SBATCH --account=eecs545w24_class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/chrenx/eecs545-sn-jersey/wrn/svhn-%x-%j.log

eval "$(conda shell.bash hook)"
conda init bash
conda activate soccernet
cd /home/chrenx/eecs545-sn-jersey/wrn
# cd /home/chrenx/eecs545-sn-jersey/yolo-cls

python -m yolo_obb_trainer
# python -m train_yolo_cls