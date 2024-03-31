#!/bin/bash
# Train interaction diffusion on the BEHAVE datase

# The interpreter used to execute the script

#"#SBATCH" directives that convey submission options:

#SBATCH --job-name=cls-train
#SBATCH --mail-user=chrenx@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --time=00-08:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10GB
#SBATCH --account=eecs545w24_class
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/chrenx/eecs545-sn-jersey/cls-train-%x-%j.log

eval "$(conda shell.bash hook)"
conda init bash
conda activate soccernet
# cd /home/chrenx/eecs545-sn-jersey/yolo-bb
cd /home/chrenx/eecs545-sn-jersey/yolo-cls

# python -m train_yolo_obb
python -m train_yolo_cls