#!/bin/bash
# Train interaction diffusion on the BEHAVE datase

# The interpreter used to execute the script

#"#SBATCH" directives that convey submission options:

#SBATCH --job-name=clip4str-train
#SBATCH --mail-user=chrenx@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --time=00-00:05:00
#SBATCH --ntasks-per-node=1
#SBATCH --mem=45GB
#SBATCH --account=eecs545w24_class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/chrenx/eecs545-sn-jersey/clip4str/clip4str-%x-%j.log

eval "$(conda shell.bash hook)"
conda init bash
conda activate clip4str
cd /home/chrenx/eecs545-sn-jersey/clip4str
# cd /home/chrenx/eecs545-sn-jersey/yolo-cls

bash scripts/vl4str_base.sh