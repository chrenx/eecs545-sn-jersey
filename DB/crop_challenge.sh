#!/bin/bash
#SBATCH --job-name=single_job_test    # Job name
#SBATCH --mail-type=END       # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=yuningc@umich.edu     # Where to send mail	
#SBATCH --nodes=1                    # Run on a single CPU
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=1g
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00               # Time limit hrs:min:sec
#SBATCH --output=crop_challenge.log
#SBATCH --account=eecs545w24_class
#SBATCH --get-user-env



pwd; hostname; date

nvidia-smi

module load python3.10-anaconda/2023.03

module load cuda/11.8.0

source ~/.bashrc

conda activate DBC

# python /home/yuningc/DB/datasets/jersey/convert.py
# [[38.32999999999999, 33.605], [58.60999999999999, 33.605], [58.60999999999999, 57.775], [38.32999999999999, 57.775]]
# CUDA_VISIBLE_DEVICES=0 python demo.py experiments/seg_detector/bbjr_resnet50.yaml --image_path /home/yuningc/jersey-2023/train/images/11/11_10.jpg --resume /home/yuningc/DB/outputs/workspace/L1BalanceCELoss/model/final --box_thresh 0.45 --visualize --image_short_side 128

# CUDA_VISIBLE_DEVICES=0 python util.py experiments/ASF/td500_resnet50_deform_thre_asf.yaml --result_dir /home/yuningc/jersey-2023/challenge_filter/images --image_path /home/yuningc/jersey-2023/challenge/images --resume /home/yuningc/DB/models/finetune_ctw1500_res50_dcn_fpn_dbv2 --box_thresh 0.15 --visualize --image_short_side 128 
# CUDA_VISIBLE_DEVICES=0 python util.py /home/yuningc/DB/experiments/seg_detector/bbjr_resnet50.yaml --result_dir /home/yuningc/jersey-2023/challenge_filter/images --image_path /home/yuningc/jersey-2023/challenge/images --resume /home/yuningc/DB/outputs/workspace/DB/SegDetectorModel-seg_detector/deformable_resnet50/final_v3 --box_thresh 0.5 --visualize --image_short_side 128 


CUDA_VISIBLE_DEVICES=0 python util.py /home/yuningc/DB/experiments/seg_detector/bbjr_asf_resnet50.yaml --result_dir /home/yuningc/jersey-2023/challenge_crop/images --image_path /home/yuningc/jersey-2023/challenge/images --resume /home/yuningc/DB/outputs/workspace/DB/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss/model/final --box_thresh 0.7 --visualize --image_short_side 128 



# CUDA_VISIBLE_DEVICES=0 python train.py experiments/seg_detector/CONVERT_ONLY.yaml --num_gpus 1 --validate --localization #--debug #jersey_test.yaml --num_gpus 1 --validate --debug
# python train.py /home/yuningc/DB/experiments/seg_detector/jersey_extra_dataset.yaml --num_gpus 1 --validate#

# "model.module.decoder.concat_attention.conv.weight", "model.module.decoder.concat_attention.conv.bias", "model.module.decoder.concat_attention.enhanced_attention.spatial_wise.0.weight", "model.module.decoder.concat_attention.enhanced_attention.spatial_wise.2.weight", "model.module.decoder.concat_attention.enhanced_attention.attention_wise.0.weight". 

# "model.module.decoder.concat_attention.enhanced_attention.channel_wise.1.weight", "model.module.decoder.concat_attention.enhanced_attention.channel_wise.3.weight". 
# rm -r /home/yuningc/jersey-2023/test/images