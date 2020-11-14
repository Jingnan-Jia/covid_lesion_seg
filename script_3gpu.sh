#!/bin/bash
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH -t 7-00:00:00
#SBATCH --mem-per-gpu=90G
#SBATCH --mail-type=end
#SBATCH --mail-user=jiajingnan2222@gmail.com

eval $(conda shell.bash hook)
conda activate py37
time=$(date +%y_%m_%d_%H_%M_%S)  # there must not be any space before and after =
export PYTHONPATH="${PYTHONPATH}:/data/jjia/jjnutils/jjnutils"

idx=0; export CUDA_VISIBLE_DEVICES=$idx; stdbuf -oL python -u train.py 2>logs/slurm-${time}_$idx.err 1>logs/slurm-${time}_$idx.out --model_names='net_lesion-net_recon-net_lobe'
#idx=1; export CUDA_VISIBLE_DEVICES=$idx; stdbuf -oL python -u train.py 2>logs/slurm-${time}_$idx.err 1>logs/slurm-${time}_$idx.out --model_names='net_lesion-net_recon' &
#idx=2; export CUDA_VISIBLE_DEVICES=$idx; stdbuf -oL python -u train.py 2>logs/slurm-${time}_$idx.err 1>logs/slurm-${time}_$idx.out --model_names='net_lesion-net_recon-net_lobe' &
#idx=3; export CUDA_VISIBLE_DEVICES=$idx; stdbuf -oL python -u train.py 2>logs/slurm-${time}_$idx.err 1>logs/slurm-${time}_$idx.out --model_names='net_lesion-net_lobe' &

