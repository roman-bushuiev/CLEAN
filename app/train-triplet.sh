#! /bin/bash
#SBATCH --job-name=train_CLEAN
#SBATCH --account=project_465001738
#SBATCH --partition=standard-g
#SBATCH --output=train_CLEAN.out
#SBATCH --error=train_CLEAN.err
#SBATCH --time=24:00:00
#SBATCH --gpus 8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1

eval "$(conda shell.bash hook)"
conda activate clean

cd /scratch/project_465001738/rbushuie/CLEAN/app

n_epochs=2500

python train-triplet.py \
    --model_name CARE_proteins_EC3split_train_triplet \
    --training_data CARE_proteins_EC3split_train_for_CLEAN \
    --epoch $n_epochs \
