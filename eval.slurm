#!/bin/bash
##SBATCH --gres=gpu:1
#SBATCH -c 4
##SBATCH -C gmem12
##SBATCH --mem=32G 
#SBATCH --time=10-00:00:00
#SBATCH --output="./slurm_logs/%x-%j.out"
#SBATCH --job-name=test

module load cudnn/10-74
module load cuda/11.3


# source eval.sh
# python top100.py cvact 90
# python top100.py cvact 180
# python top100.py cvact 270

python top100.py vigor 90
python top100.py vigor 180
python top100.py vigor 270

# python top100.py cvusa 90
# python top100.py cvusa 180
# python top100.py cvusa 270
