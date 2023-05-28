#!/bin/sh

#SBATCH --job-name=AIDM_Convergence
#SBATCH --partition=compute
#SBATCH --account=education-eemcs-msc-cs
#SBATCH --time=4:00:00
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G

module load 2022r2
module load python
module load py-setuptools
module load py-pip

python -m pip install --user box2d-py
python -m pip install --user stable-baselines3[extra]==1.8.0
python -m pip install --user gymnasium==0.28.1
python -m pip install --user gymnasium[box2d]
python -m pip install --user gym~=0.21.0

srun python main.py AIDM_Log/logs_1/ &
srun python main.py AIDM_Log/logs_2/ &
srun python main.py AIDM_Log/logs_3/ &
srun python main.py AIDM_Log/logs_4/ &
srun python main.py AIDM_Log/logs_5/
