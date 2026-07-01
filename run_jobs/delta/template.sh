#!/bin/bash
#SBATCH --mem=${mem}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=${gpus_per_task}
#SBATCH --partition=${partition}      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8 gpuH200x8
#SBATCH --account=${account}    # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=${name}
#SBATCH --time=${time}      # hh:mm:ss for the job
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

# initialize bash
source ~/.bashrc
# Initialize conda for shell script
eval "$$(conda shell.bash hook)"

# activate conda environment
conda activate ${conda_env}
conda env list

#export your required environment variables below
#################################################
export WANDB_API_KEY=$$(cat ~/keys/wandb.key)

# cd to project directory
cd ${cwd}

# run the script
${command}
