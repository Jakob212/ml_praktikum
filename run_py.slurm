#!/bin/bash
#SBATCH --job-name=ml_repo_job
#SBATCH --output=ml_repo_output_%j.log
#SBATCH --error=ml_repo_error_%j.log
#SBATCH --time=96:00:00
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB

module load lang/Python/3.11.3-GCCcore-12.3.0

cd ml_praktikum_jagoetz_wkathari

python RandomForestHyper.py
