#!/bin/bash
#SBATCH --job-name=ml_repo_job       # Name des Jobs
#SBATCH --output=ml_repo_output_%j.log  # Ausgabe-Log mit Job-ID
#SBATCH --reservation=mlcourse         # Reservation für deinen Kurs
#SBATCH --time=02:00:00                # Maximale Laufzeit: 2 Stunden
#SBATCH --partition=parallel           # Partition für parallele CPU-Jobs
#SBATCH --nodes=1                      # 1 Knoten verwenden
#SBATCH --ntasks=1                     # 1 Task pro Knoten
#SBATCH --cpus-per-task=4              # 4 CPUs pro Task
#SBATCH --mem=16GB                     # Speicherbedarf

# Modul laden (angepasst auf die erforderliche Python-Version)
module load lang/Python/3.11.3-GCCcore-12.3.0

# Wechsle in das Verzeichnis des Repositories
cd /lustre/project/ki-mlpractical/group1/ml_praktikum_jagoetz_wkathari

# Stelle sicher, dass die neueste Version des Repositories geladen wird
git pull origin main

# Führe das Python-Skript aus
python RandomForestHyper.py