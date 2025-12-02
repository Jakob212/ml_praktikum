#!/bin/bash
#SBATCH --job-name=rf_weka
#SBATCH --output=rf_weka_%j.out
#SBATCH --time=4-00:00:00
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB

module load lang/Java/11.0

cd ml_praktikum_jagoetz_wkathari

git pull origin main

WEKA_JAR="weka.jar"
CLASS_NAME="WekaRandomForestHyper"

echo "Kompiliere ${CLASS_NAME}.java..."
javac -cp .:${WEKA_JAR} ${CLASS_NAME}.java || exit 1

echo "Starte ${CLASS_NAME}..."
java -cp .:${WEKA_JAR} ${CLASS_NAME} || exit 1

echo "Job abgeschlossen."
