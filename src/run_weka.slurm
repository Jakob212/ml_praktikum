#!/bin/bash
#SBATCH --job-name=WekaRandomForest
#SBATCH --output=WekaRandomForest_%j.out
#SBATCH --error=WekaRandomForest_%j.err
#SBATCH --time=24:00:00                # Maximale Laufzeit: 24 Stunden
#SBATCH --partition=parallel           # Partition für parallele CPU-Jobs
#SBATCH --nodes=1                      # 1 Knoten verwenden
#SBATCH --ntasks=1                     # 1 Task pro Knoten
#SBATCH --cpus-per-task=4              # 4 CPUs pro Task
#SBATCH --mem=16GB                     # Speicherbedarf

# Modul laden (falls erforderlich, abhängig von der Cluster-Konfiguration)
module load lang/Java/11.0

cd /lustre/project/ki-mlpractical/group1/ml_praktikum_jagoetz_wkathari/src

git pull origin main

# Pfade zur JAR-Datei und Java-Datei
WEKA_JAR="weka.jar"
JAVA_FILE="WekaRandomForestHyper.java"
CLASS_NAME="WekaRandomForestHyper"

# Kompilieren der Java-Datei
echo "Kompilieren der Datei RandomForestHyper..."
javac -cp .:weka.jar WekaRandomForestHyper.java
if [ $? -ne 0 ]; then
    echo "Kompilierung fehlgeschlagen!"
    exit 1
fi

# Ausführen der kompilierten Java-Klasse
echo "Ausführen der Datei RandomForestHyper..."
java -cp .:weka.jar WekaRandomForestHyper
if [ $? -ne 0 ]; then
    echo "Ausführung fehlgeschlagen!"
    exit 1
fi

echo "Job abgeschlossen."
