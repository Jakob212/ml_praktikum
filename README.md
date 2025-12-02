# **ML Practical – Evaluation of Decision Trees, Random Forests & Rule-Based Models (WEKA & Python)**

## **Project Overview**
This repository contains a complete machine learning pipeline for evaluating and comparing different classification models across more than 15 real-world tabular datasets.

The project includes:

- Reproducible machine learning experiments  
- Implementations of Decision Trees, Random Forests, PART, and JRip (RIPPER)  
- Hyperparameter optimization via Grid Search  
- 3-fold Cross-Validation + repeated train/test splits (up to 15 runs)  
- Use of Java (WEKA) and Python for flexible experiment automation  
- Executable Slurm scripts for running experiments on a supercomputer  
- Structured results (accuracy, CV accuracy, runtime, standard deviation)

The implementation focuses strongly on **reproducibility and scalability**.

---

## **Unified Pipeline for All Models**
Each dataset is processed using the same workflow:

1. Load & preprocess data  
2. Identify and (if necessary) convert the target column (numeric → nominal)  
3. Shuffle with deterministic seeds  
4. 2/3 training – 1/3 testing  
5. 3-fold Cross-Validation  
6. Multiple repetitions for stable statistics  
7. Computation of accuracy, CV accuracy, standard deviation, runtime  
8. Best hyperparameters identified per run  

---

## **Hyperparameter Optimization (Grid Search)**

### **Example: Random Forest**
| Parameter | Values |
|-----------|---------|
| `n_estimators` | 50, 100, 150 |
| `min_samples_leaf` | 10, 30 |
| `max_features` | sqrt, log2 |

### **Example: Decision Tree**
| Parameter | Values |
|-----------|---------|
| `depth` | unlimited, 10, 20 |
| `min_leaf` | 1, 5, 10 |
| `max_features` | none, sqrt, log2 |

All combinations are evaluated using **3-fold Cross-Validation**.

---

## **Evaluation & Metrics**

For each run configuration, the following metrics are recorded:

- Accuracy per run  
- Mean & standard deviation  
- Cross-validation accuracy  
- Runtime per run  
- Most frequent best hyperparameter combination  
- Best-run analysis  

All results can be saved in the `results/` directory.

---

## **HPC Cluster Support (Slurm)**

This project was partially executed on a supercomputer.  
It includes production-ready Slurm scripts:

### **Java + WEKA Example**
```bash
#!/bin/bash
#SBATCH --job-name=RFWeka
#SBATCH --output=RFWeka_%j.out
#SBATCH --time=4-00:00:00
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB

module load lang/Java/11.0

cd ml_praktikum_jagoetz_wkathari/src
git pull origin main

javac -cp .:weka.jar WekaRandomForestHyper.java
java  -cp .:weka.jar WekaRandomForestHyper
```

### Python Example
```bash
#!/bin/bash
#SBATCH --job-name=ml_repo_job
#SBATCH --output=ml_repo_output_%j.log
#SBATCH --time=96:00:00
#SBATCH --partition=parallel
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB

module load lang/Python/3.11.3-GCCcore-12.3.0
cd ml_praktikum_jagoetz_wkathari
python RandomForestHyper.py
```