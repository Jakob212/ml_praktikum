from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import time

# Datei für Zwischenergebnisse öffnen
log_file = "results_log.txt"

with open(log_file, "w") as log:
    log.write("Hyperparameter-Tuning Ergebnisse\n")
    log.write("===============================\n")

# Daten laden
datasets = [
    'dat/covertype.csv',
    'dataset/clf_cat/compas-two-years.csv',
    'dataset/clf_num/covertype.csv'
]

target_columns = {
    'dat/covertype.csv': 'class',
    'dataset/clf_cat/compas-two-years.csv': 'twoyearrecid',
    'dataset/clf_num/covertype.csv': 'Y'
}

# Gesamtzeit messen
total_start_time = time.time()

for dataset in datasets:
    target_column = target_columns[dataset]
    print(f"Verarbeite Datensatz: {dataset}")

    # Ergebnisse ins Log schreiben
    with open(log_file, "a") as log:
        log.write(f"\nVerarbeite Datensatz: {dataset}\n")

    data = pd.read_csv(dataset)
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Klassen anzeigen
    print(f"Klassen: {y.unique()}")
    print(f"Verteilung: \n{y.value_counts()}")

    with open(log_file, "a") as log:
        log.write(f"Klassen: {y.unique()}\n")
        log.write(f"Verteilung:\n{y.value_counts().to_string()}\n")

    # Hyperparameter-Gitter
    param_grid = {
        'n_estimators': [50, 100, 150],
        'min_samples_leaf': [10, 30],
        'max_features': ['sqrt', 'log2']
    }

    accuracies = []
    best_params_list = []

    # Zeitmessung pro Datensatz
    dataset_start_time = time.time()

    # 10 Durchläufe
    for i in range(1):
        # Zufälliges Shufflen der Daten in jedem Durchlauf
        X_shuffled, y_shuffled = X.sample(frac=1, random_state=i), y.sample(frac=1, random_state=i)
        
        # GridSearchCV initialisieren
        grid = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            cv=10,               # 10-fache Kreuzvalidierung
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Zeitmessung pro Durchlauf
        run_start_time = time.time()
        
        # Training und Suche
        grid.fit(X_shuffled, y_shuffled)
        
        # Durchlaufzeit berechnen
        run_end_time = time.time()
        run_duration = run_end_time - run_start_time

        # Ergebnisse speichern
        accuracies.append(grid.best_score_)
        best_params_list.append(grid.best_params_)

        # Zwischenergebnisse ins Log schreiben
        with open(log_file, "a") as log:
            log.write(f"\nDurchlauf {i + 1}:\n")
            log.write(f"Beste Genauigkeit: {grid.best_score_:.4f}\n")
            log.write(f"Beste Hyperparameter: {grid.best_params_}\n")
            log.write(f"Dauer des Durchlaufs: {run_duration:.2f} Sekunden\n")

    # Durchschnittliche Ergebnisse
    mean_accuracy = np.mean(accuracies)
    dataset_end_time = time.time()
    dataset_duration = dataset_end_time - dataset_start_time

    print(f"Durchschnittliche Genauigkeit: {mean_accuracy:.4f}")
    print(f"Beste Hyperparameter pro Durchlauf: {best_params_list}")

    with open(log_file, "a") as log:
        log.write(f"\nDurchschnittliche Genauigkeit: {mean_accuracy:.4f}\n")
        log.write(f"Beste Hyperparameter pro Durchlauf: {best_params_list}\n")
        log.write(f"Dauer für Datensatz: {dataset_duration:.2f} Sekunden\n")

# Gesamtzeit berechnen
total_end_time = time.time()
total_duration = total_end_time - total_start_time

with open(log_file, "a") as log:
    log.write(f"\nGesamtdauer des Programms: {total_duration:.2f} Sekunden\n")