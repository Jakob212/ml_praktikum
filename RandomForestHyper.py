from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from joblib import Parallel, delayed
import pandas as pd
import numpy as np

# Daten laden
datasets = [
    'dataset/clf_num/pol.csv',
    'dataset/clf_cat/compas-two-years.csv',
    'dataset/clf_cat/electricity.csv'
]

target_columns = {
    'dataset/clf_cat/electricity.csv': 'class',
    'dataset/clf_cat/compas-two-years.csv': 'twoyearrecid',
    'dataset/clf_num/pol.csv': 'binaryClass'
}

# Hyperparameter-Gitter
param_grid = {
    'n_estimators': [50, 100, 150],
    'min_samples_leaf': [10, 30],
    'max_features': ['sqrt', 'log2']
}

def process_dataset(dataset, target_column):
    print(f"Verarbeite Datensatz: {dataset}")

    data = pd.read_csv(dataset)
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Klassen anzeigen
    print(f"Klassen: {y.unique()}")
    print(f"Verteilung: \n{y.value_counts()}")

    accuracies = []
    best_params_list = []

    for i in range(15):  # Schleife auf 15 erweitert
        # Zufälliges Shufflen der Daten in jedem Durchlauf
        X_shuffled, y_shuffled = X.sample(frac=1, random_state=i), y.sample(frac=1, random_state=i)

        # GridSearchCV initialisieren
        grid = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            cv=10,  # 10-fache Kreuzvalidierung
            scoring='accuracy',
            n_jobs=-1  # Parallelisiere GridSearchCV
        )

        # Training und Suche
        grid.fit(X_shuffled, y_shuffled)

        # Ergebnisse speichern
        accuracies.append(grid.best_score_)
        best_params_list.append(grid.best_params_)

    # Durchschnittliche Ergebnisse
    mean_accuracy = np.mean(accuracies)

    print(f"Durchschnittliche Genauigkeit: {mean_accuracy:.4f}")
    print(f"Beste Hyperparameter pro Durchlauf: {best_params_list}")
    return dataset, mean_accuracy, best_params_list

# Parallelisierte Verarbeitung aller Datensätze
results = Parallel(n_jobs=-1)(
    delayed(process_dataset)(dataset, target_columns[dataset]) for dataset in datasets
)

# Ergebnisse zusammenfassen
for dataset, mean_accuracy, best_params_list in results:
    print(f"Ergebnisse für {dataset}:")
    print(f"Durchschnittliche Genauigkeit: {mean_accuracy:.4f}")
    print(f"Beste Hyperparameter pro Durchlauf: {best_params_list}")
