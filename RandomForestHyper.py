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

# Funktion zur Verarbeitung eines einzelnen Datensatzes
def process_dataset(dataset, target_column):
    print(f"Verarbeite Datensatz: {dataset}")

    data = pd.read_csv(dataset).copy()  # Kopiere Daten, um Konflikte zu vermeiden
    X = data.drop(target_column, axis=1).copy()
    y = data[target_column].copy()

    # Klassen anzeigen
    print(f"Klassen: {y.unique()}")
    print(f"Verteilung: \n{y.value_counts()}")

    # Hyperparameter-Gitter
    param_grid = {
        'n_estimators': [50, 100, 150],
        'min_samples_leaf': [10, 30],
        'max_features': ['sqrt', 'log2']
    }

    accuracies = []
    best_params_list = []

    # 10 Durchläufe
    for i in range(10):
        # Zufälliges Shufflen der Daten in jedem Durchlauf
        random_state = 42 + i  # Eindeutiger Seed für jeden Durchlauf
        X_shuffled, y_shuffled = X.sample(frac=1, random_state=random_state), y.sample(frac=1, random_state=random_state)

        # GridSearchCV initialisieren
        grid = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            cv=10,               # 10-fache Kreuzvalidierung
            scoring='accuracy',
            n_jobs=2            # Parallele Verarbeitung innerhalb von GridSearch
        )

        # Training und Suche
        grid.fit(X_shuffled, y_shuffled)

        # Ergebnisse speichern
        accuracies.append(grid.best_score_)
        best_params_list.append(grid.best_params_)

    # Durchschnittliche Ergebnisse
    mean_accuracy = np.mean(accuracies)

    print(f"Durchschnittliche Genauigkeit von {dataset}: {mean_accuracy:.4f}")
    print(f"Beste Hyperparameter pro Durchlauf: {best_params_list}\n")

# Parallele Verarbeitung aller Datensätze
results = Parallel(n_jobs=2, prefer="processes")(  # Nutzt alle verfügbaren Kerne
    delayed(process_dataset)(dataset, target_columns[dataset]) for dataset in datasets
)
