from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

# Daten laden
datasets = [
    'dat/covertype.csv',
    'dat/wine.csv',
    'dat/letter-recognition.csv'
    
]

target_columns = {
    'dat/covertype.csv': 'class',
    'dat/letter-recognition.csv': 'yedgex',
    'dat/wine.csv': 'Proline'
}

for dataset in datasets:
    target_column = target_columns[dataset]
    print(f"Verarbeite Datensatz: {dataset}")

    data = pd.read_csv(dataset)
    X = data.drop(target_column, axis=1)
    y = data[target_column]

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
for i in range(1):
    # Zufälliges Shufflen der Daten in jedem Durchlauf
    X_shuffled, y_shuffled = X.sample(frac=1, random_state=i), y.sample(frac=1, random_state=i)
    
    # GridSearchCV initialisieren
    grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=10,               # 10-fache Kreuzvalidierung
        scoring='accuracy',

    )
    
    # Training und Suche
    grid.fit(X_shuffled, y_shuffled)
    
    # Ergebnisse speichern
    accuracies.append(grid.best_score_)
    best_params_list.append(grid.best_params_)

# Durchschnittliche Ergebnisse
print(f"Durchschnittliche Genauigkeit: {np.mean(accuracies):.4f}")
print(f"Beste Hyperparameter pro Durchlauf: {best_params_list}")
