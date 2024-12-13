from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Daten laden
data = pd.read_csv('dataset/clf_num/jannis.csv')
X = data.drop('class', axis=1)
y = data['class']

# Hyperparameter-Gitter
param_grid = {
    'n_estimators': [100],
    'max_depth': [None, 10],
    'criterion': ['gini'],
    'max_features': ['sqrt']
}

accuracies = []
best_params_list = []

# Mehrere Durchläufe (z. B. 10 Iterationen)
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
