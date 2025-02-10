from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

datasets = [
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\house_16H.csv', 
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\jannis.csv',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\MagicTelescope.csv',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\MiniBooNE.csv',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\pol.csv',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_cat\\compas-two-years.csv',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_cat\\road-safety.csv',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_cat\\albert.csv',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\bank-marketing.csv',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\Bioresponse.csv',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\california.csv',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\credit.csv',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\default-of-credit-card-clients.csv',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\Diabetes130US.csv',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\electricity.csv',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\eye_movements.csv',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\heloc.csv'
]

target_columns = {
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\house_16H.csv': 'binaryClass',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\jannis.csv': 'class',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\MagicTelescope.csv': 'class',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\MiniBooNE.csv': 'signal',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\pol.csv': 'binaryClass',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\Higgs.csv': 'target',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_cat\\compas-two-years.csv': 'twoyearrecid',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_cat\\default-of-credit-card-clients.csv': 'y',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_cat\\electricity.csv': 'class',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_cat\\eye_movements.csv': 'label',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_cat\\road-safety.csv': 'SexofDriver',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_cat\\albert.csv': 'class',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\bank-marketing.csv': 'Class',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\Bioresponse.csv': 'target',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\california.csv': 'price_above_median',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\credit.csv': 'SeriousDlqin2yrs',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\default-of-credit-card-clients.csv': 'y',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\Diabetes130US.csv': 'readmitted',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\electricity.csv': 'class',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\eye_movements.csv': 'label',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\heloc.csv': 'RiskPerformance',
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


    param_grid = {
        'n_estimators': [50, 100, 150],
        'min_samples_leaf': [10, 30],
        'max_features': ['sqrt', 'log2']
    }

    accuracies = []
    best_params_list = []


    # 10 Durchläufe
    for i in range(3):
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
        best_accuracy = grid.best_score_
        best_params = grid.best_params_
        
        
        # Ergebnisse speichern
        accuracies.append(grid.best_score_)
        best_params_list.append(grid.best_params_)

        # Ergebnisse der aktuellen Iteration ausgeben
        print(f"Beste Genauigkeit: {best_accuracy:.4f}")
        print(f"Beste Parameter: {best_params}\n")

    # Durchschnittliche Ergebnisse
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

     # Beste Parameter über alle Iterationen auswählen
    overall_best_params = best_params_list[np.argmax(accuracies)]

    print(f"Durchschnittliche Genauigkeit von {dataset}: {mean_accuracy:.4f}")
    print(f"Standardabweichung der Genauigkeit: {std_accuracy:.4f}")
    print(f"Beste Hyperparameter pro Durchlauf: {best_params_list}")
    print(f"Beste Hyperparameter insgesamt: {overall_best_params}\n")
