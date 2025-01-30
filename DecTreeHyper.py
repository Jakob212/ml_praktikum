import time
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

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

param_grid = {
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'max_features': [None, 'sqrt', 'log2']
}

for dataset in datasets:
    target_col = target_columns[dataset]
    print(f"\n=== Verarbeite Datensatz: {dataset} ===")
    
    # Daten laden und vorbereiten
    data = pd.read_csv(dataset)
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    # Zielvariable konvertieren falls numerisch
    if pd.api.types.is_numeric_dtype(y):
        print(f"Konvertiere Zielspalte '{target_col}' zu String.")
        y = y.astype(str)
    
    print("Klassenverteilung:")
    print(y.value_counts())
    
    test_accuracies = []
    run_times = []
    best_params_list = []

    # 10 Wiederholungen mit Grid Search
    for run in range(10):
        start_time = time.time()
        
        # 1. Shuffle und Split (2/3 - 1/3)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=1/3, 
            shuffle=True, 
            random_state=run
        )
        
        # 2. Grid Search mit 3-fach CV
        grid = GridSearchCV(
            estimator=DecisionTreeClassifier(random_state=42),
            param_grid=param_grid,
            cv=3,
            scoring='accuracy'
        )
        grid.fit(X_train, y_train)
        
        # 3. Bestes Modell evaluieren
        test_acc = grid.score(X_test, y_test)
        
        # Metriken speichern
        test_accuracies.append(test_acc)
        best_params_list.append(grid.best_params_)
        run_times.append(time.time() - start_time)
        
        print(f"Run {run+1}:")
        print(f"Test-Accuracy: {test_acc:.4f}")
        print(f"Beste Parameter: {grid.best_params_}")
        print(f"Zeit: {run_times[-1]:.2f}s\n")

    # Statistische Auswertung
    print("\n" + "="*50)
    print("Ergebniszusammenfassung:")
    
    # 1. Accuracy-Matrix
    print("\nTest-Accuracies (10 Runs):")
    print(", ".join(f"{acc:.4f}" for acc in test_accuracies))
    
    # 2. Mittelwerte und Standardabweichungen
    mean_acc = np.mean(test_accuracies)
    std_acc = np.std(test_accuracies)
    mean_time = np.mean(run_times)
    std_time = np.std(run_times)
    print(f"\nDurchschnittliche Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Durchschnittliche Laufzeit: {mean_time:.2f}s ± {std_time:.2f}s")
    
    # 3. Bester Run
    best_run_idx = np.argmax(test_accuracies)
    print(f"\nBester Run (Nr. {best_run_idx+1}):")
    print(f"Accuracy: {test_accuracies[best_run_idx]:.4f}")
    print(f"Parameter: {best_params_list[best_run_idx]}")
    
    # 4. Häufigste Parameterkombination
    param_counts = {}
    for params in best_params_list:
        key = frozenset(params.items())
        param_counts[key] = param_counts.get(key, 0) + 1
    
    most_common = max(param_counts, key=param_counts.get)
    count = param_counts[most_common]
    print(f"\nHäufigste Parameterkombination ({count}/10):")
    print(dict(most_common))