import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from statistics import mean, stdev

# Deine Datensatzliste (bitte achte darauf, dass alle Pfade korrekt sind 
# und durch Kommata getrennt!)
datasets = [
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\house_16H.csv', 
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\jannis.csv',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\MagicTelescope.csv',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\MiniBooNE.csv',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\pol.csv',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_cat\\compas-two-years.csv',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_cat\\default-of-credit-card-clients.csv',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_cat\\electricity.csv',
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_cat\\eye_movements.csv',
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
    'ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\heloc.csv',
]

# Dictionary mit Zielspalten
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

# Feste Hyperparameter für den DecisionTree
fixed_params = {
    'max_depth': None,
    'min_samples_leaf': 10,
    'max_features': 'sqrt',
    # Optional kann man einen random_state setzen, um Reproduzierbarkeit zu gewährleisten
    'random_state': 42
}

for dataset in datasets:
    # Falls ein Datensatz evtl. nicht im Dictionary ist (z.B. Tippfehler), überspringen
    if dataset not in target_columns:
        print(f"**WARNUNG**: Kein Zielspalteneintrag für {dataset}. Überspringe diesen Datensatz.")
        continue
    
    target_col = target_columns[dataset]
    
    print(f"\n=== Verarbeite Datensatz: {dataset} ===")
    
    # Daten laden
    df = pd.read_csv(dataset)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Klassenverteilung ausgeben
    print("Klassenverteilung:")
    print(y.value_counts())
    
    # Liste, um die Test-Genauigkeiten der 15 Wiederholungen zu speichern
    test_accuracies = []
    # (Optional) Liste, um die inneren CV-Genauigkeiten zu speichern
    inner_cv_accuracies = []
    
    for i in range(15):
        # 2/3 Train, 1/3 Test (Hold-Out)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=1/3, 
            random_state=i,  # Damit bei jeder Wiederholung ein anderes, aber reproduzierbares Splitting
            shuffle=True
        )
        
        # DecisionTree mit festen Hyperparametern
        clf = DecisionTreeClassifier(**fixed_params)
        
        # Innere CV (3-fach) nur zum Performance Check
        cv_scores = cross_val_score(clf, X_train, y_train, cv=3, scoring='accuracy')
        inner_cv_accuracies.append(cv_scores.mean())
        
        # Nun den Baum auf dem gesamten Trainingssplit trainieren
        clf.fit(X_train, y_train)
        
        # Testen auf dem Hold-Out
        test_accuracy = clf.score(X_test, y_test)
        test_accuracies.append(test_accuracy)
    
    # Ergebnisse berechnen
    mean_test_acc = mean(test_accuracies)
    std_test_acc = stdev(test_accuracies)
    
    print(f"Test-Accuracy über 15 Wiederholungen: {mean_test_acc:.4f} ± {std_test_acc:.4f}")
    
    # Optional die innere CV-Accuracy (falls gewünscht)
    mean_inner_cv_acc = mean(inner_cv_accuracies)
    std_inner_cv_acc = stdev(inner_cv_accuracies)
    print(f"Innere CV-Accuracy (3-fach) Mittelwert: {mean_inner_cv_acc:.4f} ± {std_inner_cv_acc:.4f}")
