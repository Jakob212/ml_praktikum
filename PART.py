#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold

def mean(values):
    return np.mean(values)

def std_dev(values):
    # Sample Standardabweichung (ddof=1)
    return np.std(values, ddof=1)

def main():
    # Liste der Datensätze (Pfade anpassen)
    datasets = [
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_num\house_16H.csv",
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_num\jannis.csv",
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_num\MagicTelescope.csv",
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_num\MiniBooNE.csv",
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_num\pol.csv",
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_cat\compas-two-years.csv",
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_cat\road-safety.csv",
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_cat\albert.csv",
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_num\bank-marketing.csv",
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_num\Bioresponse.csv",
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_num\california.csv",
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_num\credit.csv",
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_num\default-of-credit-card-clients.csv",
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_num\Diabetes130US.csv",
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_num\electricity.csv",
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_num\eye_movements.csv",
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_num\heloc.csv"
    ]

    # Mapping: Datensatzpfad -> Name der Zielspalte
    target_columns = {
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_num\house_16H.csv": "binaryClass",
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_num\jannis.csv": "class",
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_num\MagicTelescope.csv": "class",
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_num\MiniBooNE.csv": "signal",
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_num\pol.csv": "binaryClass",
        # Hinweis: "Higgs.csv" taucht hier in der Map auf, aber nicht in der datasets-Liste.
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_cat\compas-two-years.csv": "twoyearrecid",
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_cat\default-of-credit-card-clients.csv": "y",
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_cat\electricity.csv": "class",
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_cat\eye_movements.csv": "label",
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_cat\road-safety.csv": "SexofDriver",
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_cat\albert.csv": "class",
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_num\bank-marketing.csv": "Class",
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_num\Bioresponse.csv": "target",
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_num\california.csv": "price_above_median",
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_num\credit.csv": "SeriousDlqin2yrs",
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_num\default-of-credit-card-clients.csv": "y",
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_num\Diabetes130US.csv": "readmitted",
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_num\electricity.csv": "class",
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_num\eye_movements.csv": "label",
        r"ml_praktikum_jagoetz_wkathari\dataset\clf_num\heloc.csv": "RiskPerformance"
    }

    # Verarbeitung jedes Datensatzes
    for dataset in datasets:
        if dataset not in target_columns:
            print(f"**WARNUNG**: Kein Zielspalteneintrag für {dataset}. Überspringe diesen Datensatz.")
            continue

        target_col = target_columns[dataset]
        print(f"\n=== Verarbeite Datensatz: {dataset} ===")
        
        # Prüfen, ob die Datei existiert
        if not os.path.exists(dataset):
            print(f"**WARNUNG**: Datei {dataset} nicht gefunden.")
            continue

        # CSV-Datei laden
        try:
            data = pd.read_csv(dataset)
        except Exception as e:
            print(f"Fehler beim Laden von {dataset}: {e}")
            continue

        # Prüfe, ob die Zielspalte vorhanden ist
        if target_col not in data.columns:
            print(f"**WARNUNG**: Zielspalte {target_col} nicht in {dataset} gefunden.")
            continue

        # Falls das Zielattribut numerisch ist, in nominal (Kategorie) konvertieren
        if pd.api.types.is_numeric_dtype(data[target_col]):
            data[target_col] = data[target_col].astype(str)
        
        # Ausgabe der Klassenverteilung
        print("Klassenverteilung:")
        class_counts = data[target_col].value_counts()
        for klass, count in class_counts.items():
            print(f"{klass}: {count}")
        
        # Listen für Ergebnisse über 15 Durchläufe
        test_accuracies = []
        cv_accuracies = []
        run_times = []

        # 15 Durchläufe (mit unterschiedlichem Zufallssamen)
        for i in range(15):
            start_time = time.time()

            # Zufällige Durchmischung der Daten (Seed = i)
            data_shuffled = data.sample(frac=1, random_state=i).reset_index(drop=True)

            # Aufteilen in Features (X) und Ziel (y)
            X = data_shuffled.drop(columns=[target_col])
            y = data_shuffled[target_col]

            # Aufteilen in Training (2/3) und Test (1/3)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=1/3, random_state=i, shuffle=True
            )

            # Erstellen des Klassifikators (hier: Entscheidungsbaum als Ersatz für PART)
            # Parameter: min_samples_leaf=2 entspricht minNumObj=2 aus Weka.
            clf = DecisionTreeClassifier(min_samples_leaf=2, random_state=i)
            
            # 3-fache Kreuzvalidierung auf den Trainingsdaten
            # Wir definieren ein KFold mit shuffle=True und Seed=i für Reproduzierbarkeit
            cv = KFold(n_splits=3, shuffle=True, random_state=i)
            cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")
            cv_acc = np.mean(cv_scores)
            cv_accuracies.append(cv_acc)

            # Training des Klassifikators auf den Trainingsdaten
            clf.fit(X_train, y_train)

            # Evaluierung auf den Testdaten
            test_acc = clf.score(X_test, y_test)
            test_accuracies.append(test_acc)

            end_time = time.time()
            duration = end_time - start_time  # in Sekunden
            run_times.append(duration)

            print(f"Durchlauf {i+1} Dauer: {duration:.4f}s")
        
        # Ausgabe der Test-Accuracies als Matrix (1x15)
        print("\nMatrix mit den 15 Test-Accuracies:")
        formatted_accuracies = ", ".join(f"{acc:.4f}" for acc in test_accuracies)
        print(f"[{formatted_accuracies}]")

        # Berechnung von Mittelwert und Standardabweichung
        mean_test = mean(test_accuracies)
        std_test = std_dev(test_accuracies)
        mean_cv = mean(cv_accuracies)
        std_cv = std_dev(cv_accuracies)
        mean_time = mean(run_times)
        std_time = std_dev(run_times)

        print(f"\nTest-Accuracy (M ± SD): {mean_test:.4f} ± {std_test:.4f}")
        print(f"CV-Accuracy (M ± SD): {mean_cv:.4f} ± {std_cv:.4f}")
        print(f"Laufzeit (M ± SD): {mean_time:.4f}s ± {std_time:.4f}s")

if __name__ == '__main__':
    main()
