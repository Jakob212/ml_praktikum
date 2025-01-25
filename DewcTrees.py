import sys
import time
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from statistics import mean, stdev
from scipy.stats import ttest_1samp

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
    def flush(self):
        for f in self.files:
            f.flush()

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

fixed_params = {
    'max_depth': None,
    'min_samples_leaf': 10,
    'max_features': 'sqrt',
    'random_state': 42
}

f = open("decision_tree_results.txt", "w", encoding="utf-8")
original_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, f)

for dataset in datasets:
    if dataset not in target_columns:
        print(f"**WARNUNG**: Kein Zielspalteneintrag für {dataset}. Überspringe diesen Datensatz.")
        continue
    target_col = target_columns[dataset]
    print(f"\n=== Verarbeite Datensatz: {dataset} ===")
    df = pd.read_csv(dataset)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    print("Klassenverteilung:")
    print(y.value_counts())
    test_accuracies = []
    inner_cv_accuracies = []
    run_times = []
    for i in range(15):
        start_time = time.time()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1/3, random_state=i, shuffle=True
        )
        clf = DecisionTreeClassifier(**fixed_params)
        cv_scores = cross_val_score(clf, X_train, y_train, cv=3, scoring='accuracy')
        inner_cv_accuracies.append(cv_scores.mean())
        clf.fit(X_train, y_train)
        test_accuracy = clf.score(X_test, y_test)
        test_accuracies.append(test_accuracy)
        end_time = time.time()
        elapsed = end_time - start_time
        run_times.append(elapsed)
        print(f"Durchlauf {i+1} Dauer: {elapsed:.4f}s")
    mean_test_acc = mean(test_accuracies)
    std_test_acc = stdev(test_accuracies)
    print(f"Test-Accuracy über 15 Wiederholungen: {mean_test_acc:.4f} ± {std_test_acc:.4f}")
    mean_inner_cv_acc = mean(inner_cv_accuracies)
    std_inner_cv_acc = stdev(inner_cv_accuracies)
    print(f"Innere CV-Accuracy (3-fach) Mittelwert: {mean_inner_cv_acc:.4f} ± {std_inner_cv_acc:.4f}")
    mean_run_time = mean(run_times)
    std_run_time = stdev(run_times)
    print(f"Durchschnittliche Dauer: {mean_run_time:.4f}s ± {std_run_time:.4f}s")
    if y.nunique() == 2:
        stat, p_value = ttest_1samp(test_accuracies, 0.5)
        print("\nSignifikanztest (Ein-Stichproben-t-Test gegen 0.5):")
        print(f"T-Statistik: {stat:.4f}, p-Wert: {p_value:.6f}")
        if p_value < 0.05:
            print(f"==> Die mittlere Accuracy ({mean_test_acc:.3f}) ist signifikant von 0.5 verschieden (5%-Niveau).")
        else:
            print(f"==> Kein signifikanter Unterschied zu 0.5 (5%-Niveau).")
    else:
        print("\n(Signifikanztest gegen 0.5 nicht durchgeführt, da keine binäre Klassifikation.)")

sys.stdout = original_stdout
f.close()
print("Fertig! Alle Ausgaben wurden zusätzlich in 'decision_tree_results.txt' gespeichert.")
