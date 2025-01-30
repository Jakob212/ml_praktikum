import time
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from statistics import mean, stdev

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
    'criterion': 'entropy',
    'max_depth': None,
    'min_samples_leaf': 10,
    'max_features': 'sqrt',
    'random_state': 42
}

for dataset in datasets:
    if dataset not in target_columns:
        print(f"**WARNUNG**: Kein Zielspalteneintrag für {dataset}. Überspringe diesen Datensatz.")
        continue
    
    target_col = target_columns[dataset]
    print(f"\n=== Verarbeite Datensatz: {dataset} ===")
    
    df = pd.read_csv(dataset)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    if pd.api.types.is_numeric_dtype(y):
        print(f"Zielspalte '{target_col}' ist numerisch. Konvertiere zu String.")
        y = y.astype(str)
    
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
        
        cv = KFold(n_splits=3, shuffle=True, random_state=i)
        clf = DecisionTreeClassifier(**fixed_params)
        
        cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
        inner_cv_accuracies.append(cv_scores.mean())
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # Metrikenberechnung (nur für Zeitmessung)
        test_accuracy = accuracy_score(y_test, y_pred)
        _ = precision_score(y_test, y_pred, average='macro', zero_division=0)
        _ = recall_score(y_test, y_pred, average='macro', zero_division=0)
        _ = f1_score(y_test, y_pred, average='macro', zero_division=0)
        _ = confusion_matrix(y_test, y_pred)
        
        test_accuracies.append(test_accuracy)
        run_times.append(time.time() - start_time)
        print(f"Durchlauf {i+1} Dauer: {run_times[-1]:.4f}s")
    
    acc_matrix = np.array(test_accuracies).reshape(1, -1)
    print("\nMatrix mit den 15 Test-Accuracies:")
    print(np.array2string(acc_matrix, separator=', '))
    
    print(f"\nTest-Accuracy (M ± SD): {mean(test_accuracies):.4f} ± {stdev(test_accuracies):.4f}")
    print(f"CV-Accuracy (M ± SD): {mean(inner_cv_accuracies):.4f} ± {stdev(inner_cv_accuracies):.4f}")
    print(f"Laufzeit (M ± SD): {mean(run_times):.4f}s ± {stdev(run_times):.4f}s")