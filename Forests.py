import time
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from statistics import mean, stdev

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"

dataset_files = [
    "clf_num/house_16H.csv",
    "clf_num/jannis.csv",
    "clf_num/MagicTelescope.csv",
    "clf_num/MiniBooNE.csv",
    "clf_num/pol.csv",
    "clf_cat/compas-two-years.csv",
    "clf_cat/road-safety.csv",
    "clf_cat/albert.csv",
    "clf_num/bank-marketing.csv",
    "clf_num/Bioresponse.csv",
    "clf_num/california.csv",
    "clf_num/credit.csv",
    "clf_num/default-of-credit-card-clients.csv",
    "clf_num/Diabetes130US.csv",
    "clf_num/electricity.csv",
    "clf_num/eye_movements.csv",
    "clf_num/heloc.csv"
]

target_columns = {
    "house_16H.csv": "binaryClass",
    "jannis.csv": "class",
    "MagicTelescope.csv": "class",
    "MiniBooNE.csv": "signal",
    "pol.csv": "binaryClass",
    "compas-two-years.csv": "twoyearrecid",
    "default-of-credit-card-clients.csv": "y",
    "electricity.csv": "class",
    "eye_movements.csv": "label",
    "road-safety.csv": "SexofDriver",
    "albert.csv": "class",
    "bank-marketing.csv": "Class",
    "Bioresponse.csv": "target",
    "california.csv": "price_above_median",
    "credit.csv": "SeriousDlqin2yrs",
    "Diabetes130US.csv": "readmitted",
    "heloc.csv": "RiskPerformance"
}

forest_params = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1
}

for rel_path in dataset_files:
    dataset_path = DATA_DIR / rel_path
    filename = dataset_path.name

    if filename not in target_columns:
        print(f"Dataset {filename} skipped.")
        continue

    target_col = target_columns[filename]
    print(f"\n=== {filename} ===")

    df = pd.read_csv(dataset_path)
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(str)

    print("Klassenverteilung:")
    print(y.value_counts())

    test_accuracies = []
    cv_accuracies = []
    run_times = []

    for i in range(15):
        t0 = time.time()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1/3, random_state=i, shuffle=True
        )

        cv = KFold(n_splits=3, shuffle=True, random_state=i)
        clf = RandomForestClassifier(**forest_params)

        cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")
        cv_accuracies.append(cv_scores.mean())

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_score(y_test, y_pred, average="macro", zero_division=0)
        confusion_matrix(y_test, y_pred)

        test_accuracies.append(acc)
        run_times.append(time.time() - t0)

        print(f"Run {i+1}: {run_times[-1]:.2f}s")

    print("\nTest-Accuracies:")
    print(", ".join(f"{acc:.4f}" for acc in test_accuracies))

    print(f"\nTest-Accuracy (M±SD): {mean(test_accuracies):.4f} ± {stdev(test_accuracies):.4f}")
    print(f"CV-Accuracy (M±SD): {mean(cv_accuracies):.4f} ± {stdev(cv_accuracies):.4f}")
    print(f"Laufzeit (M±SD): {mean(run_times):.2f}s ± {stdev(run_times):.2f}s")

print("\nFertig.")
