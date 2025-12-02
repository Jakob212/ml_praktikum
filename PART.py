import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold

def mean(x):
    return np.mean(x)

def std_dev(x):
    return np.std(x, ddof=1)

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

def main():
    for rel_path in dataset_files:
        dataset_path = DATA_DIR / rel_path
        filename = dataset_path.name

        if filename not in target_columns:
            continue

        target_col = target_columns[filename]
        print(f"\n=== {filename} ===")

        if not dataset_path.exists():
            print("Datei fehlt.")
            continue

        try:
            data = pd.read_csv(dataset_path)
        except:
            print("Fehler beim Laden.")
            continue

        if target_col not in data.columns:
            print("Zielspalte fehlt.")
            continue

        y = data[target_col].astype(str)
        X = data.drop(columns=[target_col])

        print("Klassenverteilung:")
        print(y.value_counts())

        test_accuracies = []
        cv_accuracies = []
        run_times = []

        for i in range(15):
            t0 = time.time()

            data_shuffled = data.sample(frac=1, random_state=i).reset_index(drop=True)
            X = data_shuffled.drop(columns=[target_col])
            y = data_shuffled[target_col].astype(str)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=1/3, random_state=i, shuffle=True
            )

            clf = DecisionTreeClassifier(min_samples_leaf=2, random_state=i)
            cv = KFold(n_splits=3, shuffle=True, random_state=i)

            cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")
            cv_accuracies.append(cv_scores.mean())

            clf.fit(X_train, y_train)
            test_acc = clf.score(X_test, y_test)
            test_accuracies.append(test_acc)

            run_times.append(time.time() - t0)
            print(f"Run {i+1}: {run_times[-1]:.4f}s")

        print("\nMatrix (15 Accuracy-Werte):")
        print("[" + ", ".join(f"{acc:.4f}" for acc in test_accuracies) + "]")

        print(f"\nTest-Accuracy (M ± SD): {mean(test_accuracies):.4f} ± {std_dev(test_accuracies):.4f}")
        print(f"CV-Accuracy   (M ± SD): {mean(cv_accuracies):.4f} ± {std_dev(cv_accuracies):.4f}")
        print(f"Laufzeit      (M ± SD): {mean(run_times):.4f}s ± {std_dev(run_times):.4f}s")

if __name__ == "__main__":
    main()
