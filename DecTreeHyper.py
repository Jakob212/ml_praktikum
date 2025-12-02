from pathlib import Path
import time
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

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
    "clf_num/heloc.csv",
]

target_columns = {
    "house_16H.csv": "binaryClass",
    "jannis.csv": "class",
    "MagicTelescope.csv": "class",
    "MiniBooNE.csv": "signal",
    "pol.csv": "binaryClass",
    "Higgs.csv": "target",
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
    "heloc.csv": "RiskPerformance",
}

param_grid = {
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 5, 10],
    "max_features": [None, "sqrt", "log2"],
}

for rel_path in dataset_files:
    dataset_path = DATA_DIR / rel_path
    filename = dataset_path.name
    target_col = target_columns[filename]

    print(f"\n=== Verarbeite Datensatz: {filename} ===")

    data = pd.read_csv(dataset_path)
    X = data.drop(columns=[target_col])
    y = data[target_col]

    if pd.api.types.is_numeric_dtype(y):
        print(f"Konvertiere Zielspalte '{target_col}' zu String.")
        y = y.astype(str)

    print("Klassenverteilung:")
    print(y.value_counts())

    test_accuracies = []
    run_times = []
    best_params_list = []

    for run in range(10):
        start_time = time.time()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1 / 3, shuffle=True, random_state=run
        )

        grid = GridSearchCV(
            estimator=DecisionTreeClassifier(random_state=42),
            param_grid=param_grid,
            cv=3,
            scoring="accuracy",
        )
        grid.fit(X_train, y_train)

        test_acc = grid.score(X_test, y_test)

        test_accuracies.append(test_acc)
        best_params_list.append(grid.best_params_)
        run_times.append(time.time() - start_time)

        print(f"Run {run + 1}:")
        print(f"Test-Accuracy: {test_acc:.4f}")
        print(f"Beste Parameter: {grid.best_params_}")
        print(f"Zeit: {run_times[-1]:.2f}s\n")

    print("\n" + "=" * 50)
    print("Ergebniszusammenfassung:")

    print("\nTest-Accuracies (10 Runs):")
    print(", ".join(f"{acc:.4f}" for acc in test_accuracies))

    mean_acc = np.mean(test_accuracies)
    std_acc = np.std(test_accuracies)
    mean_time = np.mean(run_times)
    std_time = np.std(run_times)
    print(f"\nDurchschnittliche Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Durchschnittliche Laufzeit: {mean_time:.2f}s ± {std_time:.2f}s")

    best_run_idx = np.argmax(test_accuracies)
    print(f"\nBester Run (Nr. {best_run_idx + 1}):")
    print(f"Accuracy: {test_accuracies[best_run_idx]:.4f}")
    print(f"Parameter: {best_params_list[best_run_idx]}")

    param_counts = {}
    for params in best_params_list:
        key = frozenset(params.items())
        param_counts[key] = param_counts.get(key, 0) + 1

    most_common = max(param_counts, key=param_counts.get)
    count = param_counts[most_common]
    print(f"\nHäufigste Parameterkombination ({count}/10):")
    print(dict(most_common))
