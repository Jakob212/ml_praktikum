from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

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
    "n_estimators": [50, 100, 150],
    "min_samples_leaf": [10, 30],
    "max_features": ["sqrt", "log2"],
}

for rel_path in dataset_files:
    dataset_path = DATA_DIR / rel_path
    filename = dataset_path.name

    if filename not in target_columns:
        continue

    target_column = target_columns[filename]
    print(f"\nVerarbeite Datensatz: {filename}")

    data = pd.read_csv(dataset_path)

    if target_column not in data.columns:
        print("Zielspalte fehlt, Datensatz übersprungen.")
        continue

    data[target_column] = data[target_column].astype(str)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    print("Klassen:", y.unique())
    print("Verteilung:\n", y.value_counts())

    accuracies = []
    best_params_list = []

    for i in range(10):
        shuffled = data.sample(frac=1, random_state=i).reset_index(drop=True)
        X_shuffled = shuffled.drop(columns=[target_column])
        y_shuffled = shuffled[target_column]

        grid = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            cv=10,
            scoring="accuracy",
        )

        grid.fit(X_shuffled, y_shuffled)

        best_accuracy = grid.best_score_
        best_params = grid.best_params_

        accuracies.append(best_accuracy)
        best_params_list.append(best_params)

        print(f"Durchlauf {i+1}: Accuracy={best_accuracy:.4f}, Params={best_params}")

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    overall_best_params = best_params_list[np.argmax(accuracies)]

    print(f"\nDurchschnittliche Genauigkeit: {mean_accuracy:.4f}")
    print(f"Standardabweichung: {std_accuracy:.4f}")
    print(f"Beste Hyperparameter pro Durchlauf: {best_params_list}")
    print(f"Beste Hyperparameter insgesamt: {overall_best_params}")
