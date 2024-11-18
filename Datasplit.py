import pandas as pd
from sklearn.model_selection import train_test_split

# Datensatz laden
data = pd.read_csv("dataset\clf_num\eye_movements.csv")  # Ersetzen Sie den Dateinamen

# Unabhängige Variablen (Features) und Zielvariable (Target) trennen
X = data.drop(columns=["label"])  # Zielspalte ersetzen
y = data["label"]

# Daten splitten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Trainings- und Testdaten kombinieren (Features + Zielspalte)
X_train["label"] = y_train
X_test["label"] = y_test

# Speichern der Datensätze als CSV
X_train.to_csv("train_set_eye_movement.csv", index=False)
X_test.to_csv("test_set_eye_movement.csv", index=False)
