import pandas as pd

def dataframe_to_arff(df, relation_name, file_path):
    with open(file_path, "w") as f:
        # Relation
        f.write(f"@relation {relation_name}\n\n")
        
        # Attributes
        for col in df.columns:
            if df[col].dtype == "object":
                unique_vals = sorted(df[col].dropna().unique())
                unique_vals_str = ",".join(map(str, unique_vals))
                f.write(f"@attribute {col} {{{unique_vals_str}}}\n")
            else:
                f.write(f"@attribute {col} numeric\n")
        
        # Data
        f.write("\n@data\n")
        for _, row in df.iterrows():
            row_data = ",".join(map(str, row.values))
            f.write(f"{row_data}\n")

# Laden der CSV-Dateien
eye_movements_df = pd.read_csv('dataset\clf_num\eye_movements.csv')  # Pfad zur Datei
jannis_df = pd.read_csv('dataset\clf_num\jannis.csv')  # Pfad zur Datei

# Konvertierung in ARFF
dataframe_to_arff(eye_movements_df, "eye_movements", "eye_movements.arff")
dataframe_to_arff(jannis_df, "jannis", "jannis.arff")

print("ARFF-Dateien wurden erfolgreich erstellt!")
