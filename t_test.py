import numpy as np
from scipy.stats import t
from math import sqrt

def corrected_resampled_ttest(weka_acc, scikit_acc, n1, n2):
    # Arrays konvertieren (falls Listen übergeben wurden)
    weka_acc = np.array(weka_acc)
    scikit_acc = np.array(scikit_acc)
    
    # Anzahl Wiederholungen
    n = len(weka_acc)
    if len(scikit_acc) != n:
        raise ValueError("weka_acc und scikit_acc müssen gleich lang sein!")
    
    # Differenzen x_j = weka - scikit
    x = weka_acc - scikit_acc
    
    # Mittelwert und Varianz der Differenzen
    x_bar = np.mean(x)
    sigma_hat_sq = np.var(x, ddof=1)  # ddof=1 -> Stichprobenvarianz
    
    # Korrekturfaktor (1/n + n2/n1)
    correction = (1 / n) + (n2 / n1)
    
    # t-Statistik
    if sigma_hat_sq == 0:
        # Sonderfall: alle Differenzen gleich -> Varianz = 0
        # Dann kann keine Aussage über Signifikanz getroffen werden
        return float('inf'), 0.0
    
    t_stat = x_bar / np.sqrt(correction * sigma_hat_sq)
    
    # Freiheitsgrade (n-1)
    df = n - 1
    
    # zweiseitiger p-Wert
    p_value = 2 * (1 - t.cdf(abs(t_stat), df=df))
    
    return t_stat, p_value


if __name__ == "__main__":
    # -----------------------
    # 1) Beispiel-Daten
    # -----------------------
    
    # Nehmen wir an, das sind die Ergebnisse aus 15 Wiederholungen
    # (Train/Test-Splits), jeweils die Test-Accuracies:
    weka_accuracies = [0.8234, 0.8172, 0.8221, 0.7994, 0.8067, 0.8103, 0.8156, 0.8138, 0.8132, 0.8145, 0.8083, 0.8094, 0.8201, 0.8136, 0.8027]
    scikit_accuracies = [0.82317616, 0.82384342, 0.82184164, 0.81739324, 0.8147242 , 0.82295374, 0.82095196, 0.82384342, 0.8280694 , 0.82273132, 0.83096085, 0.82384342, 0.8080516 , 0.81561388, 0.81405694]
    
    # Wir nehmen an, in jedem Durchlauf wurden ~ 2/3 trainiert, 1/3 getestet.
    # Beispiel: Datensatz hat 1500 Instanzen, d. h. Train=1000, Test=500
    n1 = 10000  # Trainingsgröße
    n2 = 5000   # Testgröße
    
    # -----------------------
    # 2) Matrizenausgabe
    # -----------------------
    
    # Machen wir aus den Listen 1x15-Matrizen:
    weka_matrix = np.array(weka_accuracies).reshape(1, -1)
    scikit_matrix = np.array(scikit_accuracies).reshape(1, -1)
    
    print("WEKA-Matrix (1x15):\n", weka_matrix)
    print("scikit-Matrix (1x15):\n", scikit_matrix)
    
    # Differenz-Matrix (einfach weka minus scikit)
    diff_matrix = weka_matrix - scikit_matrix
    print("Differenzen WEKA - scikit (1x15):\n", diff_matrix)
    
    # -----------------------
    # 3) Test durchführen
    # -----------------------
    t_stat, p_val = corrected_resampled_ttest(weka_accuracies, scikit_accuracies, n1, n2)
    
    print("\nErgebnis des corrected resampled t-Tests (WEKA vs scikit):")
    print(f"t-Statistik: {t_stat:.4f}, p-Wert: {p_val:.6f}")
    
    alpha = 0.05
    if p_val < alpha:
        print("==> Signifikanter Unterschied (5%-Niveau).")
        if t_stat > 0:
            print("   WEKA hat im Mittel höhere Accuracy.")
        else:
            print("   scikit hat im Mittel höhere Accuracy.")
    else:
        print("==> Kein signifikanter Unterschied (5%-Niveau).")
