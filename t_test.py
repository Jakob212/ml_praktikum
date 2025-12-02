import numpy as np
from scipy.stats import t
from math import sqrt

def corrected_resampled_ttest(weka_acc, scikit_acc, n1, n2):
    # Arrays konvertieren (falls Listen übergeben wurden)
    weka_acc = np.array(weka_acc)
    scikit_acc = np.array(scikit_acc)
    
    # Anzahl Wiederholungen
    n = len(weka_acc)
    
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
    scikit_accuracies = [0.66916617, 0.65446911, 0.65026995, 0.65326935, 0.67966407, 0.66076785,
  0.66196761, 0.65536893, 0.65386923, 0.65836833, 0.64787043, 0.65656869,
  0.65686863, 0.67636473, 0.66076785]
    weka_accuracies = [0.6382, 0.6523, 0.6466, 0.6427, 0.6520, 0.6340, 0.6460, 0.6451, 0.6331, 0.6538, 0.6334, 0.6553, 0.6541, 0.6310, 0.6466]
    
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
