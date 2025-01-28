package ml_praktikum_jagoetz_wkathari;

import weka.core.converters.CSVLoader;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.classifiers.trees.RandomTree;
import weka.classifiers.Evaluation;

import java.io.File;
import java.util.*;
import java.util.stream.Collectors;  // <--- WICHTIG für das .stream().collect(...)
import java.util.Locale;            // <--- Damit Dezimalpunkt immer '.' ist

public class WekaDecisionTree {

    // Liste von CSV-Datensätzen
    static String[] datasets = {
        "ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\house_16H.csv",
        "ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\jannis.csv",
        "ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\MagicTelescope.csv",
        "ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\MiniBooNE.csv",
        "ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\pol.csv",
        "ml_praktikum_jagoetz_wkathari\\dataset\\clf_cat\\compas-two-years.csv",
        "ml_praktikum_jagoetz_wkathari\\dataset\\clf_cat\\road-safety.csv",
        "ml_praktikum_jagoetz_wkathari\\dataset\\clf_cat\\albert.csv",
        "ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\bank-marketing.csv",
        "ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\Bioresponse.csv",
        "ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\california.csv",
        "ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\credit.csv",
        "ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\default-of-credit-card-clients.csv",
        "ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\Diabetes130US.csv",
        "ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\electricity.csv",
        "ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\eye_movements.csv",
        "ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\heloc.csv"
    };

    // Mapping: Pfad -> Name des Zielattributs
    static Map<String, String> targetColumns = new HashMap<>();
    static {
        targetColumns.put("ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\house_16H.csv", "binaryClass");
        targetColumns.put("ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\jannis.csv", "class");
        targetColumns.put("ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\MagicTelescope.csv", "class");
        targetColumns.put("ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\MiniBooNE.csv", "signal");
        targetColumns.put("ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\pol.csv", "binaryClass");
        targetColumns.put("ml_praktikum_jagoetz_wkathari\\dataset\\clf_cat\\compas-two-years.csv", "twoyearrecid");
        targetColumns.put("ml_praktikum_jagoetz_wkathari\\dataset\\clf_cat\\road-safety.csv", "SexofDriver");
        targetColumns.put("ml_praktikum_jagoetz_wkathari\\dataset\\clf_cat\\albert.csv", "class");
        targetColumns.put("ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\bank-marketing.csv", "Class");
        targetColumns.put("ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\Bioresponse.csv", "target");
        targetColumns.put("ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\california.csv", "price_above_median");
        targetColumns.put("ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\credit.csv", "SeriousDlqin2yrs");
        targetColumns.put("ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\default-of-credit-card-clients.csv", "y");
        targetColumns.put("ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\Diabetes130US.csv", "readmitted");
        targetColumns.put("ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\electricity.csv", "class");
        targetColumns.put("ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\eye_movements.csv", "label");
        targetColumns.put("ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\heloc.csv", "RiskPerformance");
    }

    // Feste Parameter analog zum Python-Code:
    // max_depth=None => unbeschränkt => -depth 0
    // min_samples_leaf=10 => -M 10
    // max_features='sqrt' => -K sqrt(#Features)
    // random_state=42 => -S 42
    static final int MIN_SAMPLES_LEAF = 10;
    static final int SEED = 42;
    static final String MAX_DEPTH = "0";  // "0" => keine Begrenzung

    public static void main(String[] args) throws Exception {

        // Optional: Ausgabe in eine Datei umleiten
        // PrintStream out = new PrintStream("decision_tree_results.txt", "UTF-8");
        // System.setOut(out);

        for (String dataset : datasets) {
            if (!targetColumns.containsKey(dataset)) {
                System.out.println("**WARNUNG**: Kein Zielspalteneintrag für " + dataset + ". Überspringe diesen Datensatz.");
                continue;
            }
            String targetCol = targetColumns.get(dataset);

            System.out.println("\n=== Verarbeite Datensatz: " + dataset + " ===");

            // 1) CSV laden
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(dataset));
            Instances data = loader.getDataSet();

            // 2) Klasse setzen
            int classIndex = -1;
            for (int i = 0; i < data.numAttributes(); i++) {
                if (data.attribute(i).name().equalsIgnoreCase(targetCol)) {
                    classIndex = i;
                    break;
                }
            }
            if (classIndex == -1) {
                System.out.println("Zielspalte '" + targetCol + "' nicht gefunden! Überspringe.");
                continue;
            }
            data.setClassIndex(classIndex);

            // Falls die Klasse numerisch => NumericToNominal
            if (data.classAttribute().isNumeric()) {
                System.out.println("Klasse ist numerisch => konvertiere zu nominal ...");
                NumericToNominal num2nom = new NumericToNominal();
                num2nom.setOptions(new String[]{"-R", String.valueOf(classIndex + 1)});
                num2nom.setInputFormat(data);
                data = Filter.useFilter(data, num2nom);
                data.setClassIndex(classIndex);
            }

            // Klassenverteilung
            Map<String, Integer> classDist = new HashMap<>();
            for (int i = 0; i < data.numInstances(); i++) {
                String lbl = data.instance(i).stringValue(data.classIndex());
                classDist.put(lbl, classDist.getOrDefault(lbl, 0) + 1);
            }
            System.out.println("Klassenverteilung: " + classDist);

            // Wir sammeln die Metriken
            List<Double> testAccuracies = new ArrayList<>();
            List<Double> cvAccuracies   = new ArrayList<>();
            List<Double> runTimes       = new ArrayList<>();

            // (max_features='sqrt') => #Features = floor(sqrt(#Attribute - 1))
            int numAttributes = data.numAttributes() - 1; // abzügl. Klassenattribut
            int kFeatures = (int)Math.floor(Math.sqrt(numAttributes));
            if (kFeatures < 1) kFeatures = 1;  // Sicherheit

            // 15 Wiederholungen
            for (int run = 0; run < 15; run++) {
                long start = System.currentTimeMillis();

                // a) Shuffle
                Instances shuffled = new Instances(data);
                shuffled.randomize(new Random(run));

                // b) 2/3 Training, 1/3 Test
                int trainSize = (int) Math.round(shuffled.numInstances() * (2.0/3.0));
                int testSize  = shuffled.numInstances() - trainSize;

                Instances train = new Instances(shuffled, 0, trainSize);
                Instances test  = new Instances(shuffled, trainSize, testSize);

                // c) RandomTree konfigurieren
                String[] options = {
                    "-depth", MAX_DEPTH,                        // 0 => kein Limit
                    "-M", String.valueOf(MIN_SAMPLES_LEAF),     // min instances per leaf
                    "-K", String.valueOf(kFeatures),            // sqrt(#features)
                    "-S", String.valueOf(SEED)                  // random seed
                };
                RandomTree rt = new RandomTree();
                rt.setOptions(options);

                // d) 3-fach Crossvalidation auf train
                Evaluation evalCV = new Evaluation(train);
                evalCV.crossValidateModel(rt, train, 3, new Random(run));
                double cvAcc = evalCV.pctCorrect() / 100.0;
                cvAccuracies.add(cvAcc);

                // e) Auf das komplette Training fitten
                rt.buildClassifier(train);

                // f) Auf Test evaluieren
                Evaluation evalTest = new Evaluation(train);
                evalTest.evaluateModel(rt, test);
                double testAcc = evalTest.pctCorrect() / 100.0;
                testAccuracies.add(testAcc);

                // Zeit
                long end = System.currentTimeMillis();
                double sec = (end - start) / 1000.0;
                runTimes.add(sec);

                System.out.printf("Durchlauf %2d Dauer: %.4fs\n", run+1, sec);
            }

            // Nach allen 15 Wiederholungen: "Matrix" der Test-Accuracies kommasepariert ausgeben
            System.out.println("\nMatrix (Liste) der 15 Test-Accuracies kommasepariert:");
            String matrixString = testAccuracies.stream()
                    // Locale.US => immer Dezimalpunkt
                    .map(acc -> String.format(Locale.US, "%.4f", acc))
                    .collect(Collectors.joining(", "));
            System.out.println("[" + matrixString + "]");

            // 6) Statistik
            double meanTest   = mean(testAccuracies);
            double stdTest    = stddev(testAccuracies, meanTest);
            System.out.printf("Test-Accuracy (15 Wdh.): %.4f ± %.4f\n", meanTest, stdTest);

            double meanCV = mean(cvAccuracies);
            double stdCV  = stddev(cvAccuracies, meanCV);
            System.out.printf("Innere CV-Accuracy (3-fach): %.4f ± %.4f\n", meanCV, stdCV);

            double meanTime = mean(runTimes);
            double stdTime  = stddev(runTimes, meanTime);
            System.out.printf("Durchschnittliche Dauer: %.4fs ± %.4fs\n", meanTime, stdTime);

            // 7) Ein-Stichproben-t-Test gegen 0.5, falls binär
            if (data.classAttribute().isNominal() && data.classAttribute().numValues() == 2) {
                double tStat = oneSampleTTest(testAccuracies, 0.5);
                double pVal  = twoTailedPValue(tStat, testAccuracies.size() - 1);

                System.out.println("\nSignifikanztest (t-Test gegen 0.5):");
                System.out.printf("T-Statistik: %.4f, p-Wert: %.6f\n", tStat, pVal);
                if (pVal < 0.05) {
                    System.out.printf("==> Mittlere Accuracy (%.3f) ist signifikant von 0.5 verschieden.\n", meanTest);
                } else {
                    System.out.println("==> Kein signifikanter Unterschied zu 0.5 (5%-Niveau).");
                }
            } else {
                System.out.println("\n(Signifikanztest nicht durchgeführt, da keine binäre Klassifikation.)");
            }
        }

        System.out.println("Fertig! Alle Ergebnisse stehen hier in der Konsole.");
    }

    // Hilfsfunktionen für Mittelwert & Stdabw.:
    private static double mean(List<Double> vals) {
        if (vals.isEmpty()) return 0.0;
        double sum = 0.0;
        for (double v : vals) sum += v;
        return sum / vals.size();
    }

    private static double stddev(List<Double> vals, double mean) {
        if (vals.size() < 2) return 0.0;
        double sumSq = 0.0;
        for (double v : vals) {
            double diff = v - mean;
            sumSq += diff * diff;
        }
        return Math.sqrt(sumSq / (vals.size() - 1));
    }

    // Ein-Stichproben-t-Test gegen mu=0.5
    private static double oneSampleTTest(List<Double> vals, double mu) {
        int n = vals.size();
        if (n < 2) return 0.0; // kein sinnvoller t-Test bei n=1

        double m = mean(vals);
        double s = stddev(vals, m);
        if (s == 0) return 0.0;  // alle Werte identisch => t=0
        return (m - mu) / (s / Math.sqrt(n));
    }

    // Zweiseitiger p-Wert; für große df Approx. via Normalverteilung
    private static double twoTailedPValue(double tStat, int df) {
        // Betragswert => symmetrische Normalapprox
        double z = Math.abs(tStat);
        double phi = 0.5 * (1.0 + erf(z / Math.sqrt(2.0)));
        return 2.0 * (1.0 - phi);
    }

    // Einfache Approximations-Implementierung der Fehlerfunktion erf(...)
    private static double erf(double x) {
        double sign = (x < 0) ? -1.0 : 1.0;
        x = Math.abs(x);

        // Formel Approx. nach Abramowitz/Stegun
        double p = 0.3275911;
        double t = 1.0 / (1.0 + p * x);
        double a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741,
               a4 = -1.453152027, a5 = 1.061405429;
        double y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x*x);
        return sign * y;
    }
}
