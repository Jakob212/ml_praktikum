package ml_praktikum_jagoetz_wkathari;

import weka.core.converters.CSVLoader;
import weka.core.Instances;
import weka.classifiers.trees.RandomTree;
import weka.classifiers.Evaluation;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.Filter;

import java.io.File;
import java.util.*;
import java.util.stream.Collectors;

import static java.lang.Math.*;

/**
 * Entspricht grob dem Python-Beispiel mit GridSearchCV & DecisionTreeClassifier.
 * - Lädt CSV-Dateien, konvertiert die Klasse ggf. von numerisch zu nominal
 * - Führt 10 Wiederholungen mit zufälligem Shufflen (Train/Test 2/3 : 1/3) durch
 * - In jeder Wiederholung:
 *   -> Grid-Search über (max_depth, min_samples_leaf, max_features), 3-fach CV auf dem Trainingsset
 *   -> Finde bestes Param-Set, trainiere damit, evaluiere auf Testset
 * - Gibt Mittelwert, Std-Abweichung, beste und häufigste Param-Kombination aus
 */
public class WekaGridSearchDecisionTree {

    // Pfade zu den CSVs
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

    // Mapping: Datei -> Name der Zielspalte
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

    public static void main(String[] args) throws Exception {

        // Parameterbereiche wie im Python param_grid:
        //   max_depth: [None, 10, 20]  => in Weka: -depth (0=unbeschränkt)
        //   min_samples_leaf: [1, 5, 10] => -M
        //   max_features: [None, 'sqrt', 'log2'] => -K (0=alle, sqrt, log2)
        Integer[] maxDepthVals = { null, 10, 20 };    // null => 0 => unbeschränkt
        Integer[] minSamplesLeafVals = { 1, 5, 10 };
        String[] maxFeaturesVals = { null, "sqrt", "log2" };

        // Wir speichern pro Datensatz die Ergebnisse
        for (String dataset : datasets) {
            // Prüfen, ob wir eine Zielspalte haben
            if (!targetColumns.containsKey(dataset)) {
                System.out.println("Kein Zielspalteneintrag für " + dataset + " - überspringe.");
                continue;
            }
            String targetColumn = targetColumns.get(dataset);
            System.out.println("=== Datensatz: " + dataset + " ===");

            // 1) CSV laden
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(dataset));
            Instances data = loader.getDataSet();

            // 2) Klassenattribut setzen
            int classIndex = -1;
            for (int i = 0; i < data.numAttributes(); i++) {
                if (data.attribute(i).name().equalsIgnoreCase(targetColumn)) {
                    classIndex = i;
                    break;
                }
            }
            if (classIndex == -1) {
                System.out.println("Zielspalte '" + targetColumn + "' nicht gefunden - überspringe.");
                continue;
            }
            data.setClassIndex(classIndex);

            // 2a) Falls numerisch => NumericToNominal
            if (data.classAttribute().isNumeric()) {
                NumericToNominal ntm = new NumericToNominal();
                ntm.setOptions(new String[] {"-R", String.valueOf(classIndex+1)});
                ntm.setInputFormat(data);
                data = Filter.useFilter(data, ntm);
                data.setClassIndex(classIndex);
                System.out.println("Klasse war numerisch, umgewandelt zu nominal.");
            }

            // 3) 10 Wiederholungen
            List<Double> testAccuracies = new ArrayList<>();
            List<Double> runTimes = new ArrayList<>();
            List<Map<String,Object>> bestParamsList = new ArrayList<>();

            for (int run = 0; run < 10; run++) {
                long startTime = System.currentTimeMillis();

                // a) Shuffle
                Instances shuffled = new Instances(data);
                shuffled.randomize(new Random(run));
                // b) 2/3 Training, 1/3 Test
                int trainSize = (int) Math.round(shuffled.numInstances() * (2.0/3.0));
                Instances train = new Instances(shuffled, 0, trainSize);
                Instances test  = new Instances(shuffled, trainSize, shuffled.numInstances() - trainSize);

                // c) "GridSearchCV" = wir testen alle Kombis, machen 3-fach CV auf dem Trainings-Set
                double bestCVAcc = 0.0;
                String[] bestOptions = null;
                Map<String,Object> bestParamMap = null;

                // Wie in scikit-learn: Schleife über Parameter-Kombis
                int numFeatures = train.numAttributes() - 1; // abzüglich Klasse
                for (Integer md : maxDepthVals) {
                    for (Integer msl : minSamplesLeafVals) {
                        for (String mf : maxFeaturesVals) {
                            // Entsprechung scikit => Weka
                            int depth = (md == null) ? 0 : md;  // 0 => unbeschränkt
                            int minLeaf = msl;

                            // max_features
                            int k;  // -K
                            if (mf == null) {
                                k = 0; // alle
                            } else if ("sqrt".equals(mf)) {
                                k = (int) max(1, floor(sqrt(numFeatures)));
                            } else if ("log2".equals(mf)) {
                                k = (int) max(1, floor(log(numFeatures) / log(2)));
                            } else {
                                k = 0;
                            }

                            // Options für RandomTree
                            String[] opts = {
                                "-depth", String.valueOf(depth),
                                "-M", String.valueOf(minLeaf),
                                "-K", String.valueOf(k),
                                "-S", "42"   // fester Seed für Reproduzierbarkeit der Splits
                            };
                            RandomTree rt = new RandomTree();
                            rt.setOptions(opts);

                            // 3-fach CV auf 'train'
                            Evaluation evalCV = new Evaluation(train);
                            evalCV.crossValidateModel(rt, train, 3, new Random(run));
                            double cvAccuracy = evalCV.pctCorrect() / 100.0;
                            if (cvAccuracy > bestCVAcc) {
                                bestCVAcc = cvAccuracy;
                                bestOptions = opts;
                                Map<String,Object> tmpMap = new LinkedHashMap<>();
                                tmpMap.put("max_depth", md);
                                tmpMap.put("min_samples_leaf", msl);
                                tmpMap.put("max_features", mf);
                                bestParamMap = tmpMap;
                            }
                        }
                    }
                }

                // d) Haben nun bestes Param-Set => trainieren final auf komplettem Train
                RandomTree bestModel = new RandomTree();
                if (bestOptions != null) {
                    bestModel.setOptions(bestOptions);
                }
                bestModel.buildClassifier(train);

                // e) Test-Accuracy
                Evaluation evalTest = new Evaluation(train);
                evalTest.evaluateModel(bestModel, test);
                double testAcc = evalTest.pctCorrect() / 100.0;
                testAccuracies.add(testAcc);

                long endTime = System.currentTimeMillis();
                double elapsedSec = (endTime - startTime) / 1000.0;
                runTimes.add(elapsedSec);

                // Speichern der besten Parameter
                bestParamsList.add(bestParamMap);

                System.out.printf("Run %2d: Test-Acc=%.4f, Zeit=%.2fs, Params=%s\n",
                                  run+1, testAcc, elapsedSec,
                                  (bestParamMap != null) ? bestParamMap.toString() : "null");
            }

            // 4) Auswertung (wie im Python-Code)
            double meanAcc = mean(testAccuracies);
            double stdAcc = stddev(testAccuracies, meanAcc);
            double meanTime = mean(runTimes);
            double stdTime = stddev(runTimes, meanTime);

            System.out.printf("=> Durchschnittliche Accuracy: %.4f ± %.4f\n", meanAcc, stdAcc);
            System.out.printf("=> Durchschnittliche Zeit: %.2fs ± %.2fs\n", meanTime, stdTime);

            // Bester Run
            int bestRunIdx = 0;
            double bestAcc = 0.0;
            for (int i = 0; i < testAccuracies.size(); i++) {
                if (testAccuracies.get(i) > bestAcc) {
                    bestAcc = testAccuracies.get(i);
                    bestRunIdx = i;
                }
            }
            System.out.printf("Bester Durchlauf: %d mit Acc=%.4f, Param=%s\n",
                              bestRunIdx+1, bestAcc,
                              bestParamsList.get(bestRunIdx));

            // Häufigste Param-Kombi
            //   wir wandeln jede Map in einen String (o. sortierte Tupel), zählen Frequenzen
            Map<String,Integer> freqMap = new HashMap<>();
            for (Map<String,Object> pm : bestParamsList) {
                if (pm == null) continue;
                String key = pm.entrySet().stream()
                        .map(e -> e.getKey()+"="+e.getValue())
                        .sorted()   // Sortierung, damit "max_depth=10 ..." immer gleicher String
                        .collect(Collectors.joining(","));
                freqMap.put(key, freqMap.getOrDefault(key, 0) + 1);
            }
            // die häufigste:
            String mostCommonKey = null;
            int maxCount = 0;
            for (String k : freqMap.keySet()) {
                int cnt = freqMap.get(k);
                if (cnt > maxCount) {
                    maxCount = cnt;
                    mostCommonKey = k;
                }
            }
            if (mostCommonKey != null) {
                System.out.println("Am häufigsten bestParams: {" + mostCommonKey + "} (in " + maxCount + " von 10 Runs)");
            }
            System.out.println();
        }
    }

    // Mittelwert
    private static double mean(List<Double> vals) {
        if (vals.isEmpty()) return 0.0;
        double sum = 0.0;
        for (double v : vals) sum += v;
        return sum / vals.size();
    }

    // Stichproben-Standardabweichung
    private static double stddev(List<Double> vals, double mean) {
        if (vals.size() < 2) return 0.0;
        double sumSq = 0.0;
        for (double v : vals) {
            double diff = v - mean;
            sumSq += diff * diff;
        }
        return Math.sqrt(sumSq / (vals.size() - 1));
    }
}
