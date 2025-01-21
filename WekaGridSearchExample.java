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
 * Beispiel zur Hyperparameter-Optimierung mit Weka:
 * - Lädt CSV-Dateien
 * - Prüft und konvertiert das Klassenattribut ggf. von numeric auf nominal
 * - Führt 10 Wiederholungen mit zufälligem Shufflen durch
 * - Sucht in jeder Wiederholung per Grid-Ansatz die besten Parameter
 *   (max_depth, min_samples_leaf, max_features)
 * - Gibt Mittelwert, Standardabweichung und häufigste Parameterkombi aus
 */
public class WekaGridSearchExample {

    // Pfade zu den CSVs
    static String[] datasets = {
        //"ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\house_16H.csv",
        //"ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\jannis.csv",
        //"ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\MagicTelescope.csv",
        //"ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\MiniBooNE.csv",
        //"ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\pol.csv",
        //"ml_praktikum_jagoetz_wkathari\\dataset\\clf_cat\\compas-two-years.csv",
        //"ml_praktikum_jagoetz_wkathari\\dataset\\clf_cat\\default-of-credit-card-clients.csv",
        //"ml_praktikum_jagoetz_wkathari\\dataset\\clf_cat\\electricity.csv",
        //"ml_praktikum_jagoetz_wkathari\\dataset\\clf_cat\\eye_movements.csv",
        //"ml_praktikum_jagoetz_wkathari\\dataset\\clf_cat\\road-safety.csv",
        //"ml_praktikum_jagoetz_wkathari\\dataset\\clf_cat\\albert.csv",
        //"ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\bank-marketing.csv",
        //"ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\Bioresponse.csv",
        //"ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\california.csv",
        //"ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\credit.csv",
        //"ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\default-of-credit-card-clients.csv",
        //"ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\Diabetes130US.csv",
        //"ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\electricity.csv",
        //"ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\eye_movements.csv",
        //"ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\heloc.csv",
        "ml_praktikum_jagoetz_wkathari\\dataset\\clf_cat\\albert.csv"
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
        targetColumns.put("ml_praktikum_jagoetz_wkathari\\dataset\\clf_cat\\default-of-credit-card-clients.csv", "y");
        targetColumns.put("ml_praktikum_jagoetz_wkathari\\dataset\\clf_cat\\electricity.csv", "class");
        targetColumns.put("ml_praktikum_jagoetz_wkathari\\dataset\\clf_cat\\eye_movements.csv", "label");
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

        for (String dataset : datasets) {
            // 1) Laden
            String targetColumn = targetColumns.get(dataset);
            System.out.println("Verarbeite Datensatz: " + dataset);

            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(dataset));
            Instances data = loader.getDataSet();

            // 2) Klassenattribut setzen:
            //    Finde den Index, an dem das targetColumn steht
            int classIndex = -1;
            for (int i = 0; i < data.numAttributes(); i++) {
                if (data.attribute(i).name().equalsIgnoreCase(targetColumn)) {
                    classIndex = i;
                    break;
                }
            }
            if (classIndex == -1) {
                throw new IllegalArgumentException("Zielspalte '" + targetColumn + "' nicht gefunden!");
            }

            data.setClassIndex(classIndex);

            // 2a) Falls die Klasse numerisch ist, -> NumericToNominal
            if (data.classAttribute().isNumeric()) {
                System.out.println("Klasse ist numerisch, wandle in nominal um ...");
                NumericToNominal num2nom = new NumericToNominal();
                // Achtung: Weka-Filter ist 1-basiert -> classIndex + 1
                num2nom.setOptions(new String[]{"-R", String.valueOf(classIndex + 1)});
                num2nom.setInputFormat(data);

                Instances newData = Filter.useFilter(data, num2nom);
                // Nach Filter erneut setzen (Index bleibt der gleiche, aber wir müssen es Weka "mitteilen")
                newData.setClassIndex(classIndex);
                data = newData;
            }

            // Klassen anzeigen
            System.out.println("Klassen-Attribut: " + data.classAttribute().name());
            System.out.println("Is nominal? " + data.classAttribute().isNominal());
            System.out.println("Is numeric? " + data.classAttribute().isNumeric());

            // Verteilung ausgeben
            Map<String, Integer> distMap = new HashMap<>();
            for (int i = 0; i < data.numInstances(); i++) {
                // Bei nominaler Klasse .stringValue(...) -> Label
                String valStr = data.instance(i).stringValue(data.classIndex());
                distMap.put(valStr, distMap.getOrDefault(valStr, 0) + 1);
            }
            System.out.println("Verteilung: " + distMap);

            // 3) Parameterbereiche definieren
            //    In scikit-learn: max_depth=[None,10,20], min_samples_leaf=[1,5,10], max_features=[None,'sqrt','log2']
            //    Entsprechung in Weka RandomTree:
            //    -depth <int>    (0 => unbeschränkt)
            //    -M <int>        => minimal #inst pro Blatt
            //    -K <int>        => #Features pro Split (0 => alle)
            int numFeatures = data.numAttributes() - 1; // abzüglich Klasse

            Integer[] maxDepthVals = { null, 10, 20 };  // null => unbeschränkt
            Integer[] minSamplesLeafVals = { 1, 5, 10 };
            String[] maxFeaturesVals = { null, "sqrt", "log2" };

            // Sammeln für die 10 Wiederholungen
            List<Double> accuracies = new ArrayList<>();
            List<Map<String, Object>> bestParamsList = new ArrayList<>();

            // 4) 10 Wiederholungen
            for (int run = 0; run < 10; run++) {
                // Zufälliges Shufflen
                data.randomize(new Random(run));

                double bestAcc = 0.0;
                Map<String, Object> bestParamsInRun = null;

                // 5) "Grid Search" => Schleife über alle Kombis
                for (Integer md : maxDepthVals) {
                    for (Integer msl : minSamplesLeafVals) {
                        for (String mf : maxFeaturesVals) {

                            // -depth
                            int depth = (md == null) ? 0 : md;  // 0 => unbeschränkt
                            // -M
                            int minLeaf = msl;
                            // -K
                            int k;
                            if (mf == null) {
                                k = 0;  // alle Features
                            } else if ("sqrt".equals(mf)) {
                                k = (int) max(1, floor(sqrt(numFeatures)));
                            } else if ("log2".equals(mf)) {
                                k = (int) max(1, floor(log(numFeatures) / log(2)));
                            } else {
                                k = 0;
                            }

                            // RandomTree konfigurieren
                            String[] options = {
                                "-K", String.valueOf(k),
                                "-M", String.valueOf(minLeaf),
                                "-depth", String.valueOf(depth)
                            };
                            RandomTree rt = new RandomTree();
                            rt.setOptions(options);

                            // 10-fold CV
                            Evaluation eval = new Evaluation(data);
                            eval.crossValidateModel(rt, data, 10, new Random(1));
                            double accuracy = eval.pctCorrect() / 100.0;

                            if (accuracy > bestAcc) {
                                bestAcc = accuracy;
                                Map<String, Object> tmp = new LinkedHashMap<>();
                                tmp.put("max_depth", md);
                                tmp.put("min_samples_leaf", msl);
                                tmp.put("max_features", mf);
                                bestParamsInRun = tmp;
                            }
                        }
                    }
                }

                // Ergebnis dieses Durchlaufs
                accuracies.add(bestAcc);
                bestParamsList.add(bestParamsInRun);
            }

            // 6) Auswertung über 10 Läufe
            double mean = accuracies.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
            double variance = accuracies.stream()
                .mapToDouble(a -> (a - mean) * (a - mean))
                .sum() / accuracies.size();
            double stdDev = Math.sqrt(variance);

            // Bester Lauf
            int bestRunIdx = 0;
            double bestRunAcc = 0.0;
            for (int i = 0; i < accuracies.size(); i++) {
                if (accuracies.get(i) > bestRunAcc) {
                    bestRunAcc = accuracies.get(i);
                    bestRunIdx = i;
                }
            }
            Map<String, Object> bestRunParams = bestParamsList.get(bestRunIdx);

            System.out.printf("Durchschnittliche Genauigkeit von %s: %.4f (Std: %.4f)\n",
                    dataset, mean, stdDev);
            System.out.printf("Bester Durchlauf mit Accuracy: %.4f\n", bestRunAcc);
            System.out.println("Hyperparameter: " + bestRunParams);

            // Häufigste Parameterkombi
            Map<String, Integer> freqMap = new HashMap<>();
            for (Map<String, Object> bp : bestParamsList) {
                String key = mapToString(bp);
                freqMap.put(key, freqMap.getOrDefault(key, 0) + 1);
            }
            String mostCommonKey = null;
            int mostCommonCount = 0;
            for (Map.Entry<String, Integer> e : freqMap.entrySet()) {
                if (e.getValue() > mostCommonCount) {
                    mostCommonCount = e.getValue();
                    mostCommonKey = e.getKey();
                }
            }
            System.out.println("Häufigste Parameterkombination: " + mostCommonKey
                    + " (in " + mostCommonCount + " von 10 Durchläufen)\n");
        }
    }

    /**
     * Hilfsfunktion: Map -> String (z. B. {max_depth=10, min_samples_leaf=1, max_features=sqrt})
     */
    private static String mapToString(Map<String, Object> map) {
        return map.entrySet().stream()
            .map(e -> e.getKey() + "=" + e.getValue())
            .sorted()
            .collect(Collectors.joining(", ", "{", "}"));
    }
}
