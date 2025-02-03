package ml_praktikum_jagoetz_wkathari;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import java.io.File;
import java.util.*;

public class WekaRandomForestGridSearch {

    static String[] datasets = {
            //"ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\house_16H.csv",
            //"ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\jannis.csv",
            "ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\MagicTelescope.csv",
            //"ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\MiniBooNE.csv",
            "ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\pol.csv",
            "ml_praktikum_jagoetz_wkathari\\dataset\\clf_cat\\compas-two-years.csv",
            //"ml_praktikum_jagoetz_wkathari\\dataset\\clf_cat\\road-safety.csv",
            //"ml_praktikum_jagoetz_wkathari\\dataset\\clf_cat\\albert.csv",
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
        
        // Hyperparameter-Ranges
        String[] numTreesOptions = {"50", "100", "150"};
        String[] minLeafOptions = {"10", "30"};
        String[] maxFeaturesOptions = {"sqrt", "log2"};

        for (String dataset : datasets) {
            if (!targetColumns.containsKey(dataset)) {
                System.out.println("Überspringe " + dataset + " - keine Zielspalte definiert");
                continue;
            }
            
            String targetCol = targetColumns.get(dataset);
            System.out.println("\n=== Verarbeite " + dataset + " ===");

            // 1. Daten laden
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(dataset));
            Instances data = loader.getDataSet();

            // 2. Klassenattribut setzen
            int classIndex = data.attribute(targetCol).index();
            data.setClassIndex(classIndex);

            // 3. Numerische Klasse konvertieren
            if (data.classAttribute().isNumeric()) {
                NumericToNominal filter = new NumericToNominal();
                filter.setOptions(new String[]{"-R", String.valueOf(classIndex + 1)});
                filter.setInputFormat(data);
                data = Filter.useFilter(data, filter);
                data.setClassIndex(classIndex);
                System.out.println("Klasse zu nominal konvertiert");
            }

            // 4. Grid Search Durchläufe
            List<Double> accuracies = new ArrayList<>();
            List<Map<String, String>> bestParamsList = new ArrayList<>();

            for (int run = 0; run < 3; run++) {
                System.out.println("\n--- Run " + (run + 1) + " ---");
                long startTime = System.currentTimeMillis();

                // 4a. Daten shufflen
                Instances shuffledData = new Instances(data);
                shuffledData.randomize(new Random(run));

                // 4b. Grid Search
                double bestRunAcc = 0;
                Map<String, String> bestParams = new HashMap<>();
                int numFeatures = shuffledData.numAttributes() - 1;

                for (String nTrees : numTreesOptions) {
                    for (String minLeaf : minLeafOptions) {
                        for (String maxFeat : maxFeaturesOptions) {
                            
                            // Max Features berechnen
                            int k = 0;
                            if ("sqrt".equals(maxFeat)) {
                                k = (int) Math.max(1, Math.floor(Math.sqrt(numFeatures)));
                            } else if ("log2".equals(maxFeat)) {
                                k = (int) Math.max(1, Math.floor(Math.log(numFeatures) / Math.log(2)));
                            }

                            // Modell konfigurieren
                            RandomForest rf = new RandomForest();
                            rf.setOptions(new String[]{
                                "-I", nTrees,
                                "-M", minLeaf,
                                "-K", String.valueOf(k),
                                "-S", "42"  // Random Seed
                            });

                            // 10-fache Cross-Validation
                            Evaluation eval = new Evaluation(shuffledData);
                            eval.crossValidateModel(rf, shuffledData, 10, new Random(run));
                            double currentAcc = eval.pctCorrect();

                            if (currentAcc > bestRunAcc) {
                                bestRunAcc = currentAcc;
                                bestParams.clear();
                                bestParams.put("n_estimators", nTrees);
                                bestParams.put("min_samples_leaf", minLeaf);
                                bestParams.put("max_features", maxFeat);
                            }
                        }
                    }
                }

                // 4c. Ergebnisse speichern
                accuracies.add(bestRunAcc);
                bestParamsList.add(bestParams);
                double elapsed = (System.currentTimeMillis() - startTime) / 1000.0;

                System.out.printf("Beste Accuracy: %.2f%%\n", bestRunAcc);
                System.out.println("Parameter: " + bestParams);
                System.out.printf("Zeit: %.2fs\n", elapsed);
            }

            // 5. Statistische Auswertung
            System.out.println("\n===== Ergebniszusammenfassung =====");
            
            // Mittelwert & Standardabweichung
            double meanAcc = accuracies.stream()
                .mapToDouble(Double::doubleValue)
                .average().orElse(0);
            
            double stdAcc = Math.sqrt(accuracies.stream()
                .mapToDouble(a -> Math.pow(a - meanAcc, 2))
                .average().orElse(0));

            System.out.printf("Durchschnittliche Genauigkeit: %.2f%% ± %.2f%%\n", meanAcc, stdAcc);
            
            // Häufigste Parameterkombination
            Map<String, Integer> paramCount = new HashMap<>();
            for (Map<String, String> params : bestParamsList) {
                String key = params.toString();
                paramCount.put(key, paramCount.getOrDefault(key, 0) + 1);
            }
            
            String mostCommon = Collections.max(
                paramCount.entrySet(), 
                Map.Entry.comparingByValue()
            ).getKey();
            
            System.out.println("Häufigste Parameter: " + mostCommon);
        }
    }
}