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
        // Ihre Dataset-Pfade hier einfügen
    };

    static Map<String, String> targetColumns = new HashMap<>();
    static {
        // Target-Spalten Mapping hier einfügen
    }

    public static void main(String[] args) throws Exception {
        
        // Parameterbereiche
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

            // 1) Daten laden
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(dataset));
            Instances data = loader.getDataSet();

            // 2) Klassenattribut setzen
            int classIndex = data.attribute(targetCol).index();
            data.setClassIndex(classIndex);

            // 3) Numerische Klasse konvertieren
            if (data.classAttribute().isNumeric()) {
                NumericToNominal filter = new NumericToNominal();
                filter.setOptions(new String[]{"-R", String.valueOf(classIndex + 1)});
                filter.setInputFormat(data);
                data = Filter.useFilter(data, filter);
                data.setClassIndex(classIndex);
            }

            // 4) Grid Search Durchläufe
            List<Double> bestAccuracies = new ArrayList<>();
            List<Map<String,String>> bestParams = new ArrayList<>();
            
            for (int run = 0; run < 3; run++) {
                System.out.println("\n--- Run " + (run+1) + " ---");
                
                // 4a) Daten shufflen
                Instances shuffledData = new Instances(data);
                shuffledData.randomize(new Random(run));
                
                // 4b) Parameterkombinationen testen
                double bestRunAcc = 0;
                Map<String,String> bestRunParams = new HashMap<>();
                int numFeatures = shuffledData.numAttributes() - 1;

                for (String nTrees : numTreesOptions) {
                    for (String minLeaf : minLeafOptions) {
                        for (String maxFeat : maxFeaturesOptions) {
                            
                            // Max Features berechnen
                            int k = 0;
                            if ("sqrt".equals(maxFeat)) {
                                k = (int) Math.floor(Math.sqrt(numFeatures));
                            } else if ("log2".equals(maxFeat)) {
                                k = (int) Math.floor(Math.log(numFeatures)/Math.log(2));
                            }
                            k = Math.max(1, k);

                            // 4c) Modell konfigurieren
                            RandomForest rf = new RandomForest();
                            rf.setOptions(new String[]{
                                "-I", nTrees,
                                "-M", minLeaf,
                                "-K", String.valueOf(k),
                                "-S", "42"
                            });

                            // 4d) 10-fache Cross-Validation
                            Evaluation eval = new Evaluation(shuffledData);
                            eval.crossValidateModel(rf, shuffledData, 10, new Random(run));
                            double acc = eval.pctCorrect();

                            // 4e) Beste Kombination merken
                            if (acc > bestRunAcc) {
                                bestRunAcc = acc;
                                bestRunParams.clear();
                                bestRunParams.put("n_estimators", nTrees);
                                bestRunParams.put("min_samples_leaf", minLeaf);
                                bestRunParams.put("max_features", maxFeat);
                            }
                        }
                    }
                }
                
                // 4f) Run-Ergebnis speichern
                bestAccuracies.add(bestRunAcc);
                bestParams.add(bestRunParams);
                System.out.printf("Beste Kombination: %s (Acc: %.2f%%)%n", 
                    bestRunParams.toString(), bestRunAcc);
            }

            // 5) Statistische Auswertung
            double meanAcc = bestAccuracies.stream()
                .mapToDouble(Double::doubleValue)
                .average().orElse(0);
            
            double stdAcc = Math.sqrt(bestAccuracies.stream()
                .mapToDouble(a -> Math.pow(a - meanAcc, 2))
                .average().orElse(0));

            System.out.println("\n===== Ergebnis =====");
            System.out.printf("Durchschnittliche Genauigkeit: %.2f ± %.2f%%%n", meanAcc, stdAcc);
            System.out.println("Beste Parameterkombinationen:");
            bestParams.forEach(System.out::println);
        }
    }
}