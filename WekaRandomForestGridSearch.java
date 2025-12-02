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

    static final String DATA_DIR = "dataset";

    static final String[] DATASET_FILES = {
        "clf_num/credit.csv",
        "clf_num/default-of-credit-card-clients.csv",
        "clf_num/Diabetes130US.csv",
        "clf_num/electricity.csv",
        "clf_num/eye_movements.csv",
        "clf_num/heloc.csv"
    };

    static final Map<String, String> TARGET_COLUMNS = new HashMap<>();
    static {
        TARGET_COLUMNS.put("house_16H.csv", "binaryClass");
        TARGET_COLUMNS.put("jannis.csv", "class");
        TARGET_COLUMNS.put("MagicTelescope.csv", "class");
        TARGET_COLUMNS.put("MiniBooNE.csv", "signal");
        TARGET_COLUMNS.put("pol.csv", "binaryClass");
        TARGET_COLUMNS.put("compas-two-years.csv", "twoyearrecid");
        TARGET_COLUMNS.put("road-safety.csv", "SexofDriver");
        TARGET_COLUMNS.put("albert.csv", "class");
        TARGET_COLUMNS.put("bank-marketing.csv", "Class");
        TARGET_COLUMNS.put("Bioresponse.csv", "target");
        TARGET_COLUMNS.put("california.csv", "price_above_median");
        TARGET_COLUMNS.put("credit.csv", "SeriousDlqin2yrs");
        TARGET_COLUMNS.put("default-of-credit-card-clients.csv", "y");
        TARGET_COLUMNS.put("Diabetes130US.csv", "readmitted");
        TARGET_COLUMNS.put("electricity.csv", "class");
        TARGET_COLUMNS.put("eye_movements.csv", "label");
        TARGET_COLUMNS.put("heloc.csv", "RiskPerformance");
    }

    public static void main(String[] args) throws Exception {

        String[] numTreesOptions = {"50", "100", "150"};
        String[] minLeafOptions = {"10", "30"};
        String[] maxFeaturesOptions = {"sqrt", "log2"};

        for (String relPath : DATASET_FILES) {

            File file = new File(DATA_DIR, relPath);
            String filename = file.getName();

            if (!TARGET_COLUMNS.containsKey(filename)) {
                System.out.println("Überspringe " + filename + " – keine Zielspalte definiert");
                continue;
            }

            String targetCol = TARGET_COLUMNS.get(filename);
            System.out.println("\n=== Verarbeite " + relPath + " ===");

            if (!file.exists()) {
                System.out.println("Datei nicht gefunden: " + file.getPath());
                continue;
            }

            CSVLoader loader = new CSVLoader();
            loader.setSource(file);
            Instances data = loader.getDataSet();

            if (data.attribute(targetCol) == null) {
                System.out.println("Zielattribut " + targetCol + " nicht gefunden – überspringe.");
                continue;
            }
            int classIndex = data.attribute(targetCol).index();
            data.setClassIndex(classIndex);

            if (data.classAttribute().isNumeric()) {
                NumericToNominal filter = new NumericToNominal();
                filter.setOptions(new String[]{"-R", String.valueOf(classIndex + 1)});
                filter.setInputFormat(data);
                data = Filter.useFilter(data, filter);
                data.setClassIndex(classIndex);
                System.out.println("Klasse zu nominal konvertiert");
            }

            List<Double> accuracies = new ArrayList<>();
            List<Map<String, String>> bestParamsList = new ArrayList<>();

            for (int run = 0; run < 3; run++) {
                System.out.println("\n--- Run " + (run + 1) + " ---");
                long startTime = System.currentTimeMillis();

                Instances shuffledData = new Instances(data);
                shuffledData.randomize(new Random(run));

                double bestRunAcc = 0.0;
                Map<String, String> bestParams = new HashMap<>();
                int numFeatures = shuffledData.numAttributes() - 1;

                for (String nTrees : numTreesOptions) {
                    for (String minLeaf : minLeafOptions) {
                        for (String maxFeat : maxFeaturesOptions) {

                            int k = 0;
                            if ("sqrt".equals(maxFeat)) {
                                k = (int) Math.max(1, Math.floor(Math.sqrt(numFeatures)));
                            } else if ("log2".equals(maxFeat)) {
                                k = (int) Math.max(1, Math.floor(Math.log(numFeatures) / Math.log(2)));
                            }

                            RandomForest rf = new RandomForest();
                            rf.setOptions(new String[]{
                                "-I", nTrees,
                                "-M", minLeaf,
                                "-K", String.valueOf(k),
                                "-S", "42"
                            });

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

                accuracies.add(bestRunAcc);
                bestParamsList.add(bestParams);

                double elapsed = (System.currentTimeMillis() - startTime) / 1000.0;
                System.out.printf("Beste Accuracy: %.2f%%%n", bestRunAcc);
                System.out.println("Parameter: " + bestParams);
                System.out.printf("Zeit: %.2fs%n", elapsed);
            }

            System.out.println("\n===== Ergebniszusammenfassung =====");

            double meanAcc = accuracies.stream()
                    .mapToDouble(Double::doubleValue)
                    .average()
                    .orElse(0.0);

            double stdAcc = Math.sqrt(
                    accuracies.stream()
                            .mapToDouble(a -> Math.pow(a - meanAcc, 2))
                            .average()
                            .orElse(0.0)
            );

            System.out.printf("Durchschnittliche Genauigkeit: %.2f%% ± %.2f%%%n", meanAcc, stdAcc);

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
