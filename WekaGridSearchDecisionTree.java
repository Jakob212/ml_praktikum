package ml_praktikum_jagoetz_wkathari;

import weka.core.converters.CSVLoader;
import weka.core.Instances;
import weka.classifiers.trees.RandomTree;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.Filter;

import java.io.File;
import java.util.*;
import java.util.stream.Collectors;
import java.util.Locale;

import static java.lang.Math.*;

public class WekaGridSearchDecisionTree {

    static final String DATA_DIR = "dataset";

    static final String[] DATASET_FILES = {
        "clf_num/house_16H.csv",
        "clf_num/jannis.csv",
        "clf_num/MagicTelescope.csv",
        "clf_num/MiniBooNE.csv",
        "clf_num/pol.csv",
        "clf_cat/compas-two-years.csv",
        "clf_cat/road-safety.csv",
        "clf_cat/albert.csv",
        "clf_num/bank-marketing.csv",
        "clf_num/Bioresponse.csv",
        "clf_num/california.csv",
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

        Integer[] maxDepthVals = { null, 10, 20 };
        Integer[] minSamplesLeafVals = { 1, 5, 10 };
        String[] maxFeaturesVals = { null, "sqrt", "log2" };

        for (String relPath : DATASET_FILES) {

            File file = new File(DATA_DIR, relPath);
            String filename = file.getName();

            if (!TARGET_COLUMNS.containsKey(filename)) {
                System.out.println("Kein Zielspalteneintrag für " + filename + " - überspringe.");
                continue;
            }

            String targetColumn = TARGET_COLUMNS.get(filename);
            System.out.println("=== Datensatz: " + relPath + " ===");

            if (!file.exists()) {
                System.out.println("Datei nicht gefunden: " + file.getPath());
                continue;
            }

            CSVLoader loader = new CSVLoader();
            loader.setSource(file);
            Instances data = loader.getDataSet();

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

            if (data.classAttribute().isNumeric()) {
                NumericToNominal ntm = new NumericToNominal();
                ntm.setOptions(new String[] {"-R", String.valueOf(classIndex + 1)});
                ntm.setInputFormat(data);
                data = Filter.useFilter(data, ntm);
                data.setClassIndex(classIndex);
                System.out.println("Klasse war numerisch, umgewandelt zu nominal.");
            }

            List<Double> testAccuracies = new ArrayList<>();
            List<Double> runTimes = new ArrayList<>();
            List<Map<String,Object>> bestParamsList = new ArrayList<>();

            for (int run = 0; run < 10; run++) {
                long startTime = System.currentTimeMillis();

                Instances shuffled = new Instances(data);
                shuffled.randomize(new Random(run));

                int trainSize = (int) Math.round(shuffled.numInstances() * (2.0 / 3.0));
                Instances train = new Instances(shuffled, 0, trainSize);
                Instances test  = new Instances(shuffled, trainSize, shuffled.numInstances() - trainSize);

                double bestCVAcc = 0.0;
                String[] bestOptions = null;
                Map<String,Object> bestParamMap = null;

                int numFeatures = train.numAttributes() - 1;

                for (Integer md : maxDepthVals) {
                    for (Integer msl : minSamplesLeafVals) {
                        for (String mf : maxFeaturesVals) {
                            int depth = (md == null) ? 0 : md;
                            int minLeaf = msl;

                            int k;
                            if (mf == null) {
                                k = 0;
                            } else if ("sqrt".equals(mf)) {
                                k = (int) max(1, floor(sqrt(numFeatures)));
                            } else if ("log2".equals(mf)) {
                                k = (int) max(1, floor(log(numFeatures) / log(2)));
                            } else {
                                k = 0;
                            }

                            String[] opts = {
                                "-depth", String.valueOf(depth),
                                "-M", String.valueOf(minLeaf),
                                "-K", String.valueOf(k),
                                "-S", "42"
                            };
                            RandomTree rt = new RandomTree();
                            rt.setOptions(opts);

                            weka.classifiers.Evaluation evalCV =
                                    new weka.classifiers.Evaluation(train);
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

                RandomTree bestModel = new RandomTree();
                if (bestOptions != null) {
                    bestModel.setOptions(bestOptions);
                }
                bestModel.buildClassifier(train);

                weka.classifiers.Evaluation evalTest =
                        new weka.classifiers.Evaluation(train);
                evalTest.evaluateModel(bestModel, test);
                double testAcc = evalTest.pctCorrect() / 100.0;
                testAccuracies.add(testAcc);

                long endTime = System.currentTimeMillis();
                double elapsedSec = (endTime - startTime) / 1000.0;
                runTimes.add(elapsedSec);

                bestParamsList.add(bestParamMap);

                System.out.printf(
                    "Run %2d: Test-Acc=%.4f, Zeit=%.2fs, Params=%s%n",
                    run + 1, testAcc, elapsedSec,
                    (bestParamMap != null) ? bestParamMap.toString() : "null"
                );
            }

            System.out.println("\nMatrix (10 Test-Accuracies):");
            String matrixString = testAccuracies.stream()
                    .map(acc -> String.format(Locale.US, "%.4f", acc))
                    .collect(Collectors.joining(", "));
            System.out.println("[" + matrixString + "]");

            double meanAcc = mean(testAccuracies);
            double stdAcc = stddev(testAccuracies, meanAcc);
            double meanTime = mean(runTimes);
            double stdTime = stddev(runTimes, meanTime);

            System.out.printf("=> Durchschnittliche Accuracy: %.4f ± %.4f%n", meanAcc, stdAcc);
            System.out.printf("=> Durchschnittliche Zeit: %.2fs ± %.2fs%n", meanTime, stdTime);

            int bestRunIdx = 0;
            double bestAcc = 0.0;
            for (int i = 0; i < testAccuracies.size(); i++) {
                if (testAccuracies.get(i) > bestAcc) {
                    bestAcc = testAccuracies.get(i);
                    bestRunIdx = i;
                }
            }
            System.out.printf(
                "Bester Durchlauf: %d mit Acc=%.4f, Param=%s%n",
                bestRunIdx + 1, bestAcc, bestParamsList.get(bestRunIdx)
            );

            Map<String,Integer> freqMap = new HashMap<>();
            for (Map<String,Object> pm : bestParamsList) {
                if (pm == null) continue;
                String key = pm.entrySet().stream()
                        .map(e -> e.getKey() + "=" + e.getValue())
                        .sorted()
                        .collect(Collectors.joining(","));
                freqMap.put(key, freqMap.getOrDefault(key, 0) + 1);
            }
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
}
