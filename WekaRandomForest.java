package ml_praktikum_jagoetz_wkathari;

import weka.core.converters.CSVLoader;
import weka.core.Instances;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.io.File;
import java.util.*;
import java.util.stream.Collectors;
import java.util.Locale;

public class WekaRandomForest {

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

        for (String relPath : DATASET_FILES) {

            File file = new File(DATA_DIR, relPath);
            String filename = file.getName();

            if (!TARGET_COLUMNS.containsKey(filename)) {
                System.out.println("Kein Zielspalteneintrag für " + filename + " – überspringe.");
                continue;
            }

            String targetCol = TARGET_COLUMNS.get(filename);

            System.out.println("\n=== Verarbeite Datensatz: " + relPath + " ===");

            if (!file.exists()) {
                System.out.println("Datei nicht gefunden: " + file.getPath());
                continue;
            }

            CSVLoader loader = new CSVLoader();
            loader.setSource(file);
            Instances data = loader.getDataSet();

            int classIndex = -1;
            for (int i = 0; i < data.numAttributes(); i++) {
                if (data.attribute(i).name().equalsIgnoreCase(targetCol)) {
                    classIndex = i;
                    break;
                }
            }
            if (classIndex == -1) {
                System.out.println("Zielspalte '" + targetCol + "' nicht gefunden – überspringe.");
                continue;
            }
            data.setClassIndex(classIndex);

            if (data.classAttribute().isNumeric()) {
                NumericToNominal num2nom = new NumericToNominal();
                num2nom.setOptions(new String[]{"-R", String.valueOf(classIndex + 1)});
                num2nom.setInputFormat(data);
                data = Filter.useFilter(data, num2nom);
                data.setClassIndex(classIndex);
            }

            System.out.println("Klassenverteilung:");
            Map<String, Integer> classDist = new HashMap<>();
            for (int i = 0; i < data.numInstances(); i++) {
                String clsVal = data.instance(i).stringValue(data.classIndex());
                classDist.put(clsVal, classDist.getOrDefault(clsVal, 0) + 1);
            }
            System.out.println(classDist);

            List<Double> testAccuracies = new ArrayList<>();
            List<Double> cvAccuracies   = new ArrayList<>();
            List<Double> runTimes       = new ArrayList<>();

            int numTrees = 100;
            int maxDepth = 0;
            int minSamplesLeaf = 10;
            int numFeatures = (int) Math.floor(Math.sqrt(data.numAttributes() - 1));
            if (numFeatures < 1) numFeatures = 1;

            for (int run = 0; run < 15; run++) {
                long start = System.currentTimeMillis();

                Instances shuffledData = new Instances(data);
                shuffledData.randomize(new Random(run));
                int trainSize = (int) Math.round(shuffledData.numInstances() * (2.0 / 3.0));
                int testSize  = shuffledData.numInstances() - trainSize;

                Instances train = new Instances(shuffledData, 0, trainSize);
                Instances test  = new Instances(shuffledData, trainSize, testSize);

                RandomForest rf = new RandomForest();
                String[] options = {
                    "-I", String.valueOf(numTrees),
                    "-depth", String.valueOf(maxDepth),
                    "-K", String.valueOf(numFeatures),
                    "-num-slots", String.valueOf(Runtime.getRuntime().availableProcessors()),
                    "-M", String.valueOf(minSamplesLeaf)
                };
                rf.setOptions(options);

                Evaluation evalCV = new Evaluation(train);
                evalCV.crossValidateModel(rf, train, 3, new Random(run));
                double cvAcc = evalCV.pctCorrect() / 100.0;
                cvAccuracies.add(cvAcc);

                rf.buildClassifier(train);

                Evaluation evalTest = new Evaluation(train);
                evalTest.evaluateModel(rf, test);
                double testAcc = evalTest.pctCorrect() / 100.0;
                testAccuracies.add(testAcc);

                long end = System.currentTimeMillis();
                double elapsedSec = (end - start) / 1000.0;
                runTimes.add(elapsedSec);

                System.out.printf(Locale.US, "Durchlauf %2d Dauer: %.4fs%n", run + 1, elapsedSec);
            }

            System.out.println("\nMatrix (15 Test-Accuracies):");
            String matrixString = testAccuracies.stream()
                    .map(acc -> String.format(Locale.US, "%.4f", acc))
                    .collect(Collectors.joining(", "));
            System.out.println("[" + matrixString + "]");

            double meanTest  = mean(testAccuracies);
            double stdTest   = stddev(testAccuracies, meanTest);
            double meanCV    = mean(cvAccuracies);
            double stdCV     = stddev(cvAccuracies, meanCV);
            double meanTime  = mean(runTimes);
            double stdTime   = stddev(runTimes, meanTime);

            System.out.printf(Locale.US,
                    "Test-Accuracy (15 Wdh.): %.4f ± %.4f%n", meanTest, stdTest);
            System.out.printf(Locale.US,
                    "CV-Accuracy   (M ± SD): %.4f ± %.4f%n", meanCV, stdCV);
            System.out.printf(Locale.US,
                    "Laufzeit      (M ± SD): %.4fs ± %.4fs%n", meanTime, stdTime);
        }

        System.out.println("Fertig.");
    }

    private static double mean(List<Double> values) {
        if (values.isEmpty()) return 0.0;
        double sum = 0.0;
        for (double v : values) sum += v;
        return sum / values.size();
    }

    private static double stddev(List<Double> values, double mean) {
        if (values.size() < 2) return 0.0;
        double sumSq = 0.0;
        for (double v : values) {
            double diff = v - mean;
            sumSq += diff * diff;
        }
        return Math.sqrt(sumSq / (values.size() - 1));
    }
}
