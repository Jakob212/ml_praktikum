package ml_praktikum_jagoetz_wkathari;

import weka.core.converters.CSVLoader;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.classifiers.trees.RandomTree;
import weka.classifiers.Evaluation;

import java.io.File;
import java.util.*;
import java.util.stream.Collectors;
import java.util.Locale;

public class WekaDecisionTree {

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

    static final int MIN_SAMPLES_LEAF = 10;
    static final int SEED = 42;
    static final String MAX_DEPTH = "0";

    public static void main(String[] args) throws Exception {

        for (String relPath : DATASET_FILES) {

            File file = new File(DATA_DIR, relPath);
            String filename = file.getName();

            if (!TARGET_COLUMNS.containsKey(filename)) {
                System.out.println("Kein Target-Mapping für " + filename + ", übersprungen.");
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
                System.out.println("Zielspalte '" + targetCol + "' nicht gefunden, übersprungen.");
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

            Map<String, Integer> classDist = new HashMap<>();
            for (int i = 0; i < data.numInstances(); i++) {
                String lbl = data.instance(i).stringValue(data.classIndex());
                classDist.put(lbl, classDist.getOrDefault(lbl, 0) + 1);
            }
            System.out.println("Klassenverteilung: " + classDist);

            List<Double> testAccuracies = new ArrayList<>();
            List<Double> cvAccuracies   = new ArrayList<>();
            List<Double> runTimes       = new ArrayList<>();

            int numAttributes = data.numAttributes() - 1;
            int kFeatures = (int) Math.floor(Math.sqrt(numAttributes));
            if (kFeatures < 1) kFeatures = 1;

            for (int run = 0; run < 15; run++) {
                long start = System.currentTimeMillis();

                Instances shuffled = new Instances(data);
                shuffled.randomize(new Random(run));

                int trainSize = (int) Math.round(shuffled.numInstances() * (2.0 / 3.0));
                int testSize  = shuffled.numInstances() - trainSize;

                Instances train = new Instances(shuffled, 0, trainSize);
                Instances test  = new Instances(shuffled, trainSize, testSize);

                String[] options = {
                    "-depth", MAX_DEPTH,
                    "-M", String.valueOf(MIN_SAMPLES_LEAF),
                    "-K", String.valueOf(kFeatures),
                    "-S", String.valueOf(SEED)
                };
                RandomTree rt = new RandomTree();
                rt.setOptions(options);

                Evaluation evalCV = new Evaluation(train);
                evalCV.crossValidateModel(rt, train, 3, new Random(run));
                double cvAcc = evalCV.pctCorrect() / 100.0;
                cvAccuracies.add(cvAcc);

                rt.buildClassifier(train);

                Evaluation evalTest = new Evaluation(train);
                evalTest.evaluateModel(rt, test);
                double testAcc = evalTest.pctCorrect() / 100.0;
                testAccuracies.add(testAcc);

                long end = System.currentTimeMillis();
                double sec = (end - start) / 1000.0;
                runTimes.add(sec);

                System.out.printf("Durchlauf %2d Dauer: %.4fs%n", run + 1, sec);
            }

            System.out.println("\nMatrix (15 Test-Accuracies):");
            String matrixString = testAccuracies.stream()
                    .map(acc -> String.format(Locale.US, "%.4f", acc))
                    .collect(Collectors.joining(", "));
            System.out.println("[" + matrixString + "]");

            double meanTest   = mean(testAccuracies);
            double stdTest    = stddev(testAccuracies, meanTest);
            System.out.printf("Test-Accuracy: %.4f ± %.4f%n", meanTest, stdTest);

            double meanCV = mean(cvAccuracies);
            double stdCV  = stddev(cvAccuracies, meanCV);
            System.out.printf("CV-Accuracy:   %.4f ± %.4f%n", meanCV, stdCV);

            double meanTime = mean(runTimes);
            double stdTime  = stddev(runTimes, meanTime);
            System.out.printf("Laufzeit:      %.4fs ± %.4fs%n", meanTime, stdTime);
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
