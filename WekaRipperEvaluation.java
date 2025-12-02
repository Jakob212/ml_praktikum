package ml_praktikum_jagoetz_wkathari;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.classifiers.rules.JRip;
import weka.classifiers.Evaluation;

import java.io.File;
import java.util.*;

public class WekaRipperEvaluation {

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
                System.out.println("**WARNUNG**: Kein Zielspalteneintrag für " + filename + ". Überspringe.");
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

            if (data.attribute(targetCol) == null) {
                System.out.println("**WARNUNG**: Zielspalte " + targetCol + " nicht gefunden in " + filename);
                continue;
            }
            int classIndex = data.attribute(targetCol).index();
            data.setClassIndex(classIndex);

            if (data.classAttribute().isNumeric()) {
                NumericToNominal numToNom = new NumericToNominal();
                numToNom.setAttributeIndices(String.valueOf(classIndex + 1));
                numToNom.setInputFormat(data);
                data = Filter.useFilter(data, numToNom);
                data.setClassIndex(classIndex);
            }

            System.out.println("Klassenverteilung:");
            int[] counts = data.attributeStats(data.classIndex()).nominalCounts;
            Enumeration<?> classValues = data.classAttribute().enumerateValues();
            int idx = 0;
            while (classValues.hasMoreElements()) {
                String val = (String) classValues.nextElement();
                System.out.println(val + ": " + counts[idx]);
                idx++;
            }

            List<Double> testAccuracies = new ArrayList<>();
            List<Double> cvAccuracies   = new ArrayList<>();
            List<Double> runTimes       = new ArrayList<>();

            for (int i = 0; i < 15; i++) {
                long startTime = System.nanoTime();

                Instances randData = new Instances(data);
                randData.randomize(new Random(i));

                int trainSize = (int) Math.round(randData.numInstances() * (2.0 / 3.0));
                int testSize  = randData.numInstances() - trainSize;
                Instances train = new Instances(randData, 0, trainSize);
                Instances test  = new Instances(randData, trainSize, testSize);

                Evaluation evalCV = new Evaluation(train);
                JRip jripCV = new JRip();
                jripCV.setSeed(42);
                jripCV.setOptimizations(2);
                jripCV.setFolds(3);
                evalCV.crossValidateModel(jripCV, train, 3, new Random(i));
                double cvAcc = evalCV.pctCorrect() / 100.0;
                cvAccuracies.add(cvAcc);

                JRip jrip = new JRip();
                jrip.setSeed(42);
                jrip.setOptimizations(2);
                jrip.setFolds(3);
                jrip.buildClassifier(train);

                Evaluation evalTest = new Evaluation(train);
                evalTest.evaluateModel(jrip, test);
                double testAcc = evalTest.pctCorrect() / 100.0;
                testAccuracies.add(testAcc);

                long endTime = System.nanoTime();
                double duration = (endTime - startTime) / 1e9;
                runTimes.add(duration);

                System.out.printf("Durchlauf %d Dauer: %.4fs%n", i + 1, duration);
            }

            System.out.println("\nMatrix mit den 15 Test-Accuracies:");
            System.out.print("[");
            for (int i = 0; i < testAccuracies.size(); i++) {
                System.out.printf("%.4f", testAccuracies.get(i));
                if (i < testAccuracies.size() - 1) {
                    System.out.print(", ");
                }
            }
            System.out.println("]");

            double meanTest = mean(testAccuracies);
            double stdTest  = stdDev(testAccuracies, meanTest);
            double meanCV   = mean(cvAccuracies);
            double stdCV    = stdDev(cvAccuracies, meanCV);
            double meanTime = mean(runTimes);
            double stdTime  = stdDev(runTimes, meanTime);

            System.out.printf("%nTest-Accuracy (M ± SD): %.4f ± %.4f%n", meanTest, stdTest);
            System.out.printf("CV-Accuracy   (M ± SD): %.4f ± %.4f%n", meanCV, stdCV);
            System.out.printf("Laufzeit      (M ± SD): %.4fs ± %.4fs%n", meanTime, stdTime);
        }
    }

    public static double mean(List<Double> values) {
        double sum = 0.0;
        for (Double d : values) sum += d;
        return values.isEmpty() ? 0.0 : sum / values.size();
    }

    public static double stdDev(List<Double> values, double mean) {
        if (values.size() < 2) return 0.0;
        double sum = 0.0;
        for (Double d : values) {
            double diff = d - mean;
            sum += diff * diff;
        }
        return Math.sqrt(sum / (values.size() - 1));
    }
}
