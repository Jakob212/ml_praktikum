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

    // List of CSV datasets
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

    // Mapping from each CSV path to the target column name
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

        // Optional: write results to a file:
        // PrintStream outStream = new PrintStream("random_forest_results.txt", "UTF-8");
        // System.setOut(outStream);

        // Iterate over all datasets
        for (String dataset : datasets) {

            // Check if we have a target column for this dataset
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

            // 2) Klassenattribut setzen
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

            // 2a) Falls die Klasse numerisch ist -> NumericToNominal
            if (data.classAttribute().isNumeric()) {
                System.out.println("Klasse ist numerisch, wandle in nominal um ...");
                NumericToNominal num2nom = new NumericToNominal();
                // Weka-Filter ist 1-basiert
                num2nom.setOptions(new String[]{"-R", String.valueOf(classIndex + 1)});
                num2nom.setInputFormat(data);
                data = Filter.useFilter(data, num2nom);
                // Klasse erneut setzen
                data.setClassIndex(classIndex);
            }

            // Klassenverteilung ausgeben
            System.out.println("Klassenverteilung:");
            Map<String, Integer> classDist = new HashMap<>();
            for (int i = 0; i < data.numInstances(); i++) {
                String clsVal = data.instance(i).stringValue(data.classIndex());
                classDist.put(clsVal, classDist.getOrDefault(clsVal, 0) + 1);
            }
            System.out.println(classDist);

            // Wir sammeln die Ergebnisse:
            List<Double> testAccuracies = new ArrayList<>();
            List<Double> cvAccuracies = new ArrayList<>();
            List<Double> runTimes = new ArrayList<>();

            // Parameter wie im Python-Code
            int numTrees = 100;
            int maxDepth = 0; // 0 => unbeschränkt
            int minSamplesLeaf = 10;
            // sqrt => K = floor(sqrt(#attribute - 1))
            int numFeatures = (int) Math.floor(Math.sqrt(data.numAttributes() - 1));
            if (numFeatures < 1) {
                numFeatures = 1;  // Sicherheit
            }

            // 15 Wiederholungen
            for (int run = 0; run < 15; run++) {
                long start = System.currentTimeMillis();

                // a) Split in Train/Test (2/3 : 1/3) mit seed = run
                Instances shuffledData = new Instances(data);
                shuffledData.randomize(new Random(run));
                int trainSize = (int) Math.round(shuffledData.numInstances() * (2.0/3.0));
                int testSize = shuffledData.numInstances() - trainSize;

                Instances train = new Instances(shuffledData, 0, trainSize);
                Instances test = new Instances(shuffledData, trainSize, testSize);

                // b) RandomForest konfigurieren
                RandomForest rf = new RandomForest();
                String[] options = {
                    "-I", String.valueOf(numTrees),   // number of trees
                    "-depth", String.valueOf(maxDepth),
                    "-K", String.valueOf(numFeatures),
                    "-num-slots", String.valueOf(Runtime.getRuntime().availableProcessors()),
                    "-M", String.valueOf(minSamplesLeaf)
                };
                rf.setOptions(options);

                // c) 3-fach CV auf dem Trainingsset
                Evaluation evalCV = new Evaluation(train);
                evalCV.crossValidateModel(rf, train, 3, new Random(run));
                double cvAcc = evalCV.pctCorrect() / 100.0;
                cvAccuracies.add(cvAcc);

                // d) Auf komplettes Training fitten
                rf.buildClassifier(train);

                // e) Test Accuracy messen
                Evaluation evalTest = new Evaluation(train);
                evalTest.evaluateModel(rf, test);
                double testAcc = evalTest.pctCorrect() / 100.0;
                testAccuracies.add(testAcc);

                long end = System.currentTimeMillis();
                double elapsedSec = (end - start) / 1000.0;
                runTimes.add(elapsedSec);

                System.out.printf("Durchlauf %2d Dauer: %.4fs\n", run+1, elapsedSec);
            }

            // --- Hier geben wir die "Matrix" (Liste) aller 15 Test-Accuracies kommasepariert aus ---
            System.out.println("\nMatrix (Liste) der 15 Test-Accuracies kommasepariert:");
            String matrixString = testAccuracies.stream()
                    .map(acc -> String.format(Locale.US, "%.4f", acc))
                    .collect(Collectors.joining(", "));
            System.out.println("[" + matrixString + "]");

            // 6) Ergebnisstatistik
            double meanTest = mean(testAccuracies);
            double stdTest = stddev(testAccuracies, meanTest);
            System.out.printf("Test-Accuracy (15 Wiederholungen): %.4f ± %.4f\n", meanTest, stdTest);

            double meanCV = mean(cvAccuracies);
            double stdCV = stddev(cvAccuracies, meanCV);
            System.out.printf("Innere CV-Accuracy (3-fach, Durchschnitt): %.4f ± %.4f\n", meanCV, stdCV);

            double meanTime = mean(runTimes);
            double stdTime = stddev(runTimes, meanTime);
            System.out.printf("Durchschnittliche Dauer: %.4fs ± %.4fs\n", meanTime, stdTime);

            // -- t-Test-Block entfernt --

        }

        System.out.println("Fertig! Alle Ausgaben erscheinen hier in der Konsole.");
    }

    // Mittelwert einer Liste
    private static double mean(List<Double> values) {
        if (values.isEmpty()) return 0.0;
        double sum = 0.0;
        for (double v : values) {
            sum += v;
        }
        return sum / values.size();
    }

    // Stichproben-Standardabweichung
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
