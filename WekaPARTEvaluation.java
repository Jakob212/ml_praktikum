package ml_praktikum_jagoetz_wkathari;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.classifiers.rules.PART;
import weka.classifiers.Evaluation;

import java.io.File;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class WekaPARTEvaluation {

    public static void main(String[] args) throws Exception {
        // Liste der Datensätze (Pfade anpassen)
        String[] datasets = {
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

        // Mapping: Datensatzpfad -> Name der Zielspalte
        Map<String, String> targetColumns = new HashMap<>();
        targetColumns.put("ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\house_16H.csv", "binaryClass");
        targetColumns.put("ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\jannis.csv", "class");
        targetColumns.put("ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\MagicTelescope.csv", "class");
        targetColumns.put("ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\MiniBooNE.csv", "signal");
        targetColumns.put("ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\pol.csv", "binaryClass");
        targetColumns.put("ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\Higgs.csv", "target");
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

        // Verarbeitung jedes Datensatzes
        for (String dataset : datasets) {
            if (!targetColumns.containsKey(dataset)) {
                System.out.println("**WARNUNG**: Kein Zielspalteneintrag für " + dataset + ". Überspringe diesen Datensatz.");
                continue;
            }
            
            String targetCol = targetColumns.get(dataset);
            System.out.println("\n=== Verarbeite Datensatz: " + dataset + " ===");

            // CSV-Datei laden
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(dataset));
            Instances data = loader.getDataSet();

            // Zielattribut anhand des Namens suchen und als Klasse festlegen
            int classIndex = data.attribute(targetCol) != null ? data.attribute(targetCol).index() : -1;
            if (classIndex == -1) {
                System.out.println("**WARNUNG**: Zielspalte " + targetCol + " nicht gefunden in " + dataset);
                continue;
            }
            data.setClassIndex(classIndex);

            // Falls das Zielattribut numerisch ist, in nominal konvertieren
            if (data.classAttribute().isNumeric()) {
                NumericToNominal numToNom = new NumericToNominal();
                // WEKA verwendet 1-basierte Indizes
                numToNom.setAttributeIndices("" + (classIndex + 1));
                numToNom.setInputFormat(data);
                data = Filter.useFilter(data, numToNom);
            }

            // Ausgabe der Klassenverteilung
            System.out.println("Klassenverteilung:");
            int[] counts = data.attributeStats(data.classIndex()).nominalCounts;
            Enumeration<?> classValues = data.classAttribute().enumerateValues();
            int idx = 0;
            while (classValues.hasMoreElements()) {
                String val = (String) classValues.nextElement();
                System.out.println(val + ": " + counts[idx]);
                idx++;
            }

            // Listen für Ergebnisse über 15 Durchläufe
            List<Double> testAccuracies = new ArrayList<>();
            List<Double> cvAccuracies = new ArrayList<>();
            List<Double> runTimes = new ArrayList<>();

            // 15 Durchläufe (mit unterschiedlichem Zufallssamen für die Aufteilung)
            for (int i = 0; i < 15; i++) {
                long startTime = System.nanoTime();

                // Zufällige Durchmischung der Daten (Seed = i)
                Instances randData = new Instances(data);
                randData.randomize(new Random(i));

                // Aufteilen in Training (2/3) und Test (1/3)
                int trainSize = (int) Math.round(randData.numInstances() * (2.0 / 3));
                int testSize = randData.numInstances() - trainSize;
                Instances train = new Instances(randData, 0, trainSize);
                Instances test = new Instances(randData, trainSize, testSize);

                // Kreuzvalidierung (3-fach) auf den Trainingsdaten mit PART
                Evaluation evalCV = new Evaluation(train);
                PART partCV = new PART();
                // Setze Parameter (hier: exemplarisch – anpassbar!)
                partCV.setConfidenceFactor(0.33f); // analog zu prune_size=0.33
                partCV.setMinNumObj(2);            // minimale Anzahl Instanzen pro Regel
                partCV.setNumFolds(3);             // für reduziertes Fehler-Pruning (3-Fold)
                evalCV.crossValidateModel(partCV, train, 3, new Random(i));
                double cvAcc = evalCV.pctCorrect() / 100.0;
                cvAccuracies.add(cvAcc);

                // Training des PART-Klassifikators auf den Trainingsdaten
                PART part = new PART();
                part.setConfidenceFactor(0.33f);
                part.setMinNumObj(2);
                part.setNumFolds(3);
                part.buildClassifier(train);

                // Evaluierung auf den Testdaten
                Evaluation evalTest = new Evaluation(train);
                evalTest.evaluateModel(part, test);
                double testAcc = evalTest.pctCorrect() / 100.0;
                testAccuracies.add(testAcc);

                long endTime = System.nanoTime();
                double duration = (endTime - startTime) / 1e9; // in Sekunden
                runTimes.add(duration);

                System.out.printf("Durchlauf %d Dauer: %.4fs%n", i + 1, duration);
            }

            // Ausgabe der Test-Accuracies als Matrix (1x15)
            System.out.println("\nMatrix mit den 15 Test-Accuracies:");
            System.out.print("[");
            for (int i = 0; i < testAccuracies.size(); i++) {
                System.out.printf("%.4f", testAccuracies.get(i));
                if (i < testAccuracies.size() - 1) {
                    System.out.print(", ");
                }
            }
            System.out.println("]");

            // Berechnung von Mittelwert und Standardabweichung
            double meanTest = mean(testAccuracies);
            double stdTest = stdDev(testAccuracies, meanTest);
            double meanCV = mean(cvAccuracies);
            double stdCV = stdDev(cvAccuracies, meanCV);
            double meanTime = mean(runTimes);
            double stdTime = stdDev(runTimes, meanTime);

            System.out.printf("%nTest-Accuracy (M ± SD): %.4f ± %.4f%n", meanTest, stdTest);
            System.out.printf("CV-Accuracy (M ± SD): %.4f ± %.4f%n", meanCV, stdCV);
            System.out.printf("Laufzeit (M ± SD): %.4fs ± %.4fs%n", meanTime, stdTime);
        }
    }

    // Berechnet den Mittelwert einer Liste von Double-Werten
    public static double mean(List<Double> values) {
        double sum = 0;
        for (Double d : values) {
            sum += d;
        }
        return sum / values.size();
    }

    // Berechnet die Standardabweichung (Stichprobe) einer Liste von Double-Werten
    public static double stdDev(List<Double> values, double mean) {
        double sum = 0;
        for (Double d : values) {
            sum += Math.pow(d - mean, 2);
        }
        return Math.sqrt(sum / (values.size() - 1));
    }
}
