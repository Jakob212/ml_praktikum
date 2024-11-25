import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;
import weka.core.converters.CSVLoader;
import weka.core.Instances;
import java.io.File;
import java.util.Random;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class WekaRandomForestExperiment {

    public static void main(String[] args) throws Exception {
        // 1. Daten laden
        System.out.println("Daten werden geladen...");
        CSVLoader loader = new CSVLoader();
        File file = new File("ml_praktikum_jagoetz_wkathari\\dataset\\clf_num\\jannis.csv");

        // Datei überprüfen
        if (!file.exists()) {
            System.out.println("Fehler: Datei nicht gefunden! Pfad: " + file.getAbsolutePath());
            return;
        }

        loader.setSource(file);
        Instances data = loader.getDataSet();
        // Klassenattribut in nominales Attribut umwandeln
        NumericToNominal convert = new NumericToNominal();
        convert.setAttributeIndices("" + (data.numAttributes())); // Konvertiere die letzte Spalte
        convert.setInputFormat(data);
        data = Filter.useFilter(data, convert);

        // Klassenattribut setzen (nach Konvertierung)
        data.setClassIndex(data.numAttributes() - 1);

        System.out.println("Daten geladen: " + data.numInstances() + " Instanzen, " + data.numAttributes() + " Attribute.");

        // Klassenattribut setzen (letzte Spalte)
        data.setClassIndex(data.numAttributes() - 1);

        // 2. Parameter
        int nExperiments = 100;
        double[] buildTimes = new double[nExperiments];
        double[] evaluateTimes = new double[nExperiments];
        double[] accuracies = new double[nExperiments];

        // 3. Experimente durchführen
        for (int seed = 0; seed < nExperiments; seed++) {
            System.out.println("\nExperiment " + (seed + 1) + " mit Seed " + seed);

            // 3.1 Zufälliges Teilen in Trainings- und Testdaten
            data.randomize(new Random(seed));
            int trainSize = (int) Math.round(data.numInstances() * 0.66);
            int testSize = data.numInstances() - trainSize;

            Instances train = new Instances(data, 0, trainSize);
            Instances test = new Instances(data, trainSize, testSize);

            // 3.2 Random Forest Modell erstellen
            RandomForest rf = new RandomForest();
            rf.setOptions(new String[]{"-I", "100"}); // Anzahl der Bäume auf 100 setzen
            rf.setMaxDepth(0); // Unbegrenzte Tiefe
            rf.setNumExecutionSlots(8); // Paralleles Training mit 8 Threads

            long startBuild = System.nanoTime();
            rf.buildClassifier(train); // Modell trainieren
            long endBuild = System.nanoTime();
            buildTimes[seed] = (endBuild - startBuild) / 1e9; // Zeit in Sekunden

            // 3.3 Modell evaluieren
            long startEvaluate = System.nanoTime();
            Evaluation eval = new Evaluation(train);
            eval.evaluateModel(rf, test);
            long endEvaluate = System.nanoTime();
            evaluateTimes[seed] = (endEvaluate - startEvaluate) / 1e9; // Zeit in Sekunden

            accuracies[seed] = eval.pctCorrect() / 100.0; // Genauigkeit in Prozent
            System.out.printf("Genauigkeit: %.2f%%\n", accuracies[seed] * 100);
        }

        // 4. Ergebnisse ausgeben
        System.out.println("\nErgebnisse der Experimente:");
        System.out.println("Trainingszeiten (build model):");
        for (double time : buildTimes) System.out.printf("%.4f ", time);
        System.out.println();

        System.out.println("Bewertungszeiten (evaluate model):");
        for (double time : evaluateTimes) System.out.printf("%.4f ", time);
        System.out.println();

        System.out.println("Genauigkeiten:");
        for (double acc : accuracies) System.out.printf("%.2f%% ", acc * 100);
        System.out.println();

        // Durchschnittswerte berechnen und ausgeben
        System.out.printf("\nDurchschnittliche Trainingszeit: %.4f Sekunden\n", average(buildTimes));
        System.out.printf("Durchschnittliche Bewertungszeit: %.4f Sekunden\n", average(evaluateTimes));
        System.out.printf("Durchschnittliche Genauigkeit: %.2f%%\n", average(accuracies) * 100);
    }

    // Hilfsmethode: Durchschnitt berechnen
    private static double average(double[] array) {
        double sum = 0;
        for (double value : array) sum += value;
        return sum / array.length;
    }
}
