import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;
import weka.core.converters.CSVLoader;
import weka.core.Instances;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class WekaRandomForestHyper {

    public static void main(String[] args) throws Exception {
        // 1. Daten laden
        System.out.println("Daten werden geladen...");
        CSVLoader loader = new CSVLoader();
        File file = new File("../dat/covertype.csv");

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

        // Klassenverteilung anzeigen
        int[] classCounts = new int[data.numClasses()];
        for (int i = 0; i < data.numInstances(); i++) {
            classCounts[(int) data.instance(i).classValue()]++;
        }
        System.out.println("Klassen: ");
        for (int i = 0; i < classCounts.length; i++) {
            System.out.println("Klasse " + i + ": " + classCounts[i] + " Instanzen");
        }

        // 2. Hyperparameter-Gitter definieren
        int[] nEstimatorsOptions = {50, 100, 150};
        int[] minSamplesLeafOptions = {10, 30};
        String[] maxFeaturesOptions = {"sqrt", "log2"};

        List<Double> accuracies = new ArrayList<>();
        List<String> bestParams = new ArrayList<>();

        // 10 Durchläufe
        for (int i = 0; i < 10; i++) {
            System.out.println("\nDurchlauf " + (i + 1));

            // Daten zufällig shufflen
            data.randomize(new Random(i));

            double bestAccuracy = 0;
            String bestParamSet = "";

            // Hyperparameter-Kombinationen testen
            for (int nEstimators : nEstimatorsOptions) {
                for (int minSamplesLeaf : minSamplesLeafOptions) {
                    for (String maxFeatures : maxFeaturesOptions) {

                        RandomForest rf = new RandomForest();
                        rf.setOptions(new String[]{"-I", String.valueOf(nEstimators), "-num-slots", "1"});
                        rf.setNumExecutionSlots(1);

                        Evaluation eval = new Evaluation(data);
                        eval.crossValidateModel(rf, data, 10, new Random(i));

                        double accuracy = eval.pctCorrect() / 100.0;
                        if (accuracy > bestAccuracy) {
                            bestAccuracy = accuracy;
                            bestParamSet = String.format("n_estimators=%d, min_samples_leaf=%d, max_features=%s", nEstimators, minSamplesLeaf, maxFeatures);
                        }
                    }
                }
            }

            accuracies.add(bestAccuracy);
            bestParams.add(bestParamSet);
            System.out.printf("Beste Genauigkeit: %.4f mit Parametern: %s\n", bestAccuracy, bestParamSet);
        }

        // Durchschnittliche Ergebnisse berechnen
        double meanAccuracy = accuracies.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        System.out.printf("\nDurchschnittliche Genauigkeit: %.4f\n", meanAccuracy);
        System.out.println("Beste Hyperparameter pro Durchlauf:");
        for (String params : bestParams) {
            System.out.println(params);
        }
    }
}
