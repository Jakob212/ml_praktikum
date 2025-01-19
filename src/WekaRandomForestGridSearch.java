import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

import java.io.File;
import java.util.Random;

public class WekaRandomForestGridSearch {
    public static void main(String[] args) throws Exception {
        // Pfad zur CSV-Datei anpassen
        String csvFile = "../dataset/clf_num/pol.csv";
                        

        // CSV laden
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(csvFile));
        Instances data = loader.getDataSet();
        
        // Letzte Spalte als Klassenattribut setzen
        data.setClassIndex(data.numAttributes() - 1);

        // Klassenattribut in nominal umwandeln (falls es numerisch ist)
        if (data.classAttribute().isNumeric()) {
            NumericToNominal convert = new NumericToNominal();
            convert.setAttributeIndices("last");
            convert.setInputFormat(data);
            data = Filter.useFilter(data, convert);
        }

        // Anzahl der Merkmale (für max_features = sqrt in sklearn)
        int numAttributes = data.numAttributes() - 1; // ohne Klassenattribut
        int sqrtFeatures = (int) Math.sqrt(numAttributes);

        // Hyperparameter-Gitter definieren
        // Entspricht grob den Parametern im Python-Code
        int[] nEstimatorsArray = {100};
        int[] maxDepthArray = {0, 10}; // 0 bedeutet kein Limit in WEKA
        int[] kFeaturesArray = {sqrtFeatures}; // sqrt der Anzahl der Features

        // Anzahl Folds für Cross-Validation
        int folds = 10;

        double bestAccuracy = -1.0;
        String bestParams = "";

        // Grid-Search manuell
        for (int nEst : nEstimatorsArray) {
            for (int depth : maxDepthArray) {
                for (int kFeat : kFeaturesArray) {
                    // RandomForest konfigurieren
                    RandomForest rf = new RandomForest();
                    rf.setNumIterations(nEst);  // -I Parameter (Anzahl Bäume)
                    rf.setMaxDepth(depth);      // -depth Parameter (0=kein Limit)
                    rf.setNumFeatures(kFeat);   // -K Parameter (Anzahl Features pro Split)

                    // Bewertung per Cross-Validation
                    Evaluation eval = new Evaluation(data);
                    eval.crossValidateModel(rf, data, folds, new Random(42));
                    
                    double accuracy = eval.pctCorrect() / 100.0; // Prozent in Dezimal
                    if (accuracy > bestAccuracy) {
                        bestAccuracy = accuracy;
                        bestParams = String.format("n_estimators=%d, max_depth=%s, K=%d",
                                nEst, (depth == 0 ? "None" : depth), kFeat);
                    }
                }
            }
        }

        System.out.println("Beste Genauigkeit: " + bestAccuracy);
        System.out.println("Beste Hyperparameter: " + bestParams);
    }
}
