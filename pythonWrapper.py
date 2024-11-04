import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random
from weka.filters import Filter

# Starten der JVM
jvm.start()

# Pfad zur CSV-Datei
file_path = "dataset/clf_cat/eye_movements.csv"

# CSV-Datei laden
loader = Loader(classname="weka.core.converters.CSVLoader")
data = loader.load_file(file_path)
data.class_is_last()

# Filter zur Umwandlung numerischer Klassen in nominale
filter = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal", options=["-R", "last"])
filter.inputformat(data)
nominal_data = filter.filter(data)

# Erstelle und trainiere das Random Forest Modell
classifier = Classifier(classname="weka.classifiers.trees.RandomForest")
classifier.build_classifier(nominal_data)

# Bewertung mit 10-facher Kreuzvalidierung
evaluation = Evaluation(nominal_data)
evaluation.crossvalidate_model(classifier, nominal_data, 10, Random(1))

# Ausgabe der Ergebnisse
print("Summary:", evaluation.summary())
print("Accuracy:", evaluation.percent_correct)
print("Precision:", evaluation.weighted_precision)
print("Recall:", evaluation.weighted_recall)
print("F1-score:", evaluation.weighted_f_measure)

# JVM stoppen
jvm.stop()
