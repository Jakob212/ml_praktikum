import itertools
import weka.core.jvm as jvm
import weka.core.converters as converters
from weka.classifiers import Classifier, Evaluation
import weka.core.classes as wcc
import numpy as np

def main():
    jvm.start()

    data = converters.load_any_file("ml_praktikum_jagoetz_wkathari\dataset\clf_cat\compas-two-years.arff")
    data.class_is_last()

    # Parameterräume (je nach J48-Optionen)
    # Hier z.B. 'C' (confidenceFactor) und 'M' (minNumObj)
    param_C = [0.001, 0.01, 0.1, 0.3, 0.5]
    param_M = [1, 5, 10, 20, 50]

    best_acc = 0.0
    best_params = (None, None)

    for c_val, m_val in itertools.product(param_C, param_M):
        # J48 aufsetzen mit den entsprechenden Optionen:
        # J48-Optionen:
        # -C <value> => confidence factor
        # -M <value> => minNumObj
        # (Weitere Optionen findest du in der WEKA-Doku)
        options = [
            "-C", str(c_val),
            "-M", str(m_val),
        ]
        j48 = Classifier(classname="weka.classifiers.trees.J48", options=options)

        # Auswertung per 10-fold CrossValidation
        evaluation = Evaluation(data)
        evaluation.crossvalidate_model(j48, data, 10, wcc.Random(1))

        acc = evaluation.percent_correct
        if acc > best_acc:
            best_acc = acc
            best_params = (c_val, m_val)

    print(f"Beste Accuracy: {best_acc:.2f}% mit C={best_params[0]}, M={best_params[1]}")
    jvm.stop()

if __name__ == "__main__":
    main()
