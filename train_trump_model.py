# train_trump_model.py
#
# Trainiert ein Deep-Learning-Modell (MLP) auf den
# aus den Swisslos-Logs extrahierten Trumpf-Daten.

import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


DATA_FILE = "Data/trump_train_sw.npz"
MODEL_FILE = "Data/trump_model_sw.joblib"


def main():
    data = np.load(DATA_FILE)
    X = data["X"]
    y = data["y"]

    print("Daten geladen:", X.shape, y.shape)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        max_iter=30,
        verbose=True,
    )

    print("Training startet...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print("Validation Accuracy:", acc)
    print(classification_report(y_val, y_pred))

    joblib.dump(clf, MODEL_FILE)
    print("Modell gespeichert als:", MODEL_FILE)


if __name__ == "__main__":
    main()
