"""
ml_model/train_classifier.py
Данные берутся из dataset.csv.
"""

import os
import csv
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

CLASS_NAMES  = {0: "Штатная", 1: "Тех. сбой", 2: "АВАРИЯ"}
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "event_classifier.pkl")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.csv")


def load_dataset() -> list:
    """Читает обучающие данные из dataset.csv."""
    with open(DATASET_PATH, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [
            (row["текст"], int(row["класс_id"]))
            for row in reader
            if row["текст"].strip()
        ]


def train_and_save() -> dict:
    data   = load_dataset()
    texts  = [t for t, _ in data]
    labels = [l for _, l in data]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=8000,
            sublinear_tf=True,
            min_df=1,
        )),
        ("clf", LogisticRegression(
            C=5.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        )),
    ])

    pipeline.fit(X_train, y_train)

    cv_scores = cross_val_score(pipeline, texts, labels, cv=5, scoring="f1_macro")
    y_pred    = pipeline.predict(X_test)

    report = classification_report(
        y_test, y_pred,
        target_names=list(CLASS_NAMES.values()),
        output_dict=True,
    )

    joblib.dump(pipeline, MODEL_PATH)

    return {
        "cv_f1_mean":    float(np.mean(cv_scores)),
        "cv_f1_std":     float(np.std(cv_scores)),
        "test_report":   report,
        "model_path":    MODEL_PATH,
        "train_size":    len(X_train),
        "test_size":     len(X_test),
        "total_samples": len(texts),
    }


def load_model():
    if not os.path.exists(MODEL_PATH):
        train_and_save()
    return joblib.load(MODEL_PATH)


def predict(text: str, model=None) -> dict:
    if model is None:
        model = load_model()
    pred  = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    return {
        "class_id":      int(pred),
        "class_name":    CLASS_NAMES[int(pred)],
        "confidence":    float(proba[pred]),
        "probabilities": {CLASS_NAMES[i]: float(p) for i, p in enumerate(proba)},
    }


if __name__ == "__main__":
    data = load_dataset()
    print(f"Датасет: {len(data)} примеров из {DATASET_PATH}")
    by_cls = {}
    for _, l in data:
        by_cls[l] = by_cls.get(l, 0) + 1
    for k, v in sorted(by_cls.items()):
        print(f"  {CLASS_NAMES[k]}: {v}")

    print("\nОбучаем модель...")
    result = train_and_save()
    print(f"CV F1 (macro): {result['cv_f1_mean']:.3f} ± {result['cv_f1_std']:.3f}")
    print("\nОтчёт на тестовой выборке:")
    for cls, metrics in result["test_report"].items():
        if isinstance(metrics, dict):
            print(f"  {cls}: precision={metrics['precision']:.2f}  "
                  f"recall={metrics['recall']:.2f}  f1={metrics['f1-score']:.2f}")
    print(f"\nМодель сохранена: {result['model_path']}")

