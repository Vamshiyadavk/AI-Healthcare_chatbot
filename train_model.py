"""
train_model.py
==============
Run this once to train and save the model:
    python train_model.py

What it does:
  1. Reads disease_symptoms.csv  (if found, uses it directly)
  2. Augments each disease into 14 training rows
  3. Trains TF-IDF + Calibrated LinearSVC + LogisticRegression ensemble
  4. Saves disease_model.pkl and tfidf_vectorizer.pkl
"""

import os, logging, random, joblib, warnings
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Synonym dictionary for augmentation ───────────────────────────────────
SYNONYMS = {
    "fatigue":             ["tiredness", "exhaustion", "lethargy"],
    "headache":            ["head pain", "head ache", "cephalgia"],
    "fever":               ["high temperature", "pyrexia", "elevated temperature"],
    "nausea":              ["feeling sick", "queasiness", "stomach upset"],
    "vomiting":            ["throwing up", "emesis", "retching"],
    "diarrhea":            ["loose stools", "watery stools", "loose motions"],
    "cough":               ["coughing", "dry cough", "persistent cough"],
    "rash":                ["skin rash", "skin eruption", "skin lesion"],
    "pain":                ["ache", "discomfort", "soreness"],
    "swelling":            ["edema", "puffiness", "inflammation"],
    "shortness of breath": ["dyspnea", "breathlessness", "difficulty breathing"],
    "itching":             ["pruritus", "scratching urge", "skin irritation"],
    "dizziness":           ["lightheadedness", "unsteadiness", "spinning sensation"],
    "weight loss":         ["unexplained weight drop", "losing weight"],
    "confusion":           ["disorientation", "mental fog", "cognitive impairment"],
    "muscle weakness":     ["muscular weakness", "loss of strength"],
    "chest pain":          ["chest discomfort", "chest tightness", "thoracic pain"],
    "joint pain":          ["arthralgia", "joint ache", "joint soreness"],
    "blurred vision":      ["vision problems", "visual disturbance", "unclear vision"],
    "abdominal pain":      ["stomach pain", "belly pain", "abdominal cramps"],
}


def synonym_swap(symptom: str) -> str:
    for term, synonyms in SYNONYMS.items():
        if term in symptom and random.random() < 0.4:
            return symptom.replace(term, random.choice(synonyms), 1)
    return symptom


def augment(disease: str, symptoms_str: str, n: int = 14) -> list:
    symptoms = [s.strip() for s in symptoms_str.split(",") if s.strip()]
    k = len(symptoms)
    rows = []

    rows.append((disease, ", ".join(symptoms)))                          # original

    for _ in range(3):                                                   # shuffled
        s = symptoms.copy(); random.shuffle(s)
        rows.append((disease, ", ".join(s)))

    for ratio in [0.6, 0.7, 0.8, 0.9]:                                  # subsets
        sub = random.sample(symptoms, max(3, int(k * ratio)))
        rows.append((disease, ", ".join(sub)))

    for _ in range(4):                                                   # synonym swaps
        sw = [synonym_swap(s) for s in symptoms]; random.shuffle(sw)
        rows.append((disease, ", ".join(sw)))

    for _ in range(4):                                                   # subset + swap
        sub = random.sample(symptoms, max(3, int(k * random.uniform(0.6, 0.85))))
        rows.append((disease, ", ".join(synonym_swap(s) for s in sub)))

    seen, unique = set(), []
    for r in rows:
        if r[1] not in seen:
            seen.add(r[1]); unique.append(r)
    return unique[:n]


def load_csv() -> pd.DataFrame:
    """Load disease_symptoms.csv — mandatory."""
    path = os.path.join(BASE_DIR, "disease_symptoms.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "disease_symptoms.csv not found.\n"
            "Make sure it is in the same folder as train_model.py."
        )
    df = pd.read_csv(path).dropna(subset=["symptoms", "disease"])
    df["disease"]  = df["disease"].str.strip()
    df["symptoms"] = df["symptoms"].str.lower().str.strip()
    log.info("Loaded disease_symptoms.csv — %d diseases", len(df))
    return df


def augment_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        rows.extend(augment(row["disease"], row["symptoms"]))
    aug = pd.DataFrame(rows, columns=["disease", "symptoms"])
    log.info("Augmented: %d rows (%.1fx per disease)", len(aug), len(aug) / len(df))
    return aug


def build_pipeline() -> Pipeline:
    tfidf = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 3),
        min_df=1, sublinear_tf=True, max_features=80_000,
    )
    svc = CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=5000, class_weight="balanced"),
        cv=3, method="sigmoid",
    )
    lr = LogisticRegression(
        C=5.0, max_iter=2000, solver="lbfgs", class_weight="balanced", n_jobs=-1,
    )
    ensemble = VotingClassifier(
        estimators=[("svc", svc), ("lr", lr)],
        voting="soft", weights=[2, 1], n_jobs=-1,
    )
    return Pipeline([("tfidf", tfidf), ("clf", ensemble)])


def evaluate(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    acc   = accuracy_score(y_test, y_pred)
    f1_m  = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    f1_w  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    log.info("Test Accuracy  : %.2f%%", acc * 100)
    log.info("Macro F1       : %.4f",   f1_m)
    log.info("Weighted F1    : %.4f",   f1_w)
    return acc, f1_m, f1_w


def main():
    # 1. Load CSV (auto-detected, mandatory)
    df  = load_csv()
    aug = augment_dataset(df)

    X, y = aug["symptoms"], aug["disease"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    log.info("Train: %d  |  Test: %d", len(X_train), len(X_test))

    # 2. Build & train
    pipeline = build_pipeline()
    log.info("Training model …")
    pipeline.fit(X_train, y_train)

    # 3. Evaluate
    acc, f1_m, f1_w = evaluate(pipeline, X_test, y_test)

    # 4. Cross-validation
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy", n_jobs=-1,
    )
    log.info("CV Accuracy    : %.2f%% ± %.2f%%", cv_scores.mean()*100, cv_scores.std()*100)

    # 5. Save model files (used by app.py)
    model_path = os.path.join(BASE_DIR, "disease_model.pkl")
    vec_path   = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
    joblib.dump(pipeline.named_steps["clf"],   model_path)
    joblib.dump(pipeline.named_steps["tfidf"], vec_path)
    log.info("Saved: %s", model_path)
    log.info("Saved: %s", vec_path)

    log.info("=" * 50)
    log.info("DONE  Accuracy=%.2f%%  MacroF1=%.4f", acc * 100, f1_m)
    log.info("=" * 50)


if __name__ == "__main__":
    main()
