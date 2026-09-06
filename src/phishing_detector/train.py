"""Entraînement, comparaison de configurations et journalisation MLflow.

Protocole :
  1. Split 70/15/15 GROUPÉ PAR DOMAINE ENREGISTRABLE (anti-fuite).
  2. Chaque configuration est entraînée sur `train`, son seuil de décision est
     choisi sur `val`, et ses métriques de `val` servent à la sélection.
  3. Le jeu de `test` n'est touché QU'UNE FOIS, par le vainqueur : c'est ce qui
     rend le chiffre publié honnête.
  4. Un run supplémentaire, volontairement fautif, rejoue le vainqueur avec un
     split ALÉATOIRE. L'écart entre les deux mesure la fuite de données.
"""

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # backend sans écran : indispensable en CI et en conteneur

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer, StandardScaler  # noqa: E402

from phishing_detector.data import load_dataset  # noqa: E402
from phishing_detector.features import FEATURE_NAMES, featurize  # noqa: E402

RANDOM_STATE = 42
# Contrainte métier : on maximise le rappel (attraper le hameçonnage) tant que
# la précision reste au-dessus de ce plancher (ne pas noyer le support sous les
# fausses alertes). Un faux négatif coûte plus cher qu'un faux positif, mais pas
# à l'infini — ce nombre est la traduction chiffrée de cet arbitrage.
MIN_PRECISION = 0.90

# Configurations comparées mais NON éligibles à la production.
# 05_tfidf_char_logreg gagne le benchmark interne (rappel 91,8 % contre 78,7 %)
# et s'effondre sur des données réelles : 89,5 % de faux positifs sur le top
# Tranco (cf. scripts/eval_external.py). Il n'a aucune notion de structure
# d'URL, il mémorise des sous-chaînes du corpus 2020. On le garde comme point
# de comparaison — un score de benchmark ne vaut pas un critère de mise en
# production.
BENCHMARK_ONLY = frozenset({"01_dummy_majoritaire", "05_tfidf_char_logreg"})

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "mlartifacts"
EXPERIMENT = "phishing-url-detector"


# --------------------------------------------------------------------------- #
# Découpage
# --------------------------------------------------------------------------- #

def split_grouped(df: pd.DataFrame, seed: int = RANDOM_STATE):
    """70/15/15 en garantissant qu'un domaine ne traverse jamais deux jeux."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, rest_idx = next(gss.split(df, groups=df["domain"]))
    rest = df.iloc[rest_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_idx, test_idx = next(gss2.split(rest, groups=rest["domain"]))
    return df.iloc[train_idx], rest.iloc[val_idx], rest.iloc[test_idx]


def split_random(df: pd.DataFrame, seed: int = RANDOM_STATE):
    """Le MÊME découpage, mais aléatoire : sert uniquement à mesurer la fuite."""
    ss = ShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, rest_idx = next(ss.split(df))
    rest = df.iloc[rest_idx]
    ss2 = ShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_idx, test_idx = next(ss2.split(rest))
    return df.iloc[train_idx], rest.iloc[val_idx], rest.iloc[test_idx]


def split_study(df, model_name, seeds=(0, 1, 2, 3, 4)):
    """Compare les deux stratégies de découpage sur PLUSIEURS tirages.

    Un tirage unique ne prouve rien : la difficulté d'un jeu de test varie d'un
    tirage à l'autre, et l'écart observé peut n'être que du bruit. On répète
    donc, et on compare sur une métrique INDÉPENDANTE DU SEUIL (PR-AUC) pour ne
    pas confondre l'effet du découpage avec celui de la calibration.
    """
    out = {}
    for kind, splitter in (("group_by_domain", split_grouped),
                           ("random_LEAKY", split_random)):
        pr, roc = [], []
        for seed in seeds:
            tr, _, te = splitter(df, seed=seed)
            pipe = build_models()[model_name]
            pipe.fit(tr["url"], tr["label"])
            proba = proba_of(pipe, te["url"])
            pr.append(average_precision_score(te["label"], proba))
            roc.append(roc_auc_score(te["label"], proba))
        out[kind] = {
            "pr_auc_mean": float(np.mean(pr)),
            "pr_auc_std": float(np.std(pr)),
            "roc_auc_mean": float(np.mean(roc)),
        }
        with mlflow.start_run(run_name=f"SPLIT_STUDY_{kind}"):
            mlflow.log_params({"model": model_name, "split": kind,
                               "n_seeds": len(seeds), "seeds": str(list(seeds))})
            mlflow.log_metrics(out[kind])
    return out


# --------------------------------------------------------------------------- #
# Modèles
# --------------------------------------------------------------------------- #

def _with_features(*steps) -> Pipeline:
    """Préfixe un pipeline par l'extraction : l'entrée est l'URL BRUTE.

    Conséquence essentielle : l'objet sérialisé fait URL -> probabilité tout
    seul. L'API n'a aucun prétraitement à réimplémenter, donc aucun risque de
    désynchronisation entre l'entraînement et le service.
    """
    features = ("features", FunctionTransformer(featurize, validate=False))
    return Pipeline([features, *steps])


def build_models() -> dict[str, Pipeline]:
    return {
        # Référence plancher : ignore les caractéristiques et répond toujours la
        # classe majoritaire. Tout modèle qui ne la bat pas est inutile.
        "01_dummy_majoritaire": _with_features(
            ("clf", DummyClassifier(strategy="most_frequent")),
        ),
        # Le scaler n'est utile qu'ici : la régression logistique est sensible
        # aux échelles (url_length ~ 75 contre is_https ~ 0/1), pas les arbres.
        "02_logreg": _with_features(
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ),
        "03_random_forest": _with_features(
            ("clf", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2,
                random_state=RANDOM_STATE, n_jobs=-1)),
        ),
        "04_hist_gradient_boosting": _with_features(
            ("clf", HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.1, random_state=RANDOM_STATE)),
        ),
        # Contraste : aucune caractéristique artisanale, on donne les n-grammes
        # de caractères bruts au modèle et on le laisse trouver les motifs.
        "05_tfidf_char_logreg": Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                      min_df=3, max_features=50_000)),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
    }


# --------------------------------------------------------------------------- #
# Évaluation
# --------------------------------------------------------------------------- #

def choose_threshold(y_true, proba, min_precision: float = MIN_PRECISION) -> tuple[float, bool]:
    """Seuil maximisant le RAPPEL sous contrainte de précision minimale.

    Le 0.5 de `predict()` n'est pas un choix métier, c'est un défaut
    d'implémentation. On le remplace par une règle explicite, calibrée sur la
    validation — jamais sur le test.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # precision/recall ont un point de plus que thresholds (le point P=1, R=0).
    eligible = precision[:-1] >= min_precision
    if not eligible.any():
        return 0.5, False
    idx = int(np.argmax(np.where(eligible, recall[:-1], -1.0)))
    return float(thresholds[idx]), True


def evaluate(y_true, proba, threshold: float, prefix: str) -> dict[str, float]:
    y_pred = (proba >= threshold).astype(int)
    return {
        f"{prefix}_recall": recall_score(y_true, y_pred, zero_division=0),
        f"{prefix}_precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}_f1": f1_score(y_true, y_pred, zero_division=0),
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        # Indépendantes du seuil : mesurent la qualité du CLASSEMENT.
        f"{prefix}_pr_auc": average_precision_score(y_true, proba),
        f"{prefix}_roc_auc": roc_auc_score(y_true, proba),
    }


def plot_confusion(y_true, proba, threshold: float, title: str, path: Path) -> Path:
    cm = confusion_matrix(y_true, (proba >= threshold).astype(int))
    labels = ["légitime", "hameçonnage"]
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("prédit")
    ax.set_ylabel("réel")
    ax.set_title(f"{title}\nseuil = {threshold:.3f}", fontsize=9)
    vmax = cm.max()
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > vmax / 2 else "black", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def proba_of(pipe: Pipeline, urls) -> np.ndarray:
    """Probabilité de la classe positive (hameçonnage)."""
    return pipe.predict_proba(urls)[:, 1]


# --------------------------------------------------------------------------- #
# Boucle d'expérimentation
# --------------------------------------------------------------------------- #

def run_config(name, pipe, train, val, split_kind, log_test=None):
    """Entraîne, calibre le seuil sur `val`, journalise. Renvoie un résumé."""
    with mlflow.start_run(run_name=name):
        t0 = time.perf_counter()
        pipe.fit(train["url"], train["label"])
        fit_seconds = time.perf_counter() - t0

        proba_val = proba_of(pipe, val["url"])
        threshold, reached = choose_threshold(val["label"], proba_val)
        metrics = evaluate(val["label"], proba_val, threshold, "val")
        metrics["fit_seconds"] = fit_seconds

        clf = pipe.steps[-1][1]
        mlflow.log_params({
            "model": name,
            "classifier": clf.__class__.__name__,
            "split": split_kind,
            "random_state": RANDOM_STATE,
            "n_features": len(FEATURE_NAMES),
            "n_train": len(train),
            "n_val": len(val),
            "min_precision_target": MIN_PRECISION,
            "threshold": round(threshold, 4),
            "threshold_target_reached": reached,
        })

        if log_test is not None:
            proba_test = proba_of(pipe, log_test["url"])
            metrics |= evaluate(log_test["label"], proba_test, threshold, "test")
            mlflow.log_param("n_test", len(log_test))
            png = plot_confusion(log_test["label"], proba_test, threshold,
                                 f"{name} — test", ARTIFACTS_DIR / f"{name}_confusion.png")
            mlflow.log_artifact(str(png))

        mlflow.log_metrics(metrics)
        print(f"  {name:26} val_recall={metrics['val_recall']:.3f} "
              f"val_precision={metrics['val_precision']:.3f} "
              f"val_pr_auc={metrics['val_pr_auc']:.3f} seuil={threshold:.3f} "
              f"({fit_seconds:.1f}s)")
        return {"name": name, "pipe": pipe, "threshold": threshold, **metrics}


def main() -> None:
    df = load_dataset()
    train, val, test = split_grouped(df)

    inter = set(train["domain"]) & set(test["domain"])
    print(f"\nCorpus : {len(df)} URLs, {df['domain'].nunique()} domaines")
    print(f"Split groupé : train={len(train)} val={len(val)} test={len(test)}")
    print(f"Domaines partagés entre train et test : {len(inter)}  (doit être 0)\n")
    assert not inter, "Fuite : un domaine se trouve des deux côtés du split."

    mlflow.set_experiment(EXPERIMENT)

    print("Configurations (sélection sur la VALIDATION) :")
    results = [run_config(name, pipe, train, val, "group_by_domain")
               for name, pipe in build_models().items()]

    # Sélection : meilleur rappel de validation, la contrainte de précision
    # étant déjà intégrée au choix du seuil.
    eligibles = [r for r in results if r["name"] not in BENCHMARK_ONLY]
    winner = max(eligibles, key=lambda r: (r["val_recall"], r["val_pr_auc"]))
    hors_concours = max(results, key=lambda r: (r["val_recall"], r["val_pr_auc"]))
    if hors_concours["name"] in BENCHMARK_ONLY:
        print(f"\nMeilleur score brut : {hors_concours['name']} "
              f"(rappel {hors_concours['val_recall']:.3f}) — écarté de la production, "
              f"cf. BENCHMARK_ONLY")
    print(f"Vainqueur éligible : {winner['name']} (rappel {winner['val_recall']:.3f})")

    # Le test n'est touché qu'ici, une seule fois.
    print("\nÉvaluation finale sur le TEST (une seule fois) :")
    final = run_config(f"FINAL_{winner['name']}", build_models()[winner["name"]],
                       train, val, "group_by_domain", log_test=test)
    with mlflow.start_run(run_name=f"FINAL_{winner['name']}_model", nested=False):
        # MLflow refuse par défaut de sérialiser une fonction personnalisée
        # (protection contre la désérialisation de code arbitraire). On déclare
        # explicitement la nôtre comme fiable : elle vient de notre paquet.
        mlflow.sklearn.log_model(
            final["pipe"], name="model",
            skops_trusted_types=["phishing_detector.features.featurize"],
        )

    # Démonstration de la fuite, sur 5 tirages pour ne pas conclure sur du bruit.
    print("\nÉtude du découpage (5 tirages, PR-AUC de test) :")
    study = split_study(df, winner["name"])
    g, r = study["group_by_domain"], study["random_LEAKY"]
    ecart = r["pr_auc_mean"] - g["pr_auc_mean"]
    print(f"{'='*66}")
    print(f"PR-AUC test, split GROUPE    : {g['pr_auc_mean']:.4f} +/- {g['pr_auc_std']:.4f}")
    print(f"PR-AUC test, split ALEATOIRE : {r['pr_auc_mean']:.4f} +/- {r['pr_auc_std']:.4f}")
    print(f"Surestimation due a la fuite : {ecart:+.4f}")
    print(f"{'='*66}\n")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # compress=3 : 32 Mo -> 7 Mo, pour 0,1 s de chargement. Le modèle est
    # versionné dans le dépôt afin que `docker compose up` fonctionne après un
    # simple clone ; en production il viendrait d'un registre de modèles.
    joblib.dump(final["pipe"], MODELS_DIR / "model.joblib", compress=3)

    # Copie de la matrice de confusion hors de mlartifacts/ (ignoré par git)
    # pour qu'elle s'affiche dans le README.
    src_png = ARTIFACTS_DIR / f"FINAL_{winner['name']}_confusion.png"
    if src_png.exists():
        docs = ROOT / "docs"
        docs.mkdir(exist_ok=True)
        (docs / "confusion_matrix.png").write_bytes(src_png.read_bytes())
    metadata = {
        "model_name": winner["name"],
        "model_version": "0.1.0",
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "threshold": round(final["threshold"], 4),
        "min_precision_target": MIN_PRECISION,
        "feature_names": FEATURE_NAMES,
        "split": "GroupShuffleSplit par domaine enregistrable, 70/15/15",
        "benchmark_only": sorted(BENCHMARK_ONLY),
        "selection_note": (
            "05_tfidf_char_logreg obtient le meilleur score interne mais 89,5 % "
            "de faux positifs sur le top Tranco : écarté de la production."
        ),
        "dataset": {"n_urls": len(df), "n_domains": int(df["domain"].nunique()),
                    "n_train": len(train), "n_val": len(val), "n_test": len(test)},
        "test_metrics": {k: round(v, 4) for k, v in final.items()
                         if k.startswith("test_")},
        "leakage_study": {
            "metric": "PR-AUC de test, moyenne sur 5 tirages",
            "grouped_split": round(g["pr_auc_mean"], 4),
            "grouped_split_std": round(g["pr_auc_std"], 4),
            "random_split": round(r["pr_auc_mean"], 4),
            "random_split_std": round(r["pr_auc_std"], 4),
            "overestimation": round(ecart, 4),
        },
    }
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Modèle écrit dans {MODELS_DIR / 'model.joblib'} "
          f"({(MODELS_DIR / 'model.joblib').stat().st_size / 1e6:.1f} Mo)")
    print("Comparer les runs : mlflow ui  ->  http://localhost:5000")


if __name__ == "__main__":
    main()
