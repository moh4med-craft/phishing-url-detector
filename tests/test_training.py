"""Tests des invariants d'entraînement et non-régression du modèle expédié.

Ces tests ne mesurent pas une performance — ils protègent les DÉCISIONS :
le découpage doit rester anti-fuite, le seuil doit rester une règle explicite,
et le modèle livré doit continuer à trancher correctement les cas manifestes.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import precision_score, recall_score

from phishing_detector.features import FEATURE_NAMES
from phishing_detector.train import MIN_PRECISION, choose_threshold, split_grouped

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"
META_PATH = ROOT / "models" / "metadata.json"

besoin_modele = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="modèle absent — lancer `python -m phishing_detector.train`",
)


# --------------------------------------------------------------------------- #
# Choix du seuil
# --------------------------------------------------------------------------- #

def test_seuil_respecte_la_contrainte_de_precision():
    """Séparation parfaite : le seuil doit tout attraper en gardant P = 1."""
    y = np.array([1] * 10 + [0] * 10)
    proba = np.concatenate([np.linspace(0.60, 0.95, 10), np.linspace(0.01, 0.20, 10)])

    seuil, atteint = choose_threshold(y, proba)

    assert atteint is True
    y_pred = (proba >= seuil).astype(int)
    assert precision_score(y, y_pred) >= MIN_PRECISION
    assert recall_score(y, y_pred) == 1.0


def test_seuil_retombe_sur_05_si_la_contrainte_est_inatteignable():
    """Quand aucun seuil ne tient la précision cible, on le signale au lieu de mentir."""
    y = np.array([1] + [0] * 9)
    proba = np.array([0.05] + [0.9] * 9)  # les négatifs sont mieux notés que le positif

    seuil, atteint = choose_threshold(y, proba)

    assert atteint is False
    assert seuil == 0.5


def test_seuil_nest_pas_le_defaut_05():
    """Le 0.5 de predict() est un défaut d'implémentation, pas un choix métier."""
    y = np.array([1] * 5 + [0] * 5)
    proba = np.array([0.9, 0.85, 0.8, 0.75, 0.7, 0.2, 0.15, 0.1, 0.05, 0.01])
    seuil, atteint = choose_threshold(y, proba)
    assert atteint is True
    assert seuil != 0.5


# --------------------------------------------------------------------------- #
# Découpage anti-fuite — l'invariant central du projet
# --------------------------------------------------------------------------- #

@pytest.fixture
def corpus_synthetique() -> pd.DataFrame:
    """60 domaines, 1 à 4 URLs chacun : reproduit le motif du vrai corpus."""
    lignes = []
    for i in range(60):
        domaine = f"site{i}.com"
        for j in range((i % 4) + 1):
            lignes.append({"url": f"http://{domaine}/p{j}", "label": i % 2, "domain": domaine})
    return pd.DataFrame(lignes)


def test_aucun_domaine_ne_traverse_le_split(corpus_synthetique):
    """LE test du projet : un domaine ne doit jamais être des deux côtés.

    Sans cet invariant, le modèle mémorise les domaines et le score de test
    mesure de la récitation, pas de la généralisation.
    """
    train, val, test = split_grouped(corpus_synthetique)

    assert not set(train["domain"]) & set(val["domain"])
    assert not set(train["domain"]) & set(test["domain"])
    assert not set(val["domain"]) & set(test["domain"])


def test_le_split_ne_perd_ni_ne_duplique_aucune_ligne(corpus_synthetique):
    train, val, test = split_grouped(corpus_synthetique)
    assert len(train) + len(val) + len(test) == len(corpus_synthetique)
    assert set(train["url"]) | set(val["url"]) | set(test["url"]) == set(corpus_synthetique["url"])


def test_proportions_approximatives_70_15_15(corpus_synthetique):
    """Groupé, donc approximatif : on découpe des domaines, pas des lignes."""
    train, val, test = split_grouped(corpus_synthetique)
    n = len(corpus_synthetique)
    assert 0.60 <= len(train) / n <= 0.80
    assert 0.08 <= len(val) / n <= 0.25
    assert 0.08 <= len(test) / n <= 0.25


def test_le_split_est_reproductible(corpus_synthetique):
    a = split_grouped(corpus_synthetique, seed=7)[0]["url"].tolist()
    b = split_grouped(corpus_synthetique, seed=7)[0]["url"].tolist()
    assert a == b


# --------------------------------------------------------------------------- #
# Modèle expédié
# --------------------------------------------------------------------------- #

@besoin_modele
def test_metadonnees_coherentes_avec_le_code():
    """Si FEATURE_NAMES change sans réentraînement, le modèle devient faux en silence."""
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    assert meta["feature_names"] == FEATURE_NAMES
    assert 0.0 < meta["threshold"] < 1.0
    assert meta["min_precision_target"] == MIN_PRECISION


@besoin_modele
def test_le_pipeline_prend_une_url_brute():
    """Contrat : URL -> probabilité, sans prétraitement à réimplémenter côté API."""
    modele = joblib.load(MODEL_PATH)
    proba = modele.predict_proba(["http://exemple.com"])[:, 1]
    assert proba.shape == (1,)
    assert 0.0 <= float(proba[0]) <= 1.0


@besoin_modele
@pytest.mark.parametrize(
    "url",
    [
        "http://paypal.com.secure-verify.tk/login?id=1",
        "http://192.168.1.1/account/verify.php",
        "http://secure-login-appleid.xyz/confirm?session=8271932",
    ],
)
def test_non_regression_hameconnage_manifeste(url):
    modele = joblib.load(MODEL_PATH)
    seuil = json.loads(META_PATH.read_text(encoding="utf-8"))["threshold"]
    assert float(modele.predict_proba([url])[:, 1][0]) >= seuil


@besoin_modele
@pytest.mark.parametrize("url", ["https://www.wikipedia.org/", "https://www.google.com"])
def test_non_regression_legitime_manifeste(url):
    modele = joblib.load(MODEL_PATH)
    seuil = json.loads(META_PATH.read_text(encoding="utf-8"))["threshold"]
    assert float(modele.predict_proba([url])[:, 1][0]) < seuil
