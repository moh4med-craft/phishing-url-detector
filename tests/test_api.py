"""Tests de l'API.

TestClient appelle l'application EN MÉMOIRE : aucun serveur lancé, aucun port
ouvert. Les tests d'API deviennent aussi rapides que des tests unitaires.

Détail qui piège tout le monde : TestClient doit être utilisé comme
gestionnaire de contexte (`with TestClient(app) as client`). Sans le `with`,
les événements de cycle de vie ne se déclenchent pas et le modèle n'est jamais
chargé.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from phishing_detector import api as api_module
from phishing_detector.api import app

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="modèle absent — lancer `python -m phishing_detector.train`",
)


@pytest.fixture(scope="module")
def client():
    """Portée « module » : le modèle n'est chargé qu'une fois pour tous les tests."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def seuil():
    return json.loads((ROOT / "models" / "metadata.json").read_text(encoding="utf-8"))["threshold"]


# --------------------------------------------------------------------------- #
# Disponibilité
# --------------------------------------------------------------------------- #

def test_health_signale_le_modele_charge(client):
    r = client.get("/health")
    assert r.status_code == 200
    corps = r.json()
    assert corps["status"] == "ok"
    assert corps["model_loaded"] is True
    assert 0.0 < corps["threshold"] < 1.0


def test_racine_redirige_vers_la_documentation(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (307, 302)
    assert r.headers["location"] == "/docs"


# --------------------------------------------------------------------------- #
# Prédiction
# --------------------------------------------------------------------------- #

def test_predict_renvoie_un_contrat_complet(client, seuil):
    r = client.post("/predict", json={"url": "http://exemple.com/page"})
    assert r.status_code == 200
    corps = r.json()
    assert 0.0 <= corps["probability"] <= 1.0
    assert corps["label"] in {"phishing", "legitimate"}
    # Seuil et version rendent la prédiction auditable a posteriori.
    assert corps["threshold"] == pytest.approx(seuil)
    assert corps["model_version"]


def test_label_coherent_avec_le_seuil(client, seuil):
    corps = client.post("/predict", json={"url": "http://a-b-c.com/login?x=1"}).json()
    attendu = "phishing" if corps["probability"] >= seuil else "legitimate"
    assert corps["label"] == attendu


@pytest.mark.parametrize(
    "url",
    [
        "http://paypal.com.secure-verify.tk/login?id=1",
        "http://192.168.1.1/account/verify.php",
    ],
)
def test_hameconnage_manifeste_est_signale(client, url):
    assert client.post("/predict", json={"url": url}).json()["label"] == "phishing"


def test_site_legitime_manifeste_nest_pas_signale(client):
    corps = client.post("/predict", json={"url": "https://www.wikipedia.org/"}).json()
    assert corps["label"] == "legitimate"


def test_url_malformee_ne_provoque_pas_derreur_serveur(client):
    """Le contrat « ne jamais lever » de features.py doit tenir jusqu'ici."""
    for url in ("???", "http://[", "a" * 2000):
        assert client.post("/predict", json={"url": url}).status_code == 200


# --------------------------------------------------------------------------- #
# Validation : c'est Pydantic qui refuse, avant notre code
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "payload",
    [
        {},                          # champ manquant
        {"url": 42},                 # mauvais type
        {"url": ""},                 # sous la longueur minimale
        {"url": "a" * 3000},         # au-dessus de la longueur maximale
    ],
)
def test_payload_invalide_renvoie_422(client, payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    assert "detail" in r.json()


# --------------------------------------------------------------------------- #
# Lot
# --------------------------------------------------------------------------- #

def test_batch_renvoie_autant_de_resultats_que_durls(client):
    urls = ["http://a.com", "https://b.org/x", "http://1.2.3.4/y"]
    r = client.post("/predict/batch", json={"urls": urls})
    assert r.status_code == 200
    assert [item["url"] for item in r.json()] == urls


def test_batch_trop_grand_est_refuse(client):
    r = client.post("/predict/batch", json={"urls": ["http://a.com"] * 101})
    assert r.status_code == 422


# --------------------------------------------------------------------------- #
# LA question d'entretien : où le modèle est-il chargé ?
# --------------------------------------------------------------------------- #

def test_le_modele_est_charge_une_seule_fois(monkeypatch):
    """Preuve que le chargement est au démarrage, pas par requête."""
    appels = []
    vrai_load = api_module.joblib.load

    def load_compte(chemin, *args, **kwargs):
        appels.append(chemin)
        return vrai_load(chemin, *args, **kwargs)

    monkeypatch.setattr(api_module.joblib, "load", load_compte)

    with TestClient(app) as c:
        for _ in range(5):
            assert c.post("/predict", json={"url": "http://exemple.com"}).status_code == 200

    assert len(appels) == 1, f"{len(appels)} chargements pour 5 requêtes"
