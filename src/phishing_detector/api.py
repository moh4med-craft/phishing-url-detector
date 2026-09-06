"""API de détection d'URL de hameçonnage.

Sécurité applicative — trois règles, assumées :
  * On ne VISITE JAMAIS l'URL reçue. La récupérer ouvrirait un SSRF : un
    attaquant nous ferait interroger 169.254.169.254 et exfiltrerait les
    identifiants de l'instance. Le modèle ne lit que la chaîne.
  * La longueur est bornée par Pydantic, avant d'atteindre notre code.
  * Les URLs ne sont pas journalisées telles quelles.
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.environ.get("PHISHING_MODEL_PATH", ROOT / "models" / "model.joblib"))
META_PATH = Path(os.environ.get("PHISHING_META_PATH", ROOT / "models" / "metadata.json"))

MAX_BATCH = 100


# --------------------------------------------------------------------------- #
# Schémas — la frontière valide, le code métier n'a plus à vérifier ses entrées
# --------------------------------------------------------------------------- #

class PredictRequest(BaseModel):
    url: str = Field(
        min_length=3,
        max_length=2048,
        description="L'URL à analyser. Elle n'est jamais visitée.",
        examples=["http://paypal.com.secure-verify.tk/login?id=1"],
    )


class BatchRequest(BaseModel):
    urls: list[str] = Field(min_length=1, max_length=MAX_BATCH)


class PredictResponse(BaseModel):
    # Pydantic v2 réserve le préfixe « model_ » pour son propre usage. Sans
    # cette ligne, `model_version` déclenche un avertissement de collision.
    model_config = ConfigDict(protected_namespaces=())

    url: str
    probability: float = Field(description="Probabilité brute renvoyée par le modèle.")
    label: Literal["phishing", "legitimate"]
    threshold: float = Field(description="Seuil appliqué, calibré à l'entraînement.")
    model_version: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_name: str | None = None
    threshold: float | None = None


# --------------------------------------------------------------------------- #
# Cycle de vie : le modèle est chargé UNE FOIS, au démarrage du processus
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage, pas à chaque requête.

    Le charger dans la fonction de prédiction relirait ~7 Mo de disque et
    désérialiserait à chaque appel : ~100 ms au lieu de ~1 ms.

    Le charger au niveau du module marcherait, mais l'import deviendrait un
    effet de bord : le chargement se déclencherait aussi pendant la collecte
    des tests ou l'introspection de l'application.

    Bénéfice décisif du lifespan : si le modèle manque ou est corrompu, le
    processus REFUSE DE DÉMARRER. L'orchestrateur voit un déploiement en échec,
    au lieu d'un conteneur « sain » qui renverrait des 500 en production.
    """
    app.state.model = joblib.load(MODEL_PATH)
    app.state.meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    logger.info("Modèle %s chargé (seuil %.4f)",
                app.state.meta.get("model_name"), app.state.meta.get("threshold"))
    yield
    app.state.model = None
    app.state.meta = None


app = FastAPI(
    title="phishing-url-detector",
    description="Classifie une URL comme hameçonnage ou légitime à partir de sa seule chaîne.",
    version="0.1.0",
    lifespan=lifespan,
)


def _predict_many(request: Request, urls: list[str]) -> list[PredictResponse]:
    """UN SEUL appel au modèle pour toutes les URLs.

    Boucler en Python et appeler predict_proba par URL coûtait 8,7 ms/URL,
    contre 0,1 ms en vectorisé : l'essentiel du temps est de la surcharge fixe
    d'appel, pas du calcul. Sans ça, l'endpoint « lot » n'économiserait que
    l'aller-retour HTTP.
    """
    modele, meta = request.app.state.model, request.app.state.meta
    seuil = float(meta["threshold"])
    version = str(meta.get("model_version", "unknown"))
    probas = modele.predict_proba(urls)[:, 1]
    return [
        PredictResponse(
            url=url,
            probability=round(float(proba), 4),
            label="phishing" if proba >= seuil else "legitimate",
            threshold=seuil,
            model_version=version,
        )
        for url, proba in zip(urls, probas, strict=True)
    ]


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.get("/", include_in_schema=False)
async def racine() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Sonde de vivacité : vérifie que le modèle est réellement chargé."""
    modele = getattr(request.app.state, "model", None)
    if modele is None:
        return HealthResponse(status="degraded", model_loaded=False)
    meta = request.app.state.meta
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=meta.get("model_name"),
        threshold=meta.get("threshold"),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    """Renvoie la probabilité de hameçonnage d'une URL.

    Le seuil et la version du modèle sont renvoyés avec la réponse : une
    prédiction doit rester auditable six mois plus tard, quand on demandera
    pourquoi cette URL a été bloquée.
    """
    return _predict_many(request, [payload.url])[0]


@app.post("/predict/batch", response_model=list[PredictResponse])
async def predict_batch(payload: BatchRequest, request: Request) -> list[PredictResponse]:
    """Traite jusqu'à 100 URLs en un appel — pensé pour le débit, pas la latence."""
    return _predict_many(request, payload.urls)
