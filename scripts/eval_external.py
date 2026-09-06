"""Évaluation externe : le modèle tient-il hors de son corpus d'origine ?

Le score de test mesure la généralisation à de nouveaux domaines DU MÊME
corpus, collecté en 2020. Il ne dit rien de la dérive de distribution. On
mesure donc deux choses sur des données réelles d'aujourd'hui :

  * RAPPEL  sur du hameçonnage frais (flux PhishTank), postérieur de six ans au
    corpus d'entraînement ;
  * TAUX DE FAUX POSITIFS sur des domaines légitimes de premier plan (Tranco).

Les domaines déjà vus à l'entraînement sont exclus : sans ça, on remesurerait
de la mémorisation.
"""

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from phishing_detector.data import DATA_DIR, load_dataset
from phishing_detector.features import registrable_domain
from phishing_detector.train import build_models, choose_threshold, proba_of, split_grouped

PHISHTANK_URL = "https://data.phishtank.com/data/online-valid.csv"
TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"
EXT_DIR = DATA_DIR / "external"
N_SAMPLE = 1000
CANDIDATS = ["03_random_forest", "05_tfidf_char_logreg"]


def _fetch(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"Téléchargement de {dest.name}...")
        urllib.request.urlretrieve(url, dest)
    return dest


def load_phishtank() -> pd.Series:
    path = _fetch(PHISHTANK_URL, EXT_DIR / "phishtank.csv")
    return pd.read_csv(path, usecols=["url"])["url"].dropna()


def load_tranco() -> pd.Series:
    path = _fetch(TRANCO_URL, EXT_DIR / "tranco.zip")
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw), names=["rank", "domain"])
    return df.sort_values("rank")["domain"]


def main() -> None:
    df = load_dataset()
    train, val, _ = split_grouped(df)
    domaines_connus = set(df["domain"])

    phish = load_phishtank()
    phish = phish[~phish.map(registrable_domain).isin(domaines_connus)]
    phish = phish.sample(min(N_SAMPLE, len(phish)), random_state=0)

    tranco = load_tranco()
    tranco = tranco[~tranco.map(registrable_domain).isin(domaines_connus)]
    tranco = tranco.head(N_SAMPLE * 5).sample(N_SAMPLE, random_state=0)

    print(f"\nPhishTank : {len(phish)} URLs fraîches, domaines inconnus du corpus")
    print(f"Tranco    : {len(tranco)} domaines légitimes de premier plan\n")

    print(f"{'modèle':<26} {'test interne':>13} {'rappel frais':>13} {'faux positifs':>14}")
    print("-" * 70)
    for nom in CANDIDATS:
        pipe = build_models()[nom]
        pipe.fit(train["url"], train["label"])
        seuil, _ = choose_threshold(val["label"], proba_of(pipe, val["url"]))

        _, _, test = split_grouped(df)
        interne = (proba_of(pipe, test["url"]) >= seuil).astype(int)
        rappel_interne = (interne[test["label"].to_numpy() == 1]).mean()

        rappel_frais = (proba_of(pipe, phish) >= seuil).mean()
        faux_positifs = (proba_of(pipe, tranco) >= seuil).mean()

        print(f"{nom:<26} {rappel_interne:>12.1%} {rappel_frais:>13.1%} {faux_positifs:>14.1%}")


if __name__ == "__main__":
    main()
