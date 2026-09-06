"""Chargement du corpus d'URLs.

Corpus principal : ``pirocheto/phishing-url`` (Hugging Face), 11 430 URLs
équilibrées 50/50, dérivé de l'article arXiv 2010.12847.

Pourquoi celui-là plutôt que « PhishTank + Tranco » : PhishTank fournit des URLs
complètes, Tranco des domaines nus. Les mélanger apprendrait au modèle la règle
« il y a un chemin après le domaine -> hameçonnage », qui atteint 99 % sur le
papier et s'effondre en production. PhishTank et Tranco servent donc uniquement
à l'évaluation externe (phase 7), pas à l'entraînement.

Le corpus fournit 87 caractéristiques précalculées : on les ignore volontairement
et on ne garde que l'URL brute, puisque l'extraction est le cœur du projet.
"""

import os
import urllib.request
from pathlib import Path

import pandas as pd

from phishing_detector.features import registrable_domain

HF_BASE = "https://huggingface.co/datasets/pirocheto/phishing-url/resolve/main/data"
RAW_FILES = ("train.parquet", "test.parquet")

DATA_DIR = Path(os.environ.get("PHISHING_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
RAW_DIR = DATA_DIR / "raw"


def download_raw(force: bool = False) -> list[Path]:
    """Télécharge les parquets bruts dans ``data/raw/`` (mis en cache)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in RAW_FILES:
        dest = RAW_DIR / name
        if force or not dest.exists():
            print(f"Téléchargement de {name}...")
            urllib.request.urlretrieve(f"{HF_BASE}/{name}", dest)
        paths.append(dest)
    return paths


def load_dataset(force_download: bool = False) -> pd.DataFrame:
    """Renvoie un DataFrame ``url`` / ``label`` / ``domain``.

    ``label`` : 1 = hameçonnage, 0 = légitime. La classe positive est celle que
    l'on cherche à détecter — convention indispensable pour que le rappel et la
    précision de scikit-learn portent bien sur le hameçonnage.

    ``domain`` : domaine enregistrable, clé de regroupement du split anti-fuite.

    On réunit volontairement les splits train/test d'origine : ils ont été
    découpés aléatoirement, ce qui disperse un même domaine des deux côtés. On
    refait notre propre découpage groupé en phase 2.
    """
    paths = download_raw(force=force_download)
    frames = [pd.read_parquet(p, columns=["url", "status"]) for p in paths]
    df = pd.concat(frames, ignore_index=True)

    df = df.dropna(subset=["url", "status"])
    df["label"] = (df["status"].str.lower() == "phishing").astype(int)
    df = df.drop(columns=["status"])

    before = len(df)
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    if before != len(df):
        print(f"{before - len(df)} URL(s) en double supprimée(s).")

    df["domain"] = df["url"].map(registrable_domain)
    return df


def main() -> None:
    df = load_dataset()
    print(f"\n{len(df)} URLs | {df['domain'].nunique()} domaines distincts")
    repartition = df["label"].value_counts().rename({0: "légitime", 1: "hameçonnage"})
    print(f"Répartition des classes :\n{repartition}")
    print(f"\nURLs par domaine (moyenne) : {len(df) / df['domain'].nunique():.2f}")
    print("\nExemples :")
    print(df.sample(5, random_state=0).to_string(index=False))


if __name__ == "__main__":
    main()
