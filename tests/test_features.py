"""Tests de l'extraction de caractéristiques.

On ne teste pas « le modèle atteint 95 % » : ce n'est pas un test unitaire, ça
dépend de l'aléa et c'est instable. On teste la LOGIQUE DÉTERMINISTE et les
INVARIANTS — c'est du code ordinaire, testable ordinairement.
"""

import math

import pytest

from phishing_detector.features import (
    FEATURE_NAMES,
    extract_features,
    featurize,
    normalize,
    registrable_domain,
    shannon_entropy,
)

# --------------------------------------------------------------------------- #
# Comptages élémentaires
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("url", "attendu"),
    [
        ("http://a.com", {"n_dots": 1, "n_hyphens": 0, "is_https": 0}),
        ("https://a-b-c.example.com/x", {"n_dots": 2, "n_hyphens": 2, "is_https": 1}),
        ("http://a.com/p/q/r", {"path_depth": 3}),
        ("http://a.com/?x=1&y=2", {"query_length": 7}),
    ],
)
def test_comptages_elementaires(url, attendu):
    f = extract_features(url)
    for cle, valeur in attendu.items():
        assert f[cle] == valeur, f"{cle} sur {url}"


def test_url_length_est_la_longueur_brute():
    assert extract_features("http://a.com")["url_length"] == len("http://a.com")


# --------------------------------------------------------------------------- #
# Détection d'IP : le piège de la regex naïve
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("url", "est_une_ip"),
    [
        ("http://192.168.0.1/x", True),
        ("http://255.255.255.255/", True),
        ("http://exemple.com", False),
        ("http://1000.1.1.1/", False),   # une regex \d+\.\d+\.\d+\.\d+ l'accepterait
        ("http://999.1.1.1/", False),
        ("http://1.2.3/", False),        # trois octets seulement
    ],
)
def test_detection_ip(url, est_une_ip):
    assert extract_features(url)["has_ip"] == float(est_une_ip)


def test_ip_dans_userinfo_nest_pas_lhote():
    """Tout ce qui précède « @ » est ignoré par le navigateur : c'est un leurre.

    L'hôte réel est le domaine de hameçonnage, pas l'IP affichée.
    """
    f = extract_features("http://192.168.1.1@paypal.com.evil.tk/login")
    assert f["has_ip"] == 0.0
    assert f["has_at"] == 1.0
    assert f["has_suspicious_tld"] == 1.0


# --------------------------------------------------------------------------- #
# Marqueurs d'attaque
# --------------------------------------------------------------------------- #

def test_punycode_detecte():
    assert extract_features("http://xn--pple-43d.com/")["is_punycode"] == 1.0
    assert extract_features("http://apple.com/")["is_punycode"] == 0.0


def test_raccourcisseur_detecte():
    assert extract_features("http://bit.ly/3xYz")["is_shortener"] == 1.0
    assert extract_features("http://example.com/3xYz")["is_shortener"] == 0.0


def test_mots_sensibles_comptes():
    f = extract_features("http://x.com/login/verify/account")
    assert f["n_sensitive_words"] >= 3


# --------------------------------------------------------------------------- #
# Entropie
# --------------------------------------------------------------------------- #

def test_entropie_chaine_uniforme_est_nulle():
    assert shannon_entropy("aaaa") == 0.0


def test_entropie_croit_avec_le_desordre():
    assert shannon_entropy("aaaa") < shannon_entropy("aabb") < shannon_entropy("abcd")


def test_entropie_vaut_log2_pour_caracteres_distincts():
    assert shannon_entropy("abcd") == pytest.approx(2.0)


def test_entropie_chaine_vide():
    assert shannon_entropy("") == 0.0


# --------------------------------------------------------------------------- #
# Domaine enregistrable
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("url", "domaine"),
    [
        ("http://a.b.exemple.co.uk/x", "exemple.co.uk"),
        ("https://www.google.com", "google.com"),
        ("http://google.com", "google.com"),
        ("http://evil.tk", "evil.tk"),
        ("http://192.168.1.1/x", "192.168.1.1"),
        # Choix conservateur : blogspot.com n'est pas traité comme suffixe public,
        # donc tous ses sous-domaines forment UN SEUL groupe au moment du split.
        ("http://xyz.blogspot.com/p", "blogspot.com"),
    ],
)
def test_domaine_enregistrable(url, domaine):
    assert registrable_domain(url) == domaine


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #

def test_normalize_ajoute_le_schema_manquant():
    """Sans schéma, urlsplit range tout dans `path` et laisse `hostname` vide.

    Le cas est réel : la liste Tranco ne contient que des domaines nus.
    """
    assert normalize("google.com/a") == "http://google.com/a"
    assert extract_features("google.com/a")["hostname_length"] == len("google.com")


def test_normalize_preserve_un_schema_existant():
    assert normalize("https://a.com") == "https://a.com"
    assert normalize("HTTPS://a.com") == "HTTPS://a.com"


# --------------------------------------------------------------------------- #
# Contrat : ne jamais lever, ordre figé
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "url",
    ["", "   ", "???", "http://", "https://", "http://[", "http://a:bb/",
     "a" * 5000, "http://[::1]/x", "https://élysée.fr/café?q=🎉",
     "javascript:alert(1)", "//evil.com", "http://exemple.com:99999/"],
)
def test_ne_leve_jamais_et_ordre_stable(url):
    """Une URL malformée doit produire un vecteur, pas une erreur 500."""
    f = extract_features(url)
    assert list(f) == FEATURE_NAMES
    assert len(f) == len(FEATURE_NAMES)


def test_entree_non_textuelle_toleree():
    """Un None ou un NaN de pandas ne doit pas faire tomber l'API."""
    for valeur in (None, 42, float("nan")):
        assert list(extract_features(valeur)) == FEATURE_NAMES


def test_toutes_les_valeurs_sont_des_flottants_finis():
    f = extract_features("https://a-b.exemple.co.uk:8443/p/q?x=1#f")
    for nom, valeur in f.items():
        assert isinstance(valeur, float), nom
        assert math.isfinite(valeur), nom


def test_noms_de_caracteristiques_uniques():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))


# --------------------------------------------------------------------------- #
# featurize
# --------------------------------------------------------------------------- #

def test_featurize_forme_et_colonnes():
    urls = ["http://a.com", "https://b.org/x", "http://1.2.3.4/y"]
    X = featurize(urls)
    assert X.shape == (3, len(FEATURE_NAMES))
    assert list(X.columns) == FEATURE_NAMES
    assert X.notna().all().all()


def test_featurize_entree_vide_garde_les_colonnes():
    """Un DataFrame sans structure ferait planter le modèle plus loin."""
    X = featurize([])
    assert X.shape == (0, len(FEATURE_NAMES))
    assert list(X.columns) == FEATURE_NAMES
