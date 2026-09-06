"""Extraction de caractéristiques lexicales d'une URL.

Contrainte de conception : AUCUN accès réseau (pas de DNS, pas de WHOIS, pas de
requête HTTP). Un détecteur doit répondre en millisecondes et ne doit jamais
télécharger le contenu qu'il analyse. Toutes les caractéristiques ci-dessous se
calculent donc à partir de la seule chaîne de caractères.
"""

import math
import re
from collections import Counter
from collections.abc import Iterable
from urllib.parse import urlsplit

import pandas as pd

# --------------------------------------------------------------------------- #
# Tables de référence
# --------------------------------------------------------------------------- #

# IPv4 strict : rejette 1000.1.1.1 et 999.1.1.1, qu'une regex naïve accepterait.
_IPV4_RE = re.compile(
    r"^(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
    r"(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}$"
)

_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.\-]*://", re.IGNORECASE)

# TLD gratuits ou massivement abusés par le hameçonnage.
_SUSPICIOUS_TLDS = frozenset({
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "work", "click", "link", "gdn",
    "loan", "download", "review", "country", "stream", "racing", "win", "bid",
    "party", "science", "date", "faith", "zip", "mov", "cam", "rest", "buzz",
    "icu", "cyou", "sbs", "fit", "quest", "monster", "surf", "biz",
})

# Raccourcisseurs : masquent la destination réelle.
_SHORTENERS = frozenset({
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bit.do", "cutt.ly", "rb.gy", "shorturl.at", "tiny.cc",
    "rebrand.ly", "s.id", "t.ly", "shorte.st", "soo.gd", "clck.ru",
})

# Vocabulaire d'ingénierie sociale et marques les plus usurpées.
_SENSITIVE_WORDS = (
    "login", "signin", "verify", "verification", "account", "secure",
    "security", "update", "confirm", "banking", "bank", "paypal", "apple",
    "microsoft", "amazon", "netflix", "facebook", "google", "wallet",
    "password", "support", "invoice", "payment", "suspend", "alert",
    "recover", "unlock", "webscr", "billing",
)

_SPECIAL_CHARS = "@%=&?+~,;$!*'()"

# Suffixes composés fréquents. La référence exacte est la Public Suffix List, qui
# suppose une dépendance et une mise à jour réseau : on couvre ici les cas les
# plus courants et on documente la limite dans le README.
_COMPOUND_SUFFIXES = frozenset({
    "co.uk", "org.uk", "ac.uk", "gov.uk", "me.uk", "net.uk", "sch.uk",
    "com.au", "net.au", "org.au", "edu.au", "gov.au",
    "co.jp", "or.jp", "ne.jp", "ac.jp", "go.jp",
    "com.br", "net.br", "org.br", "gov.br",
    "co.in", "net.in", "org.in", "gov.in", "ac.in",
    "com.cn", "net.cn", "org.cn", "gov.cn",
    "co.za", "org.za", "co.nz", "co.kr", "co.id", "co.il", "co.th",
    "com.mx", "com.ar", "com.tr", "com.sg", "com.hk", "com.tw", "com.my",
    "com.ph", "com.vn", "com.pk", "com.ua", "com.pl", "com.ru", "com.eg",
    "com.sa", "com.ng", "com.co", "com.pe", "com.ve", "com.ec",
})

# Ordre FIGÉ des colonnes. Source de vérité unique, partagée par l'entraînement
# et l'API : si les deux ne produisent pas les colonnes dans le même ordre, le
# modèle prédit n'importe quoi SANS lever d'erreur.
FEATURE_NAMES: list[str] = [
    # Longueurs
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    # Comptages
    "n_dots",
    "n_hyphens",
    "n_digits",
    "n_special_chars",
    "path_depth",
    "n_subdomains",
    "tld_length",
    "n_sensitive_words",
    # Ratios et mesures
    "digit_ratio",
    "hostname_digit_ratio",
    "hostname_entropy",
    # Booléens
    "has_ip",
    "has_at",
    "has_double_slash_in_path",
    "has_port",
    "is_https",
    "is_punycode",
    "is_shortener",
    "has_suspicious_tld",
]


# --------------------------------------------------------------------------- #
# Fonctions utilitaires
# --------------------------------------------------------------------------- #

def normalize(url: str) -> str:
    """Ajoute un schéma si absent.

    Sans schéma, ``urlsplit`` range tout dans ``path`` et laisse ``hostname``
    vide : ``google.com/a`` donnerait une longueur d'hôte de 0. Le cas est réel,
    la liste Tranco ne contient que des domaines nus.
    """
    url = (url or "").strip()
    if not url:
        return ""
    if _SCHEME_RE.match(url):
        return url
    return "http://" + url


def shannon_entropy(text: str) -> float:
    """Mesure le désordre d'une chaîne, en bits par caractère.

    ``google`` a une entropie basse (structure de mot), ``x7fj3kq2`` une entropie
    haute : marqueur classique des domaines générés algorithmiquement (DGA).
    """
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _split(url: str) -> tuple[str, str, str, str, bool]:
    """Découpe une URL en (hostname, path, query, scheme, port_present).

    Ne lève jamais : une URL malformée renvoie des morceaux vides.
    """
    try:
        parts = urlsplit(normalize(url))
        hostname = (parts.hostname or "").lower()
        try:
            has_port = parts.port is not None
        except ValueError:  # port non numérique, ex. http://a:bb/
            has_port = True
        return hostname, parts.path, parts.query, parts.scheme.lower(), has_port
    except ValueError:  # IPv6 malformé, caractères interdits dans netloc...
        return "", "", "", "", False


def registrable_domain(url: str) -> str:
    """Renvoie le domaine enregistrable (eTLD+1) : ``a.b.exemple.co.uk`` -> ``exemple.co.uk``.

    Sert de clé de regroupement pour le split anti-fuite de la phase 2.
    """
    hostname, *_ = _split(url)
    if not hostname or _IPV4_RE.match(hostname) or hostname.startswith("["):
        return hostname
    labels = hostname.split(".")
    if len(labels) < 2:
        return hostname
    if len(labels) >= 3 and ".".join(labels[-2:]) in _COMPOUND_SUFFIXES:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

def extract_features(url: str) -> dict[str, float]:
    """Extrait les caractéristiques d'une URL. Aucun accès réseau, jamais d'exception.

    Renvoie toujours un dictionnaire dont les clés sont ``FEATURE_NAMES``, dans
    cet ordre exact.
    """
    if not isinstance(url, str):
        url = ""
    url = url.strip()
    lowered = url.lower()

    hostname, path, query, scheme, has_port = _split(url)
    labels = hostname.split(".") if hostname else []
    tld = labels[-1] if len(labels) >= 2 else ""

    reg_domain = registrable_domain(url)
    n_reg_labels = len(reg_domain.split(".")) if reg_domain else 0
    n_subdomains = max(len(labels) - n_reg_labels, 0) if labels else 0

    n_digits = sum(c.isdigit() for c in url)
    host_digits = sum(c.isdigit() for c in hostname)
    is_ip = bool(_IPV4_RE.match(hostname)) or hostname.startswith("[")

    values = {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "query_length": len(query),

        "n_dots": url.count("."),
        "n_hyphens": url.count("-"),
        "n_digits": n_digits,
        "n_special_chars": sum(url.count(c) for c in _SPECIAL_CHARS),
        "path_depth": len([seg for seg in path.split("/") if seg]),
        "n_subdomains": n_subdomains,
        "tld_length": len(tld),
        "n_sensitive_words": sum(w in lowered for w in _SENSITIVE_WORDS),

        "digit_ratio": n_digits / len(url) if url else 0.0,
        "hostname_digit_ratio": host_digits / len(hostname) if hostname else 0.0,
        "hostname_entropy": shannon_entropy(hostname),

        "has_ip": is_ip,
        # Tout ce qui précède "@" est ignoré par le navigateur : redirection déguisée.
        "has_at": "@" in url,
        # "//" dans le chemin : redirection ouverte.
        "has_double_slash_in_path": "//" in path,
        "has_port": has_port,
        "is_https": scheme == "https",
        # Punycode : attaque homographe (аpple.com avec un "а" cyrillique).
        "is_punycode": "xn--" in hostname,
        "is_shortener": reg_domain in _SHORTENERS,
        "has_suspicious_tld": tld in _SUSPICIOUS_TLDS,
    }

    # Reconstruction dans l'ordre de FEATURE_NAMES : garantit l'invariant, et
    # lève immédiatement si une caractéristique manque (plutôt qu'en silence).
    return {name: float(values[name]) for name in FEATURE_NAMES}


def featurize(urls: Iterable[str]) -> pd.DataFrame:
    """Applique ``extract_features`` à un itérable d'URLs."""
    rows = [extract_features(u) for u in urls]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)
