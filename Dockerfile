# syntax=docker/dockerfile:1

# =========================================================================== #
# Étape 1 — CONSTRUCTION
# Installe les dépendances dans un préfixe isolé. Tout ce qui est nécessaire
# pour construire (pip, caches, éventuels compilateurs) reste ici et ne sera
# jamais copié dans l'image finale.
# =========================================================================== #
FROM python:3.14-slim AS builder

WORKDIR /app

# Les dépendances AVANT le code source. Docker met chaque instruction en cache :
# si requirements.txt ne change pas, cette couche coûteuse n'est pas rejouée
# quand on modifie une ligne de Python. Copier le code d'abord annulerait le
# cache à chaque commit.
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# =========================================================================== #
# Étape 2 — EXÉCUTION
# Repart d'une base propre : ni pip, ni cache, ni outillage de test.
# =========================================================================== #
FROM python:3.14-slim

# Un conteneur qui tourne en root reste root : une évasion de conteneur
# devient une compromission de l'hôte.
RUN useradd --create-home --uid 1000 appuser

# On ne récupère QUE les paquets installés à l'étape précédente.
COPY --from=builder /install /usr/local

WORKDIR /app
COPY src/ ./src/
COPY models/ ./models/

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER appuser
EXPOSE 8000

# Sonde de vivacité : vérifie que le modèle est réellement chargé, pas
# seulement que le processus existe.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

# --host 0.0.0.0 est indispensable : sans lui, uvicorn n'écoute que la boucle
# locale À L'INTÉRIEUR du conteneur, et le port publié ne mène à rien.
CMD ["uvicorn", "phishing_detector.api:app", "--host", "0.0.0.0", "--port", "8000"]
