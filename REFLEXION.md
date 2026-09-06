# Réflexion

Ce que ce projet m'a appris au-delà du code. Chaque affirmation ici est mesurée sur mes propres données ; les commandes de reproduction sont indiquées.

---

## 1. Un bon score ne dit pas qu'un modèle est bon

Le meilleur modèle de mon banc d'essai était un TF-IDF de n-grammes de caractères : **91,8 % de rappel** sur mon jeu de test, contre 78,7 % pour la forêt aléatoire. Treize points d'écart.

Confronté à des données réelles, il classait **89,5 % des domaines du top Tranco comme du hameçonnage**. Neuf sites légitimes sur dix.

| Modèle | Test interne | Hameçonnage frais | Faux positifs Tranco |
|---|---|---|---|
| `03_random_forest` | 78,7 % | 73,1 % | 1,2 % |
| `05_tfidf_char_logreg` | 91,8 % | 67,0 % | **89,5 %** |

L'explication tient en une phrase : il n'a aucune notion de la structure d'une URL, il mémorise des sous-chaînes du corpus de 2020. Tant qu'on l'évalue sur ce corpus, la mémorisation ressemble à de la compétence.

**J'ai donc expédié le modèle qui perd le banc d'essai de treize points**, et j'ai encodé cette décision dans le code plutôt que dans un commentaire : la constante `BENCHMARK_ONLY` empêche la sélection automatique de retenir un modèle disqualifié. Un score de test est une mesure, pas un critère de mise en production.

> Reproduction : `python scripts/eval_external.py`

---

## 2. L'effet du taux de base — la leçon centrale

Mon corpus est équilibré à 50/50. Le trafic réel contient **moins de 1 % de hameçonnage**. Le modèle ne change pas d'une situation à l'autre, mais ce qu'il produit change du tout au tout.

En appliquant mes propres taux mesurés (78,7 % de rappel, 13,5 % de faux positifs sur des URLs légitimes complètes) à 10 000 URLs :

| | Mon corpus (50 %) | Trafic réel (1 %) |
|---|---|---|
| Hameçonnages attrapés | 3 935 | 79 |
| Fausses alertes | 675 | **1 336** |
| Hameçonnages manqués | 1 065 | 21 |
| **Précision réelle** | **85,4 %** | **5,6 %** |

**En production, dix-neuf alertes sur vingt seraient fausses.**

Ce n'est pas un défaut de mon modèle, c'est une propriété de toute détection d'événement rare : quand la classe positive est rare, même un faible taux de faux positifs produit, en valeur absolue, bien plus de fausses alertes que de vraies détections. C'est la raison profonde pour laquelle l'exactitude ne veut rien dire ici — et pourquoi une précision mesurée sur un corpus équilibré ne se transporte pas telle quelle.

En conséquence, un déploiement réel n'utiliserait jamais ce modèle seul : liste blanche des grands domaines en amont, combinaison avec d'autres signaux, et alerte plutôt que blocage automatique.

---

## 3. Le seuil de décision est un curseur métier, pas un défaut d'implémentation

Le `0.5` de `predict()` n'est le résultat d'aucune réflexion. J'ai fixé le mien à **0,559** en appliquant une règle explicite : le seuil le plus bas qui maintient une précision ≥ 90 % sur la validation. Mais ce 90 % est lui-même un choix, et voici ce qu'il coûte :

| Seuil | Rappel | Précision | Faux positifs |
|---|---|---|---|
| 0,300 | 92,6 % | 83,0 % | 28,6 % |
| 0,400 | 88,8 % | 86,6 % | 20,8 % |
| 0,500 | 83,0 % | 88,3 % | 16,5 % |
| **0,559** | **78,7 %** | **89,8 %** | **13,5 %** |
| 0,700 | 70,7 % | 93,3 % | 7,7 % |
| 0,800 | 61,7 % | 95,5 % | 4,4 % |

Descendre à 0,30 attraperait 92,6 % des hameçonnages — au prix de 28,6 % de fausses alertes. Un faux négatif coûte plus cher qu'un faux positif, mais **pas à l'infini** : c'est ce rapport de coûts, propre à chaque organisation, qui devrait fixer le curseur. Une passerelle bancaire et un filtre grand public ne choisiraient pas la même ligne de ce tableau.

---

## 4. Mon modèle a appris un raccourci, et il coûte trois caractères à défaire

C'est la découverte la plus dérangeante du projet.

Dans mon corpus d'entraînement :

| | URLs commençant par `www.` |
|---|---|
| Légitimes | **66,8 %** (3 817 / 5 715) |
| Hameçonnage | **20,5 %** (1 174 / 5 714) |

Le modèle a donc appris que `www.` est un marqueur de légitimité. Conséquence directe, sur des URLs proches du seuil :

| Sans `www.` | Avec `www.` | URL |
|---|---|---|
| 0,759 → hameçonnage | **0,130 → légitime** | `cloudstore.net/s` |
| 0,750 → hameçonnage | **0,121 → légitime** | `openfolder.com/x` |
| 0,738 → hameçonnage | **0,144 → légitime** | `quickview.net/d` |

**Le score est divisé par six, et le verdict bascule, pour trois caractères que n'importe qui peut ajouter.**

Ce n'est pas un signal causal — rien n'empêche un site de hameçonnage d'utiliser `www.` — c'est une corrélation propre au corpus de 2020, que le modèle a prise pour une règle. On appelle ça un raccourci d'apprentissage, et le problème est qu'il est invisible tant qu'on n'évalue que sur des données de même provenance.

Test complémentaire : en construisant huit URLs qui imitent simplement le profil d'un site légitime — courtes, en `.com`, en HTTPS, sans tiret, sans chiffre, sans aucun mot de mon dictionnaire d'ingénierie sociale — **cinq passent au travers**. Un attaquant qui lit ce dépôt sait exactement quoi faire.

C'est la limite structurelle des caractéristiques lexicales : elles décrivent ce à quoi ressemblaient les attaques d'hier, et un attaquant adaptatif n'a qu'à ne plus y ressembler.

---

## 5. Certains hameçonnages sont hors de portée par construction

Sur mon jeu de test, **255 hameçonnages sur 1 197 passent au travers**. En regardant lesquels, un motif apparaît :

```
0,211  http://www.zzhomes.com/fonts/dropboxz/proposal/
0,247  https://www.taliraphaely.com/lds/Linkedin/Linkedin/
0,312  http://vksavesmusic.webservis.ru
0,322  http://www.annikasangster.info/journal/MARKET/
```

Ce sont des **sites légitimes compromis**. L'attaquant n'a pas acheté un domaine douteux : il a piraté le site d'une agence immobilière et déposé sa fausse page dans `/fonts/`. Le nom de domaine est authentique, ancien, sans tiret, sans mot suspect.

Aucune caractéristique lexicale ne peut attraper ça, parce que **l'information n'est pas dans la chaîne**. Il faudrait le contenu de la page, la réputation du certificat, ou un signal de comportement. C'est une limite de la conception, pas un défaut de réglage — et l'admettre vaut mieux que d'empiler des caractéristiques qui n'y changeront rien.

---

## 6. La fuite de données est une question d'unité d'observation

J'ai découpé mon corpus par **domaine enregistrable** plutôt qu'aléatoirement. Mesuré sur cinq tirages :

| Découpage | PR-AUC de test |
|---|---|
| Groupé par domaine | 0,9051 ± 0,0248 |
| Aléatoire | 0,9538 ± 0,0045 |

Le découpage aléatoire surestime le modèle de **4,9 points**, et — détail plus subtil — produit un écart-type **cinq fois plus faible**. Il flatte le score *et* donne une fausse impression de stabilité, puisque chaque tirage réévalue en grande partie les mêmes domaines.

La leçon générale dépasse ce projet : **l'unité d'observation n'est pas la ligne du tableau, c'est ce sur quoi la généralisation doit tenir.** Ici, un modèle en production ne reverra jamais un domaine déjà connu ; l'évaluation doit donc reproduire cette contrainte. Le même raisonnement vaut pour des patients dans un jeu médical, des utilisateurs dans un jeu de recommandation, ou des périodes dans une série temporelle.

51,8 % de mon corpus était exposé à cette fuite — `blogspot.com` à lui seul portait 383 URLs.

---

## 7. Entraîner et servir ne demandent pas les mêmes réglages

`n_jobs=-1` accélère l'entraînement de la forêt aléatoire. En service, sur **une seule URL**, il faisait passer la latence de 11,8 ms à **65,8 ms** : coordonner 300 arbres entre threads coûte sept fois plus cher que le calcul lui-même.

Même leçon avec l'endpoint de lot, qui appelait le modèle une fois par URL dans une boucle Python : 8,7 ms par URL, soit exactement le coût unitaire. Vectorisé en un seul appel, il tombe à 0,12 ms — soixante-dix fois moins.

Dans les deux cas, **les prédictions sont identiques** : ce sont des optimisations de service, pas des changements de modèle. Mais aucune n'apparaît dans une métrique de qualité, et aucun test unitaire ne les aurait révélées. Il a fallu mesurer le système réel.

---

## 8. Ce que je ferais avec plus de temps

Par ordre de rapport valeur/effort :

1. **Un corpus récent.** La dégradation de 78,7 % à 73,1 % sur du hameçonnage frais vient d'un corpus de 2020. Les campagnes actuelles abusent d'hébergeurs légitimes (`netlify.app`, `pages.dev`) qui n'y figurent pas. C'est le levier le plus fort, et de loin.
2. **Une liste blanche des grands domaines** en amont du modèle. Elle supprimerait l'essentiel des 13,5 % de faux positifs pour un coût négligeable.
3. **Neutraliser le raccourci `www.`** — soit en le retirant à la normalisation, soit en équilibrant le corpus sur ce critère.
4. **Une surveillance de dérive** : comparer en continu la distribution des caractéristiques en production à celle de l'entraînement, pour détecter le vieillissement du modèle avant qu'il ne devienne visible dans les incidents.
5. **La vraie Public Suffix List**, embarquée dans l'image plutôt que téléchargée, pour un regroupement exact.
6. **Une calibration des probabilités** (Platt, isotonique) : la forêt aléatoire produit des scores, pas des probabilités fiables. Or je publie un nombre nommé `probability`.

---

## Ce que ce projet n'est pas

Un détecteur déployable. C'est un **composant** : utile combiné à d'autres signaux, dangereux utilisé seul pour bloquer. Il n'est pas robuste face à un attaquant adaptatif, il vieillit, et sa précision réelle sur du trafic déséquilibré est bien inférieure à celle de son jeu de test.

Le savoir, le mesurer et l'écrire me paraît plus utile que de publier un chiffre flatteur.
