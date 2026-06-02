# Projet Football IA — M2442 Jeux Vidéo IA

**Module** : M2442 — Jeux Vidéo et Intelligence Artificielle
**Groupe** : Aymane El Akkioui & Zakariae Bellil  
**Établissement** : ENSIAS 
**Enseignant** : Pr. Youness BOUTYOUR  
**Semestre** : S4-P2 Année 2025-2026  

---

## Table des matières

1. [Description du projet](#1-description-du-projet)
2. [Fondements théoriques](#2-fondements-théoriques)
3. [Environnement Gymnasium](#3-environnement-gymnasium)
4. [Algorithme génétique](#4-algorithme-génétique)
5. [Agents de comparaison](#5-agents-de-comparaison)
6. [Protocole expérimental](#6-protocole-expérimental)
7. [Résultats et analyse](#7-résultats-et-analyse)
8. [Structure du projet](#8-structure-du-projet)
9. [Installation et utilisation](#9-installation-et-utilisation)
10. [Analyse critique et pistes d'amélioration](#10-analyse-critique-et-pistes-damélioration)

---

## 1. Description du projet

Ce projet est l'évaluation finale du module M2442. Il mobilise l'ensemble des notions vues en cours autour de la **modélisation d'environnements interactifs**, de l'**API Gymnasium** et de la **conception d'agents basés sur les algorithmes génétiques**.

### Sujet choisi : Positionnement tactique et esquive sur terrain de football quadrillé

Un agent contrôle un joueur de football sur une grille 2D de 10×10 cases. L'agent doit naviguer intelligemment depuis sa position de départ (bas du terrain) vers la **zone de but** (haut du terrain), tout en **esquivant trois défenseurs adverses** qui se déplacent dynamiquement. L'objectif global est d'optimiser les victoires sur une série d'épisodes simulant une saison pour assurer le maintien de l'équipe dans la ligue.

```
+----------+
|.....G....|   ← zone de but (G)
|..........|
|....D.....|   ← défenseurs (D) — mobiles, déplacement aléatoire
|..........|
|...D......|
|..........|
|......D...|
|..........|
|..........|
|.....A....|   ← agent (A) — position initiale
+----------+
```

**Points clés du système** (selon Chapitre 1 du cours) :
- Interaction continue : boucle pas-à-pas entre l'agent et l'environnement
- Dynamique temporelle : les défenseurs se déplacent à chaque pas
- Règles ludiques : victoire si but atteint, défaite si capturé
- Incertitude événementielle : comportement stochastique des défenseurs

---

## 2. Fondements théoriques

### 2.1 Le jeu comme système interactif (Chapitre 1)

Selon le cours, un jeu vidéo est un **système interactif** composé de cinq éléments fondamentaux :

| Élément | Dans notre projet |
|---|---|
| **Joueur / Agent** | L'agent football (contrôlé par l'AG ou une baseline) |
| **Environnement** | La grille 10×10 avec défenseurs |
| **Actions** | 4 directions : haut, bas, gauche, droite |
| **Règles** | Bords bloquants, capture par contact, but en zone cible |
| **Feedback** | Récompense numérique à chaque pas |

L'environnement implémente le **modèle PDA** (Perception–Décision–Action) vu en cours :
- **Perception** : observation normalisée des positions (agent, défenseurs, but)
- **Décision** : sélection d'action via la politique de l'agent
- **Action** : déplacement sur la grille, mise à jour de l'état

La dynamique d'état suit la formulation du cours : **s_{t+1} = f(s_t, a_t, Δt)**, où chaque pas de temps correspond à un déplacement simultané de l'agent et des défenseurs.

### 2.2 Algorithmes génétiques (Chapitre 4)

Les algorithmes génétiques sont une méthode d'optimisation **inspirée de l'évolution naturelle** (Holland, 1965) qui cherche de bonnes solutions dans de grands espaces complexes sans nécessiter de gradient ni de propriétés mathématiques fortes de la fonction objectif.

**Pourquoi les AG sont adaptés à ce problème ?**
- L'espace de recherche (poids d'une fonction d'utilité) est continu et multidimensionnel
- Il n'existe pas de formule analytique pour calculer la politique optimale
- On veut explorer plusieurs stratégies en parallèle (population d'individus)
- L'équilibre exploration / exploitation est assuré par sélection + mutation

**Correspondance biologique ↔ informatique** (Chapitre 4) :

| Terme biologique | Dans ce projet |
|---|---|
| Individu | Un vecteur de 8 poids réels |
| Population | 50 individus |
| Gène | Un poids w_i ∈ [-1, 1] |
| Fitness | Retour cumulé moyen sur K épisodes |
| Génération | Une itération complète de l'AG |
| Sélection | Tournoi de taille k=3 |
| Croisement | Arithmétique (interpolation linéaire) |
| Mutation | Gaussienne (σ=0.1, p=0.3 par gène) |
| Élitisme | Conservation des 2 meilleurs individus |

---

## 3. Environnement Gymnasium

**Fichier** : `env/football_env.py` — classe `FootballEnv(gym.Env)`

### 3.1 Espace d'observation

Vecteur de **10 flottants normalisés dans [0, 1]** :

```
obs = [agent_r, agent_c,
       def0_r,  def0_c,
       def1_r,  def1_c,
       def2_r,  def2_c,
       goal_r,  goal_c]
```

Toutes les coordonnées sont divisées par (GRID_SIZE − 1) = 9. Cette normalisation est cohérente avec les bonnes pratiques de représentation continue vues dans le Chapitre 4 (codage réel).

### 3.2 Espace d'actions

`Discrete(4)` — espace discret simple comme exigé par le sujet :

| Indice | Direction | Δ(ligne, col) |
|---|---|---|
| 0 | Haut ↑ | (−1, 0) |
| 1 | Bas ↓ | (+1, 0) |
| 2 | Gauche ← | (0, −1) |
| 3 | Droite → | (0, +1) |

Les déplacements hors grille sont bloqués par clipping (bords comme murs).

### 3.3 Fonction de récompense

À chaque pas, la récompense est calculée comme suit :

```
r_t = R_pas + R_approche + R_terminal
```

| Composante | Valeur | Description |
|---|---|---|
| `R_pas` | −1 | Coût constant par pas (encourage l'efficacité) |
| `R_approche` | +2 × Δdist | Bonus si l'agent se rapproche du but (Δdist = réduction de distance Manhattan) |
| `R_but` | +100 | Récompense terminale si la zone de but est atteinte |
| `R_capture` | −100 | Pénalité terminale si un défenseur occupe la même case |

**Propriété importante** : la somme des `R_approche` sur un épisode est égale à `2 × (dist_initiale − dist_finale)`, indépendamment du chemin emprunté (somme télescopique). Un épisode victorieux rapporte donc toujours au minimum −81 points, et un épisode de capture au maximum −85 points — les deux cas sont strictement séparés.

### 3.4 Conditions de fin d'épisode

| Condition | Type | Déclencheur |
|---|---|---|
| But atteint | `terminated = True` | Agent sur la case de but |
| Capture | `terminated = True` | Agent sur la même case qu'un défenseur (vérifié après les déplacements) |
| Timeout | `truncated = True` | Nombre de pas ≥ MAX_STEPS (200) |

**Note** : le but est vérifié avant la capture ; si un défenseur se trouve sur la case de but au même moment que l'agent, c'est un but (pas une capture).

### 3.5 Dynamique des défenseurs

À chaque pas, chaque défenseur choisit une direction uniformément aléatoire parmi les 4 disponibles (déplacement également clippé aux bords). Cela crée une **incertitude événementielle** qui justifie l'évaluation de la fitness sur plusieurs épisodes.

---

## 4. Algorithme génétique

**Fichier** : `agents/genetic_agent.py` — classes `GeneticAgent` et `GeneticAlgorithm`

### 4.1 Représentation du génome (codage réel)

Chaque individu est un **vecteur de 8 poids réels** (codage réel, Chapitre 4 §III) :

```
génome = [w1, w2, w3, w4, w5, w6, w7, w8]  ∈ [-1, 1]^8
```

Ces poids encodent une **fonction d'utilité linéaire** qui score chaque action candidate :

```
score(action) = génome · features(obs, action)
```

La politique est purement **greedy** : `action = argmax_a score(a)`.

### 4.2 Extraction des features

Pour chaque action candidate, 8 features décrivent l'état résultant simulé :

| Feature | Formule | Interprétation |
|---|---|---|
| f1 | −dist_but / dist_max | Proximité du but (plus élevé = plus proche) |
| f2 | dist_min_def / dist_max | Marge de sécurité (défenseur le plus proche) |
| f3 | dist_moy_def / dist_max | Sécurité moyenne (tous défenseurs) |
| f4 | Δdist_but / grid_size | Delta d'approche vers le but |
| f5 | 0 ou 1 | Mouvement valide (non bloqué par un bord) |
| f6 | 1 − row / grid_size | Progression verticale (1 = rangée du but) |
| f7 | −1 ou 0 | Indicateur de danger immédiat (dist ≤ 1) |
| f8 | 1.0 | Biais constant |

L'AG optimise ces 8 poids sur 100 générations pour trouver la combinaison qui maximise le retour cumulé moyen.

### 4.3 Opérateurs génétiques

#### Initialisation
Population de 50 individus avec poids tirés uniformément dans [−1, 1], garantissant une **diversité initiale élevée** (Chapitre 4 §II).

#### Évaluation de la fitness
```
F(génome) = moyenne des retours cumulés sur K=5 épisodes
```
Tous les individus d'une même génération sont évalués sur les **mêmes seeds d'épisodes** pour garantir une comparaison équitable. Les seeds sont tirés dans [0, 100 000), disjoints des seeds d'évaluation finale.

#### Sélection par tournoi (k=3)
Conformément au Chapitre 4 §IV : on tire aléatoirement 3 individus sans remise, et le vainqueur (fitness maximale) est sélectionné comme parent. Cette méthode offre un **bon équilibre exploration/exploitation** — k=3 est un choix classique qui évite la convergence prématurée de la sélection par roulette tout en maintenant une pression de sélection raisonnable.

#### Croisement arithmétique
Pour deux parents P1 et P2, avec α ~ Uniform[0, 1] :
```
Enfant1 = α·P1 + (1−α)·P2
Enfant2 = (1−α)·P1 + α·P2
```
Ce croisement est particulièrement adapté au **codage réel** (Chapitre 4 §IV, croisement arithmétique). Les enfants restent dans l'enveloppe convexe des parents, préservant la validité des solutions.

#### Mutation gaussienne
Chaque gène est muté avec probabilité p_m = 0.3 :
```
w'_i = w_i + ε,  ε ~ N(0, σ²)  avec σ = 0.1
```
La **mutation réelle gaussienne** (Chapitre 4 §IV) introduit des perturbations locales qui maintiennent la diversité sans détruire les bonnes solutions. σ=0.1 est une valeur classique pour des gènes dans [−1, 1].

#### Élitisme top-2
Les 2 meilleurs individus de chaque génération sont **copiés directement** dans la génération suivante sans modification. Ce mécanisme (Chapitre 4 §IV, élitisme) garantit que la meilleure fitness observée ne se dégrade jamais d'une génération à l'autre.

### 4.4 Hyperparamètres

| Paramètre | Valeur | Justification |
|---|---|---|
| Taille population (N) | 50 | Équilibre diversité / coût de calcul |
| Nombre de générations (T) | 100 | Convergence observée dès la génération 3-6 |
| Taille tournoi (k) | 3 | Pression de sélection modérée |
| Taux de mutation (p_m) | 0.3 par gène | Diversité maintenue sans déstabilisation |
| Écart-type mutation (σ) | 0.1 | Perturbations locales fines |
| Élites conservés | 2 | Préservation sans domination excessive |
| Épisodes par évaluation (K) | 5 | Estimation stable de la fitness stochastique |
| Seed maître | 42 | Reproductibilité totale |

---

## 5. Agents de comparaison

Le projet inclut deux **baselines** pour contextualiser les performances de l'AG, conformément aux exigences du sujet. Les trois agents partagent la même interface `select_action(obs) -> int`.

### 5.1 RandomAgent (`agents/random_agent.py`)

Sélectionne une action **uniformément aléatoire** à chaque pas, sans exploiter l'observation. Représente la borne inférieure de performance — tout agent intelligent doit la dépasser largement.

### 5.2 HeuristicAgent (`agents/heuristic_agent.py`)

À chaque pas, simule les 4 mouvements possibles et choisit celui qui **minimise la distance de Manhattan** au but. Politique **déterministe et gloutonne**, ignore complètement les défenseurs. Représente la borne supérieure d'une stratégie naïve mono-objectif.

---

## 6. Protocole expérimental

**Fichier** : `experiments/run_experiment.py`

### 6.1 Phase d'entraînement

- AG entraîné sur `FootballEnv(seed=42)` avec les hyperparamètres décrits en §4.4
- Seeds d'épisodes internes : tirés aléatoirement dans [0, 100 000) à chaque génération
- Durée : ~98 secondes (Python 3.14, machine standard)

### 6.2 Phase d'évaluation

- **50 épisodes** par agent (≥ 30 comme exigé)
- Seeds d'évaluation fixes : 200 000 à 200 049 (disjoints des seeds d'entraînement)
- Même environnement pour tous les agents (comparaison équitable)
- Métriques collectées par épisode : retour cumulé, nombre de pas, résultat (but / capture / timeout)

### 6.3 Métriques calculées

| Métrique | Définition |
|---|---|
| **Taux de victoire** | Proportion d'épisodes terminés par un but |
| **Score moyen** | Moyenne des retours cumulés sur 50 épisodes |
| **Écart-type du score** | Variabilité des performances |
| **Pas moyen** | Durée moyenne d'un épisode |

### 6.4 Reproductibilité

Toutes les sources d'aléatoire sont contrôlées par `MASTER_SEED = 42` :
- Initialisation de la population AG
- Sélection des seeds d'évaluation par génération
- Placement initial des défenseurs dans chaque épisode
- Déplacements des défenseurs pendant les épisodes

---

## 7. Résultats et analyse

### 7.1 Courbe de convergence de l'AG

L'AG converge **très rapidement** : dès la génération 3, la moyenne populationnelle atteint ~72, et dès la génération 6, la population est quasi-homogène avec une moyenne > 108. Cela indique que l'espace de recherche est relativement simple pour ce problème — la politique optimale (aller droit vers le but) est facile à découvrir.

Des chutes ponctuelles de la moyenne (générations 23, 40, 64, 79) sont visibles : elles correspondent à des générations évaluées sur des configurations particulièrement difficiles (défenseurs bien placés). Le mécanisme d'élitisme garantit que ces chutes n'affectent pas le meilleur individu global.

### 7.2 Tableau récapitulatif (50 épisodes, seeds 200 000–200 049)

| Agent | Taux de victoire | Score moyen | Écart-type | Pas moyen |
|---|---|---|---|---|
| **Random** | 6.0 % | −133.50 | 62.70 | 47.4 |
| **Heuristic** | 80.0 % | +67.96 | 82.08 | 8.0 |
| **Genetic** | 80.0 % | +67.96 | 82.08 | 8.0 |

### 7.3 Interprétation

**RandomAgent (6 %)** : comme attendu, une politique aléatoire peine à atteindre le but avant d'être capturée ou d'épuiser les 200 pas. Le score moyen très négatif (−133.5) confirme l'importance d'une stratégie dirigée.

**HeuristicAgent et GeneticAgent (80 %)** : les deux agents obtiennent des résultats **identiques** sur ces 50 épisodes. Ce résultat est cohérent et révélateur : l'AG a convergé vers une politique qui accorde un poids dominant au feature f4 (approche du but), ce qui revient exactement à la politique heuristique de descente de gradient Manhattan. Sur une grille simple sans obstacles fixes, la stratégie optimale est effectivement d'aller le plus directement possible vers le but.

**Meilleur génome obtenu** :
```
[-0.744, -0.099, -0.258,  0.854,  0.288,  0.646, -0.113, -0.546]
   w1      w2      w3      w4      w5      w6      w7      w8
```
- w4 = 0.854 : poids dominant sur l'approche du but → confirme la stratégie directe
- w1 = −0.744 : pénalise les positions éloignées du but (cohérent)
- w6 = 0.646 : favorise les rangées hautes (proches du but)
- w2, w3 faibles : les distances aux défenseurs ont peu d'importance car la stratégie directe est déjà efficace

---

## 8. Structure du projet

```
jeuxvideosProject/
│
├── env/
│   └── football_env.py          # Environnement Gymnasium (FootballEnv)
│
├── agents/
│   ├── genetic_agent.py         # GeneticAgent + GeneticAlgorithm
│   ├── heuristic_agent.py       # HeuristicAgent (baseline Manhattan)
│   └── random_agent.py          # RandomAgent (baseline aléatoire)
│
├── experiments/
│   └── run_experiment.py        # Protocole expérimental complet
│
├── results/                     # Sorties générées (ignorées par git)
│   ├── 01_convergence_ag.png    # Figure 1 — Courbe de convergence
│   ├── 02_winrate_comparison.png # Figure 2 — Taux de victoire
│   ├── 03_score_distribution.png # Figure 3 — Distribution des scores
│   ├── best_genome.npy          # Meilleur génome sauvegardé
│   └── eval_results.npy         # Métriques brutes d'évaluation
│
├── demo.py                      # Démonstration visuelle animée (ASCII)
├── requirements.txt             # Dépendances Python
└── README.md                    # Ce fichier
```

---

## 9. Installation et utilisation

### Prérequis

- Python ≥ 3.10 (testé sur Python 3.14.5)
- Environnement virtuel recommandé

### Installation

```bash
# Cloner ou se placer dans le répertoire du projet
cd jeuxvideosProject

# Créer l'environnement virtuel
python -m venv .venv

# Activer le venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Installer les dépendances
pip install -r requirements.txt
```

**Dépendances** (`requirements.txt`) :

```
gymnasium>=0.29.0
numpy>=1.26.0
matplotlib>=3.8.0
```

### Lancer l'expérience complète

```bash
# Entraîne l'AG + évalue les 3 agents + génère les figures + affiche le tableau
python experiments/run_experiment.py
```

Durée approximative : **~100 secondes** (100 générations × 50 individus × 5 épisodes).

### Démonstration visuelle

```bash
# Charge le meilleur génome et joue 5 épisodes en mode animé
python demo.py
```

La grille se met à jour en place dans le terminal à chaque pas (délai 0.3 s). Symboles : `A` = agent (vert), `G` = but (jaune), `D` = défenseur (rouge).

### Utiliser les agents séparément

```python
from env.football_env     import FootballEnv
from agents.genetic_agent import GeneticAgent
import numpy as np

genome = np.load("results/best_genome.npy")
agent  = GeneticAgent(genome)
env    = FootballEnv(seed=42)

obs, _ = env.reset()
done   = False
while not done:
    action = agent.select_action(obs)
    obs, reward, term, trunc, info = env.step(action)
    done = term or trunc
```

---

## 10. Analyse critique et pistes d'amélioration

### Limites identifiées

**1. Convergence trop rapide / problème trop simple**
L'AG converge dès la génération 3-6, ce qui suggère que la surface de fitness est unimodale pour ce problème. La grille 10×10 avec des défenseurs aléatoires ne crée pas suffisamment de pression de sélection pour différencier les stratégies sophistiquées des stratégies naïves.

**2. Indiscernabilité AG / Heuristique**
Le fait que l'AG et l'heuristique obtiennent des résultats identiques est révélateur : sur 50 épisodes, les 20 % d'échecs sont entièrement dus au hasard des défenseurs (pas à la qualité de la politique). L'AG n'apprend pas à esquiver — il apprend que la meilleure stratégie est de sprinter vers le but.

**3. Features non exploitées**
Les features de sécurité (f2, f3, f7) ont des poids faibles dans le meilleur génome. Cela indique que l'environnement actuel ne crée pas suffisamment de situations où l'esquive est nécessaire.

**4. Évaluation sur K=5 épisodes**
L'estimation de fitness sur 5 épisodes est bruitée. Avec des défenseurs stochastiques, un individu médiocre peut obtenir une bonne fitness par chance. K=10 ou K=20 donnerait des estimations plus stables.

### Pistes d'amélioration

| Axe | Proposition | Impact attendu |
|---|---|---|
| **Environnement** | Grille 15×15 avec obstacles fixes | Plus de situations d'esquive nécessaires |
| **Défenseurs** | Comportement dirigé (poursuite de l'agent) | Pression de sélection plus forte, différencie mieux les stratégies |
| **Fitness** | Augmenter K (10-20 épisodes) | Réduction du bruit, meilleure sélection |
| **Génome** | Ajouter des features non-linéaires | Représentation plus expressive |
| **AG** | Taux de mutation adaptatif (décroissant) | Exploration forte au début, exploitation en fin |
| **Évaluation** | Diversifier les seeds entre générations | Éviter l'overfitting à un sous-ensemble de configurations |
| **Multi-objectif** | Maximiser but ET minimiser pas ET maximiser survie | AG multi-objectif (NSGA-II) |
| **Saisonnalité** | Adapter la politique selon l'historique de la saison | Problème de contrôle adaptatif sur plusieurs épisodes |

---

*Projet réalisé dans le cadre du module M2442 — Jeux Vidéo IA, ENSIAS, Année 2025-2026 par Zakariae Bellil et Aymane ElAkkioui*
