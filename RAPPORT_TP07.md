# Rapport de TP07 : Explicabilité et Auditabilité des Modèles de Machine Learning
**Filière :** 5IASD (Déploiement d'IA / MLOps 2026-2027)  
**Établissement :** École Marocaine des Sciences de l'Ingénieur (EMSI)  
**Étudiant :** AYAT Youssef  
**Dépôt GitHub :** [mlops-tp7-xai](https://github.com/ayatyoussef/mlops-tp7-xai)  

---

## 1. Introduction et Objectifs du TP

Dans les TP précédents, nous avons entraîné, versionné et déployé un modèle de détection de fraude bancaire dans le registre MLflow sous le nom `fraude@production`. Cependant, un modèle haute performance ne suffit pas en milieu bancaire réglementé : il doit être **explicable** (comprendre pourquoi une transaction donnée a été bloquée) et **auditable** (pouvoir retracer et justifier a posteriori chaque décision prise avec son contexte exact).

Ce TP met en œuvre la chaîne complète :
$$\text{DVC (données)} \longrightarrow \text{MLflow (registre/modèles)} \longrightarrow \text{SHAP / LIME (explications)} \longrightarrow \text{Model Card (audit & gouvernance)}$$

---

## 2. Réponses aux Questions de Validation (Partie IV)

### Question 1 : Le registre et l'alias
*Pourquoi charge-t-on le modèle par `models:/fraude@production` plutôt que par un fichier ou un numéro de version ? Si l'alias était déplacé vers la version 2, que deviendraient les explications de l'activité 6, et comment le saurait-on ?*

- **Pourquoi l'alias :** L'URI logique `models:/fraude@production` offre un découplage total entre le code applicatif client et la version physique du modèle. Le service d'inférence interroge toujours la version certifiée en cours d'exploitation sans nécessiter de recompilation ni de modification de configuration lors d'un déploiement. De plus, MLflow garantit le chargement de l'environnement exact (`requirements.txt`).
- **Déplacement vers la version 2 :** Les explications enregistrées à l'Activité 6 restent attachées de manière immuable à la **version 1** dans l'historique de MLflow. En effet, le script `journaliser.py` a figé dans les tags `modele_version: 1`, `modele_run_id: 8119808a...`, et a rattaché le tag `explications_run` spécifiquement à la version 1 du modèle dans le registre. La version 2 n'aura pas ce tag tant qu'une campagne d'explicabilité dédiée n'aura pas été exécutée sur elle.

---

### Question 2 : Deux importances globales
*La MDI et la permutation donnent-elles le même classement ? Expliquez la différence observée sur `nuit` (0,093 contre 0,196) à partir de la façon dont chacune est calculée.*

- **Comparaison des classements :**
  - **MDI (Gini forêt sur train) :** `montant` (0.479) > `anciennete` (0.237) > `freq_24h` (0.191) > `nuit` (0.093).
  - **Permutation (perte de F1 sur test) :** `montant` (0.593) > `freq_24h` (0.269) > `anciennete` (0.268) > `nuit` (0.196).
  Le classement diverge : la MDI surévalue `anciennete` par rapport à `freq_24h`.
- **Explication pour la variable `nuit` :**
  - La **MDI (Mean Decrease in Impurity)** est calculée sur le jeu d'entraînement lors de la construction des arbres. Elle souffre d'un biais mathématique structurel bien connu : elle favorise systématiquement les variables continues ou à forte cardinalité (`montant`, `anciennete`) car elles offrent une multitude de seuils de coupure possibles, au détriment des variables binaires ($0/1$) comme `nuit` qui ne disposent que d'un unique point de scission.
  - L'**importance par permutation** est calculée sur le jeu de test indépendant et mesure la chute effective du $F_1\text{-score}$ quand les valeurs de la colonne sont mélangées aléatoirement. Lorsqu'on perturbe `nuit`, le $F_1$ s'effondre de près de $0.20$ point (0.196), prouvant que l'horaire nocturne est en réalité un discriminateur critique pour la détection des fraudes en généralisation.

---

### Question 3 : Lire le beeswarm
*Dans votre graphique SHAP global, dans quel sens agit l'ancienneté ? Qu'apporte ce graphique que la permutation ne dit pas ?*

- **Sens de l'ancienneté :** Sur le beeswarm plot (`reports/shap_global.png`), les points situés à droite de zéro (valeur SHAP positive, augmentant la probabilité de fraude) sont de couleur **bleue**, ce qui correspond à des valeurs faibles d'ancienneté (comptes récents). Les points à gauche (valeur SHAP négative, réduisant le risque) sont de couleur **rouge** (comptes anciens). Par conséquent, **une faible ancienneté augmente le risque de fraude**, tandis qu'un compte ancien sécurise la transaction.
- **Apport par rapport à la permutation :** L'importance par permutation ne fournit qu'un scalaire agrégé (l'amplitude globale de dépendance), sans indiquer :
  1. La **direction** de l'effet (si la valeur augmente ou diminue le risque) ;
  2. La **distribution et variabilité** des effets individuels ;
  3. Les **non-linéarités et seuils critiques** (ex: dans `shap_montant.png`, la contribution explose au-delà de 1 000 MAD).

---

### Question 4 : Global et local
*Pour Mme Benali, `nuit` est la première cause du blocage ; dans l'importance SHAP globale, elle n'est que troisième. Est-ce une contradiction ? Vérifiez aussi la propriété d'efficacité sur vos chiffres.*

- **Absence de contradiction :** Non. L'importance globale est une **moyenne** sur l'ensemble de la population de test, où 82 % des transactions ont lieu le jour (`nuit = 0`), pour lesquelles `nuit` a une contribution proche de zéro ou légèrement négative. En revanche, pour Mme Benali, l'opération a lieu en pleine nuit (`nuit = 1`), ce qui active pleinement la condition de risque (+0.3697). L'explication locale isole la spécificité du cas individuel.
- **Vérification de la propriété d'efficacité (Efficiency Property) :**
  $$\text{Base } E[f(X)] = 0.1157$$
  $$\sum \phi_i = (+0.3697) + (+0.3665) + (+0.0903) + (-0.0183) = +0.8082$$
  $$\text{Base } + \sum \phi_i = 0.1157 + 0.8082 = 0.9239 \approx 0.924$$
  $$\text{Prédiction directe } f(x) = P(\text{fraude}) = 0.9240$$
  L'égalité mathématique est vérifiée avec une exactitude parfaite ($0.924 = 0.924$).

---

### Question 5 : La phrase pour la cliente
*Rédigez la phrase destinée à Mme Benali. Pour la transaction 668, `nuit` a une contribution négative : pourquoi ?*

- **Phrase client pour Mme Benali :**
  > *« Madame Benali, votre paiement de 1 800 MAD a été temporairement suspendu pour des raisons de sécurité en raison de son montant élevé combiné à une exécution inhabituelle en pleine nuit (entre 0h et 5h), ce qui requiert votre confirmation préalable. »*
- **Pourquoi `nuit` est négatif pour la transaction 668 :** Pour la transaction 668, `nuit = 0.0` (transaction réalisée de jour). L'absence d'horaire nocturne écarte un facteur de risque majeur, tirant la prédiction vers la classe « normale » ($\phi_{\text{nuit}} = -0.0410$). Si elle est quand même prédite fraude à $99.6\%$, c'est dû à son montant exceptionnel (2 661.92 MAD, $\phi = +0.5221$), sa très faible ancienneté (11 mois, $\phi = +0.2185$) et sa fréquence très élevée (6 transactions en 24h, $\phi = +0.1807$).

---

### Question 6 : La stabilité de LIME
*Quelles contributions de LIME restent stables sur les cinq graines, lesquelles changent de signe ? Quelle règle adopteriez-vous avant de communiquer une explication à un client ?*

- **Tableau de synthèse LIME (2 000 voisins sur 5 graines) :**
  - `montant > 1007.36` : reste systématiquement positive (+0.482, +0.428, +0.440, +0.456, +0.455) $\to$ **Stable : Oui**.
  - `nuit > 0.00` : reste systématiquement positive (+0.183, +0.237, +0.181, +0.206, +0.230) $\to$ **Stable : Oui**.
  - `anciennete` : positive faible (+0.020 à +0.033 à 2000 voisins, mais oscillait de -0.008 à +0.060 à 500 voisins).
  - `freq_24h` : **change de signe** selon le tirage (+0.002, -0.037, +0.012, -0.014, -0.003) $\to$ **Stable : Non**.
- **Règle d'usage client :** Ne jamais motiver un blocage ou un refus client sur une variable dont le poids est instable ou susceptible de changer de signe lors d'un rééchantillonnage. Toujours limiter les explications aux prédicteurs majeurs invariables sur plusieurs tirages (ici le montant et l'horaire de nuit).

---

### Question 7 : SHAP ou LIME
*SHAP et LIME désignent-ils les mêmes causes principales pour Mme Benali ? Pourquoi ne faut-il pas comparer directement leurs nombres ? Laquelle des deux méthodes choisiriez-vous pour ce modèle, et pourquoi ?*

- **Accord sur les causes :** Oui, les deux méthodes identifient les deux mêmes facteurs dominants : le montant excessif et la nuit.
- **Incomparabilité directe des nombres :**
  - Les valeurs SHAP sont des contributions additives rigoureuses issues de la théorie des jeux coopératifs sommant exactement à la probabilité de sortie du modèle.
  - LIME estime les coefficients d'un modèle linéaire local pénalisé (Ridge) sur un voisinage perturbé pondéré par une métrique de distance. Leurs échelles et unités mathématiques sont totalement différentes.
- **Choix pour ce modèle :** **SHAP (avec `TreeExplainer`)** est la méthode à privilégier car :
  1. Elle est déterministe et reproductible (aucun aléa d'échantillonnage de voisinage) ;
  2. Elle est exacte et ultra-rapide pour les arbres et forêts d'arbres décisionnels ;
  3. Elle garantit l'efficacité et la cohérence mathématique des attributions.

---

### Question 8 : Le compromis performance / lisibilité
*La logistique avec `log10(montant)` détecte 113 fraudes sur 134, la forêt 114. Quel modèle mettriez-vous en production pour bloquer des paiements ? Justifiez avec l'écart de performance (et le bruit du jeu de test) et la lisibilité.*

- **Données observées :**
  - Forêt aléatoire : $F_1 = 0.8669$, 114 fraudes détectées / 134.
  - Régression Logistique avec $\log_{10}(\text{montant})$ : $F_1 = 0.8593$, 113 fraudes détectées / 134.
- **Recommandation :** L'écart n'est que de **1 seule transaction** sur 134 fraudes de test, ce qui est inférieur au seuil de signification statistique (bruit d'échantillonnage). En contrepartie, la régression logistique offre une transparence totale par conception : chaque variable possède un coefficient direct ($\times 4.76$ sur la cote de fraude par transaction en 24h, $\times 0.928$ par mois d'ancienneté). Pour bloquer des paiements dans un secteur fortement régulé par le RGPD et l'AI Act, **la régression logistique standardisée avec $\log_{10}(\text{montant})$ est le choix le plus robuste et justifiable**.

---

### Question 9 : Reconstituer une décision dans 6 mois
*Dans six mois, Mme Benali conteste. Quels éléments enregistrés dans ce TP permettent de rejouer la décision et son explication ? Lequel manque encore pour un audit complet ?*

- **Éléments disponibles enregistrés :**
  1. L'artefact sérialisé du modèle versionné (`fraude v1`) et ses hyperparamètres enregistrés dans MLflow ;
  2. L'environnement logiciel exact (`requirements_modele.txt`) ;
  3. Le run MLflow d'explications (`xai-fraude-v1`, `6e839b05`), incluant le hash MD5 des données (`cf9f6f86...`), les visualisations SHAP et LIME et les graines fixées ;
  4. La `MODEL_CARD.md` décrivant les usages, limites et métriques.
- **Ce qui manque pour un audit complet de production :** Le **journal d'inférence en temps réel (Payload Logging)** : la capture horodatée de la requête d'origine (payload JSON reçu par l'API avec son identifiant de corrélation unique), l'état exact des données contextuelles du compte au moment exact de la transaction, et la trace d'exécution de la décision renvoyée au système bancaire.

---

### Question 10 : La Model Card
*Citez un usage exclu et une limite de votre Model Card, et justifiez chacun par un résultat du TP.*

- **Usage exclu :** *"Ne jamais utiliser ce modèle pour le blocage définitif ou la résiliation d'un compte client sans validation par un conseiller humain."*  
  *Justification :* Le modèle présente une précision de $88.37\%$, ce qui signifie qu'il engendre **15 faux positifs** sur le jeu de test (matrice : `[[1101, 15], [20, 114]]`). 15 clients légitimes seraient arbitrairement pénalisés sans recours humain.
- **Limite :** *"Instabilité stochastique des explications locales produites par LIME."*  
  *Justification :* Les résultats de l'activité 4 démontrent que le coefficient de la variable `freq_24h` oscille entre $+0.012$ et $-0.037$ sur 5 graines aléatoires différentes avec 2 000 voisins, ce qui empêche de fonder une décision réglementaire irréfutable sur LIME seul.

---

## 3. Réponses aux Questions Complémentaires (Bonus)

### Question C1 : FLAML et Explicabilité
- **C1.a (Écart de F1 et variables) :**
  - $F_1$ FLAML (modèle `XGBClassifier`) : $0.8582$ vs Forêt : $0.8669$. L'écart ($0.0087$) est inférieur à 1 point de F1 et ne dépasse pas la variance naturelle d'un jeu de 134 fraudes.
  - Les deux modèles s'appuient sur les **mêmes variables exactement dans le même ordre de permutation** :
    1. `montant` (FLAML : 0.575 | Forêt : 0.593)
    2. `freq_24h` (FLAML : 0.275 | Forêt : 0.269)
    3. `anciennete` (FLAML : 0.253 | Forêt : 0.268)
    4. `nuit` (FLAML : 0.182 | Forêt : 0.196)
- **C1.b (Explicabilité AutoML et Model Card) :**
  - Un modèle AutoML est plus complexe à expliquer car l'algorithme sous-jacent (ex: XGBoost avec arbres de gradient boosting) n'est pas prédéterminé, et ses valeurs SHAP sont exprimées dans l'espace des log-cotes non normalisées.
  - *Ajouts à la Model Card :* Budget de recherche (60s), famille sélectionnée (`xgboost`), graines d'AutoML (seed 42), et échelle des attributions SHAP (log-cotes).

### Question C2 : Orchestration Prefect
- **C2.a (Explicabilité vs Orchestration & Garde-fou) :**
  - *Explicabilité :* Production des valeurs SHAP/LIME et rendu visuel.
  - *Orchestration :* Chaînage ordonné des tâches, journalisation unifiée, contrôle de validité et gestion des pannes.
  - *Échec du garde-fou :* Lorsque `nuit` est retirée des variables autorisées, `controler` lève un `ValueError("variables non autorisees : ['nuit']")`. Prefect bascule la tâche et le flow en `Failed`, ce qui **bloque l'exécution de `journaliser`**. C'est le comportement attendu pour interdire la persistance d'artefacts non conformes.
- **C2.b (Cycle de vie MLOps) :** Ce flow doit être exécuté automatiquement **à la fin de chaque réentraînement** (dans le pipeline du TP06) et **avant toute promotion d'un nouveau modèle vers l'alias `production`**, afin de valider à la fois la performance et la conformité explicative.

---

## 4. Tableaux Récapitulatifs

### Tableau Activité 4 : Stabilité SHAP vs LIME (Cas Mme Benali)
| Variable | SHAP (Activité 3) | LIME (2 000 voisins, graine 0) | Stable sur 5 graines ? |
| :--- | :---: | :---: | :---: |
| **montant** | $+0.367$ | $+0.482$ | **Oui** (valeurs de $+0.428$ à $+0.482$) |
| **nuit** | $+0.370$ | $+0.183$ | **Oui** (valeurs de $+0.181$ à $+0.237$) |
| **anciennete** | $+0.090$ | $+0.026$ | **Oui** (faiblement positive, $+0.020$ à $+0.033$) |
| **freq_24h** | $-0.018$ | $+0.002$ | **Non** (change de signe : $-0.037$ à $+0.012$) |

### Tableau Activité 5 : Compromis Performance / Interprétabilité
| Modèle | F1-Score | Fraudes détectées | Ce que l'on peut expliquer à la cliente sans outil |
| :--- | :---: | :---: | :--- |
| **Forêt `fraude@production`** | **0.8669** | **114 / 134** | Rien directement (boîte noire de 500 arbres ; outil SHAP obligatoire). |
| **Logistique (montant brut)** | 0.7407 | 90 / 134 | Effet additif direct par dirham, mais inadéquat aux ordres de grandeur. |
| **Logistique ($\log_{10}(\text{montant})$)** | **0.8593** | **113 / 134** | **Parfaitement lisible :** multiplicateur de cote direct ($\times 4.76$ par transaction en 24h, $\times 0.928$ par mois d'ancienneté). |
| **Arbre prof. 2 (4 feuilles)** | 0.2293 | 18 / 134 | 2 règles simples, mais 116 fraudes manquées (inutilisable). |
| **Arbre prof. 3 (8 feuilles)** | 0.5737 | 72 / 134 | 3 règles simples (« Si montant > 1505.7 et nuit > 0.5... »), mais F1 très faible. |
| **Arbre prof. 6 (50 feuilles)** | 0.7510 | 95 / 134 | Arbre trop grand pour être lu sans support visuel. |
| **Arbre sans limite (206 f.)** | 0.7556 | 102 / 134 | Totalement illisible manuellement et surajusté. |

---

## 5. Guide des Captures d'Écran et Interfaces à Consulter

Le professeur demande explicitement de consulter les interfaces web et graphiques. Voici le guide complet :

| Capture | Interface / Fichier à ouvrir | Description de la capture |
| :---: | :--- | :--- |
| **Capture 1** | Terminal VS Code & **MLflow UI** (`http://127.0.0.1:5000/#/models/fraude`) | Sortie console de `src/verifier_modele.py` + onglet **Models** montrant `fraude`, version 1 portant l'alias **production**. |
| **Capture 2** | Terminal & Fichier image `reports/shap_global.png` | Tableau console de permutation + beeswarm plot de l'importance SHAP globale. |
| **Capture 3** | Terminal & Fichier image `reports/shap_benali.png` | Sortie console de `src/xai_local.py` + graphique waterfall SHAP pour Mme Benali. |
| **Capture 4** | Terminal & **Navigateur Web** (`reports/lime_benali.html`) | Sortie console des deux blocs de 5 graines + page HTML interactive générée par LIME. |
| **Capture 5** | **MLflow UI** (`http://127.0.0.1:5000/#/experiments/3`) | Expérience `tp7-explicabilite`, run `xai-fraude-v1`, onglet **Artifacts** (dossier `explications/`) et onglet **Models** montrant le tag `explications_run`. |
| **Capture 6** | **MLflow UI** (`http://127.0.0.1:5000/#/models/fraude/versions/1`) | Page de la version 1 de `fraude` affichant la description complète `MODEL_CARD.md` et le tag `model_card`. |

---

## 6. Arborescence du Dépôt et Historique Git

```
mlops-tp7-xai/
├── .gitignore
├── requirements_modele.txt
├── MODEL_CARD.md
├── RAPPORT_TP07.md
├── data/
│   └── transactions.csv
├── reports/
│   ├── flaml_importance.json
│   ├── flow_lime_benali.html
│   ├── flow_shap_global.png
│   ├── importance.json
│   ├── interpretables.json
│   ├── lime_benali.html
│   ├── lime_benali.png
│   ├── permutation.png
│   ├── run_xai.txt
│   ├── shap_benali.png
│   ├── shap_global.png
│   ├── shap_importance.json
│   ├── shap_montant.png
│   └── shap_test_668.png
└── src/
    ├── flaml_xai.py
    ├── flow_xai.py
    ├── interpretables.py
    ├── journaliser.py
    ├── publier_card.py
    ├── verifier_modele.py
    ├── xai_commun.py
    ├── xai_global.py
    ├── xai_importance.py
    └── xai_lime.py
```

### Historique des 10 commits Git :
1. `chore: environnement du TP07 aligne sur le modele fraude`
2. `feat: chargement de fraude@production et verification`
3. `feat: explications globales (permutation, SHAP)`
4. `feat: explications locales SHAP`
5. `feat: explication LIME et stabilite`
6. `feat: modeles interpretables et compromis`
7. `feat: explications archivees dans MLflow`
8. `docs: model card du modele fraude`
9. `feat(bonus): explicabilite du modele AutoML FLAML (C1)`
10. `feat(bonus): orchestration du pipeline XAI avec Prefect (C2)`
