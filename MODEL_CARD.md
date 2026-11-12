# Model Card - fraude (version servie par l'alias production)

## Modele
Nom et version dans le registre MLflow : fraude, version 1
Famille et reglages : RandomForestClassifier (n_estimators=500, max_depth=16, random_state=42)
Run d'origine (MLflow) : 8119808af56b49399499d5e088ca4251

## Objectif et usage prevu
Detection automatique et scoring des transactions financieres suspectes de fraude bancaire en temps reel afin d'alerter le systeme de surveillance et d'orienter les revues humaines.

## Usages exclus
Ne jamais utiliser ce modele pour le blocage definitif ou la cloture arbitraire d'un compte client sans validation pralable par un conseiller bancaire humain. Ne convient pas aux transactions d'entreprises (corporate) ni aux devises etrangeres.

## Donnees
Fichier, nombre de lignes, part de fraudes, empreinte (tag donnees_md5) : data/transactions.csv, 5 000 transactions, 568 fraudes (11,36 %), md5: cf9f6f86c141e7df32c61715c493483f
Origine des donnees (reelles ou synthetiques) : Donnees synthetiques generees a partir de lois statistiques (log-normale, Poisson, binomiale) reproduisant les profils d'activite.

## Performances (jeu de test, 1 250 transactions)
F1 : 0.8669 | precision : 0.8837 | rappel : 0.8507 | AUC : 0.9872 (114 fraudes detectees sur 134)

## Methodes d'explicabilite
Globale : Importance par permutation (perte de F1 sur test) et TreeExplainer SHAP moyen (|SHAP| sur 400 transactions de test).
Locale : SHAP TreeExplainer (decompositions additives waterfall exactes) et LIME Tabular Explainer (regles locales stochastiques a 2 000 voisins).
Variables les plus influentes : Montant de la transaction (predominant dans les deux methodes), horaire nocturne (nuit), anciennete du compte, et frequence sur 24h.

## Comparaison avec un modele interpretable
La regression logistique avec standardisation et log10(montant) atteint F1 = 0.8593 et detecte 113 fraudes sur 134 (soit seulement 1 fraude de moins que la foret), tout en etant directement interpretable en termes de multiplicateur de cote. Les arbres simples (profondeur 3) sont trop rigides (F1 = 0.5737, 72 fraudes).

## Limites et risques connus
Risque de generalisation lie au caractere synthetique des donnees d'apprentissage. Instabilite des explications locales LIME selon la graine d'echantillonnage. SHAP reflete les correlations apprises par le modele et non des liens de causalite economique.

## Recommandations d'utilisation et de surveillance
Utiliser le seuil de decision pour lever des alertes et declencher des verifications (ex: SMS 2FA) plutot qu'un rejet ferme. Surveiller la distribution des montants (data drift) et recalibrer trimestriellement les seuils de detection.

## Tracabilite
Run des explications (experience tp7-explicabilite) : 6e839b0541c84556ac86e4e6c2279481
Auteurs : Equipe MLOps 5IASD EMSI
Date : 12/11/2026
Prochaine revue : 12/05/2027
