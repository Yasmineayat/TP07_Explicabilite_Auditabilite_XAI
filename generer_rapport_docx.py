import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from pathlib import Path

doc = docx.Document()

# Styles
for s in doc.styles:
    if 'Normal' in s.name:
        s.font.name = 'Calibri'
        s.font.size = Pt(11)
        s.font.color.rgb = RGBColor(0x22, 0x22, 0x22)

# Colors
PRIMARY = RGBColor(0x1B, 0x4F, 0x72)    # Dark Blue
SECONDARY = RGBColor(0x2E, 0x86, 0xC1)  # Soft Blue
HIGHLIGHT = RGBColor(0xC0, 0x39, 0x2B)  # Crimson

def add_title(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.size = Pt(22)
    run.font.bold = True
    run.font.color.rgb = PRIMARY
    p.paragraph_format.space_after = Pt(4)

def add_subtitle(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.size = Pt(13)
    run.font.italic = True
    run.font.color.rgb = SECONDARY
    p.paragraph_format.space_after = Pt(18)

def add_heading1(text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(16)
    run.font.bold = True
    run.font.color.rgb = PRIMARY
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)

def add_heading2(text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(13)
    run.font.bold = True
    run.font.color.rgb = SECONDARY
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)

def add_p(text, bold_prefix="", italic=False):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(5)
    if bold_prefix:
        r_pre = p.add_run(bold_prefix + " ")
        r_pre.bold = True
        r_pre.font.color.rgb = PRIMARY
    run = p.add_run(text)
    run.italic = italic
    return p

def add_quote(text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.5)
    p.paragraph_format.right_indent = Inches(0.5)
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run(f'« {text} »')
    run.italic = True
    run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

def add_img(path, caption, width=Inches(5.5)):
    if Path(path).exists():
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(6)
        p_img.paragraph_format.space_after = Pt(2)
        p_img.add_run().add_picture(path, width=width)
        
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_after = Pt(10)
        r_cap = p_cap.add_run(f"Figure : {caption}")
        r_cap.font.size = Pt(9.5)
        r_cap.font.italic = True
        r_cap.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

# HEADER
add_title("RAPPORT DE TP07 : EXPLICABILITÉ & AUDITABILITÉ")
add_subtitle("Expliquer, comparer et documenter le modèle de détection de fraude bancaire\nDéploiement d'IA (MLOps) – 5IASD (2026-2027) | EMSI")

# INFOS ETUDIANT
p_info = doc.add_paragraph()
p_info.alignment = WD_ALIGN_PARAGRAPH.CENTER
r_info = p_info.add_run("Étudiant : AYAT Youssef  |  Filière : 5IASD  |  Dépôt GitHub : mlops-tp7-xai\nDestinataire : Pr. H. NACIRI (h.naciri@emsi.ma)")
r_info.font.bold = True
r_info.font.color.rgb = PRIMARY
doc.add_paragraph().paragraph_format.space_after = Pt(12)

# PARTIE 1
add_heading1("1. Contexte, Architecture et Objectifs")
add_p("Ce TP met en œuvre la chaîne de valeur complète de l'explicabilité et de la gouvernance en production d'IA : DVC (données) -> MLflow (registre de modèles & runs) -> SHAP / LIME (explications globales et locales) -> Model Card (auditabilité et conformité réglementaire).")
add_p("Le modèle servi sous l'alias production est une forêt aléatoire de 500 arbres (profondeur 16) entraînée sur 5 000 transactions financières. L'objectif est d'expliquer pourquoi ce modèle classe une transaction comme suspecte et d'assurer une traçabilité totale des décisions à destination des clients, des conseillers et des auditeurs bancaires.")

# PARTIE 2 : QUESTIONS 1 A 10
add_heading1("2. Réponses aux Questions d'Évaluation (Partie IV)")

add_heading2("Question 1 : Le registre et l'alias")
add_p("Pourquoi charge-t-on le modèle par models:/fraude@production plutôt que par un fichier ou un numéro de version ? Si l'alias était déplacé vers la version 2, que deviendraient les explications de l'activité 6, et comment le saurait-on ?", bold_prefix="Énoncé :")
add_p("1. Rôle de l'alias : L'URI models:/fraude@production découple totalement le code d'inférence de la version physique sous-jacente. Elle garantit que le service d'explicabilité s'applique au modèle certifié et actuellement en service actif, tout en préservant les dépendances exactes d'environnement (requirements.txt).\n"
      "2. Déplacement vers la version 2 : Les explications générées à l'activité 6 demeurent immuablement rattachées à la version 1 dans MLflow. Le script de journalisation a consigné explicitement les tags modele_version: 1 et modele_run_id: 8119808a..., et apposé le tag explications_run uniquement sur la version 1 du registre.\n"
      "3. Comment le vérifier : Dans l'interface MLflow Models, la version 2 ne portera pas le tag explications_run (à moins de relancer la journalisation sur elle), signalant immédiatement la divergence à un auditeur.", bold_prefix="Réponse :")

add_heading2("Question 2 : Deux importances globales (MDI vs Permutation)")
add_p("La MDI et la permutation donnent-elles le même classement ? Expliquez la différence observée sur nuit (0,093 contre 0,196) à partir de la façon dont chacune est calculée.", bold_prefix="Énoncé :")
add_p("1. Classement comparatif : Non, les deux méthodes divergent. La MDI classe anciennete (0.237) devant freq_24h (0.191), alors que la permutation place freq_24h (0.269) devant anciennete (0.268).\n"
      "2. Biais de la MDI sur nuit : La MDI (Mean Decrease in Impurity) est mesurée sur le jeu d'entraînement lors de la partition des nœuds. Elle favorise structurellement les variables continues à forte dispersion (montant, ancienneté) qui offrent de multiples seuils de coupe, au détriment de la variable binaire nuit (0 ou 1).\n"
      "3. Réalité de la Permutation : Calculée sur le jeu de test indépendant, la permutation mesure la dégradation réelle de la métrique métier (F1-score). Le mélange de nuit fait chuter le F1 de 0.196, démontrant que l'horaire nocturne est un facteur de détection crucial pour les fraudes réelles non vues.", bold_prefix="Réponse :")
add_img("reports/permutation.png", "Importance par permutation des variables sur le jeu de test (perte de F1)")

add_heading2("Question 3 : Interprétation du Beeswarm SHAP global")
add_p("Dans votre graphique SHAP global, dans quel sens agit l'ancienneté ? Qu'apporte ce graphique que la permutation ne dit pas ?", bold_prefix="Énoncé :")
add_p("1. Sens de l'ancienneté : Sur le beeswarm plot, les points situés à droite de zéro (contribution SHAP positive vers la fraude) sont bleus, ce qui correspond à une valeur faible d'ancienneté (comptes récents). Les points à gauche sont rouges (forte ancienneté). Ainsi, une faible ancienneté accroît le risque de fraude, tandis qu'un compte ancien a un effet protecteur.\n"
      "2. Apport par rapport à la permutation : La permutation fournit un scalaire moyen aveugle à la direction. Le beeswarm révèle la direction d'impact (signe), la dispersion des contributions selon les profils et les seuils critiques (comme la transition nette à 1 000 MAD visible sur shap_montant.png).", bold_prefix="Réponse :")
add_img("reports/shap_global.png", "Beeswarm plot SHAP global sur 400 transactions de test")
add_img("reports/shap_montant.png", "Dépendance SHAP du montant (échelle log) colorée par l'indicateur de nuit")

add_heading2("Question 4 : Décomposition Locale vs Globale et Propriété d'Efficacité")
add_p("Pour Mme Benali, nuit est la première cause du blocage ; dans l'importance SHAP globale, elle n'est que troisième. Est-ce une contradiction ? Vérifiez aussi la propriété d'efficacité sur vos chiffres.", bold_prefix="Énoncé :")
add_p("1. Absence de contradiction : L'importance globale est une moyenne sur la population de test où 82 % des transactions ont lieu de jour (nuit = 0), neutralisant la contribution moyenne de nuit. Pour Mme Benali, la transaction a lieu en pleine nuit (nuit = 1), déclenchant une contribution locale prépondérante (+0.3697). L'explication locale capture la spécificité du cas particulier.\n"
      "2. Vérification de la propriété d'efficacité :\n"
      "   - Valeur de base E[f(X)] = 0.1157\n"
      "   - Somme des contributions = (+0.3697) + (+0.3665) + (+0.0903) + (-0.0183) = +0.8082\n"
      "   - Base + Somme = 0.1157 + 0.8082 = 0.9239 ≈ 0.924\n"
      "   - Probabilité prédite directe f(x) = 0.9240\n"
      "   La propriété d'efficacité mathématique est rigoureusement validée (0.924 = 0.924).", bold_prefix="Réponse :")
add_img("reports/shap_benali.png", "Graphique Waterfall SHAP local pour la transaction de Mme Benali")

add_heading2("Question 5 : Phrase d'explication Client et Transaction 668")
add_p("Rédigez la phrase destinée à Mme Benali. Pour la transaction 668, nuit a une contribution négative : pourquoi ?", bold_prefix="Énoncé :")
add_quote("Madame Benali, votre opération de paiement de 1 800 MAD a été temporairement suspendue par notre dispositif de sécurité en raison de son montant élevé et de son exécution inhabituelle en pleine nuit (entre 0h et 5h), ce qui requiert votre confirmation préalable.")
add_p("Pour la transaction 668, nuit = 0 (opération diurne). L'absence d'anomalie horaire joue en faveur de la normalité, d'où une contribution négative (phi_nuit = -0.0410). Si la probabilité globale atteint néanmoins 0.996, c'est en raison du montant extrême (2 661.92 MAD, phi = +0.5221), de l'ancienneté minimale (11 mois, phi = +0.2185) et d'un pic de fréquence (6 opérations en 24h, phi = +0.1807).", bold_prefix="Analyse transaction 668 :")
add_img("reports/shap_test_668.png", "Graphique Waterfall SHAP local pour la transaction suspecte test_668")

add_heading2("Question 6 : Analyse de Stabilité de LIME")
add_p("Quelles contributions de LIME restent stables sur les cinq graines, lesquelles changent de signe ? Quelle règle adopteriez-vous avant de communiquer une explication à un client ?", bold_prefix="Énoncé :")
add_p("1. Stabilité observée à 2 000 voisins sur 5 graines :\n"
      "   - montant > 1007.36 : Toujours fortement positive (+0.428 à +0.482) -> Stable (Oui).\n"
      "   - nuit > 0.00 : Toujours positive (+0.181 à +0.237) -> Stable (Oui).\n"
      "   - anciennete : Faiblement positive (+0.020 à +0.033) à 2000 voisins (mais instable à 500 voisins : -0.008 à +0.060).\n"
      "   - freq_24h : Change de signe selon le tirage stochastique (+0.002, -0.037, +0.012, -0.014, -0.003) -> Instable (Non).\n"
      "2. Règle métier : Ne jamais justifier une décision bancaire ou juridique sur des variables dont le coefficient LIME est instable ou susceptible de changer de signe. L'explication client doit reposer exclusivement sur les variables invariantes de premier plan.", bold_prefix="Réponse :")
add_img("reports/lime_benali.png", "Explication locale LIME pour Mme Benali (voisinage de 2 000 échantillons)")

add_heading2("Question 7 : Comparaison méthodologique SHAP vs LIME")
add_p("SHAP et LIME désignent-ils les mêmes causes principales pour Mme Benali ? Pourquoi ne faut-il pas comparer directement leurs nombres ? Laquelle des deux méthodes choisiriez-vous pour ce modèle, et pourquoi ?", bold_prefix="Énoncé :")
add_p("1. Causes principales : Oui, les deux méthodes s'accordent sur les deux facteurs déclencheurs majeurs : le montant excessif et l'horaire de nuit.\n"
      "2. Incomparabilité des valeurs : SHAP produit des contributions additives de Shapley issues de la théorie des jeux dont la somme égale la probabilité de sortie. LIME estime les coefficients d'un modèle linéaire local pénalisé (Ridge) ajusté sur un échantillonnage perturbé ; leurs échelles et fondements théoriques diffèrent.\n"
      "3. Choix préconisé : SHAP avec TreeExplainer est nettement supérieur pour les forêts d'arbres. Il est exact, déterministe (reproductibilité absolue sans aléa de tirage) et s'exécute beaucoup plus rapidement sans approximation stochastique.", bold_prefix="Réponse :")

add_heading2("Question 8 : Le compromis Performance / Lisibilité")
add_p("La logistique avec log10(montant) détecte 113 fraudes sur 134, la forêt 114. Quel modèle mettriez-vous en production pour bloquer des paiements ? Justifiez avec l'écart de performance (et le bruit du jeu de test) et la lisibilité.", bold_prefix="Énoncé :")
add_p("1. Analyse de performance : La forêt détecte 114 fraudes (F1 = 0.8669) contre 113 pour la régression logistique avec log10(montant) (F1 = 0.8593). L'écart est d'une unique transaction sur 134 fraudes de test, ce qui correspond au bruit d'échantillonnage statistique.\n"
      "2. Avantage de transparence : La régression logistique offre des multiplicateurs de cote unitaire directs (x 4.76 par transaction sur 24h, x 0.928 par mois d'ancienneté) compréhensibles sans explicateur additionnel.\n"
      "3. Recommandation : Dans un environnement bancaire hautement régulé (RGPD, AI Act), la régression logistique avec log10(montant) est le choix optimal car elle allie quasi-parité de détection et auditabilité native.", bold_prefix="Réponse :")

add_heading2("Question 9 : Reconstitution et Auditabilité dans le temps")
add_p("Dans six mois, Mme Benali conteste. Quels éléments enregistrés dans ce TP permettent de rejouer la décision et son explication ? Lequel manque encore pour un audit complet ?", bold_prefix="Énoncé :")
add_p("1. Éléments sauvegardés : Le modèle exact (fraude v1), ses hyperparamètres, son environnement figé (requirements_modele.txt), le run MLflow xai-fraude-v1 avec le MD5 des données (cf9f6f86...), les paramètres d'explication et les visualisations certifiées.\n"
      "2. Élément manquant pour un audit complet en production : Le journal d'inférence en temps réel (Payload Logging horodaté avec correlation ID) enregistrant la transaction brute exacte telle que reçue par l'API au moment T, le statut des données client au moment de l'événement et la réponse transmise au système monétique.", bold_prefix="Réponse :")

add_heading2("Question 10 : Exploitation de la Model Card")
add_p("Citez un usage exclu et une limite de votre Model Card, et justifiez chacun par un résultat du TP.", bold_prefix="Énoncé :")
add_p("1. Usage exclu : Interdiction de procéder à un blocage définitif ou une clôture de compte sans validation humaine. Justification : La précision du modèle est de 88.37 %, engendrant 15 faux positifs sur le jeu de test (matrice [[1101, 15], [20, 114]]). Sans recours humain, 15 clients honnêtes subiraient un préjudice indu.\n"
      "2. Limite connue : L'instabilité stochastique des explications locales LIME. Justification : À l'activité 4, la variable freq_24h change de signe selon la graine de tirage aléatoire, interdisant son usage comme preuve légale unique.", bold_prefix="Réponse :")

# PARTIE 3 : ACTIVITES COMPLEMENTAIRES
add_heading1("3. Activités Complémentaires (Bonus)")

add_heading2("Activité C1 : FLAML AutoML et Explicabilité")
add_p("1. Résultats empiriques : FLAML a sélectionné un algorithme xgboost (XGBClassifier) atteignant F1 = 0.8582 (contre 0.8669 pour la forêt). L'écart (0.0087) est inférieur au bruit d'échantillonnage de 134 fraudes.\n"
      "2. Concordance des importances par permutation : L'ordre d'importance des variables est strictement identique entre FLAML et la forêt :\n"
      "   1. montant (FLAML : 0.575 | Forêt : 0.593)\n"
      "   2. freq_24h (FLAML : 0.275 | Forêt : 0.269)\n"
      "   3. anciennete (FLAML : 0.253 | Forêt : 0.268)\n"
      "   4. nuit (FLAML : 0.182 | Forêt : 0.196)\n"
      "3. Spécificité SHAP : Pour XGBoost, les valeurs SHAP sont exprimées dans l'espace des log-cotes (montant: 4.209, ancienneté: 2.329), ce qui nécessite d'ajouter à la Model Card la famille retenue et l'unité d'explication.", bold_prefix="Analyse C1 :")

add_heading2("Activité C2 : Orchestration du Pipeline XAI avec Prefect")
add_p("1. Distinction des rôles : SHAP et LIME génèrent le calcul mathématique des explications ; Prefect orchestre l'ordonnancement séquentiel des tâches (charger -> predire -> expliquer_shap -> expliquer_lime -> controler -> journaliser), les alertes et les reprises sur incident.\n"
      "2. Fonctionnement du garde-fou : Le test a consisté à exclure nuit des variables autorisées. La tâche controler a immédiatement levé ValueError('variables non autorisees : [nuit]'). Prefect a basculé le flow à l'état Failed, bloquant automatiquement l'exécution de journaliser. Ce garde-fou empêche toute fuite de variables interdites vers le registre d'audit MLflow.\n"
      "3. Intégration cycle de vie : Ce flow doit être déclenché automatiquement après chaque réentraînement de modèle (workflow reentrainement-fraude du TP06) et préalablement à toute promotion d'un modèle sous l'alias production.", bold_prefix="Analyse C2 :")

# PARTIE 4 : TABLEAUX DE SYNTHESE
add_heading1("4. Tableaux de Synthèse Récapitulatifs")

add_heading2("Tableau 1 : Stabilité SHAP vs LIME (Mme Benali)")
t1 = doc.add_table(rows=1, cols=4)
t1.alignment = WD_TABLE_ALIGNMENT.CENTER
hdr_cells = t1.rows[0].cells
hdr_cells[0].text = "Variable"
hdr_cells[1].text = "SHAP (Activité 3)"
hdr_cells[2].text = "LIME (2 000 v., g. 0)"
hdr_cells[3].text = "Stable sur 5 graines ?"

t1_data = [
    ("montant", "+0.367", "+0.482", "Oui (+0.428 à +0.482)"),
    ("nuit", "+0.370", "+0.183", "Oui (+0.181 à +0.237)"),
    ("anciennete", "+0.090", "+0.026", "Oui (+0.020 à +0.033)"),
    ("freq_24h", "-0.018", "+0.002", "Non (change de signe : -0.037 à +0.012)")
]
for row in t1_data:
    r = t1.add_row().cells
    for i, val in enumerate(row):
        r[i].text = val

doc.add_paragraph().paragraph_format.space_after = Pt(10)

add_heading2("Tableau 2 : Compromis Performance vs Interprétabilité (Activité 5)")
t2 = doc.add_table(rows=1, cols=4)
t2.alignment = WD_TABLE_ALIGNMENT.CENTER
hdr2 = t2.rows[0].cells
hdr2[0].text = "Modèle"
hdr2[1].text = "F1-Score"
hdr2[2].text = "Fraudes détectées"
hdr2[3].text = "Explicabilité sans outil"

t2_data = [
    ("Forêt fraude@production", "0.8669", "114 / 134", "Boîte noire (SHAP requis)"),
    ("Logistique (montant brut)", "0.7407", "90 / 134", "Linéaire, mais inadapté aux échelles"),
    ("Logistique (log10 montant)", "0.8593", "113 / 134", "Lisibilité native (cotes directes)"),
    ("Arbre prof. 2 (4 feuilles)", "0.2293", "18 / 134", "2 règles simples (116 fraudes manquées)"),
    ("Arbre prof. 3 (8 feuilles)", "0.5737", "72 / 134", "3 règles simples, mais F1 très faible"),
    ("Arbre prof. 6 (50 feuilles)", "0.7510", "95 / 134", "Trop complexe pour lecture manuelle"),
    ("Arbre sans limite (206 f.)", "0.7556", "102 / 134", "Surajusté et illisible")
]
for row in t2_data:
    r = t2.add_row().cells
    for i, val in enumerate(row):
        r[i].text = val

# PARTIE 5 : MODEL CARD COMPLETE
add_heading1("5. Contenu Intégral de la Model Card")
card_text = Path("MODEL_CARD.md").read_text(encoding="utf-8")
p_card = doc.add_paragraph()
p_card.paragraph_format.left_indent = Inches(0.2)
p_card.paragraph_format.right_indent = Inches(0.2)
r_card = p_card.add_run(card_text)
r_card.font.name = "Consolas"
r_card.font.size = Pt(9)

# Sauvegarde
out_file = "AYAT_Youssef_5IASD_G1.docx"
doc.save(out_file)
print(f"Rapport généré avec succès : {out_file}")
