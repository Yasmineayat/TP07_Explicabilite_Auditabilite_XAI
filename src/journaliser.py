import sys
sys.stdout.reconfigure(encoding='utf-8')
import hashlib
from pathlib import Path
import mlflow
from mlflow import MlflowClient
from sklearn.metrics import f1_score, precision_score, recall_score
from xai_commun import NOM, ALIAS, VERSION, m, Xte, yte

def empreinte(chemin):  # meme principe que DVC (md5)
    return hashlib.md5(Path(chemin).read_bytes()).hexdigest()

pred = m.predict(Xte)
mlflow.set_experiment('tp7-explicabilite')
with mlflow.start_run(run_name=f'xai-{NOM}-v{VERSION.version}') as run:
    mlflow.set_tags({'modele': NOM, 'alias': ALIAS,  # QUEL modele est explique
                     'modele_version': VERSION.version,
                     'modele_run_id': VERSION.run_id,
                     'donnees': 'data/transactions.csv',  # SUR quelles donnees
                     'donnees_md5': empreinte('data/transactions.csv'),
                     'methodes': 'permutation ; SHAP TreeExplainer ; LIME'})
    mlflow.log_params({'shap_echantillon': 400, 'lime_num_samples': 2000,
                       'lime_random_state': 0, 'permutation_n_repeats': 10})
    mlflow.log_metrics({'f1': f1_score(yte, pred), 'precision': precision_score(yte, pred),
                        'recall': recall_score(yte, pred)})
    for f in sorted(Path('reports').glob('*.*')):  # graphiques, LIME, JSON
        if f.suffix not in ('.png', '.html', '.json'):
            continue
        mlflow.log_artifact(str(f), artifact_path='explications')
    Path('reports/run_xai.txt').write_text(run.info.run_id)

MlflowClient().set_model_version_tag(NOM, VERSION.version, 'explications_run', run.info.run_id)
print('run', run.info.run_id[:8], ': explications du modele', NOM, 'v' + VERSION.version, 'archivees')
