import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from prefect import flow, task, get_run_logger
from prefect.cache_policies import NO_CACHE
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

AUTORISEES = {'montant', 'anciennete', 'freq_24h', 'nuit'}  # connues au moment de la decision
CAS = pd.DataFrame([{'montant': 1800.0, 'anciennete': 30, 'freq_24h': 4, 'nuit': 1}])

@task(cache_policy=NO_CACHE)
def charger():
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    m = mlflow.sklearn.load_model('models:/fraude@production')
    df = pd.read_csv('data/transactions.csv')
    X, y = df.drop(columns='fraude'), df['fraude']
    return m, train_test_split(X, y, test_size=0.25, random_state=42)

@task(cache_policy=NO_CACHE)
def predire(m, Xte, yte):
    f1 = f1_score(yte, m.predict(Xte))
    get_run_logger().info(f'F1 sur le test : {f1:.4f}')
    return f1

@task(cache_policy=NO_CACHE)
def expliquer_shap(m, Xte):
    Xs = Xte.sample(400, random_state=0)
    sv = shap.TreeExplainer(m)(Xs)[:, :, 1]
    shap.plots.beeswarm(sv, show=False)
    plt.tight_layout()
    plt.savefig('reports/flow_shap_global.png', dpi=120)
    plt.close('all')
    imp = pd.Series(np.abs(sv.values).mean(0), index=Xs.columns).sort_values(ascending=False)
    get_run_logger().info('importance SHAP : ' + ', '.join(f'{k} {v:.3f}' for k, v in imp.items()))
    return imp

@task(cache_policy=NO_CACHE)
def expliquer_lime(m, Xtr):
    ex = LimeTabularExplainer(Xtr.values, feature_names=list(Xtr.columns),
                             class_names=['normal', 'fraude'], mode='classification',
                             random_state=0)
    e = ex.explain_instance(CAS.values[0], m.predict_proba, num_features=4, num_samples=2000)
    e.save_to_file('reports/flow_lime_benali.html')
    return e.as_list()

@task(cache_policy=NO_CACHE)
def controler(imp):  # garde-fou : l'explication est-elle plausible ?
    interdites = set(imp.index) - AUTORISEES
    if interdites:
        raise ValueError(f'variables non autorisees : {sorted(interdites)}')
    part = imp.iloc[0] / imp.sum()
    get_run_logger().info(f'{imp.index[0]} en tete ({part:.0%} de l importance totale)')

@task(cache_policy=NO_CACHE)
def journaliser(f1, imp, regles):
    mlflow.set_experiment('tp7-explicabilite')
    with mlflow.start_run(run_name='xai-flow-prefect'):
        mlflow.log_metric('f1', f1)
        mlflow.log_dict({'shap': imp.round(4).to_dict(), 'lime': regles}, 'explications/resume.json')
        for f in ['flow_shap_global.png', 'flow_lime_benali.html']:
            mlflow.log_artifact('reports/' + f, artifact_path='explications')

@flow(name='explication-fraude', log_prints=True)
def explication():
    m, (Xtr, Xte, ytr, yte) = charger()
    f1 = predire(m, Xte, yte)
    imp = expliquer_shap(m, Xte)
    regles = expliquer_lime(m, Xtr)
    controler(imp)  # echoue -> la journalisation n'a pas lieu
    journaliser(f1, imp, regles)

if __name__ == '__main__':
    explication()
