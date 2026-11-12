import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xai_commun import m, Xte

Xs = Xte.sample(400, random_state=0)  # 400 transactions de test
ex = shap.TreeExplainer(m)  # exact pour les arbres
sv = ex(Xs)[:, :, 1]  # classe 1 = fraude
print('valeur de base :', round(float(sv.base_values[0]), 4))
imp = pd.Series(np.abs(sv.values).mean(0), index=Xs.columns)
print(imp.sort_values(ascending=False).round(4).to_string())
imp.round(4).to_json('reports/shap_importance.json')
shap.plots.beeswarm(sv, show=False)
plt.tight_layout()
plt.savefig('reports/shap_global.png', dpi=120)
plt.close('all')
plt.scatter(Xs['montant'], sv[:, 'montant'].values, c=Xs['nuit'], cmap='coolwarm', s=12)
plt.xscale('log')
plt.xlabel('montant (MAD)')
plt.ylabel('contribution SHAP du montant')
plt.colorbar(label='nuit')
plt.tight_layout()
plt.savefig('reports/shap_montant.png', dpi=120)
print('graphiques ecrits')
