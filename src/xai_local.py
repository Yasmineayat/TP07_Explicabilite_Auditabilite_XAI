import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from xai_commun import m, Xte, CAS

ex = shap.TreeExplainer(m)
proba = m.predict_proba(Xte)[:, 1]
idx = Xte.index[proba.argmax()]  # la transaction de test la plus suspecte
for nom, x in [('benali', CAS), (f'test_{idx}', Xte.loc[[idx]])]:
    e = ex(x)[0, :, 1]  # 1 transaction, classe fraude
    print(f'--- {nom} :', x.iloc[0].to_dict())
    print('valeur de base :', round(float(e.base_values), 4))
    for col, c in sorted(zip(x.columns, e.values), key=lambda t: -abs(t[1])):
        print(f' {col:11s} {c:+.4f}')
    print('base + somme :', round(float(e.base_values + e.values.sum()), 4),
          '| prediction :', round(m.predict_proba(x)[0, 1], 4))
    shap.plots.waterfall(e, show=False)
    plt.tight_layout()
    plt.savefig(f'reports/shap_{nom}.png', dpi=120)
    plt.close('all')
