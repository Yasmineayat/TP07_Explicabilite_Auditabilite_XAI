from lime.lime_tabular import LimeTabularExplainer
from xai_commun import m, X, Xtr, CAS

def expliquer(graine, n):
    ex = LimeTabularExplainer(Xtr.values, feature_names=list(X.columns),
                             class_names=['normal', 'fraude'],
                             mode='classification', random_state=graine)
    return ex.explain_instance(CAS.values[0], m.predict_proba,
                               num_features=4, num_samples=n)

e = expliquer(0, 2000)  # explication de reference
print('regles :', [(r, round(w, 3)) for r, w in e.as_list()])
e.save_to_file('reports/lime_benali.html')  # page HTML interactive
fig = e.as_pyplot_figure()
fig.tight_layout()
fig.savefig('reports/lime_benali.png', dpi=120)

for n in [500, 2000]:  # taille du voisinage
    print(f'--- {n} voisins montant nuit anciennete freq_24h')
    for graine in range(5):  # 5 tirages : est-ce stable ?
        d = dict(expliquer(graine, n).as_list())
        w = [next(v for k, v in d.items() if col in k) for col in X.columns]
        print(f'graine {graine} {w[0]:+.3f} {w[3]:+.3f} {w[1]:+.3f} {w[2]:+.3f}')
