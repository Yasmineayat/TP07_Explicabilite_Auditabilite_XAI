import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from xai_commun import m, X, Xte, yte

r = permutation_importance(m, Xte, yte, scoring='f1', n_repeats=10,
                           random_state=0)  # 10 melanges par colonne
t = pd.DataFrame({'MDI (foret)': m.feature_importances_,
                  'permutation': r.importances_mean,
                  'ecart-type': r.importances_std}, index=X.columns).round(3)
t = t.sort_values('permutation', ascending=False)
print(t.to_string())
t.to_json('reports/importance.json')
t['permutation'].plot.barh(xerr=t['ecart-type'], color='#4592EF')
plt.xlabel('perte de F1 quand la colonne est melangee')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('reports/permutation.png', dpi=120)
