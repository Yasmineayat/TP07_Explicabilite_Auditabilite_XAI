import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri('http://127.0.0.1:5000')
c = MlflowClient()
v = c.get_model_version_by_alias('fraude', 'production')
run_id = Path('reports/run_xai.txt').read_text().strip()  # le run de l'activite 6
with mlflow.start_run(run_id=run_id):  # on rouvre ce run
    mlflow.log_artifact('MODEL_CARD.md')

card = Path('MODEL_CARD.md').read_text(encoding='utf-8')
c.update_model_version('fraude', v.version, description=card)  # visible dans le registre
c.set_model_version_tag('fraude', v.version, 'model_card', 'MODEL_CARD.md, run ' + run_id[:8])
print('Model Card attachee a fraude v' + v.version, 'et au run', run_id[:8])
