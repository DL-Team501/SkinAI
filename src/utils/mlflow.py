import json
import os

import mlflow

MLFLOW_ARTIFACTS = 'mlflow_artifacts'
os.makedirs(MLFLOW_ARTIFACTS, exist_ok=True)


def log_dict_as_mlflow_artifact(dict_to_log, artifact_name):
    artifact_path = f'{MLFLOW_ARTIFACTS}/{artifact_name}'
    with open(artifact_path, 'w') as file:
        json.dump(dict_to_log, file, indent=4)
    mlflow.log_artifact(artifact_path, artifact_path='')
