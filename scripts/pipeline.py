import os
import fire
import torch
import random
import shutil
import pickle
import hashlib
import warnings
import numpy as np

from pathlib import Path
from datetime import datetime

from src.task import Task
from src.builders import MODELS
from src.utils import import_config


CONFIGS = [
    'model',
    'train_test_split',
    'feature',
    'label',
    'feature_transformation',
    'label_transformation'
]


def hash_string(string):
    sha256_hash = hashlib.sha256()
    sha256_hash.update(string.encode('utf-8'))
    hash_value = sha256_hash.hexdigest()
    truncated_hash = hash_value[:32]
    return truncated_hash


def timestamp(marker: bool = False):
    template = '%Y-%m-%d %H:%M:%S' if marker else '%Y%m%d%H%M%S'
    return datetime.now().strftime(template)


def set_seed(seed: int):
    print(f'Seed is set to {seed}.')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_config(config_path: str, workspace: str) -> dict:
    config_path = Path(config_path)
    configs = import_config(config_path, CONFIGS)

    # Determine the workspace
    if configs['model'].get('workspace') is not None:
        workspace = Path(configs['model'].get('workspace'))
    elif workspace is not None:
        if workspace.strip().lower() == 'none':
            workspace = None
        else:
            workspace = Path(workspace)
    else:
        workspace = Path.cwd() / 'workspaces' / config_path.stem
        warnings.warn(f'Setting workspace to {str(workspace)}. If you '
                       'do not want any information to be stored, '
                       'explicitly call with flag `--workspace none`.')

    if workspace is not None and workspace.exists():
        assert workspace.is_dir(), workspace

    if workspace is not None and not workspace.exists():
        os.makedirs(workspace)

    configs['workspace'] = workspace

    return configs


def recursive_dump_string(data):
    if isinstance(data, list):
        return '_'.join([recursive_dump_string(x) for x in data])
    if isinstance(data, dict):
        return '_'.join([
            recursive_dump_string(data[key])
            for key in sorted(data.keys())
        ])
    return str(data)


def build_dataset(configs: dict, device: str):
    strings = []
    fields = ['label', 'feature', 'train_test_split',
              'feature_transformation', 'label_transformation']
    for field in fields:
        strings.append(recursive_dump_string(configs[field]))
    filename = hash_string('+'.join(strings))
    cache_dir = Path('cache')
    if not cache_dir.exists():
        cache_dir.mkdir()
    cache_file = Path(cache_dir / f'battery_cache_{filename}.pkl')

    if cache_file.exists():
        warnings.warn(f'Load datasets from cache {str(cache_file)}.')
        with open(cache_file, 'rb') as f:
            dataset = pickle.load(f)
    else:
        dataset = Task(
            label_annotator=configs['label'],
            feature_extractor=configs['feature'],
            train_test_splitter=configs['train_test_split'],
            feature_transformation=configs['feature_transformation'],
            label_transformation=configs['label_transformation']).build()
        # store cache
        with open(cache_file, 'wb') as f:
            pickle.dump(dataset, f)
    return dataset.to(device)


def main(config_path: str,
         workspace: str = None,
         train: bool = False,
         evaluate: bool = True,
         checkpoint: str = None,
         device: str = 'cpu',
         metric: str = 'RMSE,MAE,MAPE',
         skip_training_if_exists: bool = False,
         skip_evaluation_if_exists: bool = False,
         build_cache_only: bool = False,
         seed: int = 0,
         epochs: int = None):
    set_seed(seed)
    configs = load_config(config_path, workspace)

    if build_cache_only:
        dataset = build_dataset(configs, device)
        return

    if isinstance(metric, str):
        metric = metric.split(',')

    dataset = None

    if epochs is not None:
        configs['model']['epochs'] = epochs

    configs['model']['seed'] = seed

    model = MODELS.build(configs['model'])
    if model.workspace is None:
        model.workspace = configs['workspace']

    if checkpoint is not None:
        model.load_checkpoint(checkpoint)

    if torch.__version__ >= '2' and isinstance(model, torch.nn.Module):
        model = torch.compile(model)

    model = model.to(device)

    ts = timestamp()
    if model.workspace is not None:
        shutil.copyfile(config_path, model.workspace / f'config_{ts}.yaml')

    if train:
        if skip_training_if_exists and model.workspace is not None and \
                (model.workspace / 'latest.ckpt').exists():
            warnings.warn(f'Skip training for {configs["workspace"]} '
                           'as the checkpoint exists.')
        else:
            if dataset is None:
                dataset = build_dataset(configs, device)
            model.fit(dataset, timestamp=ts)

    if evaluate:
        # Evaluate and save predictions
        if skip_evaluation_if_exists and model.workspace is not None and \
                any(model.workspace.glob('predictions_*.pkl')):
            warnings.warn(f'Skip evaluation for {configs["workspace"]} '
                           'as the prediction exists.')
        else:
            if dataset is None:
                dataset = build_dataset(configs, device)
            prediction = model.predict(dataset)
            scores = {
                m: dataset.evaluate(prediction, m) for m in metric
            }
            if model.workspace is not None:
                obj = {
                    'prediction': prediction,
                    'scores': scores,
                    'data': dataset.to('cpu'),
                    'seed': seed,
                }
                with open(
                    model.workspace / f'predictions_seed_{seed}_{ts}.pkl', 'wb'
                ) as f:
                    pickle.dump(obj, f)

            # Print metrics
            print(
                ' '.join([f'{m}: {s:.2f}' for m, s in scores.items()]),
                flush=True)


if __name__ == '__main__':
    fire.Fire(main)
