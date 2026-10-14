import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.80'
from typing import Any, Dict
import yaml


def print_b(*message): # blue
    print('\033[96m', *message, '\033[0m')


def print_y(*message): # yellow
    print('\033[93m', *message, '\033[0m')


def print_r(*message): # red
    print('\033[91m', *message, '\033[0m')


def make_dir(path):
    os.makedirs(path, exist_ok=True)


def save_params(path: str, hyperparams: Dict[str, Any]) -> None:
    with open(path + '/params.yml', 'w+') as f:
        yaml.dump(hyperparams, f, default_flow_style=False)
