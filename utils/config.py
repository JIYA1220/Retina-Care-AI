"""
config.py
Utility to load hyperparameters from config.yaml.
Returns a nested attribute-accessible object.
"""

import yaml
from types import SimpleNamespace

def dict_to_namespace(d):
    """Recursively convert dict to SimpleNamespace for dot notation access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d

def load_config(path="config.yaml"):
    """Load and parse the YAML config file."""
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)
    return dict_to_namespace(config_dict)
