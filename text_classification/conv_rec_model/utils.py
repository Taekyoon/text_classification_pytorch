import os
import json


def load_json(path):
    if not os.path.exists(path):
        return None

    with open(path) as f:
        data = json.load(f)

    return data


def load_configs(path):
    _path = path
    print(_path)

    configs = load_json(_path)

    if configs is None:
        raise FileNotFoundError()

    return configs
