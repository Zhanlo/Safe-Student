import json
import numpy as np
import pandas as pd
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.autonotebook import tqdm
from collections import defaultdict
import copy
import os


class Accumulator:
    def __init__(self):
        self.metrics = defaultdict(lambda: 0.)

    def add(self, key, value):
        self.metrics[key] += value

    def add_dict(self, dict):
        for key, value in dict.items():
            self.add(key, value)

    def __getitem__(self, item):
        return self.metrics[item]

    def __setitem__(self, key, value):
        self.metrics[key] = value

    def get_dict(self):
        return copy.deepcopy(dict(self.metrics))

    def items(self):
        return self.metrics.items()

    def __str__(self):
        return str(dict(self.metrics))

    def __truediv__(self, other):
        newone = Accumulator()
        for key, value in self.items():
            if isinstance(other, str):
                if other != key:
                    newone[key] = value / self[other]
                else:
                    newone[key] = value
            else:
                newone[key] = value / other
        return newone

def save_json_file(path, data):
    with open(f'{path}/log.json', "w") as outfile:
        json.dump(data, outfile, indent=2)


def save_json_file_withname(path, name, data):
    with open(f'{path}/{name}log.json', "w") as outfile:
        json.dump(data, outfile, indent=2)