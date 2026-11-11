"""Training and validation on full daily trading paths."""
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from .campaign_data import write_new
from .formal71 import train_update71, evaluate71


def jsonable(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def compressed_new(path, value):
    with Path(path).open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as f:
            f.write(json.dumps(jsonable(value), allow_nan=False).encode())


def run_train(model, optimizer, path, plan, callback):
    dates, sample, rows, window, ledger, _ = path
    return train_update71(model, optimizer, dates, sample, rows, window=window,
        ledger=ledger, plan=plan, callback=callback)


def run_eval(model, path, plan, callback):
    dates, sample, rows, window, ledger, _ = path
    return evaluate71(model, dates, sample, rows, window=window, ledger=ledger, plan=plan, callback=callback)


def save_evaluation(out, epoch, evaluated):
    stem = f'validation_{epoch:03d}'
    # Keep every day's accounting returns. Large per-stock accounting arrays
    # are not duplicated in 100 validation artifacts; source + checkpoint is
    # enough to reconstruct them exactly. Never omit a return date.
    parts = {k: v.detach().cpu().numpy() for k, v in evaluated['parts'].items()
        if torch.is_tensor(v) and v.ndim <= 1}
    with (out/(stem+'.npz')).open('xb') as f:
        np.savez_compressed(f, **parts)
    compressed_new(out/(stem+'_diagnostics.json.gz'), evaluated['decision_diagnostics'])
    write_new(out/(stem+'.json'), jsonable(evaluated['metrics']))
    return evaluated['metrics']
