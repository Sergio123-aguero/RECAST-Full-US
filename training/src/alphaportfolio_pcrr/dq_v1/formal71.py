"""71D formal model preparation. No remote calls, implicit training or test IO."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import random

import numpy as np
import torch

from .config import DQConfig
from .daily import DQDailyPortfolio
from .feature_contract import feature71_contract, contract_sha256
from . import training


DEFAULT_PLAN = Path(__file__).resolve().parents[3] / "recast.json"


def read_plan(path=DEFAULT_PLAN):
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    plan["dq_config"]["retrieval_depths"] = tuple(plan["dq_config"]["retrieval_depths"])
    return plan


def make_model(config, monthly_srem_state, *, source_binding, seed=20261015,
               dq_config=None, chunk_size=32, checkpoint_encoders=True, decision_mode='full'):
    """Copy only monthly SREM; all 71 daily columns are active at initialization.

    The supplied state must be trusted/hash-verified by the loader. This lower
    level factory also permits small hidden widths for synthetic unit tests;
    it does not change the 252x71/24x51 feature contract or feature ordering.
    """
    if (type(seed) is not int or not 0 <= seed < 2**63 or not source_binding
            or config.model.input_dim != 51 or config.model.lookback_months != 12):
        raise ValueError('explicit monthly initialization binding and 51D monthly architecture required')
    policy = dq_config or replace(DQConfig(), statistic_dim=71)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = DQDailyPortfolio(copy.deepcopy(config),
            daily_config=replace(config.model, input_dim=71, lookback_months=252),
            dq_config=policy, chunk_size=chunk_size, checkpoint_encoders=checkpoint_encoders,
            feature_contract=feature71_contract(), decision_mode=decision_mode)
    model.monthly_srem.load_state_dict(monthly_srem_state, strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('nonfinite initializer parameters')
    model.initialization_binding = dict(mode='monthly_SREM_only_native_daily71', seed=seed,
        source=copy.deepcopy(source_binding), daily_and_DQ_fresh=True,
        daily_input_initialization='all_71_columns_native_fan_in',
        feature_contract_sha256=contract_sha256(model.feature_contract))
    return model.requires_grad_(True).train()


def check_model_variant(model, plan):
    if getattr(model, 'ablation', None) != plan.get('ablation'):
        raise ValueError('model ablation differs from the explicit training plan')


def optimizer_for71(model, plan=None):
    plan = read_plan() if plan is None else plan
    check_model_variant(model, plan)
    if (model.decision_mode != plan.get('decision_mode', 'full')
            or model.feature_contract != plan['feature_contract_spec']
            or model.dq_config.to_dict() != plan['dq_config']):
        raise ValueError('formal model and preparation differ')
    return training.optimizer_for(model, **plan['learning_rates'])


def seed_fresh_training71(plan=None):
    """Explicit fresh-attempt RNG start; NEVER call on checkpoint continuation.

    Model construction isolates its initialization RNG. Dropout/exploration
    training also need a declared starting RNG. Resume instead restores the
    saved RNG with training.load_training_checkpoint(restore_training_rng=True).
    """
    plan = read_plan() if plan is None else plan
    value = plan['seed']
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError('valid integer seed required')
    random.seed(value)
    np.random.seed(value % 2**32)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def train_update71(model, optimizer, dates, get_sample, items, *, window, ledger, plan=None, callback=None):
    """Formal configuration adapter; caller must supply an audited real ledger."""
    plan = read_plan() if plan is None else plan
    check_model_variant(model, plan)
    if 'execution_valuation_policy' in plan:
        from .valuation import validate_binding
        validate_binding(ledger, items)
    if type(optimizer) is not torch.optim.AdamW:
        raise ValueError('formal71 uses AdamW, not an implicitly substituted optimizer')
    if (window.role != 'train' or window.signal_start < plan['train'][0]
            or window.signal_end > plan['train'][1]
            or training.aware(window.outcome_cutoff).date().isoformat() > plan['train'][1]):
        raise ValueError('training path crosses the formal train split')
    # Verify actual optimizer groups/rates, not just a JSON description.
    expected = model.parameter_groups(**plan['learning_rates'])
    if len(optimizer.param_groups) != len(expected):
        raise ValueError('formal optimizer groups differ')
    for supplied, wanted in zip(optimizer.param_groups, expected, strict=True):
        if (supplied.get('name') != wanted['name'] or supplied['lr'] != wanted['lr']
                or supplied.get('weight_decay', 0.) != plan['weight_decay']
                or [id(p) for p in supplied['params']] != [id(p) for p in wanted['params']]):
            raise ValueError('formal optimizer learning rate/regularization differs')
    if (model.decision_mode != plan.get('decision_mode', 'full')
            or model.feature_contract != plan['feature_contract_spec']
            or model.dq_config.to_dict() != plan['dq_config']):
        raise ValueError('formal model and preparation differ')
    return training.train_full_path(model, optimizer, dates, get_sample, items,
        window=window, ledger=ledger, clip=plan['gradient_clip'], callback=callback,
        sharpe_weight=plan['sharpe_weight'], growth_weight=plan['growth_weight'])


def prepare_execution71(items, *, window, source_binding, clocks=None, evidence=(), plan=None):
    """Same policy for all declared roles; test preparation is explicit, never automatic."""
    from .valuation import prepare_path, policy_spec
    plan = read_plan() if plan is None else plan
    if plan.get('execution_valuation_policy') != policy_spec():
        raise ValueError('an explicitly approved assumed-valuation preparation is required')
    _check_window71(window, plan)
    rows, ledger, receipt = prepare_path(items, source_binding=dict(source_binding, role=window.role,
        window=vars(window), preparation_sha256=training.fingerprint(plan)), clocks=clocks, evidence=evidence)
    window.validate([r['signal_date'] for r in rows], rows)
    return rows, ledger, receipt


def _check_window71(window, plan):
    if window.role not in ('train', 'validation', 'test'):
        raise ValueError('unknown formal path role')
    bounds = plan[window.role]
    if (window.signal_start < bounds[0] or window.signal_end > bounds[1]
            or training.aware(window.outcome_cutoff).date().isoformat() > bounds[1]):
        raise ValueError('path crosses the formal role split')


def evaluate71(model, dates, get_sample, items, *, window, ledger, plan=None, callback=None):
    from .valuation import validate_binding
    plan = read_plan() if plan is None else plan
    check_model_variant(model, plan)
    if window.role not in ('validation', 'test'):
        raise ValueError('formal evaluation requires validation/test role')
    _check_window71(window, plan)
    if 'execution_valuation_policy' in plan:
        validate_binding(ledger, items)
    if (model.decision_mode != plan.get('decision_mode', 'full')
            or model.feature_contract != plan['feature_contract_spec'] or model.dq_config.to_dict() != plan['dq_config']):
        raise ValueError('formal model and evaluation preparation differ')
    return training.evaluate_path(model, dates, get_sample, items, window=window, ledger=ledger, callback=callback)
