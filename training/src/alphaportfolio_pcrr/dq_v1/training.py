"""Independent full-path training; no AutoDL, implicit data policy or test tuning.

An explicitly supplied execution ledger carries holdings and supplies the net
path. Exact two-pass vector-Jacobian replay avoids retaining every day's neural
graph. Only final weights (gross <= 1 with explicit cash fallback) enter that ledger; auxiliary supervision
is a separate loss and is never deducted from reported portfolio returns.
"""
from dataclasses import dataclass
from contextlib import nullcontext
from datetime import date, datetime
import copy
import hashlib
import math
import os
from pathlib import Path
import random

import numpy as np
import torch

from .daily import DQDailyPortfolio
from .audit_cache import AuditReplayCache
from .evaluation_cache import MonthlyEvaluationCache
from .inputs import DEFAULT_STATISTICS, select_diagnostic_statistics, validate_decision_clock
from .losses import auxiliary_loss, net_path_objective
from .feature_contract import resolve_statistics, validate_sample_contract
from .hash_snapshot import tensor_snapshots


def aware(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00')) if isinstance(value, str) else value
    if not isinstance(parsed, datetime) or parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError('explicit timezone-aware timestamp required')
    return parsed


@dataclass(frozen=True)
class PathWindow:
    role: str
    signal_start: str
    signal_end: str
    outcome_cutoff: str

    def validate(self, dates, items):
        if self.role not in {'train', 'validation', 'test'}:
            raise ValueError('unknown path role')
        first, last = date.fromisoformat(self.signal_start), date.fromisoformat(self.signal_end)
        cutoff = aware(self.outcome_cutoff)
        days = [date.fromisoformat(str(day)) for day in dates]
        if (len(days) < 2 or days != sorted(set(days)) or len(days) != len(items)
                or first > last or not all(first <= day <= last for day in days)):
            raise ValueError('ordered unique dates must lie in the declared signal window')
        for day, item in zip(days, items, strict=True):
            if str(item['signal_date']) != str(day):
                raise ValueError('execution signal dates not aligned')
            # No date-only fallback: availability must come from the data binding.
            available = aware(item['outcome_available_at'])
            if available > cutoff or available.date() < day:
                raise ValueError('an outcome crosses the declared label cutoff')
        return self


@dataclass(frozen=True)
class LedgerContract:
    """Caller-selected ledger, with an explicit fee-unit conversion.

    For existing ledgers charging .001 * .5*|trade|, fee_per_abs_weight=.0005.
    Do not pass .001 here and silently double the optimizer's cost estimate.
    ``run`` takes (list[weights], items) and returns a dict containing net.
    """
    policy_id: str
    fee_per_abs_weight: float
    run: object
    report: object = None
    binding: object = None

    def validate(self, model):
        if not self.policy_id or not callable(self.run):
            raise ValueError('an explicit callable execution ledger policy is required')
        if (not math.isfinite(self.fee_per_abs_weight) or self.fee_per_abs_weight < 0
                or not math.isclose(self.fee_per_abs_weight, model.dq_config.cost_rate, abs_tol=1e-12, rel_tol=0)):
            raise ValueError('decision-layer and execution-ledger fee units differ')
        if self.report is not None and not callable(self.report):
            raise ValueError('ledger report must be callable when supplied')


def capture_rng():
    nr = np.random.get_state()
    return dict(torch=torch.get_rng_state().clone(),
        cuda=[v.clone() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else [],
        python=random.getstate(), numpy=(nr[0], nr[1].tolist(), nr[2], nr[3], nr[4]))


def restore_rng(state):
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])
    random.setstate(state['python'])
    name, keys, position, gaussian, cached = state['numpy']
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, gaussian, cached))


def fingerprint(value, *, batch_tensors=False):
    """Content hash for replay safety, including values with zero forward effect."""
    digest = hashlib.sha256()
    snapshots = tensor_snapshots(value) if batch_tensors else {}
    def add(x):
        if isinstance(x, torch.Tensor):
            x = snapshots[id(x)] if id(x) in snapshots else x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            if x.dtype.hasobject:
                raise ValueError('object arrays are not valid numerical replay inputs')
            a = np.ascontiguousarray(x)
            digest.update(repr((a.dtype.str, x.shape)).encode())
            if a.size:
                digest.update(memoryview(a).cast('B'))
        elif isinstance(x, dict):
            for key in sorted(x):
                digest.update(str(key).encode()+b'\0')
                add(x[key])
        elif isinstance(x, (tuple, list)):
            digest.update(str(len(x)).encode()+b'[')
            for item in x:
                add(item)
        else:
            digest.update(repr(x).encode()+b'\0')
    add(value)
    return digest.hexdigest()


def forward_sample(model, sample, *, statistic_names=None, return_aux=True):
    """Read only a feature whitelist; no execution or outcome columns are read."""
    ref = next(model.parameters())
    validate_sample_contract(model, sample)
    statistic_names = resolve_statistics(model, statistic_names)
    keys = ('asset_ids','daily','monthly','daily_observed','monthly_observed','daily_feature_names',
            'signal_date','daily_available_at','monthly_available_at','decision_at','execution_at')
    selected = {key: sample[key] for key in keys}
    if model.feature_contract is not None:
        selected['monthly_feature_names'] = sample['monthly_feature_names']
        selected['feature_contract_sha256'] = sample['feature_contract_sha256']
    validate_decision_clock(**{key: aware(sample[key]) for key in
        ('daily_available_at','monthly_available_at','decision_at','execution_at')})
    if len(statistic_names) != model.dq_config.statistic_dim:
        raise ValueError('diagnostic statistic schema differs from the model')
    def tensor(key, dtype):
        return torch.as_tensor(sample[key], dtype=dtype, device=ref.device).unsqueeze(0)
    daily, monthly = tensor('daily', ref.dtype), tensor('monthly', ref.dtype)
    do, mo = tensor('daily_observed', torch.bool), tensor('monthly_observed', torch.bool)
    raw_ids = np.asarray(sample['asset_ids'])
    if raw_ids.ndim != 1 or not np.issubdtype(raw_ids.dtype, np.integer):
        raise ValueError('permanent asset identifiers must be exact integers')
    statistics, known = select_diagnostic_statistics(daily,do,sample['daily_feature_names'],
                                                    statistic_names=statistic_names)
    decision_inputs = {}
    if model.dq_config.access_policy == 'hierarchical_v1':
        # Date/update keyed exploration is replayable and never outcome keyed.
        decision_inputs['decision_keys'] = [date.fromisoformat(str(sample['signal_date'])).toordinal()]
        selected['access_update_step'] = model.completed_updates
    for key in ('valid_assets','long_allowed','short_allowed'):
        if key in sample:
            arr = np.asarray(sample[key])
            if arr.shape != raw_ids.shape or arr.dtype != np.bool_:
                raise ValueError('decision-time eligibility must be an explicit boolean asset vector')
            decision_inputs[key] = tensor(key,torch.bool)
            selected[key] = sample[key]
    monthly_cache = getattr(model, '_evaluation_monthly_cache', None)
    if monthly_cache is not None: monthly_cache.bind_asset_ids(raw_ids)
    out = model(daily,monthly,daily_observed=do,monthly_observed=mo,
        asset_ids=tensor('asset_ids',torch.int64),statistics=statistics,
        statistics_observed=known,return_aux=return_aux,**decision_inputs)
    weight = out.normalized_weights[0]
    gross = weight.abs().sum()
    if weight.shape != raw_ids.shape or not torch.isfinite(weight).all() or gross > 1+2e-5:
        raise FloatingPointError('finite final weights within the declared gross/cash policy required')
    cash = out.unused_risk_budget_cash[0]
    if (not torch.isfinite(cash) or cash < 0
            or not torch.isclose(gross+cash, weight.new_tensor(1.), atol=2e-5, rtol=0)):
        raise FloatingPointError('target gross and unused cash budget do not reconcile')
    return out, fingerprint(selected, batch_tensors=getattr(model, 'batch_replay_hashes', False))


def admission_report(dates, rows):
    """Target policy events only; executed cash/positions remain ledger outputs."""
    return dict(schema='dq.admission_fallback_report.v1', sessions=len(dates),
        fallback_sessions=sum(bool(row['fallback_used']) for row in rows),
        cash_sessions=sum(row['unused_risk_budget_cash'] > 1e-6 for row in rows),
        mean_target_unused_risk_budget_cash=sum(row['unused_risk_budget_cash'] for row in rows)/len(rows),
        max_target_unused_risk_budget_cash=max(row['unused_risk_budget_cash'] for row in rows),
        events=[dict(signal_date=str(day), **row) for day, row in zip(dates, rows, strict=True)
                if row['fallback_used']], dates_skipped=False, target_not_execution_report=True)


def _bind(sample, item, day, window):
    if (str(sample['signal_date']) != str(day) or str(item['signal_date']) != str(day)
            or not np.array_equal(np.asarray(sample['asset_ids']),np.asarray(item['target_asset_ids']))):
        raise ValueError('feature, accounting target axis or date mismatch')
    execution, available = aware(sample['execution_at']), aware(item['outcome_available_at'])
    if execution.date().isoformat() != str(item['execution_date']):
        raise ValueError('declared execution time differs from ledger interval')
    if not execution < available <= aware(window.outcome_cutoff):
        raise ValueError('outcome must follow execution and precede the path cutoff')


def _labels(out, item):
    ids, targets, index = (np.asarray(item[k]) for k in ('asset_ids','target_asset_ids','target_indices'))
    if (index.shape != targets.shape or not np.issubdtype(index.dtype,np.integer)
            or (index < 0).any() or (index >= len(ids)).any() or not np.array_equal(ids[index],targets)):
        raise ValueError('invalid accounting-to-target label map')
    returns, observed, stale = (np.asarray(item[k]) for k in ('mark_returns','observed_return','stale_mark'))
    if (returns.shape != ids.shape or observed.shape != ids.shape or stale.shape != ids.shape
            or observed.dtype != np.bool_ or stale.dtype != np.bool_):
        raise ValueError('invalid label values/observability')
    ref = out.normalized_weights
    r = torch.as_tensor(returns[index],dtype=ref.dtype,device=ref.device)[None]
    available = observed & ~stale
    if 'dq_valuation_assumed' in item:
        assumed = np.asarray(item['dq_valuation_assumed'])
        if assumed.dtype != np.bool_ or assumed.shape != ids.shape:
            raise ValueError('invalid assumed-valuation label mask')
        available &= ~assumed
    known = torch.as_tensor(available[index],dtype=torch.bool,device=ref.device)[None]
    return r, known


def optimizer_for(model, **learning_rates):
    return torch.optim.AdamW(model.parameter_groups(**learning_rates),weight_decay=0.)


def train_full_path(model, optimizer, dates, get_sample, items, *, window, ledger,
                    statistic_names=None, sharpe_weight=1., growth_weight=.25,
                    clip=5., callback=None, cache_access_labels=True):
    """One optimizer update per complete net path, with exact stochastic replay.

    Does not read validation/test inputs, save over old checkpoints or restart.
    If replay/input checks fail, no optimizer step is taken and gradients clear.
    """
    if type(model) is not DQDailyPortfolio or window.role != 'train':
        raise ValueError('DQ model and train-role window required')
    window.validate(dates,items)
    ledger.validate(model)
    if not math.isfinite(clip) or clip <= 0:
        raise ValueError('positive finite gradient clip required')
    params = list(model.parameters())
    supplied = [p for group in optimizer.param_groups for p in group['params']]
    if (not all(p.requires_grad for p in params) or len(supplied) != len(params)
            or len({id(p) for p in supplied}) != len(supplied)
            or {id(p) for p in supplied} != {id(p) for p in params}):
        raise ValueError('every parameter must be optimized exactly once')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    values, penalties, rngs, signatures, item_hashes = [], [], [], [], []
    decision_signatures, decision_stopping, admission_rows = [], [], []
    batch_hashes = getattr(model, 'batch_replay_hashes', False)
    content_hash = lambda value: fingerprint(value, batch_tensors=batch_hashes)
    audit_cache = (AuditReplayCache(content_hash) if cache_access_labels
                   and model.decision_mode != 'candidate_only'
                   and model.dq_config.access_policy == 'hierarchical_v1' else None)
    def predict(i):
        sample = get_sample(dates[i])
        _bind(sample,items[i],dates[i],window)
        out, signature = forward_sample(model, sample, statistic_names=statistic_names)
        if model.decision_mode == 'candidate_only':
            # Candidate control learns directly from the same net economic
            # path; no unused refinement labels or auxiliary modules remain.
            aux = {'loss': out.normalized_weights.sum()*0}
        else:
            r, known = _labels(out,items[i])
            aux = auxiliary_loss(out,r,future_observed=known,cost_rate=model.dq_config.cost_rate,
                gain_scale=model.dq_config.gain_scale, access_replay_cache=audit_cache,
                access_replay_key=(model.completed_updates, i, str(dates[i])),
                reject_failed_base_audit=getattr(model, 'reject_failed_base_audit', False))
        return (out.normalized_weights[0], aux['loss'], signature,
                content_hash(out.dq_diagnostics), out.dq_diagnostics['decision_stopping'],
                out.dq_diagnostics['admission_fallback'][0])
    for i in range(len(dates)):
        rngs.append(capture_rng())
        # This pass records values only. Its graphs were previously discarded
        # without backward; the exact replay below owns all parameter gradients.
        # Keep train mode/dropout/RNG unchanged and retain the exact replay gate.
        # Explicit opt-in until a restored-checkpoint equivalence probe passes.
        with (torch.no_grad() if getattr(model, 'no_grad_path_record', False) else nullcontext()):
            weight, penalty, signature, decision_signature, stopping, admission = predict(i)
        values.append(weight.detach()); penalties.append(penalty.detach())
        signatures.append(signature); item_hashes.append(fingerprint(items[i]))
        decision_signatures.append(decision_signature); decision_stopping.append(stopping)
        admission_rows.append(admission)
        del weight, penalty
        if callback: callback('forward',i+1,len(dates))
    end_rng = capture_rng()
    try:
        if any(fingerprint(item) != expected for item,expected in zip(items,item_hashes,strict=True)):
            raise RuntimeError('execution data changed during forward pass')
        leaves = [weight.clone().requires_grad_() for weight in values]
        parts = ledger.run(leaves,items)
        valuation_report = None if ledger.report is None else ledger.report(parts, items)
        economic = net_path_objective(parts['net'],sharpe_weight=sharpe_weight,growth_weight=growth_weight)
        adjoints = torch.autograd.grad(economic['loss'], leaves)
        # Replay the same decision-time samples and stochastic draws for backpropagation.
        if audit_cache is not None:
            audit_cache.begin_replay()
        for i in range(len(dates)):
            if fingerprint(items[i]) != item_hashes[i]:
                raise RuntimeError('execution data changed during gradient replay')
            restore_rng(rngs[i])
            weight, penalty, signature, decision_signature, _, _ = predict(i)
            if (signature != signatures[i] or not torch.equal(weight.detach(),values[i])
                    or not torch.equal(penalty.detach(),penalties[i])
                    or decision_signature != decision_signatures[i]):
                raise RuntimeError('input, decision stopping or stochastic replay mismatch; optimizer not stepped')
            ((weight*adjoints[i].detach()).sum()+penalty/len(dates)).backward()
            del weight, penalty
            if callback: callback('backward',i+1,len(dates))
        if audit_cache is not None:
            audit_cache.finish_replay()
        missing = [name for name,p in model.named_parameters() if p.grad is None]
        if all(p.grad is None for p in params) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
            raise FloatingPointError('missing all or nonfinite DQ gradients')
        norm = torch.nn.utils.clip_grad_norm_(params,clip,error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in params):
            raise FloatingPointError('nonfinite parameter after optimizer update; do not resume automatically')
        model.mark_completed_update()
        return dict(schema='dq.training_update.v1',completed_updates=model.completed_updates,
            model_schema=model.checkpoint_schema,
            decision_mode=model.decision_mode,
            audit_replay_cache=None if audit_cache is None else audit_cache.report(),
            engineering_execution=dict(batched_fine_evidence=getattr(model.core, 'batch_fine_evidence', False),
                batched_access_features=getattr(model.core, 'batch_access_features', False),
                reused_fine_queries=getattr(model.core, 'reuse_fine_queries', False),
                audit_cache_enabled=audit_cache is not None, precision_changed=False,
                no_grad_path_record=getattr(model, 'no_grad_path_record', False),
                direct_encoder_autograd=getattr(model, 'direct_encoder_autograd', False),
                direct_monthly_autograd=getattr(model, 'direct_monthly_autograd', False),
                reject_failed_base_audit=getattr(model, 'reject_failed_base_audit', False),
                batch_replay_hashes=batch_hashes,
                model_or_objective_changed=False),
            economic_loss=float(economic['loss'].detach()),auxiliary_loss=float(torch.stack(penalties).mean()),
            admission_fallback_policy=model.dq_config.admission_spec(),
            admission_fallback_report=admission_report(dates, admission_rows),
            gradient_scope='piecewise_fixed_discrete_selection',
            gradient_norm_before_clip=float(norm),all_replays_exact=True,sessions=len(dates),
            all_decision_replays_exact=True, decision_stopping=decision_stopping,
            stopping_policy=model.dq_config.stopping_spec(),
            parameters_without_gradient=missing,ledger_policy_id=ledger.policy_id,
            execution_valuation_report=valuation_report,
            training_sharpe=float(economic['sharpe'].detach()),
            training_annual_log_growth=float(economic['annualized_log_growth'].detach()),
            local_solver_cost_scope='repair_only',final_weights_only_executed=True,
            auxiliary_penalties_in_returns=False,validation_or_test_used_for_gradients=False)
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise
    finally:
        restore_rng(end_rng)


def evaluate_path(model, dates, get_sample, items, *, window, ledger,
                  statistic_names=None, callback=None):
    """Fixed weights and no auxiliary future-label call; preserve model mode/RNG."""
    if window.role not in {'validation','test'}:
        raise ValueError('evaluation requires validation/test role')
    window.validate(dates,items)
    ledger.validate(model)
    state, mode = capture_rng(), model.training
    weights, diagnostics = [], []
    if getattr(model, '_evaluation_monthly_cache', None) is not None:
        raise ValueError('nested monthly evaluation cache is forbidden')
    monthly_cache = (MonthlyEvaluationCache() if window.role == 'validation'
                     and getattr(model,'cache_validation_monthly',False) else None)
    cache_report = None
    try:
        model.eval()
        if monthly_cache is not None: model._evaluation_monthly_cache = monthly_cache
        with torch.no_grad():
            for i,day in enumerate(dates):
                sample = get_sample(day)
                _bind(sample,items[i],day,window)
                out, _ = forward_sample(model, sample, statistic_names=statistic_names, return_aux=False)
                weights.append(out.normalized_weights[0])
                diagnostics.append(out.dq_diagnostics)
                if callback: callback(window.role,i+1,len(dates))
            parts = ledger.run(weights,items)
            net = parts['net']
            net_path_objective(net)  # finite/NAV check only
            growth = 252*torch.log1p(net).mean()
            lognav = torch.cat([net.new_zeros(1),torch.log1p(net).cumsum(0)])
            metrics = dict(role=window.role,sessions=len(dates),net_cagr=float(torch.expm1(growth)),
                model_schema=model.checkpoint_schema,
                net_sharpe=float(math.sqrt(252)*net.mean()/net.std(unbiased=True).clamp_min(1e-8)),
                net_max_drawdown=float(torch.expm1(lognav-lognav.cummax(0).values).min()),
                annualization_sessions=252,sharpe_ddof=1,risk_free_rate=0.,
                ledger_policy_id=ledger.policy_id,weights_fixed=True,auxiliary_penalties_in_returns=False)
            metrics.update(admission_fallback_policy=model.dq_config.admission_spec(),
                admission_fallback_report=admission_report(dates, [d['admission_fallback'][0] for d in diagnostics]))
            if ledger.report is not None:
                metrics['execution_valuation_report'] = ledger.report(parts, items)
    finally:
        if monthly_cache is not None:
            cache_report = monthly_cache.report()
            monthly_cache.clear()
            if hasattr(model, '_evaluation_monthly_cache'): del model._evaluation_monthly_cache
        model.train(mode); restore_rng(state)
    result = dict(parts=parts,metrics=metrics,decision_diagnostics=diagnostics)
    if cache_report is not None: result['monthly_encoder_cache'] = cache_report
    return result


def _cpu_tree(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {key:_cpu_tree(item) for key,item in value.items()}
    if isinstance(value,list): return [_cpu_tree(item) for item in value]
    if isinstance(value,tuple): return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def save_training_checkpoint(path, model, optimizer, *, binding):
    """Append-only portable model + optimizer + RNG; legacy schemas untouched."""
    if not isinstance(binding,dict) or not binding:
        raise ValueError('explicit data/protocol binding required')
    if type(optimizer) not in (torch.optim.AdamW,torch.optim.SGD):
        raise TypeError('portable resume supports explicit AdamW or SGD optimizers')
    names = {id(p):name for name,p in model.named_parameters()}
    groups = [[names[id(p)] for p in group['params']] for group in optimizer.param_groups]
    flattened = [name for group in groups for name in group]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(names.values()):
        raise ValueError('checkpoint optimizer must cover the whole model exactly once')
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.partial')
    if path.exists() or temporary.exists():
        raise FileExistsError('append-only checkpoint path required')
    path.parent.mkdir(parents=True,exist_ok=True)
    payload = dict(schema='dq.training_checkpoint.v2',model=model.checkpoint_payload(),
        optimizer=_cpu_tree(optimizer.state_dict()),optimizer_class=type(optimizer).__name__,
        optimizer_parameter_names=groups,binding=copy.deepcopy(binding),rng=capture_rng())
    payload['content_sha256'] = fingerprint(payload)
    with temporary.open('xb') as stream:
        torch.save(payload,stream)
        stream.flush()
        os.fsync(stream.fileno())
    check = torch.load(temporary,map_location='cpu',weights_only=True)
    if check.pop('content_sha256') != fingerprint(check):
        raise ValueError('checkpoint readback mismatch; partial file retained')
    # Atomic commit without replacement, including on POSIX AutoDL hosts.
    os.link(temporary,path)
    temporary.unlink()
    return dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                completed_updates=model.completed_updates)


def load_training_checkpoint(path, *, binding, expected_sha256, device='cpu', restore_training_rng=False):
    """Explicit resume only. No launch/restart is performed by loading."""
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('checkpoint file hash mismatch')
    payload = torch.load(path,map_location='cpu',weights_only=True)
    content_hash = payload.pop('content_sha256',None)
    if (payload.get('schema') != 'dq.training_checkpoint.v2' or payload.get('binding') != binding
            or content_hash != fingerprint(payload)):
        raise ValueError('checkpoint schema, data binding or contents differ')
    caller_rng = capture_rng()
    try:
        model = DQDailyPortfolio.from_checkpoint_payload(payload['model']).to(device)
        classes = {'AdamW':torch.optim.AdamW,'SGD':torch.optim.SGD}
        if payload['optimizer_class'] not in classes:
            raise ValueError('unsupported optimizer class')
        named = dict(model.named_parameters())
        group_names = payload['optimizer_parameter_names']
        flat = [name for group in group_names for name in group]
        if len(flat) != len(set(flat)) or set(flat) != set(named):
            raise ValueError('optimizer parameter mapping differs from model')
        groups = [dict(params=[named[name] for name in group]) for group in group_names]
        optimizer = classes[payload['optimizer_class']](groups,lr=1e-4)
        optimizer.load_state_dict(payload['optimizer'])
    finally:
        restore_rng(caller_rng)
    if restore_training_rng:
        restore_rng(payload['rng'])
    return model,optimizer,payload
