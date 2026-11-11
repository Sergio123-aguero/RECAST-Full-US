"""Prepare and validate terminal valuations and signed settlement inputs."""
from collections import Counter
from datetime import date
import json
from pathlib import Path
import numpy as np
from .daily_portable_v7 import ARRAY_KEYS, DATE_KEYS, inventory, verify_manifest, write_new
from .daily_settlement_v16 import POLICY_ID as PARENT_POLICY

POLICY_ID = 'V17_RESEARCH_IMMEDIATE_TERMINAL_CLOSE_CURRENT_MARK_ELSE_LAST_CARRY'

KEYS = (*ARRAY_KEYS, 'receivable_release_fraction', 'assumed_liquidation_fraction')


def apply_terminal_liquidation(items, lifecycle_receipt=None):
    if not items:
        raise ValueError('nonempty independent cash-start path required')
    # The old adapter represents delayed CASH as a locked stock, not a claim.
    # Its recorded start must therefore also be closed, without reading the
    # later payment or using that future price at the earlier date.
    pending = {}
    for e in (lifecycle_receipt or {}).get('lifecycle_repair', {}).get('evidence', []):
        if e.get('kind') == 'DELAYED_CASH_START':
            key = (str(e['signal_date']), int(e['permno']))
            if key in pending:
                raise ValueError('duplicate delayed-cash identity')
            if not np.isfinite(e['anchor_price']) or e['anchor_price'] <= 0:
                raise ValueError('delayed-cash proxy needs a valid previous carrying price')
            pending[key] = e
    revised, events, retired, matched = [], [], set(), set()
    cleared = 0
    ids = items[0]['asset_ids']
    for original in items:
        if original.get('settlement_policy_id') != PARENT_POLICY:
            raise ValueError('verified V16 parent required')
        if not np.array_equal(ids, original['asset_ids']):
            raise ValueError('fixed accounting union required')
        item = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in original.items()}
        n = len(ids)
        item['settlement_policy_id'] = POLICY_ID
        item['terminal_liquidation_assumptions'] = []
        item['settlement_evidence'] = []  # parent file preserves original evidence
        item['assumed_liquidation_fraction'] = np.zeros(n, np.float64)
        for j, permno in enumerate(ids):
            p = int(permno)
            if p in retired:
                if item['tradable_fill'][j]:
                    raise ValueError('new tradable lifecycle after terminal close needs separate classification')
                cleared += int(item['receivable_carry'][j])
                for k in ('tradable_fill', 'observed_return', 'stale_mark', 'terminal_end',
                          'terminal_cash_settlement_observed', 'receivable_start',
                          'receivable_carry', 'receivable_start_last_mark', 'receivable_mark_update'):
                    item[k][j] = False
                item['requires_zero_holding'][j] = True
                for k in ('mark_returns', 'receivable_cash_fraction', 'receivable_release_fraction'):
                    item[k][j] = 0.
                continue
            key = (str(item['signal_date']), p)
            delayed = pending.get(key)
            if delayed is not None:
                if (item['terminal_end'][j] or item['observed_return'][j]
                        or not item['stale_mark'][j] or item['mark_returns'][j] != 0
                        or item['requires_zero_holding'][j]
                        or str(item['execution_date']) != delayed['anchor_date']):
                    raise ValueError('delayed cash evidence does not bind to last-mark start')
                item['terminal_end'][j] = item['receivable_start'][j] = True
                item['receivable_start_last_mark'][j] = True
                matched.add(key)
            if item['receivable_carry'][j] or item['receivable_release_fraction'][j] != 0:
                raise ValueError('orphan carried claim: start must be inside this independent path')
            if not item['receivable_start'][j]:
                continue
            fallback = bool(item['receivable_start_last_mark'][j])
            ret = float(item['mark_returns'][j])
            cash = float(item['receivable_cash_fraction'][j])
            if (not np.isfinite(ret) or ret < -1 or not np.isfinite(cash) or not 0 <= cash <= 1
                    or not item['terminal_end'][j] or item['terminal_cash_settlement_observed'][j]
                    or item['requires_zero_holding'][j]
                    or (fallback and (ret != 0 or item['observed_return'][j] or not item['stale_mark'][j]))
                    or (not fallback and (not item['observed_return'][j] or item['stale_mark'][j]))):
                raise ValueError('invalid terminal valuation or last-carry fallback')
            item['assumed_liquidation_fraction'][j] = 1-cash
            proof = {'permno': p, 'signal_date': str(item['signal_date']),
                'settlement_date': str(item['outcome_date']),
                'basis': 'LAST_CARRY_PROXY' if fallback else 'CURRENT_TERMINAL_MARK',
                'mark_return': ret, 'known_cash_fraction': cash, 'assumed_fraction': 1-cash,
                'origin': 'DELAYED_CASH_START' if delayed else 'TERMINAL_CLAIM_START',
                'actual_execution_certified': False, 'future_value_backdated': False}
            if delayed:
                proof['proxy_anchor_price'] = float(delayed['anchor_price'])
                proof['proxy_anchor_date'] = delayed['anchor_date']
            item['terminal_liquidation_assumptions'].append(proof)
            events.append(proof)
            retired.add(p)
        revised.append(item)
    if matched != set(pending):
        raise ValueError('delayed cash start outside supplied execution path')
    validate_terminal_path(revised)
    return revised, {'schema': 'pcrr.assumed-terminal-liquidation.v17', 'policy_id': POLICY_ID,
        'status': 'PASS_EXPLICIT_RESEARCH_ASSUMPTION', 'events': events,
        'terminal_identities_closed': len(retired),
        'valuation_basis_counts': dict(Counter(e['basis'] for e in events)),
        'delayed_cash_starts_closed': len(matched), 'old_carry_asset_days_removed': cleared,
        'execution_certified': False, 'future_terminal_values_used_early': False,
        'unknown_value_replaced_with_zero': False, 'short_liabilities_erased': False,
        'last_carry_proxy_can_bias_results_in_either_direction': True,
        'subsequent_claim_payments_and_marks_belong_to_assumed_buyer': True,
        'normal_suspension_policy_unchanged': True, 'additional_stock_exclusions': [],
        'raw_sources_and_target_axes_modified': False, 'parent_checkpoint_overwritten': False,
        'fee_rule': 'existing cost_rate times half-L1; add half absolute residual liquidation notional',
        'same_interval_reinvestment': False, 'old_results_recomputed': False}


def validate_terminal_path(items):
    """Bind assumptions to current dates/masks; never label them observed cash."""
    retired, old = set(), None
    for item in items:
        if item.get('settlement_policy_id') != POLICY_ID:
            raise ValueError('explicit V17 research policy required')
        if not item['signal_date'] < item['execution_date'] < item['outcome_date']:
            raise ValueError('signal/fill/outcome clock invalid')
        if old is not None and (old['outcome_date'] != item['execution_date']
                or not np.array_equal(old['asset_ids'], item['asset_ids'])):
            raise ValueError('noncontiguous path or changing accounting union')
        ids = item['asset_ids']; n = len(ids)
        for k in KEYS:
            v = np.asarray(item[k])
            if v.ndim != 1 or not np.isfinite(v).all():
                raise ValueError('nonfinite or invalid execution field: ' + k)
            if k not in ('target_asset_ids', 'target_indices') and len(v) != n:
                raise ValueError('wrong accounting shape: ' + k)
        if np.any(item['receivable_carry']) or np.any(item['receivable_mark_update']) or np.any(item['receivable_release_fraction']):
            raise ValueError('post-close carry/payment/revaluation is forbidden')
        sim = item['assumed_liquidation_fraction']; start = item['receivable_start']
        if np.any((sim < 0) | (sim > 1)) or np.any(sim[~start] != 0):
            raise ValueError('invalid assumed liquidation fraction or phase')
        if not np.array_equal(sim[start], 1-item['receivable_cash_fraction'][start]):
            raise ValueError('known cash plus assumed close must equal one')
        proofs = item['terminal_liquidation_assumptions']
        expected = set(map(int, ids[start]))
        if {e['permno'] for e in proofs} != expected or len(proofs) != len(expected):
            raise ValueError('missing or duplicate terminal assumption')
        for e in proofs:
            j = int(np.searchsorted(ids, e['permno']))
            fallback = bool(item['receivable_start_last_mark'][j])
            if (e['settlement_date'] != str(item['outcome_date']) or e['signal_date'] != str(item['signal_date'])
                    or e['basis'] != ('LAST_CARRY_PROXY' if fallback else 'CURRENT_TERMINAL_MARK')
                    or e['mark_return'] != float(item['mark_returns'][j])
                    or e['known_cash_fraction'] != float(item['receivable_cash_fraction'][j])
                    or e['assumed_fraction'] != float(sim[j])
                    or e['actual_execution_certified'] is not False or e['future_value_backdated'] is not False):
                raise ValueError('misdated or inconsistent research assumption')
        for p in retired:
            j = int(np.searchsorted(ids, p))
            if (item['tradable_fill'][j] or item['terminal_end'][j] or item['mark_returns'][j] != 0
                    or not item['requires_zero_holding'][j]):
                raise ValueError('closed claim resurrected')
        retired.update(expected)
        old = item


def export_terminal_execution(root, items, receipt):
    validate_terminal_path(items)
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    packed = {f'{i:03}_{k}': r[k] for i,r in enumerate(items) for k in KEYS}
    with (root/'execution_v17.npz').open('xb') as f:
        np.savez_compressed(f, **packed)
    write_new(root/'EXECUTION_V17.json', {'policy_id': POLICY_ID, 'keys': KEYS,
        'dates': [{k: str(r[k]) for k in DATE_KEYS} for r in items],
        'assumptions': [r['terminal_liquidation_assumptions'] for r in items]})
    write_new(root/'LIQUIDATION_RECEIPT.json', receipt)
    write_new(root/'MANIFEST.json', {'files': inventory(root)})


def load_terminal_execution(root):
    root = Path(root); verify_manifest(root)
    meta = json.loads((root/'EXECUTION_V17.json').read_text(encoding='utf-8'))
    if meta['policy_id'] != POLICY_ID or meta['keys'] != list(KEYS):
        raise ValueError('wrong terminal execution format')
    result = []
    with np.load(root/'execution_v17.npz', allow_pickle=False) as z:
        expected = {f'{i:03}_{k}' for i in range(len(meta['dates'])) for k in KEYS}
        if set(z.files) != expected:
            raise ValueError('missing or surplus terminal execution arrays')
        for i, (dates, assumptions) in enumerate(zip(meta['dates'], meta['assumptions'], strict=True)):
            r = {k: date.fromisoformat(v) for k,v in dates.items()}
            r.update({k: z[f'{i:03}_{k}'] for k in KEYS})
            r.update(settlement_policy_id=POLICY_ID, terminal_liquidation_assumptions=assumptions)
            result.append(r)
    validate_terminal_path(result)
    return result
