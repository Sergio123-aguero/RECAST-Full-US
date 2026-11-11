"""Translate execution inputs into terminal marks and signed cash settlements."""
from collections import Counter
import numpy as np
from .daily_portable_v7 import ARRAY_KEYS, DATE_KEYS
from .daily_terminal_liquidation_v17 import POLICY_ID as KERNEL_POLICY, validate_terminal_path

SCHEMA = 'pcrr.quality-fixed.assumed-terminal-liquidation.v47'

POLICY_ID = 'V47_V46_CURRENT_TERMINAL_MARK_ELSE_LAST_CARRY_SIGNED_CLOSE'

BOOL_KEYS = ARRAY_KEYS[3:-2]


def disclosure():
    return dict(synthetic_terminal_liquidation_assumed=True, close_only_on_recorded_terminal_event=True,
                future_value_backdated=False, same_interval_reinvestment=False)


def adapt_v46_path(items):
    """Convert one independent cash-start path; never mutate parent arrays."""
    if len(items) < 2:
        raise ValueError('at least two independent cash-start observations required')
    ids = np.asarray(items[0]['asset_ids'])
    n = len(ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or not np.array_equal(ids, np.unique(ids)):
        raise ValueError('sorted unique accounting identities required')
    expected = np.zeros(n, bool)
    retired = np.zeros(n, bool)
    revised, events = [], []
    removed_carry_days = 0
    observed_cash_events = 0
    old = None
    for source in items:
        if not np.array_equal(ids, source['asset_ids']):
            raise ValueError('fixed accounting union required')
        if not source['signal_date'] < source['execution_date'] < source['outcome_date']:
            raise ValueError('invalid signal/fill/outcome clock')
        if old is not None and old['outcome_date'] != source['execution_date']:
            raise ValueError('noncontiguous independent path')
        for key in ARRAY_KEYS:
            a = np.asarray(source[key])
            if a.ndim != 1 or not np.isfinite(a).all():
                raise ValueError('nonfinite execution array: ' + key)
            if key not in ('target_asset_ids', 'target_indices') and a.shape != (n,):
                raise ValueError('wrong execution shape: ' + key)
            if key in BOOL_KEYS and a.dtype != np.bool_:
                raise ValueError('boolean masks required: ' + key)
        target, index = source['target_asset_ids'], source['target_indices']
        if (not np.issubdtype(target.dtype, np.integer) or not np.issubdtype(index.dtype, np.integer)
                or not np.array_equal(target, np.unique(target)) or len(target) != len(index)
                or np.any(index < 0) or np.any(index >= n) or not np.array_equal(ids[index], target)):
            raise ValueError('invalid target to accounting mapping')
        start, carry, terminal, cash, zero, observed, stale, fallback, mark_update = (
            source[k] for k in ('receivable_start', 'receivable_carry', 'terminal_end',
                'terminal_cash_settlement_observed', 'requires_zero_holding', 'observed_return',
                'stale_mark', 'receivable_start_last_mark', 'receivable_mark_update'))
        ret, fraction = source['mark_returns'], source['receivable_cash_fraction']
        if not np.array_equal(carry, expected):
            raise ValueError('orphan or discontinuous parent frozen claim')
        if np.any(ret < -1) or np.any((fraction < 0) | (fraction > 1) | (~start & (fraction != 0))):
            raise ValueError('invalid parent return or known cash fraction')
        if np.any(~observed & ~stale & ~zero) or np.any(stale & (observed | zero | (ret != 0))):
            raise ValueError('unclassified or inconsistent parent valuation')
        if np.any(cash & (~terminal | ((~observed | stale) & ~zero))):
            raise ValueError('invalid observed terminal cash')
        if np.any(start & (~terminal | cash | zero | carry)):
            raise ValueError('close requires a recorded terminal claim start')
        if np.any(start & ~fallback & (~observed | stale)):
            raise ValueError('ordinary terminal start needs a current mark')
        if np.any(fallback & (~start | observed | ~stale | (ret != 0))):
            raise ValueError('last-carry proxy must not become an observed zero return')
        if np.any(mark_update & (~carry | start | ~observed | stale)):
            raise ValueError('invalid delayed parent mark update')
        if np.any(carry & (source['tradable_fill'] | zero | terminal |
                          (~mark_update & (observed | ~stale | (ret != 0))))):
            raise ValueError('invalid frozen parent transition')
        if np.any(retired & source['tradable_fill']):
            raise ValueError('tradable lifecycle after terminal close needs separate classification')
        row = {k: source[k].copy() for k in ARRAY_KEYS}
        row.update({k: source[k] for k in DATE_KEYS})
        row.update(settlement_policy_id=KERNEL_POLICY,
                   terminal_liquidation_assumptions=[], settlement_evidence=[],
                   receivable_release_fraction=np.zeros(n, np.float64),
                   assumed_liquidation_fraction=np.zeros(n, np.float64))
        removed_carry_days += int(np.count_nonzero(carry & retired))
        for key in BOOL_KEYS:
            row[key][retired] = key == 'requires_zero_holding'
        for key in ('mark_returns', 'receivable_cash_fraction'):
            row[key][retired] = 0.
        for j in np.flatnonzero(row['receivable_start']):
            row['assumed_liquidation_fraction'][j] = 1 - fraction[j]
            proof = dict(permno=int(ids[j]), signal_date=str(source['signal_date']),
                settlement_date=str(source['outcome_date']),
                basis='LAST_CARRY_PROXY' if fallback[j] else 'CURRENT_TERMINAL_MARK',
                mark_return=float(ret[j]), known_cash_fraction=float(fraction[j]),
                assumed_fraction=float(1-fraction[j]), origin='V46_TERMINAL_CLAIM_START',
                actual_execution_certified=False, future_value_backdated=False)
            row['terminal_liquidation_assumptions'].append(proof)
            events.append(proof)
        observed_cash_events += int(np.count_nonzero(cash & ~retired))
        retired |= start | cash
        expected = carry | (start & (fraction < 1))
        revised.append(row)
        old = source
    validate_terminal_path(revised)
    return revised, dict(schema=SCHEMA, source_ledger='V46_V4_NOT_V16_SETTLEMENT_EVIDENCE',
        policy_id=POLICY_ID, kernel_policy_id=KERNEL_POLICY,
        events=events, terminal_claim_events=len(events),
        verified_cash_terminal_events=observed_cash_events,
        valuation_basis_counts=dict(Counter(e['basis'] for e in events)),
        old_carry_asset_days_removed=removed_carry_days, **disclosure())
