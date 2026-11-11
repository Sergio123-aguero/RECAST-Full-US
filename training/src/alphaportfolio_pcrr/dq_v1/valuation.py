"""Differentiable, explicitly assumed valuation for DQ train/validation/test.

The fill and mark kernels carry executed holdings across days and supply net returns for training.
Assumptions are accounting outcomes, never observed auxiliary training labels.
"""
import copy
from collections import Counter

import numpy as np
import torch

from ..daily_terminal_liquidation_v17 import POLICY_ID as TERMINAL_POLICY, validate_terminal_path
from ..quality_liquidation_v47 import adapt_v46_path
from ..test_value_assumptions_v47a import bind_events
from .training import LedgerContract, aware, fingerprint


POLICY_ID = 'DQ71_V47_TERMINAL_V47A_RESIDUAL_BOOK_ASSUMED_VALUATION_V1'
COST_RATE = .001


def policy_spec():
    return dict(policy_id=POLICY_ID, schema='dq.assumed_valuation.v1',
        ordinary_suspension='last_available_carry_mark_until_resumption_no_fabricated_sale',
        terminal='current_terminal_value_else_last_carry_signed_close_once',
        unknown_property='V47A_known_component_plus_nonnegative_residual_book_mark',
        other_bound_unknown='last_carry_mark_then_original_marks_resume',
        confirmed_zero_recovery_preserved=True, cash_and_property_counted_once=True,
        short_liabilities_erased=False, future_recoveries_backdated=False,
        later_claim_recoveries_belong_to_assumed_buyer=True,
        same_interval_reinvestment_of_assumed_proceeds=False,
        ordinary_missing_price_implies_delisting=False,
        missing_value_assumed_zero=False, assumed_values_used_as_observed_auxiliary_labels=False,
        fee_per_abs_executed_weight=.0005, fee_per_abs_synthetic_disposal=.0005,
        cash_interest_borrow_spread_impact_included=False,
        source_observations_or_stock_pool_modified=False,
        actual_execution_certified=False, assumption_may_bias_returns_and_risk=True,
        same_policy_for_train_validation_test=True)


def _clock(source, clocks):
    supplied = {} if clocks is None else clocks[str(source['signal_date'])]
    values = {k: supplied.get(k, source.get(k)) for k in
              ('execution_at', 'fill_available_at', 'outcome_available_at')}
    execution, fill, outcome = (aware(values[k]) for k in values)
    if (execution.date().isoformat() != str(source['execution_date'])
            or not execution <= fill < outcome
            or outcome.date().isoformat() < str(source['outcome_date'])):
        raise ValueError('invalid explicit fill/outcome availability clock')
    return {k: aware(v).isoformat() for k, v in values.items()}


def prepare_path(items, *, source_binding, clocks=None, evidence=()):
    """Bind basic/V4/V17 paths and dated unknown-event evidence without IO.

    `clocks` is keyed by signal date and supplies timezone-aware execution,
    fill-publication and outcome-publication instants; never infer a clock from
    a date. Unknown property/cash gaps require the existing V47A event evidence
    plus `available_at`. Unclassified/corrupt records still fail admission.
    """
    if len(items) < 2 or not isinstance(source_binding, dict) or not source_binding:
        raise ValueError('independent cash-start path and explicit source binding required')
    source_hash = fingerprint(items)
    original = copy.deepcopy(items)
    times = [_clock(row, clocks) for row in original]
    policies = {r.get('settlement_policy_id') for r in original}
    if policies == {TERMINAL_POLICY}:
        validate_terminal_path(original)
        rows = copy.deepcopy(original)
        conversion = dict(source='already_bound_V17', policy_id=TERMINAL_POLICY)
    elif policies == {None}:
        # Basic execution paths have no claim-phase arrays. Admit their absence
        # only as an entire group, never fill in a partially corrupted schema.
        extra = ('receivable_start', 'receivable_carry', 'receivable_start_last_mark',
                 'receivable_mark_update', 'receivable_cash_fraction')
        for row in original:
            present = [k in row for k in extra]
            if any(present) and not all(present):
                raise ValueError('incomplete receivable phase schema')
            if not any(present):
                for k in extra:
                    row[k] = np.zeros(len(row['asset_ids']), dtype=float if k.endswith('fraction') else bool)
        rows, conversion = adapt_v46_path(original)
    else:
        raise ValueError('mixed or unsupported execution valuation policy')
    by_date = {str(r['signal_date']): i for i, r in enumerate(rows)}
    if len(by_date) != len(rows):
        raise ValueError('duplicate execution signal date')
    for row, time in zip(rows, times, strict=True):
        row.update(time)
    for a, b in zip(rows, rows[1:]):
        if aware(a['outcome_available_at']) > aware(b['fill_available_at']):
            raise ValueError('preceding outcome unavailable at next fill publication')
    # Reuse the existing event formula/binding checks rather than silently
    # changing historical accounting assumptions for the new model.
    events = bind_events(rows, list(evidence))
    grouped = {}
    for e in events:
        row = rows[e['row']]
        if ('available_at' not in e or aware(e['available_at']) > aware(row['outcome_available_at'])
                or aware(e['available_at']).date().isoformat() < str(row['outcome_date'])):
            raise ValueError('unknown-event evidence has invalid/future availability')
        grouped.setdefault(e['row'], []).append(e)
    gap_retired = np.zeros(len(rows[0]['asset_ids']), bool)
    counts = Counter()
    for i, row in enumerate(rows):
        n = len(row['asset_ids'])
        r = row['mark_returns'].astype(np.float64, copy=True)
        extra = np.zeros(n, np.float64)
        close = (row['terminal_cash_settlement_observed'] | row['receivable_start']).copy()
        exit_ratio = (1+r)*row['assumed_liquidation_fraction']
        permitted = np.zeros(n, bool)
        assumed = (row['stale_mark'] | row['receivable_start']).copy()
        notes = [dict(e, kind='terminal_claim_close') for e in row['terminal_liquidation_assumptions']]
        # A genuinely new tradable lifecycle can open a new position. Old
        # payments of an already disposed claim must never credit it twice.
        gap_retired &= ~row['tradable_fill']
        for e in grouped.get(i, []):
            j, value = e['column'], e['valuation']
            if gap_retired[j]:
                raise ValueError('an already disposed claim cannot receive another assumption')
            permitted[j] = assumed[j] = True
            kind = value['kind']
            if kind == 'terminal_residual_close':
                total = value['known_value_ratio']+value['residual_ratio']
                r[j] = total-1
                close[j] = True
                exit_ratio[j] = max(0., total-value['known_paid_cash_ratio'])
            elif kind == 'nonterminal_property_split':
                r[j] = value['partial_return']
                extra[j] = value['residual_ratio']
                exit_ratio[j] = extra[j]
            elif kind == 'last_book_interval':
                r[j] = 0.
            else:
                raise ValueError('unknown residual-book valuation kind')
            notes.append(dict(value, permno=int(e['permno']), kind=kind,
                signal_date=str(row['signal_date']), settlement_date=str(row['outcome_date']),
                available_at=e['available_at'],
                assumption_is_not_vendor_observation=True))
        r[gap_retired] = extra[gap_retired] = exit_ratio[gap_retired] = 0.
        close[gap_retired] = False
        unresolved = (row['requires_zero_holding'] & ~permitted) | gap_retired
        assumed[gap_retired] = False
        if (not np.isfinite(r).all() or (r < -1).any()
                or not np.isfinite(extra).all() or (extra < 0).any()
                or not np.isfinite(exit_ratio).all() or (exit_ratio < 0).any()):
            raise ValueError('invalid assumed mark or disposal ratio')
        for note in notes:
            counts[note['kind']] += 1
        row.update(dq_valuation_policy_id=POLICY_ID,
            dq_valuation_assumed=assumed, dq_valuation_return=r,
            dq_valuation_extra_cash_ratio=extra, dq_valuation_close=close,
            dq_valuation_exit_ratio=exit_ratio, dq_valuation_unresolved=unresolved,
            dq_valuation_events=notes)
        gap_retired |= close & permitted
    spec = policy_spec()
    receipt = dict(schema='dq.assumed_valuation_binding.v1', policy=spec,
        source_binding=copy.deepcopy(source_binding), source_path_sha256=source_hash,
        clocks_sha256=fingerprint(times), event_evidence_sha256=fingerprint(evidence),
        prepared_path_sha256=fingerprint(rows), policy_sha256=fingerprint(spec),
        sessions=len(rows), accounting_assets=len(rows[0]['asset_ids']),
        bound_event_counts=dict(counts), terminal_conversion=conversion,
        source_mutated=False, stocks_deleted=False, days_skipped=False)
    return rows, ledger_for(rows, receipt), receipt


def _fill(raw, row, prior):
    """Execution-only kernel: must not inspect this interval's future marks."""
    if (raw.ndim != 1 or raw.numel() != len(row['target_asset_ids'])
            or not torch.isfinite(raw).all() or bool(raw.abs().sum() > 1+2e-5)):
        raise ValueError('invalid target portfolio')
    w = raw.to(dtype=torch.float64)
    index = torch.as_tensor(row['target_indices'], device=w.device)
    target = w.new_zeros(len(row['asset_ids'])).index_add(0, index, w)
    h = torch.zeros_like(target) if prior is None else prior
    tradable = torch.as_tensor(row['tradable_fill'], device=w.device)
    locked, desired = torch.where(tradable, 0., h), torch.where(tradable, target, 0.)
    locked_gross, desired_gross = locked.abs().sum(), desired.abs().sum()
    if bool(locked_gross > 1+1e-10):
        raise FloatingPointError('ordinary suspended holdings exceed gross budget')
    def positions(fee):
        scale = torch.minimum(w.new_tensor(1.), (1-fee-locked_gross).clamp_min(0.)/desired_gross.clamp_min(1e-30))
        return locked+scale*desired
    fee = w.new_zeros(())
    for _ in range(16):
        fee = .5*COST_RATE*(positions(fee)-h).abs().sum()
    p = positions(fee)
    turnover = .5*(p-h).abs().sum()
    actual_fee = COST_RATE*turnover
    residual = (fee-actual_fee).abs()
    postfee = p.abs().sum()/(1-actual_fee)
    if bool(residual > 1e-11) or bool(locked_gross+actual_fee > 1+1e-10) or bool(postfee > 1+1e-10):
        raise FloatingPointError('infeasible fee/carry budget')
    return dict(executed_positions=p, pretrade_positions=h, rebalance_fees=actual_fee,
        rebalance_turnover=turnover, locked_gross=locked_gross, postfee_gross=postfee,
        fee_fixedpoint_residual=residual,
        cancelled_target_gross=torch.where(tradable, 0., (target-h).abs()).sum())


def _mark(fill, row):
    p = fill['executed_positions']
    def tensor(key):
        return torch.as_tensor(row[key], device=p.device)
    r, extra_ratio, exit_ratio = (tensor(k).to(p.dtype) for k in
        ('dq_valuation_return', 'dq_valuation_extra_cash_ratio', 'dq_valuation_exit_ratio'))
    close = tensor('dq_valuation_close')
    if bool(((p != 0) & tensor('dq_valuation_unresolved')).any()):
        raise ValueError('unclassified held value needs dated assumption evidence')
    if bool(((p != 0) & tensor('terminal_end') & ~close).any()):
        raise ValueError('terminal holding lacks a bound close policy')
    v = p*(1+r)
    extra_cash, disposal = p*extra_ratio, p*exit_ratio
    exit_turn = .5*disposal.abs().sum()
    exit_fee = COST_RATE*exit_turn
    fee = fill['rebalance_fees']+exit_fee
    gross = (p*r).sum()+extra_cash.sum()
    net = gross-fee
    if not torch.isfinite(net) or bool(net <= -1):
        raise FloatingPointError('nonpositive or invalid research NAV')
    settled = torch.where(close, v, 0.)
    end = (v-settled)/(1+net)
    cash = (1-fill['rebalance_fees']-p.sum()+settled.sum()+extra_cash.sum()-exit_fee)/(1+net)
    residual = (end.sum()+cash-1).abs()
    if bool(residual > 1e-10):
        raise FloatingPointError('cash plus signed holdings do not reconcile')
    assumed = tensor('dq_valuation_assumed')
    return dict(**fill, net=net, gross=gross, fees=fee,
        turnover=fill['rebalance_turnover']+exit_turn, terminal_exit_fees=exit_fee,
        terminal_cash_value=settled.sum(), property_cash=extra_cash.sum(),
        assumed_disposal_gross=disposal.abs().sum(),
        assumed_long_proceeds=disposal.clamp_min(0).sum(),
        assumed_short_payment=-disposal.clamp_max(0).sum(),
        assumed_exposure=torch.where(assumed, p.abs(), 0.).sum(),
        assumed_affected_count=((p != 0) & assumed).sum(),
        stale_gross=torch.where(tensor('stale_mark'), p.abs(), 0.).sum(),
        end_positions=end, end_cash=cash, reconciliation_residual=residual)


class AssumedPath:
    """One immutable dataset/role/path binding; no hidden state between runs."""
    def __init__(self, items, receipt):
        self.items, self.receipt = items, copy.deepcopy(receipt)
        self.check(items)

    def check(self, items):
        if (self.receipt['policy'] != policy_spec()
                or self.receipt['policy_sha256'] != fingerprint(policy_spec())
                or fingerprint(items) != self.receipt['prepared_path_sha256']):
            raise ValueError('valuation path/policy differs from the bound preparation')

    def __call__(self, weights, items):
        self.check(items)
        if len(weights) != len(items):
            raise ValueError('unaligned valuation path targets')
        prior, series = None, {}
        for raw, row in zip(weights, items, strict=True):
            marked = _mark(_fill(raw, row, prior), row)
            prior = marked['end_positions']
            for key, value in marked.items():
                series.setdefault(key, []).append(value)
        return {key: torch.stack(value) for key, value in series.items()}

    def report(self, parts, items):
        self.check(items)
        def scalar(key, op='sum'):
            return float(getattr(parts[key].detach(), op)().cpu())
        events = []
        for i, row in enumerate(items):
            p = parts['executed_positions'][i].detach().cpu().numpy()
            for event in row['dq_valuation_events']:
                j = int(np.searchsorted(row['asset_ids'], event['permno']))
                if p[j] != 0:
                    events.append(dict(event, actual_signed_weight=float(p[j]),
                        assumed_disposal_signed=float(p[j]*row['dq_valuation_exit_ratio'][j]),
                        property_cash_signed=float(p[j]*row['dq_valuation_extra_cash_ratio'][j])))
        return dict(policy=policy_spec(), binding=copy.deepcopy(self.receipt), events=events,
            affected_asset_intervals=int(scalar('assumed_affected_count')),
            assumed_exposure_sum=scalar('assumed_exposure'),
            assumed_long_proceeds_sum=scalar('assumed_long_proceeds'),
            assumed_short_payment_sum=scalar('assumed_short_payment'),
            assumed_exit_fees_sum=scalar('terminal_exit_fees'),
            reconciliation_max=scalar('reconciliation_residual', 'max'),
            fee_fixedpoint_residual_max=scalar('fee_fixedpoint_residual', 'max'),
            target_and_ledger_cash_separated=True, actual_execution_certified=False)


def ledger_for(items, receipt):
    bound = AssumedPath(items, receipt)
    return LedgerContract(POLICY_ID, .0005, bound,
                          report=bound.report, binding=copy.deepcopy(receipt))


def validate_binding(ledger, items):
    if (ledger.policy_id != POLICY_ID or not isinstance(ledger.run, AssumedPath)
            or ledger.binding != ledger.run.receipt
            or ledger.report != ledger.run.report):
        raise ValueError('formal71 requires the bound execution and valuation ledger')
    ledger.run.check(items)
