"""Apply decision-date and valuation-date checks to unpriced corporate actions."""
import math
import numpy as np
from .daily_terminal_liquidation_v17 import validate_terminal_path


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def policy_for_event(e):
    """Use only the current interval; later settlement fields are never used."""
    day, execution = e['outcome_date'], e['execution_date']
    now = [r for r in e['daily_rows'] if str(r['dlycaldt'])[:10] == day]
    before = [r for r in e['daily_rows'] if str(r['dlycaldt'])[:10] == execution]
    if len(now) > 1 or len(before) > 1:
        raise ValueError('ambiguous daily evidence')
    now = now[0] if now else {}
    before = before[0] if before else {}
    same_interval = str(now.get('dlyprevdt'))[:10] == execution
    price = before.get('dlyprc')
    if not finite(price) or price <= 0:
        price = now.get('dlyprevprc') if same_interval else None
    partial = now.get('dlyret')
    partial_ok = (e['return_missing_flag'] == 'MV' and same_interval
                  and finite(partial) and partial >= -1)
    cash_value, cash_paid = 0., 0.
    for r in e['distribution_rows']:
        if str(r['disexdt'])[:10] != day:
            raise ValueError('distribution outside the event date')
        amount = r.get('disdivamt')
        if r['dispaymenttype'] == 'USD' and finite(amount) and amount >= 0:
            if not finite(price) or price <= 0:
                raise ValueError('known cash lacks a positive prior share price')
            ratio = amount/price
            cash_value += ratio
            if r.get('dispaydt') and str(r['dispaydt'])[:10] <= day:
                cash_paid += ratio
    if e['terminal_end']:
        known = max(cash_value, 1+partial if partial_ok else 0.)
        return dict(kind='terminal_residual_close', known_value_ratio=known,
            residual_ratio=max(0., 1-known), known_paid_cash_ratio=cash_paid,
            known_cash_value_ratio=cash_value, partial_return=partial if partial_ok else None)
    if partial_ok:
        # CRSP partial return already includes its priced distributions. Do not
        # add those cash/stock components a second time.
        return dict(kind='nonterminal_property_split', known_value_ratio=1+partial,
            residual_ratio=max(0., -partial), known_paid_cash_ratio=0.,
            known_cash_value_ratio=cash_value, partial_return=partial)
    if cash_value:
        raise ValueError('unresolved unit conversion with cash needs separate classification')
    return dict(kind='last_book_interval', known_value_ratio=1., residual_ratio=0.,
        known_paid_cash_ratio=0., known_cash_value_ratio=0., partial_return=None)


def bind_events(items, evidence):
    validate_terminal_path(items)
    by_date = {str(r['signal_date']):i for i,r in enumerate(items)}
    ids = items[0]['asset_ids']
    result, seen = [], set()
    for e in evidence:
        if e['permno'] not in ids:
            continue
        key = (e['signal_date'], int(e['permno']))
        if key in seen or key[0] not in by_date:
            raise ValueError('duplicate or undated assumption')
        seen.add(key)
        i = by_date[key[0]]
        j = int(np.searchsorted(ids, key[1]))
        row = items[i]
        if (not row['requires_zero_holding'][j] or row['observed_return'][j]
            or row['stale_mark'][j] or row['receivable_start'][j]
            or bool(row['terminal_end'][j]) != e['terminal_end']
            or str(row['execution_date']) != e['execution_date']
            or str(row['outcome_date']) != e['outcome_date']):
            raise ValueError('assumption does not bind to an original unresolved slot')
        result.append(dict(e, row=i, column=j, valuation=policy_for_event(e)))
    return result
