"""Causal nested 29/49/71 daily inputs; no targets, learned scaler or remote IO.

Industry means use each day's recorded SIC division and lagged cap weights.
This is a current-vintage research construction, not certification of the
database's historical publication vintage. OHLC requires true traded prices;
bid/ask are NEVER substituted for high/low. Invalid new observations remain NaN.
"""
import numpy as np
import pandas as pd

from ..daily_multiscale_data import DAILY_FEATURE_NAMES, _normalize_cross_section
from ..daily_feature_expansion_v11 import STOCK_NAMES, rolling


EXTRA20 = (*STOCK_NAMES,
    'trend_efficiency_21', 'trend_efficiency_63', 'up_day_fraction_21', 'up_day_fraction_63',
    'trend_r_squared_63', 'trend_r_squared_126', 'dollar_volume_cv_21', 'turnover_cv_21',
    'spread_std_21', 'return_dollar_volume_correlation_21', 'return_skewness_63', 'worst_return_21')
EXTRA22 = ('industry_relative_return_21', 'industry_relative_return_63',
    'industry_relative_volatility_21', 'industry_relative_volatility_63',
    'industry_beta_63', 'industry_correlation_63', 'industry_size_rank', 'industry_turnover_rank',
    'up_market_beta_126', 'down_market_beta_126', 'up_market_mean_return_126',
    'down_market_mean_return_126', 'down_market_correlation_126', 'market_coskewness_126',
    'historical_tail_loss_63', 'abnormal_move_fraction_63',
    'overnight_gap', 'open_to_close_return', 'intraday_range', 'close_location',
    'upper_shadow', 'lower_shadow')
FEATURE_NAMES = (*DAILY_FEATURE_NAMES, *EXTRA20, *EXTRA22)
assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 71


def divide(a, b, floor=1e-12):
    a, b = np.broadcast_arrays(a, b)
    out = np.full(a.shape, np.nan, np.float64)
    np.divide(a, b, out=out, where=np.isfinite(a) & np.isfinite(b) & (np.abs(b) > floor))
    return out


def compound(r, n):
    known = np.isfinite(r) & (r >= -1)
    logs = np.log1p(np.where(known & (r > -1), r, 0.))
    out = np.expm1(rolling(np.where(known, logs, np.nan), n, 'sum'))
    zeros = rolling(np.where(known, r == -1, np.nan), n, 'sum')
    out[zeros > 0] = -1.
    return out


def regression(y, x, n, mask=None):
    """Pairwise complete, past-window OLS moments. Conditional OLS needs 20 rows."""
    x = np.broadcast_to(x, y.shape)
    pair = np.isfinite(x) & np.isfinite(y)
    if mask is not None:
        pair &= mask
    count = rolling(pair.astype(float), n, 'sum')
    moments = []
    for a in (x, y, x*x, y*y, x*y):
        moments.append(divide(rolling(np.where(pair, a, 0.), n, 'sum'), count))
    ex, ey, ex2, ey2, exy = moments
    vx, vy = np.maximum(ex2-ex*ex, 0.), np.maximum(ey2-ey*ey, 0.)
    cov = exy-ex*ey
    beta = divide(cov, vx)
    corr = np.clip(divide(cov, np.sqrt(vx*vy), 1e-20), -1., 1.)
    idio = np.sqrt(np.maximum(vy-beta*cov, 0.)*252)
    enough = count >= (n if mask is None else 20)
    return tuple(np.where(enough, a, np.nan) for a in (beta, corr, idio, ey))


def market_proxy(r, caps, eligible, minimum=20):
    lag = np.zeros_like(caps, dtype=np.float64)
    lag[1:] = np.where(eligible[:-1] & np.isfinite(caps[:-1]) & (caps[:-1] > 0), caps[:-1], 0.)
    valid = np.isfinite(r) & (lag > 0)
    denominator = np.where(valid, lag, 0.).sum(1)
    market = divide((np.where(valid, r, 0.)*lag).sum(1), denominator)
    market[valid.sum(1) < minimum] = np.nan
    return market, lag


def sic_division(sic):
    """Broad, predeclared SIC divisions (not ex-post return clusters)."""
    groups = np.full(sic.shape, -1, np.int16)
    for i, (a, b) in enumerate(((100, 999),(1000,1499),(1500,1799),(2000,3999),
                               (4000,4999),(5000,5199),(5200,5999),(6000,6799),
                               (7000,8999),(9100,9729))):
        groups[(sic >= a) & (sic <= b)] = i
    return groups


def industry_context(r, cap, eligible, sic, turnover):
    groups = sic_division(sic)
    _, lag = market_proxy(r, cap, eligible)
    benchmark, size_rank, turn_rank = (np.full(r.shape, np.nan) for _ in range(3))
    for g in range(10):
        member = (groups == g) & eligible
        valid = member & np.isfinite(r) & (lag > 0)
        weights = np.where(valid, lag, 0.)
        ret = divide((weights*np.where(valid, r, 0.)).sum(1), weights.sum(1))
        ret[valid.sum(1) < 5] = np.nan
        benchmark = np.where(groups == g, ret[:, None], benchmark)
        for source, target in ((cap, size_rank), (turnover, turn_rank)):
            ranked = pd.DataFrame(np.where(member & np.isfinite(source), source, np.nan)).rank(axis=1, pct=True).to_numpy()
            target[member] = ranked[member]
    return benchmark, size_rank, turn_rank


def trend_fit(log_returns, n):
    # Shifts before the window add a constant and cannot alter an OLS R^2.
    y = np.nancumsum(log_returns, axis=0, dtype=np.float64)
    index = np.arange(len(y), dtype=float)[:, None]
    _, corr, _, _ = regression(y, index, n)
    complete = np.isfinite(rolling(log_returns, n, 'sum'))
    return np.where(complete, corr*corr, np.nan)


def expanded_features(raw, eligible, *, selected_columns=None, output_start=0, progress=None):
    """Produce the extra 42 columns, normalizing on the full dated reference.

    Only selected output columns are retained, so a local small-cohort check
    need not allocate a full-market T*N*71 cube. All raw arrays share T,N.
    """
    required = ('dlyret','dlycap','dlyvol','dlyprc','shrout','dlybid','dlyask','siccd',
                'dlyopen','dlyclose','dlyhigh','dlylow','dlyprevprc','dlyfacprc')
    if any(k not in raw for k in required):
        raise ValueError('71D requires real historical SIC and OHLC; no placeholder replacement')
    shape = np.asarray(raw['dlyret']).shape
    if any(np.asarray(raw[k]).shape != shape for k in required) or eligible.shape != shape:
        raise ValueError('feature axes differ')
    cols = np.arange(shape[1]) if selected_columns is None else np.asarray(selected_columns)
    result = np.full((shape[0]-output_start, len(cols), 42), np.nan, np.float32)
    def put(name, value):
        normalized = _normalize_cross_section(value[output_start:], eligible[output_start:])
        result[:, :, (*EXTRA20, *EXTRA22).index(name)] = normalized[:, cols]
        if progress:
            progress(name)
    r = np.asarray(raw['dlyret'], dtype=np.float64).copy()
    r[~np.isfinite(r) | (r < -1)] = np.nan
    caps = np.asarray(raw['dlycap'],dtype=np.float64)
    market, _ = market_proxy(r, caps, eligible)
    x = market[:,None]
    betas = {}
    for n in (63,126):
        beta, rho, idio, _ = regression(r, x, n)
        betas[n] = beta
        put(f'market_beta_{n}', beta)
        put(f'market_correlation_{n}', rho)
        put(f'idiosyncratic_volatility_{n}', idio)
    lag_beta = np.full_like(r,np.nan); lag_beta[1:] = betas[126][:-1]
    residual = r-lag_beta*x
    for n in (21,63):
        put(f'residual_momentum_{n}', rolling(residual,n,'sum'))
    del betas, lag_beta, residual
    logs = np.log1p(np.where(r > -1, r, np.nan))
    for n in (21,63):
        put(f'trend_efficiency_{n}', divide(np.abs(rolling(logs,n,'sum')),rolling(np.abs(logs),n,'sum')))
        put(f'up_day_fraction_{n}', rolling(np.where(np.isfinite(r),r>0,np.nan),n,'mean'))
    for n in (63,126):
        put(f'trend_r_squared_{n}',trend_fit(logs,n))
    volume, price, shares = (np.asarray(raw[k],dtype=np.float64) for k in ('dlyvol','dlyprc','shrout'))
    dv = np.where((price>0)&(volume>=0),price*volume,np.nan)
    turnover = np.where(volume>=0,divide(volume,shares*1000),np.nan)
    bid, ask = raw['dlybid'], raw['dlyask']
    spread = np.where((bid>0)&(ask>=bid),divide(ask-bid,.5*(ask+bid)),np.nan)
    for name, value in (('dollar_volume_cv_21',dv),('turnover_cv_21',turnover)):
        put(name,divide(rolling(value,21,'std'),rolling(value,21,'mean')))
    put('spread_std_21',rolling(spread,21,'std'))
    put('return_dollar_volume_correlation_21',regression(r,dv,21)[1])
    put('return_skewness_63',rolling(r,63,'skew'))
    put('worst_return_21',rolling(r,21,'min'))
    industry, irank, trank = industry_context(r,caps,eligible,raw['siccd'],turnover)
    for n in (21,63):
        put(f'industry_relative_return_{n}',compound(r,n)-compound(industry,n))
        put(f'industry_relative_volatility_{n}',rolling(r,n,'std')-rolling(industry,n,'std'))
    beta, corr, _, _ = regression(r,industry,63)
    put('industry_beta_63',beta); put('industry_correlation_63',corr)
    put('industry_size_rank',irank); put('industry_turnover_rank',trank)
    del industry, irank, trank, beta, corr
    for name, condition in (('up',x>0),('down',x<0)):
        beta, corr, _, mean = regression(r,x,126,mask=condition)
        put(f'{name}_market_beta_126',beta); put(f'{name}_market_mean_return_126',mean)
        if name == 'down':
            put('down_market_correlation_126',corr)
    # Standardized central E[(r-Er)*(m-Em)^2], using complete historical windows.
    pair = np.isfinite(r)&np.isfinite(x)
    y, mx = np.where(pair,r,np.nan),np.where(pair,x,np.nan)
    ey, em = rolling(y,126,'mean'),rolling(mx,126,'mean')
    em2, eym = rolling(mx*mx,126,'mean'),rolling(y*mx,126,'mean')
    numerator = rolling(y*mx*mx,126,'mean')-2*em*eym-ey*em2+2*ey*em*em
    put('market_coskewness_126',divide(numerator,rolling(y,126,'std')*rolling(mx,126,'var')))
    # Mean loss of the worst ceil(.1*63)=7 realized days, not a forecast of CVaR.
    tail = np.full_like(r,np.nan)
    for j in range(r.shape[1]):
        windows = np.lib.stride_tricks.sliding_window_view(r[:,j],63)
        known = np.isfinite(windows).all(1)
        value = -np.partition(windows,6,axis=1)[:,:7].mean(1)
        tail[62:,j] = np.where(known,value,np.nan)
    put('historical_tail_loss_63',tail)
    mean, vol = rolling(r,63,'mean'),rolling(r,63,'std')
    previous_mean, previous_vol = np.full_like(r,np.nan),np.full_like(r,np.nan)
    previous_mean[1:],previous_vol[1:] = mean[:-1],vol[:-1]
    surprise = np.where(np.isfinite(r)&np.isfinite(previous_mean)&(previous_vol>1e-12),
                        np.abs(r-previous_mean)>3*previous_vol,np.nan)
    put('abnormal_move_fraction_63',rolling(surprise,63,'mean'))
    op, close, high, low, prior, factor = (np.asarray(raw[k],dtype=np.float64) for k in
        ('dlyopen','dlyclose','dlyhigh','dlylow','dlyprevprc','dlyfacprc'))
    traded = np.asarray(raw.get('traded', np.zeros(shape,dtype=bool)),dtype=bool)
    good = traded&(volume>0)&(op>0)&(close>0)&(low>0)&(high>=np.maximum(op,close))&(low<=np.minimum(op,close))&(high>=low)
    # Exclude any distribution/adjustment-factor day from overnight gap. No
    # assumption about factor orientation, and no unadjusted split-gap feature.
    ordinary = np.asarray(raw.get('dlyorddivamt',np.full(shape,np.nan)))
    special = np.asarray(raw.get('dlynonorddivamt',np.full(shape,np.nan)))
    gap_good = good&(prior>0)&(factor==1)&(ordinary==0)&(special==0)
    values = (divide(op,prior)-1,divide(close,op)-1,divide(high-low,op),
              divide(close-low,high-low),divide(high-np.maximum(op,close),op),
              divide(np.minimum(op,close)-low,op))
    for name,value in zip(EXTRA22[-6:],values,strict=True):
        put(name,np.where(gap_good if name=='overnight_gap' else good,value,np.nan))
    if np.isinf(result).any():
        raise FloatingPointError('nonfinite daily expansion')
    return result


def build_daily71(rows, sessions, *, output_start, target_ids=None, progress=None,
                  enforce_memory=True):
    """Build named daily71 on FULL-US quality/clock-checked raw rows.

    Input is the upstream dated reference, NOT a target-pool-filtered frame.
    Source receipt verification, return-duration checks and a genuine exchange
    calendar remain the raw loader's responsibility. This function does not
    certify metadata, delete stocks, substitute prices or read future targets.
    Target projection occurs only AFTER all reference normalizations.
    """
    from ..daily_multiscale_data import NUMERIC_COLUMNS, compute_daily_features, require_memory
    from .feature_contract import feature71_contract, contract_sha256, estimate_feature71_bytes
    extra = ('siccd', 'dlyopen', 'dlyclose', 'dlyhigh', 'dlylow', 'dlyprevprc',
             'dlyfacprc', 'dlyorddivamt', 'dlynonorddivamt')
    required = ('permno', 'dlycaldt', 'entry_eligible', 'traded', *NUMERIC_COLUMNS, *extra)
    if any(name not in rows for name in required):
        raise ValueError('daily71 requires full raw SIC/OHLC/action fields and dated eligibility/traded masks')
    if rows.empty or rows.duplicated(['permno', 'dlycaldt']).any():
        raise ValueError('nonempty unique source rows required')
    if (not pd.api.types.is_integer_dtype(rows.permno.dtype)
            or rows.permno.isna().any()
            or any(rows[k].dtype != np.bool_ for k in ('entry_eligible', 'traded'))):
        raise ValueError('exact permanent IDs and explicit boolean eligibility/traded masks required')
    dates = pd.DatetimeIndex(sessions)
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError('unique ordered exchange sessions required')
    start = int(dates.get_indexer([pd.Timestamp(output_start)])[0])
    if start < 252:
        raise ValueError('daily71 output needs at least 252 earlier raw sessions for rolling warmup')
    ids = np.sort(rows.permno.unique()).astype(np.int64)
    targets = ids if target_ids is None else np.asarray(target_ids)
    if (targets.ndim != 1 or not np.issubdtype(targets.dtype, np.integer)
            or len(targets) < 1 or not np.array_equal(targets, np.unique(targets))
            or not np.isin(targets, ids).all()):
        raise ValueError('target IDs must be sorted unique members of the full raw reference')
    cols = np.searchsorted(ids, targets)
    ti = dates.get_indexer(pd.to_datetime(rows.dlycaldt))
    if (ti < 0).any():
        raise ValueError('raw rows outside the supplied exchange calendar')
    ai = np.searchsorted(ids, rows.permno.to_numpy())
    estimate = estimate_feature71_bytes(sessions=len(dates), reference_assets=len(ids),
        output_sessions=len(dates)-start, target_assets=len(targets),
        source_rows_bytes=int(rows.memory_usage(deep=True).sum()))
    memory = require_memory(estimate) if enforce_memory else dict(estimated_peak_bytes=estimate)
    base = compute_daily_features(rows, sessions, output_start=output_start,
                                  progress=progress, enforce_memory=False)
    values = np.full((len(dates)-start, len(targets), 71), np.nan, np.float32)
    values[:, :, :29] = base['values'][:, cols]
    eligible_output = base['entry_eligible'][:, cols].copy()
    del base
    shape = (len(dates), len(ids))
    raw = {}
    for name in (*NUMERIC_COLUMNS, *extra):
        raw[name] = np.full(shape, np.nan, np.float32)
        raw[name][ti, ai] = rows[name].to_numpy(dtype=np.float32)
    eligible = np.zeros(shape, bool)
    eligible[ti, ai] = rows.entry_eligible
    raw['traded'] = np.zeros(shape, bool)
    raw['traded'][ti, ai] = rows.traded
    values[:, :, 29:] = expanded_features(raw, eligible, selected_columns=cols,
        output_start=start, progress=(None if progress is None else
            lambda name: progress(dict(stage='daily_feature', feature=name, shape=list(values.shape)))))
    if np.isinf(values).any():
        raise FloatingPointError('infinite daily71 feature')
    contract = feature71_contract()
    return dict(schema='dq.daily71.cache.v1', dates=tuple(d.date() for d in dates[start:]),
        asset_ids=targets.copy(), values=values, observed=np.isfinite(values),
        entry_eligible=eligible_output, daily_feature_names=FEATURE_NAMES,
        raw_siccd=raw['siccd'][start:, cols].copy(),
        sic_observed=np.isfinite(raw['siccd'][start:, cols]),
        feature_contract=contract, feature_contract_sha256=contract_sha256(contract), memory=memory,
        reference_assets=len(ids), normalization_before_target_projection=True,
        source_provenance_certified=False)
