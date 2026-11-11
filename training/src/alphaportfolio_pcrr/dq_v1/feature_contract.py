"""Named, versioned 71D feature semantics, bound to new checkpoints only.

Width alone is not a schema: swapping two columns must fail rather than silently
alter the strategy. Metadata is an assertion by a data builder, not proof of its
source vintage, raw-cap provenance or execution correctness.
"""
import copy
import hashlib
import json

from alphaportfolio_original.data.schema import FEATURE_NAMES as MONTHLY_NAMES
from .feature_expansion import FEATURE_NAMES


def feature71_contract():
    return dict(schema='dq.features.daily71_monthly51.v1',
        daily_names=list(FEATURE_NAMES), monthly_names=list(MONTHLY_NAMES),
        statistic_names=list(FEATURE_NAMES), daily_lookback=252, monthly_lookback=24,
        daily_builder='dq.feature_expansion.71.v1',
        daily_normalization='dated_full_US_common_stock_winsor_01_99_zscore_ddof0',
        monthly_normalization='existing_monthly51_source_no_second_normalization',
        missing='finite_zero_placeholder_with_explicit_observed_mask_no_backfill',
        industry='dated_recorded_SIC_division_lag_cap_weighted_min5',
        ohlc='traded_prices_only_no_bid_ask_substitution',
        execution_inputs='separate_raw_price_volume_spread_cap_ledger_not_normalized_features',
        vintage='current_vintage_research_not_certified_first_publication')


def contract_sha256(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def validate_feature_contract(contract, *, daily_config, monthly_config, statistic_dim):
    if contract != feature71_contract():
        raise ValueError('unsupported or altered 71D feature contract')
    if (daily_config.input_dim != 71 or daily_config.lookback_months != 252
            or monthly_config.input_dim != 51 or monthly_config.lookback_months != 12
            or statistic_dim != 71):
        raise ValueError('71D contract requires daily 252x71, monthly 24x51 and 71 grounding inputs')
    return copy.deepcopy(contract)


def resolve_statistics(model, requested=None):
    from .inputs import DEFAULT_STATISTICS
    contract = model.feature_contract
    expected = tuple(contract['statistic_names']) if contract else DEFAULT_STATISTICS
    selected = expected if requested is None else tuple(requested)
    if contract and selected != expected:
        raise ValueError('diagnostic statistic names/order differ from the checkpoint contract')
    if len(selected) != model.dq_config.statistic_dim:
        raise ValueError('diagnostic statistic schema differs from the model')
    return selected


def validate_sample_contract(model, sample):
    contract = model.feature_contract
    if contract is None:
        return
    if (tuple(sample.get('daily_feature_names', ())) != tuple(contract['daily_names'])
            or tuple(sample.get('monthly_feature_names', ())) != tuple(contract['monthly_names'])
            or sample.get('feature_contract_sha256') != contract_sha256(contract)):
        raise ValueError('sample feature names/order/contract do not match the 71D model')


def estimate_feature71_bytes(*, sessions, reference_assets, output_sessions, target_assets,
                             source_rows_bytes=0):
    """Conservative working-set estimate, not a measured device/RAM guarantee.

Raw arrays and rolling moments use the reference universe even for a small
target pool. A loader should shard by time (with warmup) rather than materialize
overlapping 252-day samples for all decision days at once.
"""
    counts = (sessions, reference_assets, output_sessions, target_assets)
    if any(type(n) is not int or n < 1 for n in counts) or output_sessions > sessions:
        raise ValueError('positive coherent feature dimensions required')
    if target_assets > reference_assets or source_rows_bytes < 0:
        raise ValueError('target/reference dimensions or source size invalid')
    return int(256*sessions*reference_assets + 4*29*output_sessions*reference_assets
               + 5*71*output_sessions*target_assets + source_rows_bytes + 256*2**20)
