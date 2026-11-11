"""Daily execution, valuation, and borrowing costs for the Full US portfolio."""
from datetime import datetime
import numpy as np

POLICY = "DQ71_V47A_P10_SPREAD1_BORROW_1_5_10_30_RESEARCH_V2"
SPEC = dict(
    initial_nav_usd=1_000_000.,
    participation=.10,
    full_spread_cap=.01,
    spread_definition="(closing_ask-closing_bid)/closing_midpoint",
    commission_per_absolute_order=.0005,
    spread_cost="half_full_spread_times_absolute_executed_order",
    volume_rule="10pct_of_min(execution_day_close_times_volume,prior_20_session_mean)",
    missing_or_invalid_quote_or_volume="no_market_trade; retain_existing_holding",
    opening_gross_budget=1.,
    unavoidable_carried_gross_above_one="reduce_as_constraints_allow; no_new_risk",
    borrow_rates_annual=dict(large=.01, mid=.05, small=.10, micro=.30),
    size_thresholds_usd=dict(large_at_least=10_000_000_000., mid_at_least=2_000_000_000.,
                             small_at_least=300_000_000.),
    size_asof="latest_observed_positive_cap_at_or_before_signal_date; never_backward_fill",
    missing_cap_rate="micro_cap_30pct",
    borrow_basis="actual_carried_short_market_notional_at_execution; ACT/360 calendar days",
    collateral_markup=1.,
    hypothetical_borrow_rates=True,
    real_borrow_availability_verified=False,
    recall_or_locate_constraints=False,
    collateral_rounding_or_margin_checks=False,
    cash_interest_or_short_proceeds_rebate=False,
    synthetic_settlement_exempt_from_market_trade_caps=True,
    intraday_fill_feasibility_certified=False,
)


def tiers_and_rates(cap_usd):
    cap = np.asarray(cap_usd, dtype=np.float64)
    tier = np.where(np.isfinite(cap) & (cap >= 1e10), 0,
                    np.where(np.isfinite(cap) & (cap >= 2e9), 1,
                             np.where(np.isfinite(cap) & (cap >= 3e8), 2, 3)))
    return tier, np.array([SPEC['borrow_rates_annual'][k] for k in ('large', 'mid', 'small', 'micro')])[tier]


def fill(target, prior, nav, tradable, spread, volume, adv20, *, constrained=True):
    """Cost-feasible reduce-then-increase orders; never waive blocked exits."""
    target, prior = np.asarray(target, float), np.asarray(prior, float)
    if (target.shape != prior.shape or target.ndim != 1 or not np.isfinite(target).all()
            or not np.isfinite(prior).all() or np.abs(target).sum() > 1.00002 or nav <= 0):
        raise ValueError("Invalid target/prior/NAV")
    spread, volume, adv20 = (np.asarray(a, float) for a in (spread, volume, adv20))
    allowed = np.asarray(tradable, bool).copy()
    valid = (np.isfinite(spread) & (spread >= 0) & (spread <= .01)
             & np.isfinite(volume) & (volume > 0) & np.isfinite(adv20) & (adv20 > 0))
    if constrained:
        allowed &= valid
        limit = np.where(allowed, .1 * np.minimum(volume, adv20) / nav, 0.)
        unit_cost = .0005 + np.where(allowed, spread * .5, 0.)
    else:
        limit = np.where(allowed, np.inf, 0.)
        unit_cost = np.full_like(prior, .0005)
    proposed = prior + np.clip(target - prior, -limit, limit)
    reduced = np.where(prior * proposed > 0,
                       np.sign(prior) * np.minimum(np.abs(prior), np.abs(proposed)), 0.)
    increase = proposed - reduced
    reduction_cost = float(np.dot(np.abs(reduced - prior), unit_cost))
    growth_cost = float(np.dot(np.abs(increase), unit_cost))
    available = max(0., 1. - np.abs(reduced).sum() - reduction_cost)
    scale = min(1., available / max(np.abs(increase).sum() + growth_cost, 1e-30))
    position = reduced + scale * increase
    order = position - prior
    fee = float(np.dot(np.abs(order), unit_cost))
    commission = .0005 * np.abs(order).sum()
    traded = np.abs(order) > 1e-14
    if np.any(traded & ~allowed) or np.any(np.abs(order) > limit + 1e-11):
        raise ValueError("Trade permission/cap violated")
    unavoidable = np.abs(reduced).sum() + reduction_cost > 1 + 1e-10
    if np.abs(position).sum() + fee > 1 + 1e-10 and not unavoidable:
        raise ValueError("New orders violated gross/cost budget")
    return dict(position=position, prior=prior.copy(), order=order,
                commission=float(commission), spread_fee=float(fee-commission),
                rebalance_fees=fee, pretrade_nav=float(nav),
                unavoidable_gross_excess=bool(unavoidable),
                rejected_weight=float(np.abs(target-position).sum()),
                postfee_gross=float(np.abs(position).sum() / (1-fee)),
                max_day_participation=(float(np.max(np.abs(order[traded])*nav/volume[traded]))
                                       if constrained and traded.any() else 0.),
                max_adv_participation=(float(np.max(np.abs(order[traded])*nav/adv20[traded]))
                                       if constrained and traded.any() else 0.),
                max_filled_spread=(float(spread[traded].max()) if constrained and traded.any() else 0.))


def mark(filled, row, cap_usd, *, borrow):
    p, nav = filled["position"], filled["pretrade_nav"]
    r, extra, exit_ratio = (np.asarray(row[k], float) for k in
        ("dq_valuation_return", "dq_valuation_extra_cash_ratio", "dq_valuation_exit_ratio"))
    close = np.asarray(row["dq_valuation_close"], bool)
    unresolved = np.asarray(row["dq_valuation_unresolved"], bool)
    if np.any((p != 0) & unresolved):
        raise ValueError("Held unresolved valuation needs explicit dated evidence")
    if np.any((p != 0) & row["terminal_end"] & ~close):
        raise ValueError("Terminal holding lacks bound close assumption")
    days = (datetime.fromisoformat(str(row["outcome_date"])) -
            datetime.fromisoformat(str(row["execution_date"]))).days
    if days < 1 or not np.isfinite(r).all() or np.any(r < -1):
        raise ValueError("Invalid accrual interval or mark")
    tier, rates = tiers_and_rates(cap_usd)
    short = np.maximum(-p, 0.)
    borrow_by_asset = short * rates * days / 360 if borrow else np.zeros_like(p)
    by_tier = np.array([borrow_by_asset[tier == k].sum() for k in range(4)])
    exit_fee = .0005 * np.abs(p * exit_ratio).sum()
    total_fees = filled["rebalance_fees"] + exit_fee + borrow_by_asset.sum()
    gross = float(np.dot(p, r+extra))
    net = gross-total_fees
    if not np.isfinite(net) or net <= -1:
        raise ValueError("Nonpositive NAV")
    value = p * (1+r)
    settled = np.where(close, value, 0.)
    ending = (value-settled)/(1+net)
    cash = (1-p.sum()-total_fees+settled.sum()+np.dot(p, extra))/(1+net)
    # Independent dollar cash/position reconstruction.
    prior_dollars = filled["prior"]*nav
    orders = filled["order"]*nav
    cash_dollars = (nav-prior_dollars.sum()-orders.sum()-total_fees*nav
                    + settled.sum()*nav+np.dot(p, extra)*nav)
    ending_nav = nav*(1+net)
    residual = abs(cash_dollars+(value-settled).sum()*nav-ending_nav)/nav
    if residual > 1e-10 or abs(ending.sum()+cash-1) > 1e-10:
        raise ValueError("Dollar cash/holdings reconciliation failed")
    return dict(**filled, net=float(net), gross=gross, fees=float(total_fees),
                exit_fee=float(exit_fee), borrow_fee=float(borrow_by_asset.sum()),
                borrow_by_tier=by_tier, borrow_dollars=float(borrow_by_asset.sum()*nav),
                short_exposure_by_tier=np.array([short[tier == k].sum() for k in range(4)]),
                borrow_calendar_days=days, end_position=ending, end_cash=float(cash),
                nav_after=float(ending_nav), reconciliation_residual=float(residual),
                unknown_cap_short_weight=float(short[~np.isfinite(cap_usd)].sum()),
                assumed_exposure=float(np.abs(p[np.asarray(row["dq_valuation_assumed"], bool)]).sum()),
                long_gross_profit=float(np.dot(np.maximum(p, 0), r+extra)),
                short_gross_profit=float(np.dot(np.minimum(p, 0), r+extra)))


class CausalBook:
    """Only publish execution/marks once their bound availability time arrives."""
    policy_id = POLICY

    def __init__(self, rows, market, *, borrow, constrained=True):
        self.rows = {str(r["signal_date"]): (i, r) for i, r in enumerate(rows)}
        self.ids = rows[0]["asset_ids"]
        self.market, self.borrow, self.constrained = market, borrow, constrained
        self.queue, self.next_fill, self.active = [], 0, None
        self.prior = np.zeros(len(self.ids), float)
        self.nav = SPEC["initial_nav_usd"]
        self.last_snapshot_at = self.last_event_at = None
        self.results, self.submitted = [], set()

    @staticmethod
    def at(s):
        t = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if t.tzinfo is None:
            raise ValueError("Explicit timezone required")
        return t

    def advance(self, at):
        while True:
            if self.active is not None:
                i, row, filled = self.active
                if self.at(row["outcome_available_at"]) <= at:
                    result = mark(filled, row, self.market["cap_usd"][i], borrow=self.borrow)
                    result.update(signal=str(row["signal_date"]), execution=str(row["execution_date"]),
                                  outcome=str(row["outcome_date"]))
                    self.results.append(result)
                    self.prior, self.nav = result["end_position"], result["nav_after"]
                    self.active = None
                    self.last_event_at = row["outcome_available_at"]
            if self.next_fill == len(self.queue):
                break
            i, row, target = self.queue[self.next_fill]
            if self.at(row["fill_available_at"]) > at:
                break
            if self.active is not None:
                raise ValueError("Prior outcome not published before next execution")
            filled = fill(target, self.prior, self.nav, row["tradable_fill"],
                          self.market["spread"][i], self.market["volume"][i],
                          self.market["adv20"][i], constrained=self.constrained)
            self.active = (i, row, filled)
            self.next_fill += 1
            self.last_event_at = row["fill_available_at"]

    def snapshot(self, sample):
        at = self.at(sample["decision_at"])
        if self.last_snapshot_at is not None and at < self.last_snapshot_at:
            raise ValueError("Decision clock moved backwards")
        self.last_snapshot_at = at
        self.advance(at)
        if self.active is None:
            p = self.prior
        else:
            filled = self.active[2]
            p = filled["position"]/(1-filled["rebalance_fees"])
        ids = np.asarray(sample["asset_ids"])
        col = np.searchsorted(self.ids, ids)
        if np.any(col >= len(self.ids)) or not np.array_equal(self.ids[col], ids):
            raise ValueError("Decision assets outside accounting union")
        return dict(schema="dq.executed_decision_book.v1", basis="executed_and_marked",
                    ledger_policy_id=POLICY, signal_date=sample["signal_date"],
                    asof=sample["decision_at"],
                    last_event_available_at=self.last_event_at or sample["decision_at"],
                    asset_ids=ids.copy(), positions=p[col].copy())

    def submit(self, sample, raw):
        i, row = self.rows[str(sample["signal_date"])]
        if (str(row["signal_date"]) in self.submitted
                or not np.array_equal(row["target_asset_ids"], sample["asset_ids"])
                or self.at(sample["execution_at"]) != self.at(row["execution_at"])
                or self.at(sample["decision_at"]) >= self.at(row["execution_at"])):
            raise ValueError("Duplicate/noncausal/unaligned submission")
        target = np.zeros(len(self.ids), float)
        target[row["target_indices"]] = np.asarray(raw, float)
        self.queue.append((i, row, target))
        self.submitted.add(str(row["signal_date"]))

    def finish(self, cutoff):
        self.advance(self.at(cutoff))
        if self.active is not None or len(self.results) != len(self.rows):
            raise ValueError("Incomplete causal replay")
        return self.results
