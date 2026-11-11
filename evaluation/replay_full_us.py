"""Replay a frozen Full US RECAST model or ablation in the common execution ledger."""
import argparse
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / "training/src")]
from data_tools.verify_full_us_data import digest, verify
from evaluation import ledger
from training.checkpoints import load_for_replay, selected_checkpoint
from training.train import ARMS
def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))

def save(p, value):
    with p.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)

def model_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def prepare(root):
    import numpy as np
    import pandas as pd
    from alphaportfolio_pcrr.dq_v1.ablation_campaign import FastSharedSource
    from alphaportfolio_pcrr.dq_v1.formal71 import prepare_execution71
    from alphaportfolio_pcrr.dq_v1.training import PathWindow
    from alphaportfolio_pcrr.dq_v1.valuation import policy_spec
    from alphaportfolio_pcrr.daily_terminal_liquidation_v17 import load_terminal_execution
    if root.name != "test":
        raise ValueError("--test must point to the actual test directory, not its parent")
    verification = verify(root.parent, "test")
    data = root/"data"
    source = FastSharedSource(data/"dq-test", "US_FIXED", verify_files=False)
    rows = load_terminal_execution(data/"execution/US_FIXED")
    by_date = {str(r["signal_date"]): r for r in rows}
    bridge = data/"common-carry/COMMON_CARRY_BRIDGE.json"
    for proof in read(bridge)["records"]:
        r = by_date[proof["signal"]]
        j = int(np.searchsorted(r["asset_ids"], proof["permno"]))
        if j == len(r["asset_ids"]) or r["asset_ids"][j] != proof["permno"] or not r["requires_zero_holding"][j]:
            continue
        if r["terminal_end"][j] or r["receivable_start"][j] or r["stale_mark"][j]:
            raise ValueError("Conflicting carry evidence")
        r["requires_zero_holding"][j] = False
        r["observed_return"][j] = True
        r["mark_returns"][j] = proof["return_value"]
        r["tradable_fill"][j] = proof["fill_tradable"]
    for r in rows:
        r["target_asset_ids"] = np.intersect1d(r["target_asset_ids"], source.roster)
        r["target_indices"] = np.searchsorted(r["asset_ids"], r["target_asset_ids"])
    dates = [str(r["signal_date"]) for r in rows]
    if len(dates) != 1253 or dates != sorted(set(dates)) or len(source.roster) != 3454:
        raise ValueError("Unexpected Full-US calendar or roster")
    cal = source.calendar
    def after(date):
        return (pd.Timestamp(cal[str(date)]) + timedelta(microseconds=1)).isoformat()
    clocks = {str(r["signal_date"]): dict(execution_at=cal[str(r["execution_date"])],
              fill_available_at=after(r["execution_date"]), outcome_available_at=after(r["outcome_date"])) for r in rows}
    events = read(data/"EVIDENCE.json")["events"]
    for e in events:
        e["available_at"] = after(e["outcome_date"])
    window = PathWindow("test", dates[0], dates[-1], clocks[dates[-1]]["outcome_available_at"])
    plan = {"test": ["2021-01-01","2025-12-31"], "execution_valuation_policy": policy_spec()}
    rows, _, receipt = prepare_execution71(rows, window=window, plan=plan, clocks=clocks, evidence=events,
        source_binding={"feature_manifest": verification["test_features"]["manifest_sha256"],
                        "carry_bridge_sha256": digest(bridge)})
    verification["prepared_path_sha256"] = receipt["prepared_path_sha256"]
    path = root / "market/market.npz"
    with np.load(path, allow_pickle=False) as archive:
        market = {k: archive[k] for k in archive.files}
    if (not np.array_equal(dates, market["signals"])
            or not np.array_equal([str(r["execution_date"]) for r in rows], market["execution_dates"])
            or any(not np.array_equal(r["asset_ids"], market["ids"]) for r in rows)):
        raise ValueError("Stock identifiers or market dates are misaligned")
    return source, rows, window, market, verification

def smoke():
    import numpy as np
    _, rates = ledger.tiers_and_rates([1e10,2e9,3e8,1,np.nan])
    np.testing.assert_array_equal(rates, [.01,.05,.1,.3,.3])
    fills = ledger.fill([.3,-.3,-.19,.19], [0.]*4, 1e6, [True]*4,
                   [.01,.0101,.001,np.nan], [1e6]*4, [2e6,1e6,5e5,1e6])
    np.testing.assert_allclose(fills["position"], [.1,0,-.05,0], rtol=0, atol=1e-12)
    np.testing.assert_allclose([fills["commission"],fills["spread_fee"]], [.000075,.000525])
    return {"status":"pass","uses_real_test_rows":False,"fill_cap_and_borrow_tiers_checked":True}

def metrics(r):
    import numpy as np
    r = np.asarray(r, dtype=float)
    if len(r)<2 or not np.isfinite(r).all() or np.any(r<=-1):
        raise ValueError("Invalid net return path")
    wealth=np.r_[1., np.exp(np.log1p(r).cumsum())]
    sd = float(r.std(ddof=1))
    return dict(sessions=len(r), net_cagr=float(np.expm1(252*np.log1p(r).mean())),
        net_sharpe=float(np.sqrt(252)*r.mean()/max(sd,1e-12)),
        annualized_volatility=float(np.sqrt(252)*sd),
        max_drawdown=float(np.min(wealth/np.maximum.accumulate(wealth)-1.)), cumulative_return=float(wealth[-1]-1.))

def replay(args):
    import numpy as np
    import torch
    from alphaportfolio_pcrr.dq_v1 import training
    file, selection = (selected_checkpoint(args.run_dir) if args.run_dir is not None
                       else (args.checkpoint, None))
    model, identity = load_for_replay(file, args.variant)
    if selection and (selection["epoch"] != identity["completed_updates"]
                      or selection["sha256"] != identity["checkpoint_sha256"]):
        raise ValueError("Validation selection and loaded checkpoint disagree.")
    before = model_digest(model)
    if args.load_check:
        return dict(**identity, strict_load="pass", frozen_tensor_digest=before, selection=selection)
    if args.output is None or args.output.exists():
        raise ValueError("Supply a new --output directory; existing results are never overwritten")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    model.to(args.device)
    source, rows, window, market, provenance = prepare(args.test.resolve())
    if args.probe_days:
        rows=rows[:args.probe_days]
        market={k:v if k=="ids" else v[:args.probe_days] for k,v in market.items()}
        window=training.PathWindow("test",str(rows[0]["signal_date"]),str(rows[-1]["signal_date"]),rows[-1]["outcome_available_at"])
    input_lock = dict(**identity, selection=selection, state_sha256=before, verified_inputs=provenance,
        path_sha256=provenance["prepared_path_sha256"], budget=ledger.SPEC, device=args.device, torch=torch.__version__,
        partial=bool(args.probe_days), target_sessions=len(rows))
    book=ledger.CausalBook(rows,market,borrow=True,constrained=True)
    with torch.no_grad():
        for i,r in enumerate(rows):
            day=str(r["signal_date"])
            sample=source.sample(day,r["target_asset_ids"],r["execution_date"])
            training._bind(sample,r,day,window)
            book.snapshot(sample)  # Advance fills and marks to the decision time.
            decision, _ = training.forward_sample(model, sample, return_aux=False)
            target=decision.normalized_weights[0].detach().cpu().numpy()
            if not np.isfinite(target).all() or np.abs(target).sum()>1.00002:
                raise ValueError("Invalid target")
            book.submit(sample,target)
            if (i+1)%25==0:
                print(f"{i+1}/{len(rows)} completed",flush=True)
        records=book.finish(window.outcome_cutoff)
    if len(records)!=len(rows) or model_digest(model)!=before:
        raise ValueError("Incomplete timeline or frozen parameters changed")
    daily=[]
    errors=[]
    for i,(r,t) in enumerate(zip(rows,records)):
        pos, order=t["position"],np.abs(t["order"])
        independent=(np.dot(pos,r["dq_valuation_return"]+r["dq_valuation_extra_cash_ratio"]),
                     .0005*sum(order),.5*np.dot(order,np.nan_to_num(market["spread"][i],nan=0)),
                     np.dot(np.maximum(-pos,0.),ledger.tiers_and_rates(market["cap_usd"][i])[1])*t["borrow_calendar_days"]/360.)
        delta=max(abs(x-t[k]) for x,k in zip(independent,("gross","commission","spread_fee","borrow_fee")))
        errors.append(max(delta,abs(t["gross"]-t["fees"]-t["net"])))
        if (errors[-1]>1e-10 or max(t["max_day_participation"],t["max_adv_participation"])>.1+1e-10
                or t["max_filled_spread"]>.01+1e-10):
            raise ValueError("Independent execution audit failed at "+t["signal"])
        daily.append({k:(v.item() if isinstance(v,np.generic) else v) for k,v in t.items() if np.isscalar(v)})
    report=dict(status="pass", **identity, selection=selection, partial=bool(args.probe_days), metrics=metrics([r["net"] for r in records]),
        independently_reconciled_max_error=max(errors), frozen_tensor_digest=before,
        yearly={str(y):metrics(a) for y in range(2021,2026)
            if len(a:=[r["net"] for r in records if r["outcome"].startswith(str(y))])>=2})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent, prefix=".replay-") as tmp:
        stage = Path(tmp) / "results"
        stage.mkdir()
        save(stage/"input_lock.json",input_lock)
        save(stage/"daily.json",daily)
        save(stage/"summary.json",report)
        if args.output.exists():
            raise FileExistsError(args.output)
        stage.rename(args.output)
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant",choices=ARMS,help="Optionally require this checkpoint architecture")
    p.add_argument("--test",type=Path,default=ROOT/"data/test")
    p.add_argument("--output",type=Path,help="new output directory (results are licensed research derivatives)")
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--probe-days",type=int,default=0,help="2..1252 runs a prefix for engineering checks, not a full test")
    group=p.add_mutually_exclusive_group()
    group.add_argument("--load-check",action="store_true",help="check frozen weight loading only; no panel required")
    group.add_argument("--policy-smoke",action="store_true",help="verify execution and borrow thresholds on synthetic orders only")
    weights = p.add_mutually_exclusive_group()
    weights.add_argument("--run-dir", type=Path, help="Select the highest validation net Sharpe in a training run")
    weights.add_argument("--checkpoint", type=Path, help="Explicit trained Full-US checkpoint")
    args=p.parse_args()
    if not args.policy_smoke and args.run_dir is None and args.checkpoint is None:
        p.error("Supply --run-dir or --checkpoint.")
    if args.threads<1 or (args.probe_days!=0 and not 2<=args.probe_days<1253):
        p.error("Expected positive threads and 0 or 2..1252 probe days.")
    print(json.dumps(smoke() if args.policy_smoke else replay(args),indent=2,allow_nan=False))
