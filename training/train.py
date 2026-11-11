"""Train the Full-US RECAST policy or an ablation using a full trading path per update."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE / "src")]
from data_tools.verify_full_us_data import digest, read_json, verify
from training.checkpoints import update_best

ARMS = ("recast", "candidate_only", "no_explicit_diagnostic_context",
        "no_solver_feedback_search", "no_relation_writeback", "no_monthly", "no_shared_summary")
MONTHLY = ROOT / "weights/portable/monthly_encoder.safetensors"


def configure(variant, seed=None):
    if variant not in ARMS:
        raise ValueError(f"Unknown Full-US variant: {variant}")
    plan = read_json(HERE / "recast.json")
    plan.update(variant=variant, decision_mode="candidate_only" if variant == "candidate_only" else "full",
                ablation=variant if variant in ARMS[2:] else None)
    if seed is not None:
        if not 0 <= seed < 2**63:
            raise ValueError("Seed must be an integer between 0 and 2^63 - 1.")
        plan["seed"] = seed
    cfg = plan["dq_config"]
    cfg["retrieval_depths"] = tuple(cfg["retrieval_depths"])
    return plan


def fresh(plan, monthly):
    from safetensors.torch import load_file
    from alphaportfolio_original.config import AlphaPortfolioConfig, ModelConfig
    from alphaportfolio_pcrr.dq_v1.config import DQConfig
    from alphaportfolio_pcrr.dq_v1.formal71 import make_model, seed_fresh_training71
    from alphaportfolio_pcrr.dq_v1.ablation_campaign import specialize_initial_model, apply_engineering

    seed_fresh_training71(plan)
    model = make_model(AlphaPortfolioConfig(model=ModelConfig(**plan["monthly_model"])),
                       load_file(str(monthly), device="cpu"), seed=plan["seed"],
                       source_binding={"dataset": "US_FIXED", "sha256": digest(monthly)},
                       dq_config=DQConfig.from_dict(plan["dq_config"]), decision_mode=plan["decision_mode"])
    if plan["ablation"]:
        model, _ = specialize_initial_model(model, plan["ablation"])
        apply_engineering(model)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check-data", "check-model", "train"))
    parser.add_argument("--variant", choices=ARMS, default=ARMS[0])
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--monthly", type=Path, default=MONTHLY)
    parser.add_argument("--output", type=Path, help="New run directory for checkpoints and validation metrics")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--seed", type=int, help="Daily model, dropout and training RNG seed")
    args = parser.parse_args()
    plan = configure(args.variant, args.seed)
    if args.command != "check-model":
        checks = verify(args.data.resolve(), "train_validation")
        if args.command == "check-data":
            print(json.dumps(checks, indent=2))
            return
        if not args.output or args.output.exists() or not 1 <= args.updates <= plan["full_path_updates"]:
            parser.error("Supply a new --output and 1 to 100 --updates.")
    import torch
    from alphaportfolio_pcrr.dq_v1.formal71 import optimizer_for71
    from alphaportfolio_pcrr.dq_v1.campaign_worker import run_train, run_eval, save_evaluation
    from alphaportfolio_pcrr.dq_v1.training import save_training_checkpoint
    from alphaportfolio_pcrr.dq_v1.ablation_campaign import FastSharedSource

    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    model = fresh(plan, args.monthly).to("cpu" if args.command == "check-model" else args.device)
    opt = optimizer_for71(model, plan)
    if args.command == "check-model":
        print(json.dumps(dict(arm=args.variant, mode=model.decision_mode, ablation=model.ablation,
                              seed=plan["seed"], params=sum(p.numel() for p in model.parameters()))))
        return
    shared = FastSharedSource(args.data / "train_validation/shared", "US_FIXED", verify_files=False)
    train, val = (shared.role(role, plan) for role in ("train", "validation"))
    binding = dict(plan=plan, train_input_digest=checks["train_validation"]["manifest_sha256"],
                   recipe="recast_full_us")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "run.json").write_text(json.dumps(binding, indent=2), encoding="utf-8")
    best = None
    while model.completed_updates < args.updates:
        result = run_train(model, opt, train, plan, None)
        step = model.completed_updates
        checkpoint = args.output / f"update_{step:03d}.pt"
        receipt = save_training_checkpoint(checkpoint, model, opt, binding=binding)
        info = {"update": step, "economic_loss": float(result["economic_loss"])}
        metrics = save_evaluation(args.output, step, run_eval(model, val, plan, None))
        best = update_best(args.output, best, step, checkpoint, receipt["sha256"], metrics["net_sharpe"])
        info["validation"] = metrics
        print(json.dumps(info, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
