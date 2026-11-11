"""Validation-based checkpoint selection and strict Full-US replay loading."""
import hashlib
import json
import math

from data_tools.verify_full_us_data import digest, read_json


def update_best(run, previous, epoch, checkpoint, checksum, sharpe):
    score = float(sharpe)
    if not math.isfinite(score):
        raise ValueError(f"Non-finite validation Sharpe at update {epoch}")
    if previous is not None and score <= previous["validation_net_sharpe"]:
        return previous
    best = dict(epoch=epoch, validation_net_sharpe=score, checkpoint=checkpoint.name, sha256=checksum)
    temporary = run / ".best.json.tmp"
    temporary.write_text(json.dumps(best, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(run / "best.json")
    return best


def selected_checkpoint(run):
    """Also read earlier training runs that saved per-update validation files only."""
    scores = []
    for file in sorted(run.glob("validation_[0-9][0-9][0-9].json")):
        epoch = int(file.stem.rsplit("_", 1)[1])
        score = float(read_json(file)["net_sharpe"])
        if not math.isfinite(score):
            raise ValueError(f"Non-finite validation score in {file.name}")
        scores.append((score, epoch))
    if not scores:
        raise ValueError(f"No validation metrics found in {run}")
    score, epoch = max(scores, key=lambda item: (item[0], -item[1]))
    checkpoint = run / f"update_{epoch:03d}.pt"
    selection = dict(epoch=epoch, validation_net_sharpe=score, checkpoint=checkpoint.name, sha256=digest(checkpoint))
    if (run / "best.json").is_file() and read_json(run / "best.json") != selection:
        raise ValueError("best.json disagrees with the validation scores or checkpoint bytes.")
    return checkpoint, selection


def load_for_replay(file, expected_variant=None):
    import torch
    from alphaportfolio_pcrr.dq_v1.config import DQConfig
    from alphaportfolio_pcrr.dq_v1.daily import DQDailyPortfolio
    from alphaportfolio_pcrr.dq_v1.feature_contract import feature71_contract
    from alphaportfolio_pcrr.dq_v1.training import fingerprint
    from training.train import ARMS, configure

    with file.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=True)
    binding = {}
    if payload.get("schema") == "dq.training_checkpoint.v2":
        seal = payload.pop("content_sha256", None)
        if not seal or fingerprint(payload) != seal:
            raise ValueError("Training checkpoint integrity check failed.")
        binding = payload["binding"]
        payload = payload["model"]
    model = DQDailyPortfolio.from_checkpoint_payload(payload)
    init = model.initialization_binding
    plan = binding.get("plan", {})
    dataset_tags = [s for s in (plan.get("dataset"), binding.get("dataset"),
                                init.get("source", {}).get("dataset")) if s]
    if not dataset_tags or any(s != "US_FIXED" for s in dataset_tags):
        raise ValueError("Expected a checkpoint trained on the Full-US universe.")
    variant = model.ablation or ("candidate_only" if model.decision_mode == "candidate_only" else "recast")
    if variant not in ARMS or (expected_variant is not None and expected_variant != variant):
        raise ValueError(f"Checkpoint variant {variant!r} differs from the requested model.")
    cfg = configure(variant)
    if (model.feature_contract != feature71_contract() or model.decision_mode != cfg["decision_mode"]
            or model.dq_config != DQConfig.from_dict(cfg["dq_config"])):
        raise ValueError("Checkpoint architecture or portfolio policy differs from the Full US configuration.")
    if (plan.get("decision_mode", model.decision_mode) != model.decision_mode
            or plan.get("ablation", model.ablation) != model.ablation
            or ("dq_config" in plan and DQConfig.from_dict(plan["dq_config"]) != model.dq_config)):
        raise ValueError("Training configuration and checkpoint portfolio policy disagree.")
    recorded_variant = plan.get("variant", variant)
    if recorded_variant != variant and (recorded_variant in ARMS or binding.get("recipe") == "recast_full_us"
            or not {"decision_mode", "ablation", "dq_config"}.issubset(plan)):
        raise ValueError("Training configuration and checkpoint architecture disagree.")
    info = dict(variant=variant, completed_updates=model.completed_updates,
                training_seed=plan.get("seed", init.get("seed")), checkpoint_sha256=checksum)
    return model.requires_grad_(False).eval(), info
