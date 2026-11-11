import json

import numpy as np
import pytest
import torch

from data_tools.verify_full_us_data import digest
from evaluation.replay_full_us import metrics
from training.checkpoints import load_for_replay, selected_checkpoint, update_best
from training.train import ARMS, MONTHLY, configure, fresh
from alphaportfolio_pcrr.dq_v1.solver import solve_repairs
from alphaportfolio_pcrr.dq_v1.training import fingerprint


def test_selection_uses_validation_sharpe_and_detects_changed_checkpoint(tmp_path):
    previous = None
    for epoch, sharpe in enumerate([0.4, 0.7, 0.7], start=1):
        checkpoint = tmp_path / f"update_{epoch:03d}.pt"
        checkpoint.write_bytes(bytes([epoch]))
        (tmp_path / f"validation_{epoch:03d}.json").write_text(json.dumps({"net_sharpe": sharpe}))
        previous = update_best(tmp_path, previous, epoch, checkpoint, digest(checkpoint), sharpe)
    file, selection = selected_checkpoint(tmp_path)
    assert selection["epoch"] == 2 and file.name == "update_002.pt"
    file.write_bytes(b"changed")
    with pytest.raises(ValueError, match="best.json"):
        selected_checkpoint(tmp_path)


@pytest.mark.parametrize("variant", ARMS)
def test_recast_and_ablations_roundtrip(tmp_path, variant):
    torch.set_num_threads(2)
    plan = configure(variant, seed=81)
    assert plan["dq_config"]["long_policy"] == "off"
    assert plan["dq_config"]["long_technology_policy"] == "off"
    assert plan["dq_config"]["long_turnover_penalty"] == 0
    model = fresh(plan, MONTHLY)
    file = tmp_path / "model.pt"
    torch.save(model.checkpoint_payload(), file)
    reloaded, record = load_for_replay(file, variant)
    assert record["training_seed"] == 81
    assert record["variant"] == variant
    assert not reloaded.training and all(not p.requires_grad for p in reloaded.parameters())


@pytest.mark.parametrize("field,value", [
    ("long_policy", "different_policy"),
    ("long_technology_policy", "different_policy"),
    ("long_turnover_penalty", 0.5),
])
def test_replay_rejects_different_portfolio_policy(tmp_path, field, value):
    plan = configure("recast")
    payload = fresh(plan, MONTHLY).checkpoint_payload()
    payload["constructor"]["dq_config"][field] = value
    file = tmp_path / "different_policy.pt"
    torch.save(payload, file)
    with pytest.raises(ValueError, match="portfolio policy differs"):
        load_for_replay(file, "recast")


def test_joint_transfers_respect_shared_source_capital():
    weights = torch.tensor([0.10, 0., 0., -0.10])
    owners = torch.tensor([1., 1., 1., -1.])
    # Both proposals draw on the same long position.
    directions = torch.tensor([[-0.10, -0.10], [0.10, 0.], [0., 0.10], [0., 0.]])
    result = solve_repairs(weights, directions, owners, torch.tensor([0.10, 0.09]),
                           0.01 * torch.eye(2), iterations=96)
    transferred = result.weights[1:3].sum().item()
    assert 0 < transferred <= weights[0].item() + 1e-6
    assert result.weights[0].item() == pytest.approx(weights[0].item() - transferred, abs=1e-6)
    torch.testing.assert_close(result.weights.abs().sum(), weights.abs().sum())
    assert result.diagnostics["feasible"]
    assert result.diagnostics["objective_after"] < result.diagnostics["objective_before"]


def test_replay_checks_training_policy_even_if_variant_name_matches(tmp_path):
    plan = configure("recast")
    model_payload = fresh(plan, MONTHLY).checkpoint_payload()
    plan["dq_config"]["long_policy"] = "different_policy"
    payload = dict(schema="dq.training_checkpoint.v2", model=model_payload,
                   binding={"recipe": "recast_full_us", "plan": plan})
    payload["content_sha256"] = fingerprint(payload)
    file = tmp_path / "model.pt"
    torch.save(payload, file)
    with pytest.raises(ValueError, match="portfolio policy"):
        load_for_replay(file, "recast")


def test_daily_only_bypasses_monthly_information():
    model = fresh(configure("no_monthly", seed=7), MONTHLY).eval()
    generator = torch.Generator().manual_seed(7)
    daily = torch.randn(1, 3, 252, 71, generator=generator)
    monthly = torch.randn(1, 3, 24, 51, generator=generator)

    def unexpected_monthly_call(module, inputs):
        pytest.fail("The daily-only model called the monthly encoder.")

    with model.monthly_srem.register_forward_pre_hook(unexpected_monthly_call), torch.no_grad():
        first, _ = model.encode_features(daily, monthly)
        second, _ = model.encode_features(daily, -monthly, monthly_observed=torch.zeros_like(monthly, dtype=torch.bool))
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_summary_ablation_passes_stock_states_directly_to_the_index():
    model = fresh(configure("no_shared_summary", seed=7), MONTHLY)
    ids = torch.arange(5)[None]
    valid = ids != 3
    hidden = torch.randn(1, 5, model.core.dimension, generator=torch.Generator().manual_seed(7))
    store = model.core._store(hidden, valid, ids)
    torch.testing.assert_close(store["relation"], hidden * valid[..., None])
    assert torch.equal(store["groups"], torch.zeros_like(ids))
    assert all("summary_" not in name for name, _ in model.core.named_parameters())


def test_daily_metrics_match_paper_units():
    r = np.array([0.01, -0.02, 0.03])
    result = metrics(r)
    assert result["annualized_volatility"] == pytest.approx(r.std(ddof=1) * np.sqrt(252))
    assert result["net_sharpe"] == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
    assert result["max_drawdown"] == pytest.approx(-0.02)
    assert result["net_cagr"] == pytest.approx(np.prod(1 + r) ** (252 / len(r)) - 1)
