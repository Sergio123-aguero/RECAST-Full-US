"""Run one Full US portfolio decision on a synthetic cross-section."""
import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.train import ARMS, MONTHLY, configure, fresh


def synthetic_decision(variant="recast"):
    model = fresh(configure(variant, seed=7), MONTHLY).eval()
    rng = torch.Generator().manual_seed(7)
    count = 12
    ids = torch.arange(count)
    daily = torch.randn(1, count, 252, 71, generator=rng)
    monthly = torch.randn(1, count, 24, 51, generator=rng)
    with torch.no_grad():
        decision = model(
            daily, monthly, asset_ids=ids, statistics=daily[:, :, -1, :],
            decision_keys=[1], return_aux=False,
        )
    return torch.stack((decision.dq_result.initial_weights[0], decision.normalized_weights[0]), dim=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=ARMS, default="recast")
    args = parser.parse_args()
    torch.set_num_threads(2)
    print("Candidate weight, final target weight")
    print(synthetic_decision(args.variant))
