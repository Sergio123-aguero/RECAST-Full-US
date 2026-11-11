# Frozen Full-US replay

To replay the update with the highest 2018–2020 validation net Sharpe, pass the training run directory. The executable reads the validation files and checks the selection against `best.json` when present. To evaluate a specific saved update, pass its path.

```sh
python evaluation/replay_full_us.py --run-dir runs/full_us --device cuda --output runs/test_selected
python evaluation/replay_full_us.py --checkpoint runs/full_us/update_001.pt --variant recast --device cuda --output runs/test_update_001
```

Use a new output directory for each run. The loader checks the model architecture, training universe and portfolio configuration. `--variant` additionally checks the requested experimental arm. The full model and all six ablations share the same execution and valuation code.

The ledger charges 5 basis points per traded side plus half the observed spread. It caps full spread at 1% and participation at 10% of the smaller of execution-day dollar volume and lagged 20-day average dollar volume. Short positions accrue annual borrowing rates of 1%, 5%, 10% or 30% by market-cap tier. The model ranks eligible stocks using decision-time features. The ledger carries filled positions forward and applies the appropriate borrowing fee to each held short.

## Checks

```sh
python evaluation/replay_full_us.py --policy-smoke
python evaluation/replay_full_us.py --run-dir runs/full_us --load-check
python evaluation/replay_full_us.py --run-dir runs/full_us --probe-days 2 --device cuda --output runs/test_first_two_days
```

The first command checks fill, fee and borrowing rules with synthetic orders. The second checks the selected checkpoint. The third replays two actual sessions from the beginning of the test period. An ordinary full run covers 1,253 sessions. The ledger is advanced at each decision, including when orders remain unfilled. Replay checks for gross target exposure above one, reconciles the trading and borrowing costs, and verifies the frozen model digest at completion.

## Outputs

- `input_lock.json` records the model variant, checkpoint digest, completed updates, seed, validation selection (when supplied), data seals and ledger settings.
- `daily.json` contains one row per outcome date with the realized P&L and fee components.
- `summary.json` reports net CAGR, net Sharpe, annualized volatility, signed maximum drawdown and cumulative return. Returns and volatility use decimal units. The annualization factor is 252 and daily volatility uses sample standard deviation. The `partial` field identifies prefix runs.
