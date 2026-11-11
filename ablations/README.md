# Full US ablations

Use the same variant name for training and replay. All variants use the Full US execution and valuation rules. Each is trained on 2016–2017 and selected by the highest net Sharpe over 2018–2020.

| Paper label | `--variant` | Change from RECAST |
| --- | --- | --- |
| Candidate only | `candidate_only` | Execute the initial allocation directly. |
| w/o Alloc-Diag. | `no_explicit_diagnostic_context` | Remove explicit allocation context from the diagnostic query while retaining source-capital constraints. |
| w/o Solver-Feedback | `no_solver_feedback_search` | Withhold the provisional joint solution from subsequent search within a round. |
| w/o Rel-Writeback | `no_relation_writeback` | Update the allocation while keeping stock relational states fixed across rounds. |
| w/o Monthly | `no_monthly` | Use the daily representation alone and retain the full refinement loop. |
| w/o Summaries | `no_shared_summary` | Bypass shared-summary enrichment and use a common index root with non-learned recursive splits. |

```sh
python training/train.py train --variant no_solver_feedback_search --seed 7 --device cuda --output runs/solver_feedback
python evaluation/replay_full_us.py --variant no_solver_feedback_search --run-dir runs/solver_feedback --device cuda --output runs/solver_feedback_test
```
