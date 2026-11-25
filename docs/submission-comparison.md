# Dissertation submission comparison

The baseline is development commit `b6bd57f8ba9daa67ff7c5b7c78bd1e2bbf171636`. The submitted ZIP has SHA-256 `b0dcf6b49f7ca58efdb74688fdc75075734ebc62d7e1fab2f4f87f7e8c424e8f`. The initial comparison was completed before editing the reconciliation branch. Its complete per-file inventory is stored locally with the verified backup because it includes private filenames and hashes.

| Classification | All original and submission files | Originally tracked files |
| --- | ---: | ---: |
| Byte-identical | 430 | 41 |
| Comment-only | 1 | 1 |
| Docstring differences, potentially with comments or formatting | 26 | 26 |
| Formatting or encoding only | 2 | 2 |
| Executable text | 1 | 1 |
| Documentation | 1 | 1 |
| Dataset | 2 | 0 |
| Generated planning evidence | 1 | 0 |
| Original only | 73,299 | 110 |
| Submission only | 0 | 0 |
| Line endings only | 0 | 0 |

The 464 submitted files all have a path counterpart in the original checkout. The 73,299 original-only files include Git storage, virtual environments, caches, archives, local datasets, trained models and additional experiment results. The comparison used SHA-256 first, LF-normalised text comparisons for line endings, tokenised Python comments, and Python ASTs including type comments. Docstrings were compared separately because edits to them change runtime introspection. Source diffs and offline validation were used to assess behaviour.

- The only executable-text change is in `src/evaluation/run_proposal_four_strategy_backtest.py`. The submitted version shortens the heading emitted in the all-periods Markdown report. The formulas, parameters, strategy implementations and stored reports are preserved.
- `data/raw/NIFTY50_spot.csv` and `data/raw/NIFTY_futures_continuous.csv` each contain 3,585 submitted rows through 19 June 2026. The development files have the same overlapping rows plus 30 rows through 3 August 2026. The four-strategy evaluation ends in March 2026.
- The generated live dashboard PDDL problem differs in two numeric initial-state values. This is a different captured state. Each snapshot is preserved privately.
- All other shared non-source research evidence, including the tracked four-strategy outputs and the common-scenario VAL reports, is byte-identical. The original-only tracked archives supply earlier forecasting, hedging and planning evidence omitted from the ZIP.
- The submission README describes the evidence package and contains a student identifier. The repository README is written for collaborators and carries forward the submission's research-use and tool-availability disclosures.
