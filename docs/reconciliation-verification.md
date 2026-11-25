# Reconciliation verification

The original development branch is preserved at `b6bd57f8ba9daa67ff7c5b7c78bd1e2bbf171636`. Local and GitHub `main` were both at `6803576aa68dd1a3f8b2bee85a8223b6468a86b3` at the start. The feature branch was 18 commits ahead, and `main` had no independent commits. A timestamped local copy matched all 73,763 file hashes and 11,329 directories. The all-refs Git bundle was verified with a fresh mirror restore and `git fsck --full`. The private copy contains the original ZIP, hash manifests, the full comparison and validation logs.

## Changes and preserved evidence

- Reconciled 30 source paths against the ZIP. The only executable AST difference from the original development branch is the generated four-strategy Markdown heading. Source edits also retain explicit read-only and simulation disclosures in the broker modules.
- Preserved the archived experiment reports, figures, tracked inputs and models. Added only 18 PDDL fixtures and two edge-case summaries from the submission. All 20 copies matched the ZIP byte for byte before Git's checkout line-ending conversion.
- Replaced the proposal-era README and documented the environment, methodology, evidence access, experiment order, benchmark timing and limitations. Repaired the mixed encoding in `.gitignore`, covered the original local audit files and retained all original dependency versions. The Windows-only dependency now has a platform marker.
- Kept raw and processed data, additional prediction and decision logs, local models, validator executables, private audits and credentials outside the changes. Archived evidence that was already tracked remains accessible in the repository. A post-test hash check matched all 3,692 original research, tool and environment files inspected.

## Verification results

| Area | Result |
| --- | --- |
| Python and environment | Compiled 46 tracked Python files, imported 15 core third-party modules and checked all 112 dependency pins and `pip check` on Windows with Python 3.12.10. |
| Planning and simulated execution | Rule-guided, autonomous symbolic, numerical, futures sizing and integrated planning demos completed. Contract notional, position changes and benchmark hedge arithmetic were verified. |
| Common scenarios and PDDL | Unified Planning parsed and validated all 36 saved plans across the three variants. The archived tables report all 36 cases as valid with VAL. Windows Application Control blocked a fresh run of the local VAL executable. |
| Dashboard | Streamlit reported healthy on loopback. The missing-credential path, mocked authentication, historical tabs and date changes passed. A mock broker snapshot produced an executable simulated recommendation. The local GJR-GARCH signal was checked. No orders, capital, real credentials or live broker session were used. |
| Final benchmark evidence | The selection and confirmatory tables matched the combined results. All eight summary rows were recalculated from 6,208 archived daily rows. The 1,552 planner records matched the daily numerical hedge arithmetic. |
| Benchmark regeneration | The selection rerun reproduced its daily and summary CSVs byte for byte. Applying the dissertation cutoff to the submitted spot and futures inputs reproduced both original upstream CSVs and the confirmatory daily and summary CSVs byte for byte. Across both reruns only elapsed planning time differed in the planner logs. |
| Classification | Inference with the archived locked LSTM and its saved test payload reproduced every submitted test metric. Random Forest and XGBoost completed, but neither the submitted feature split nor fresh five-day features from either market-data snapshot reproduced the earlier classical results. |
| Volatility regression and extended hedges | The two archived regression runs had byte-identical fold, summary, confirmatory and selection files. The DCC, smoothing and extended comparison reports were reviewed separately from the four-strategy benchmark. The complete regression grid, DCC suite and LSTM training grid were not rerun. |

Running the financial benchmark directly with the later spot and futures rows in the ZIP changed the 30 March 2026 rule-based hedge and the numerical target ratio. In that rerun the rule-based confirmatory return was −7.25% versus the archived −7.87%. The numerical strategy's return and summary metrics were unchanged. The March cutoff restored the archived values. The code, datasets and original results are preserved.

The exploratory classical reports have a different validation class distribution from the submitted processed split and report 306 test rows instead of 548. The recorded horizon in the submitted split metadata is 10 days, while a note in that file says five days. The exact inputs that produced the older classical reports were not established. Their archived results are preserved.

The secret scan looked for common token and private-key formats and credential assignments in reachable historical text blobs. It raised no matches. Binary artifacts were excluded from the historical text scan. The submitted README's student identifier was excluded from the rewritten README. The protected backup keeps original credentials and local data on this machine.

The reconciliation branch is the review candidate. Promotion to `main` and release tagging are held pending a supported VAL rerun and resolution of the earlier classical experiment's input provenance.
