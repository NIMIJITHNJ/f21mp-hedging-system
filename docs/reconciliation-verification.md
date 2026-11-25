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
| Common scenarios and PDDL | Native Linux VAL revalidated all 36 saved plans on Ubuntu 24.04 under WSL 2. All results matched the archived validity, return code and final objective value. Unified Planning independently validated the same 36 plans. |
| Dashboard | Streamlit reported healthy on loopback. The missing-credential path, mocked authentication, historical tabs and date changes passed. A mock broker snapshot produced an executable simulated recommendation. The local GJR-GARCH signal was checked. No orders, capital, real credentials or live broker session were used. |
| Final benchmark evidence | The selection and confirmatory tables matched the combined results. All eight summary rows were recalculated from 6,208 archived daily rows. The 1,552 planner records matched the daily numerical hedge arithmetic. |
| Benchmark regeneration | The selection rerun reproduced its daily and summary CSVs byte for byte. Applying the dissertation cutoff to the submitted spot and futures inputs reproduced both original upstream CSVs and the confirmatory daily and summary CSVs byte for byte. Across both reruns only elapsed planning time differed in the planner logs. |
| Classification | Inference with the archived locked LSTM and its saved test payload reproduced every submitted test metric. Retraining Random Forest and XGBoost from the reconstructed May 2026 inputs reproduced both archived model binaries byte for byte and matched all five classical reports after normalising line endings. |
| Volatility regression and extended hedges | The two archived regression runs had byte-identical fold, summary, confirmatory and selection files. The DCC, smoothing and extended comparison reports were reviewed separately from the four-strategy benchmark. The complete regression grid, DCC suite and LSTM training grid were not rerun. |

Running the financial benchmark directly with the later spot and futures rows in the ZIP changed the 30 March 2026 rule-based hedge and the numerical target ratio. In that rerun the rule-based confirmatory return was −7.25% versus the archived −7.87%. The numerical strategy's return and summary metrics were unchanged. The March cutoff restored the archived values. The code, datasets and original results are preserved.

## Follow-up verification on 25 November 2026

**VAL.** Windows Application Control blocked the old Windows executable. The follow-up built the upstream Linux `Validate` target at revision `3c7a1f330bdab0ba28a4762bb45c3f06c27fb6d4` in the existing Ubuntu 24.04 WSL 2 distribution using CMake 3.28.3 and GCC 13.3.0. The build ran as the normal Ubuntu user and required no changes to Windows security policy. The Linux binary had SHA-256 `5cc5c5c3aeac6d22bbdba9a8653cd94cf55d9b7b66d51c2f193d155d367dff13`. The archived comparison CSV and the 36 sets of domain, problem, saved plan and VAL report matched byte for byte between the submission ZIP and the development backup. Each saved plan also agreed with its recorded action sequence.

| Planner variant | Saved plans | Valid in archived VAL | Valid in WSL VAL | Matching final values and return codes |
| --- | ---: | ---: | ---: | ---: |
| Rule-guided Unified Planning | 12 | 12 | 12 | 12 |
| Autonomous symbolic | 12 | 12 | 12 | 12 |
| Numerical | 12 | 12 | 12 | 12 |

The deliberate invalid-plan control was rejected with exit code 1. The archived Windows binary's exact upstream revision was unidentified. The comparison covers validity, return code and final objective value rather than machine-dependent output paths. [VAL upstream source revision](https://github.com/KCL-Planning/VAL/tree/3c7a1f330bdab0ba28a4762bb45c3f06c27fb6d4)

**Classical classifiers.** Git commit `438c5f9` introduced the original reports with the May 2026 feature generator, 25-feature configuration and historical macro snapshot. The archived Random Forest has 976 fitted rows. The historical five-day label generator, historical macro snapshot and clipping the preserved raw market rows to 1 January 2018 through 28 March 2025 recreated the reported split and class counts. The trainer entered Git later in `7e24923`. Its settings match both archived models.

| Split | Reconstructed dates | Rows | Low / Medium / High |
| --- | --- | ---: | --- |
| Training | 23 January 2019–30 December 2022 | 976 | 326 / 325 / 325 |
| Validation | 2 January–29 December 2023 | 246 | 169 / 74 / 3 |
| Test | 1 January 2024–21 March 2025 | 306 | 132 / 135 / 39 |

| Model | Validation accuracy | Test accuracy |
| --- | ---: | ---: |
| Random Forest | 30.8943% | 36.9281% |
| XGBoost | 27.6423% | 33.0065% |

Inference from the archived model files reproduced the two classification reports. Both retrained binaries matched the archives byte for byte. The comparison CSV, two classification reports and two feature-importance CSVs matched after LF normalisation. The archived and retrained models agreed on every prediction across all three reconstructed splits. The original six processed CSVs and an exact May dependency lock were not recovered. The submission's later processed split has 548 test rows and different class counts and belongs to a different experiment. See [Reproduction and evidence](reproduction.md) for the verified historical recipe.

The secret scan looked for common token and private-key formats and credential assignments in reachable historical text blobs. It raised no matches. Binary artifacts were excluded from the historical text scan. The submitted README's student identifier was excluded from the rewritten README. The protected backup keeps original credentials and local data on this machine.

The two follow-up checks support merging the reconciliation branch and creating an annotated dissertation release tag after review. The release notes should state that the classical artifacts were reproduced from reconstructed inputs, since the original processed CSVs were unavailable.
