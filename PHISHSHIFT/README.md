# PhishShift Message Risk Scanner (ML-07)

Educational triage tool: classifies **synthetic, defanged** helpdesk emails/SMS as likely phishing or likely
legitimate, **abstains** when unsure, and explains the risk cues it found. It never opens links, fetches
content or collects real credentials. Output is decision support, not a guarantee of safety.

## Setup
```bash
pip install -r requirements.txt
python generate_data.py        # builds data/raw_messages.csv (1,275 synthetic msgs, 17 campaigns)
python evaluate.py             # dedup -> campaign holdout -> models -> evidence in results/ and figures/
python tests/test_phishshift.py   # or: pytest -q
python scan.py --subject "Mailbox full" --text "Dear user, sign in at hxxps://secure-x1[.]xyz/a today"
python scan.py --interactive
```
Mentor variation: `python evaluate.py --holdout PH_VENDOR,PH_SURVEY,LG_VENDOR,LG_COURIER --tau 0.6`
(swap in whichever campaigns the mentor assigns, or replace `data/raw_messages.csv` with the mentor's set;
it must keep the schema fields in `phishshift.REQUIRED_COLS`).

## Layout
| File | Purpose |
|---|---|
| `generate_data.py` | Locally authored synthetic corpus (benign + phishing simulations), all URLs defanged |
| `phishshift.py` | Features, rules baseline, TF-IDF LR, RF on cue features, dedup, metrics, abstain, explanations, perturbations |
| `evaluate.py` | Produces every piece of evidence (tables, logs, plots, saved model) |
| `scan.py` | CLI prototype (graceful errors, saved model version, masked/defanged output) |
| `tests/` | Unit tests for dedup, defanging, abstain, no-network-code check, CLI error handling |

## Requirement-to-test mapping
| Req | Where demonstrated |
|---|---|
| FR-01 dedup before split | `evaluate.py` dedup step; `results/dedup_log.csv`, `results/dedup_audit.txt`; `test_dedup_removes_near_duplicates_only` |
| FR-02 rules vs TF-IDF linear vs third model | `results/results_table.csv` (rules, `tfidf_lr`, `rf_cues`) |
| FR-03 campaign-separated holdout | `--holdout` campaigns; `results/results_table.csv` (random vs campaign split), `results/loco_by_campaign.csv` |
| FR-04 abstain + safe explanation | `results/abstain_sweep.csv`; `scan.py` output; `results/error_examples.csv` |
| FR-05 FPR on legit urgent + drift | `results/legit_urgent_fpr.csv`, `results/robustness_drift.csv` |
| NFR-01 never click/fetch/execute | no network/exec imports (`test_no_network_or_exec_imports_in_core_code`); `results/sanitization_log.txt` |
| NFR-02 no real credentials/PII | synthetic data only; `scan.py` masks emails and long numbers |
| NFR-03 decision-support notice | printed by `scan.py` on every result |
| AT-01 legit password-reset | `LG_PWRESET` row in `loco_by_campaign.csv`; `legit_urgent_fpr.csv` |
| AT-02 unfamiliar phrasing | `PH_VENDOR`, `PH_SURVEY` held out of training |
| AT-03 defang/punctuation changes | `results/robustness_drift.csv`, `results/cp5_perturbation_demo.csv` |
| CP-1..CP-5 | sanitization log, dedup audit, confusion matrices + LOCO, error examples, CP-5 demo CSV |

## What the results show (seed 42; re-run to refresh)
- Dedup removes about a third of the raw messages. Before dedup, ~44% of a random test split had a near-twin in
  training; after dedup and a campaign split, 0%.
- A random split makes TF-IDF LR look perfect (F1 1.00). On unseen campaigns it drops to about 0.88, and it
  misses many phishing messages whose wording avoids the training keywords (e.g. vendor bank-detail change).
- The Random Forest on cue features generalises worst (it over-flags legitimate urgent messages).
- Abstain trades coverage for accuracy: raising `--tau` removes errors but sends more messages to a human.

## Known limitations (put these in report section 7)
- Corpus is synthetic and template-composed, so absolute scores are optimistic; swap in a licensed public
  corpus (record its licence in the provenance sheet) for stronger claims.
- `sender_domain_type` is excluded from features to avoid a synthetic shortcut.
- Probabilities are not calibrated; add calibration if you want to claim meaningful confidence.
- Perturbations are text-only; URL features are not recomputed except where noted in `evaluate.py`.

## Still to do yourself
`SRS.pdf`, `final_report.pdf`, the 3-5 minute demo, your own experiment journal and git history,
and an **AI-usage disclosure**. You must be able to explain and re-create all of this code.
