"""Run the full PhishShift evaluation and write evidence into results/ and figures/.

Order matters (FR-01): validate -> sanitisation check -> dedup -> split -> train -> evaluate.

Usage:
    python generate_data.py
    python evaluate.py
    python evaluate.py --holdout PH_VENDOR,PH_SURVEY,LG_VENDOR,LG_COURIER --tau 0.7
"""
import argparse
import datetime
import json
import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import phishshift as ps

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "results"
FIG = ROOT / "figures"
LOGLINES = []


def log(msg=""):
    print(msg)
    LOGLINES.append(str(msg))


def fit_predict(name, train, test):
    m = ps.MODEL_FACTORIES[name]().fit(train)
    return m, m.proba(test)


# CP-5: 8 new harmless simulated messages (hand-written, defanged, no real data)
CP5 = [
    ("email", "Parking permit renewal", "Hi Meera, your parking permit renewal is due by Friday. Renew at hxxps://portal[.]examplecorp[.]test/parking/ab12.", 0),
    ("email", "Urgent: printer maintenance", "Urgent notice: the third floor printer will be offline immediately after lunch. Please use the second floor printer.", 0),
    ("email", "Password policy reminder", "Reminder: choose a new password every 90 days. Instructions at hxxps://it[.]examplecorp[.]test/policy/q9. IT will never ask for it.", 0),
    ("sms", "", "Your meeting room booking for Thursday is confirmed. Reply CANCEL to release it.", 0),
    ("email", "Action needed: account locked", "Dear user, your account is locked. Verify your login at hxxps://secure-staff12[.]xyz/k3m within 6 hours or it will be closed.", 1),
    ("sms", "", "Parcel PKG-44120 held. Pay the Rs 99 fee now: hxxps://bit[.]ly/x7Yq2a", 1),
    ("email", "Quick favour", "Are you free? I need gift cards for a client today. Keep this between us and send the codes when done.", 1),
    ("email", "Staff appreciation", "Thanks for all your work! Claim your Rs 250 thank-you voucher here: hxxps://tinyurl[.]com/staff9z", 1),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/raw_messages.csv"))
    ap.add_argument("--holdout", default="PH_VENDOR,PH_SURVEY,LG_VENDOR,LG_COURIER",
                    help="campaign groups excluded from training (mentor can change this)")
    ap.add_argument("--dedup-threshold", type=float, default=0.85)
    ap.add_argument("--tau", type=float, default=0.6, help="abstain below this confidence")
    ap.add_argument("--seed", type=int, default=ps.SEED)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)
    hold = [h.strip() for h in args.holdout.split(",") if h.strip()]

    # ---- load + schema ----
    raw = pd.read_csv(args.data, keep_default_na=False)
    ps.validate_schema(raw)
    log(f"Loaded {len(raw)} raw messages, {raw.campaign_group.nunique()} campaigns")

    # ---- CP-1 sanitisation log ----
    live = raw[~raw.body_text.map(ps.is_inert)]
    defanged_n = int(raw.body_text.str.contains("hxxp", case=False).sum())
    san = [f"rows={len(raw)}", f"rows_with_defanged_urls={defanged_n}", f"rows_with_live_urls={len(live)}",
           f"rows_with_missing_subject={(raw.subject_or_title == '').sum()} (all SMS: {(raw[raw.subject_or_title == ''].channel == 'sms').all()})",
           "attachments=excluded (no attachment fields in schema)"]
    (OUT / "sanitization_log.txt").write_text("\n".join(san))
    log("Sanitisation: " + "; ".join(san[:3]))
    assert len(live) == 0, "live URL found in corpus - fix before continuing"

    # ---- FR-01 dedup (before any split) ----
    clean, dlog = ps.dedup_near_duplicates(raw, args.dedup_threshold)
    dlog.to_csv(OUT / "dedup_log.csv", index=False)
    clean.to_csv(ROOT / "data/clean_messages.csv", index=False)
    log(f"Dedup (cos>={args.dedup_threshold}): {len(raw)} -> {len(clean)} "
        f"({len(dlog)} removed; {int((~dlog.same_label).sum()) if len(dlog) else 0} label conflicts)")

    # ---- CP-2 audit: leakage across splits before vs after dedup ----
    thr = args.dedup_threshold
    audit = []
    for tag, d in [("raw", raw), ("deduplicated", clean)]:
        tr, te = train_test_split(d, test_size=0.3, stratify=d.label, random_state=args.seed)
        sim = ps.cross_split_similarity(tr, te)
        audit.append(f"{tag} random split: {100 * (sim >= thr).mean():.1f}% of test msgs have a train twin (cos>={thr}); max={sim.max():.3f}")
    train = clean[~clean.campaign_group.isin(hold)]
    test = clean[clean.campaign_group.isin(hold)]
    sim = ps.cross_split_similarity(train, test)
    audit.append(f"deduplicated campaign split: {100 * (sim >= thr).mean():.1f}% twins; max={sim.max():.3f}")
    (OUT / "dedup_audit.txt").write_text("\n".join(audit))
    log("\n".join(audit))
    assert set(test.label) == {0, 1}, "holdout needs both classes"
    log(f"Train={len(train)}  Holdout={len(test)}  holdout campaigns={hold}")

    # ---- T3 EDA ----
    eda = {"class_counts": clean.label.value_counts().to_dict(),
           "campaign_counts": clean.campaign_group.value_counts().to_dict(),
           "missing_values": {c: int((clean[c].astype(str) == "").sum()) for c in clean.columns},
           "body_len": clean.body_text.str.len().describe().round(1).to_dict(),
           "date_range": [clean.collected_date.min(), clean.collected_date.max()],
           "leakage_notes": ["campaign_group + collected_date are split keys, never model inputs",
                             "sender_domain_type excluded from features (synthetic shortcut risk)",
                             "random splits leak templates; campaign holdout is the headline metric"]}
    (OUT / "eda_summary.json").write_text(json.dumps(eda, indent=2, default=str))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ct = clean.groupby(["campaign_group", "label"]).size().unstack(fill_value=0)
        ax = ct.plot(kind="bar", stacked=True, figsize=(9, 4), color=["#2a9d8f", "#e76f51"])
        ax.set_title("Messages per campaign (0 = legitimate, 1 = phishing)")
        ax.set_ylabel("messages (after dedup)")
        plt.tight_layout(); plt.savefig(FIG / "fig1_campaign_counts.png", dpi=130); plt.close()
        fig, ax = plt.subplots(figsize=(7, 4))
        for lab, c in [(0, "#2a9d8f"), (1, "#e76f51")]:
            ax.hist(clean[clean.label == lab].body_text.str.len(), bins=25, alpha=.6, color=c,
                    label=["legitimate", "phishing"][lab])
        ax.set_xlabel("body length (characters)"); ax.set_ylabel("messages"); ax.legend()
        ax.set_title("Body length by class")
        plt.tight_layout(); plt.savefig(FIG / "fig2_length_hist.png", dpi=130); plt.close()
        prev = pd.DataFrame({lab: pd.Series([k for _, r in g.iterrows() for k in ps.triggered_cues(r)]).value_counts() / len(g)
                             for lab, g in clean.groupby("label")}).fillna(0)
        prev.columns = ["legitimate", "phishing"]
        ax = prev.plot(kind="barh", figsize=(8, 4.5), color=["#2a9d8f", "#e76f51"])
        ax.set_title("Share of messages triggering each cue"); ax.set_xlabel("share of class")
        plt.tight_layout(); plt.savefig(FIG / "fig3_cue_prevalence.png", dpi=130); plt.close()
    except Exception as e:  # plotting must never block the evidence run
        log(f"(plots skipped: {e})")

    # ---- FR-02/03 main comparison: random split (leaky) vs campaign holdout ----
    rows, holdout_proba = [], {}
    rtr, rte = train_test_split(clean, test_size=0.3, stratify=clean.label, random_state=args.seed)
    for name in ps.MODEL_FACTORIES:
        _, p = fit_predict(name, rtr, rte)
        rows.append({"split": "random_70_30", "model": name, **ps.binary_metrics(rte.label, p)})
        _, p = fit_predict(name, train, test)
        holdout_proba[name] = p
        rows.append({"split": "campaign_holdout", "model": name, **ps.binary_metrics(test.label, p)})
    res = pd.DataFrame(rows).round(3)
    res.to_csv(OUT / "results_table.csv", index=False)
    log("\n" + res[["split", "model", "accuracy", "precision", "recall", "f1", "fpr", "roc_auc"]].to_string(index=False))

    # ---- CP-3 confusion matrices on campaign holdout ----
    cm = []
    for name, p in holdout_proba.items():
        t = pd.crosstab(pd.Series(test.label.values, name="actual"), pd.Series((p >= .5).astype(int), name="predicted"))
        cm.append(f"== {name} (campaign holdout) ==\n{t}\n")
    (OUT / "confusion_matrices.txt").write_text("\n".join(cm))

    # ---- AT-02 / FR-03: leave-one-campaign-out ----
    loco = []
    for camp in clean.campaign_group.unique():
        tr, te = clean[clean.campaign_group != camp], clean[clean.campaign_group == camp]
        for name in ps.MODEL_FACTORIES:
            _, p = fit_predict(name, tr, te)
            for mid, lab, urg, pp in zip(te.message_id, te.label, ps.is_legit_urgent(te), p):
                loco.append(dict(campaign=camp, model=name, message_id=mid, label=lab, legit_urgent=urg, p=pp))
    loco = pd.DataFrame(loco)
    summ = []
    for (camp, name), g in loco.groupby(["campaign", "model"]):
        lab = int(g.label.iloc[0])
        summ.append(dict(campaign=camp, model=name, true_class="phishing" if lab else "legitimate", n=len(g),
                         metric="recall" if lab else "false_positive_rate",
                         value=round(float((g.p >= .5).mean()), 3)))
    pd.DataFrame(summ).to_csv(OUT / "loco_by_campaign.csv", index=False)
    log("\nLeave-one-campaign-out (recall for phishing campaigns, FPR for legitimate):")
    log(pd.DataFrame(summ).pivot(index=["campaign", "metric"], columns="model", values="value").to_string())

    # ---- FR-05 / AT-01: false positives on legitimate urgent messages ----
    fpr_rows = []
    lu = loco[loco.legit_urgent]
    for name, g in lu.groupby("model"):
        fpr_rows.append(dict(model=name, scope="LOCO all legit-urgent", n=len(g), false_positive_rate=round(float((g.p >= .5).mean()), 3)))
    tu = test[ps.is_legit_urgent(test)]
    for name in ps.MODEL_FACTORIES:
        sel = ps.is_legit_urgent(test).values
        fpr_rows.append(dict(model=name, scope="holdout legit-urgent", n=int(sel.sum()),
                             false_positive_rate=round(float((holdout_proba[name][sel] >= .5).mean()), 3)))
    pd.DataFrame(fpr_rows).to_csv(OUT / "legit_urgent_fpr.csv", index=False)
    log("\nFalse-positive rate on legitimate urgent messages (AT-01):\n" + pd.DataFrame(fpr_rows).to_string(index=False))

    # ---- choose best ML model on campaign-holdout F1 ----
    ml = res[(res.split == "campaign_holdout") & (res.model != "rules")]
    best = ml.sort_values("f1", ascending=False).iloc[0].model
    log(f"\nBest ML model by campaign-holdout F1: {best}")

    # ---- FR-04 abstain sweep ----
    ab = pd.DataFrame([ps.abstain_metrics(test.label, holdout_proba[best], t) for t in [.5, .6, .7, .8, .9]]).round(3)
    ab.to_csv(OUT / "abstain_sweep.csv", index=False)
    log("\nAbstain sweep (" + best + ", campaign holdout):\n" + ab.to_string(index=False))

    # ---- AT-03 / FR-05: robustness drift ----
    drift = []
    models = {n: ps.MODEL_FACTORIES[n]().fit(train) for n in ps.MODEL_FACTORIES}
    for name, m in models.items():
        base = m.proba(test)
        for pert in ps.PERTURBATIONS:
            q = m.proba(ps.perturb(test, pert, args.seed))
            ph = test.label.values == 1
            drift.append(dict(model=name, perturbation=pert,
                              mean_abs_shift=round(float(np.abs(q - base).mean()), 3),
                              phish_mean_shift=round(float((q - base)[ph].mean()), 3),
                              flip_rate=round(float(((q >= .5) != (base >= .5)).mean()), 3),
                              recall_after=round(float((q[ph] >= .5).mean()), 3),
                              fpr_after=round(float((q[~ph] >= .5).mean()), 3)))
    drift = pd.DataFrame(drift)
    drift.to_csv(OUT / "robustness_drift.csv", index=False)
    log("\nRobustness drift on campaign holdout:\n" + drift.to_string(index=False))

    # ---- CP-4: error examples (missed phish + flagged legit) ----
    p = holdout_proba[best]
    err = test.assign(p_phish=p.round(3), pred=(p >= .5).astype(int))
    err = err[err.pred != err.label]
    fn, fp = err[err.label == 1], err[err.label == 0]
    pick = pd.concat([fn.head(2), fp.head(2)]).head(3 if len(err) >= 3 else len(err))
    expl = []
    mdl = models["tfidf_lr"]
    for _, r in pick.iterrows():
        e = ps.explain(r, mdl)
        expl.append(dict(message_id=r.message_id, campaign=r.campaign_group, actual=r.label, p_phish=r.p_phish,
                         subject=r.subject_or_title, body=ps.mask_personal_info(r.body_text),
                         cues="; ".join(k for k, _ in e["cues"]) or "(none)", top_terms=", ".join(e["top_terms"])))
    pd.DataFrame(expl).to_csv(OUT / "error_examples.csv", index=False)
    log(f"\nErrors by best model on holdout: {len(fn)} missed phish, {len(fp)} flagged legit -> results/error_examples.csv")

    # ---- CP-5: 8 new harmless simulated messages with controlled perturbations ----
    cp = pd.DataFrame([dict(message_id=f"CP5-{i + 1}", channel=c, subject_or_title=s, body_text=b,
                            sanitized_url_features=ps.url_features_from_text(b), sender_domain_type="unknown",
                            label=l, campaign_group="CP5", collected_date=str(datetime.date.today()))
                       for i, (c, s, b, l) in enumerate(CP5)])
    cp_rows = []
    m = models[best]
    for variant in ["original"] + ["synonym_swap", "punctuation_noise", "undefang_url", "typos"]:
        d = cp if variant == "original" else ps.perturb(cp, variant, args.seed)
        if variant == "undefang_url":
            d["sanitized_url_features"] = [ps.url_features_from_text(b) for b in d.body_text]
        pr = m.proba(d)
        for mid, lab, x in zip(cp.message_id, cp.label, pr):
            cp_rows.append(dict(message_id=mid, true_label=lab, variant=variant, p_phish=round(float(x), 3),
                                decision=ps.decide(float(x), args.tau)))
    cp_df = pd.DataFrame(cp_rows)
    cp_df.to_csv(OUT / "cp5_perturbation_demo.csv", index=False)
    log("\nCP-5 (original vs perturbed, " + best + "):\n" + cp_df.pivot(index="message_id", columns="variant", values="p_phish").to_string())

    # ---- save final model (trained on ALL deduplicated data) + experiment journal ----
    final = ps.MODEL_FACTORIES[best]().fit(clean)
    version = f"phishshift-v1-{best}-{datetime.date.today():%Y%m%d}"
    joblib.dump({"model": final, "model_name": best, "version": version, "tau": args.tau,
                 "trained_rows": len(clean), "holdout_used_for_eval": hold}, ROOT / "models/phishshift_v1.joblib")
    entry = dict(time=datetime.datetime.now().isoformat(timespec="seconds"), version=version, seed=args.seed,
                 holdout=hold, dedup_threshold=args.dedup_threshold, tau=args.tau, n_raw=len(raw), n_clean=len(clean),
                 holdout_f1={r.model: r.f1 for r in res[res.split == "campaign_holdout"].itertuples()})
    with (OUT / "experiment_log.jsonl").open("a") as f:
        f.write(json.dumps(entry) + "\n")
    log(f"\nSaved model {version}")
    (OUT / "run_log.txt").write_text("\n".join(LOGLINES))


if __name__ == "__main__":
    main()
