"""PhishShift CLI: classify one message, abstain when unsure, explain risk cues safely.

Never opens, fetches or executes anything in the message (NFR-01). Anything shown back is
defanged and has e-mail addresses / long numbers masked (NFR-02).

Examples:
    python scan.py --subject "Mailbox full" --text "Sign in at hxxps://secure-x1[.]xyz/a to keep access today"
    python scan.py --file message.txt
    python scan.py --interactive
"""
import argparse
import pathlib
import sys

import joblib
import pandas as pd

import phishshift as ps

MODEL_PATH = pathlib.Path(__file__).parent / "models" / "phishshift_v1.joblib"
MAX_CHARS = 5000


def load_bundle(path=MODEL_PATH):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        sys.exit(f"Model not found at {path}. Run `python evaluate.py` first.")


def scan(text: str, subject: str = "", channel: str = "email", bundle=None) -> dict:
    text = (text or "").strip()
    if not text:
        raise ValueError("Message text is empty.")
    if channel not in ("email", "sms"):
        raise ValueError("channel must be 'email' or 'sms'.")
    text = text[:MAX_CHARS]
    bundle = bundle or load_bundle()
    safe_text = ps.defang_for_display(ps.mask_personal_info(text))
    safe_subj = ps.defang_for_display(ps.mask_personal_info(subject or ""))
    row = pd.DataFrame([dict(message_id="scan", channel=channel, subject_or_title=safe_subj, body_text=safe_text,
                             sanitized_url_features=ps.url_features_from_text(safe_text),
                             sender_domain_type="unknown", label=0, campaign_group="scan",
                             collected_date="")])
    model = bundle["model"]
    p = float(model.proba(row)[0])
    decision = ps.decide(p, bundle["tau"])
    # explanation uses the linear model when available, else cues only
    lr = None
    if model.name == "tfidf_lr":
        lr = model
    expl = ps.explain(row.iloc[0], lr)
    return dict(decision=decision, p_phish=round(p, 3), cues=expl["cues"], top_terms=expl["top_terms"],
                model_version=bundle["version"], safe_text=safe_text)


def render(res: dict) -> str:
    head = {"LIKELY_PHISHING": "HIGH RISK - likely phishing",
            "LIKELY_LEGITIMATE": "LOW RISK - likely legitimate",
            "ABSTAIN": "UNSURE - send to a human analyst"}[res["decision"]]
    lines = [f"{head}  (P(phishing)={res['p_phish']}, model={res['model_version']})"]
    if res["cues"]:
        lines.append("Risk cues found:")
        lines += [f"  - {k.replace('_', ' ')}: {why}" for k, why in res["cues"]]
    else:
        lines.append("No rule-based risk cues found.")
    if res["top_terms"]:
        lines.append("Words/phrases that pushed the score up: " + ", ".join(res["top_terms"]))
    lines.append("Note: " + ps.DISCLAIMER)
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text")
    ap.add_argument("--file")
    ap.add_argument("--subject", default="")
    ap.add_argument("--channel", default="email", choices=["email", "sms"])
    ap.add_argument("--interactive", action="store_true")
    a = ap.parse_args(argv)
    bundle = load_bundle()
    if a.interactive:
        print("Paste a message body, then press Enter. Empty line quits.")
        while True:
            t = input("> ").strip()
            if not t:
                break
            try:
                print(render(scan(t, a.subject, a.channel, bundle)), "\n")
            except ValueError as e:
                print("Error:", e)
        return 0
    text = a.text
    if a.file:
        try:
            text = pathlib.Path(a.file).read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"Error: cannot read file ({e.strerror})", file=sys.stderr)
            return 2
    try:
        print(render(scan(text, a.subject, a.channel, bundle)))
    except ValueError as e:
        print("Error:", e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
