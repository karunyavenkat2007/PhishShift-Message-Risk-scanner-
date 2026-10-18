"""Unit tests. Run with `pytest -q` or, without pytest, `python tests/test_phishshift.py`."""
import pathlib
import re
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import phishshift as ps  # noqa: E402


def _row(subject, body, mid="x"):
    return dict(message_id=mid, channel="email", subject_or_title=subject, body_text=body,
                sanitized_url_features=ps.url_features_from_text(body), sender_domain_type="unknown",
                label=0, campaign_group="t", collected_date="2026-10-01")


def test_schema_validation_rejects_missing_column():
    df = pd.DataFrame([_row("a", "b")]).drop(columns=["body_text"])
    try:
        ps.validate_schema(df)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_dedup_removes_near_duplicates_only():
    a = _row("Mailbox full", "Dear user, your mailbox is 95% full. Sign in at hxxps://x[.]xyz/ab to restore access.", "1")
    b = _row("Mailbox full", "Dear user, your mailbox is 97% full. Sign in at hxxps://y[.]xyz/cd to restore access.", "2")
    c = _row("Lunch", "The team lunch is on Friday, please reply with dietary needs.", "3")
    kept, log = ps.dedup_near_duplicates(pd.DataFrame([a, b, c]), 0.85)
    assert list(kept.message_id) == ["1", "3"] and len(log) == 1


def test_defang_and_masking():
    out = ps.defang_for_display("Go to http://evil.example.com/path now")
    assert "http://" not in out and "hxxp" in out and "[.]" in out
    assert "[email]" in ps.mask_personal_info("mail me at a.b@corp.com") and ps.is_inert(out)


def test_url_features_never_need_network():
    f = ps.url_features_from_text("see hxxps://bit[.]ly/abc123")
    assert "shortener=1" in f and "defanged=1" in f


def test_abstain_logic():
    assert ps.decide(0.55, 0.7) == "ABSTAIN"
    assert ps.decide(0.95, 0.7) == "LIKELY_PHISHING" and ps.decide(0.05, 0.7) == "LIKELY_LEGITIMATE"


def test_no_network_or_exec_imports_in_core_code():
    banned = re.compile(r"^\s*(import|from)\s+(requests|urllib|http\.client|socket|webbrowser|subprocess)\b", re.M)
    for f in ["phishshift.py", "scan.py", "evaluate.py", "generate_data.py"]:
        assert not banned.search((ROOT / f).read_text(encoding="utf-8")), f"{f} imports a network/exec module"


def test_scan_rejects_empty_and_bad_channel():
    import scan
    for args in [("", "", "email"), ("hello", "", "fax")]:
        try:
            scan.scan(*args, bundle={"model": ps.RulesModel(), "tau": 0.7, "version": "t"})
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_scan_runs_on_saved_model_if_present():
    import scan
    if not scan.MODEL_PATH.exists():
        return
    r = scan.scan("Verify your password at http://login-x1.xyz/a today or your account is suspended")
    assert r["decision"] in {"LIKELY_PHISHING", "LIKELY_LEGITIMATE", "ABSTAIN"}
    assert "http://" not in r["safe_text"]


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1; print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)