"""PhishShift core library: features, rules baseline, models, dedup, metrics, perturbations.

Safety (NFR-01/02): nothing in this module opens links, fetches content, or executes
message text. Messages are treated purely as strings. URLs are defanged (hxxp / [.]).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

SEED = 42
DISCLAIMER = ("Decision support only - not a guarantee of safety. "
              "Never click links or open attachments in a suspicious message.")

REQUIRED_COLS = ["message_id", "channel", "subject_or_title", "body_text",
                 "sanitized_url_features", "sender_domain_type", "label",
                 "campaign_group", "collected_date"]

# ---------------------------------------------------------------------------
# URL handling (defanged only)
# ---------------------------------------------------------------------------
URL_RE = re.compile(r"(?:hxxps?|https?)://[^\s]+", re.I)
SHORTENERS = ("bit.ly", "tinyurl.com", "t.co", "is.gd", "rb.gy", "goo.gl", "cutt.ly")


def url_features_from_text(text: str) -> str:
    """Return a sanitized feature string; never resolves or fetches the URL."""
    urls = URL_RE.findall(text or "")
    if not urls:
        return "n_urls=0"
    u = urls[0]
    host = re.sub(r"^(?:hxxps?|https?)://", "", u, flags=re.I).split("/")[0]
    host = host.replace("[.]", ".").lower()
    tld = host.rsplit(".", 1)[-1] if "." in host else "none"
    defanged = int("hxxp" in u.lower() or "[.]" in u)
    return (f"n_urls={len(urls)};shortener={int(host in SHORTENERS)};tld={tld};"
            f"defanged={defanged};host_digits={int(any(c.isdigit() for c in host))}")


def parse_url_feats(s: str) -> dict:
    out = {}
    for part in (s or "").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def defang_for_display(text: str) -> str:
    """Make any live-looking URL inert before it is printed or stored."""
    def _d(m):
        u = m.group(0)
        u = re.sub(r"^https?", lambda x: "hxxp" + x.group(0)[4:], u, flags=re.I)
        scheme, _, rest = u.partition("://")
        host, slash, path = rest.partition("/")
        return f"{scheme}://{host.replace('.', '[.]')}{slash}{path}"
    return URL_RE.sub(_d, text)


def mask_personal_info(text: str) -> str:
    """NFR-02: mask e-mail addresses and long digit strings in anything we display."""
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[email]", text)
    return re.sub(r"\b\d{9,}\b", "[number]", text)


def is_inert(text: str) -> bool:
    """True if the text contains no live (non-defanged) URL."""
    return not re.search(r"\bhttps?://", text or "", re.I)


# ---------------------------------------------------------------------------
# Risk cues (used by rules baseline, RF features and safe explanations)
# ---------------------------------------------------------------------------
CUES = {
    "urgency": (r"\b(urgent(ly)?|immediately|asap|right away|final notice|within \d+ (hours?|minutes?)|today)\b", 1.0,
                "Pressure to act quickly."),
    "credential_request": (r"((re-?enter|verify|confirm|validate|update)[^.!?]{0,40}(password|login|credentials|account details|identity)|sign in (here|at|to))", 2.0,
                           "Asks you to sign in or hand over login details."),
    "threat": (r"\b(suspend(ed)?|locked|terminated|penalty|declined|bounce|on hold|overdue|disabled)\b", 1.5,
               "Threatens a negative consequence."),
    "payment_request": (r"\b(gift cards?|card codes?|wire|transfer|pay now|unpaid fee|fee|bank|remittance|account details)\b", 1.5,
                        "Mentions money, cards or bank details."),
    "reward": (r"\b(prize|reward|voucher|refund|bonus|gift)\b", 1.0,
               "Offers a reward or refund."),
    "secrecy": (r"(don'?t discuss|keep it between|confidential|do not ignore|don'?t call)", 1.5,
                "Discourages you from checking with others."),
    "generic_greeting": (r"\b(dear (user|employee|customer)|attention,)", 1.0,
                         "Generic greeting instead of your name."),
    "shortened_link": (r"\b(bit\[?\.\]?ly|tinyurl|t\[?\.\]?co|is\[?\.\]?gd|rb\[?\.\]?gy)", 1.5,
                       "Link is shortened, hiding the real destination."),
    "lookalike_domain": (r"(secure|login|verify|portal)-\w+\[?\.\]?(xyz|top|info|online)", 2.0,
                         "Link domain imitates a trusted name."),
}
CUE_NAMES = list(CUES)
_CUE_RE = {k: re.compile(v[0], re.I) for k, v in CUES.items()}


def _subject_body(row) -> str:
    return f"{row['subject_or_title']} {row['body_text']}"


def triggered_cues(row) -> list[str]:
    text = _subject_body(row)
    feats = parse_url_feats(row.get("sanitized_url_features", ""))
    hit = []
    for k in CUE_NAMES:
        if _CUE_RE[k].search(text) or (k == "shortened_link" and feats.get("shortener") == "1"):
            hit.append(k)
    return hit


def cue_counts(df: pd.DataFrame) -> np.ndarray:
    rows = []
    for _, r in df.iterrows():
        text = _subject_body(r)
        feats = parse_url_feats(r["sanitized_url_features"])
        c = [min(len(_CUE_RE[k].findall(text)), 3) for k in CUE_NAMES]
        if feats.get("shortener") == "1":
            c[CUE_NAMES.index("shortened_link")] = max(c[CUE_NAMES.index("shortened_link")], 1)
        rows.append(c)
    return np.array(rows, dtype=float)


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    base = cue_counts(df)
    extra = []
    for _, r in df.iterrows():
        body = r["body_text"]
        f = parse_url_feats(r["sanitized_url_features"])
        letters = [c for c in body if c.isalpha()]
        extra.append([
            int(f.get("n_urls", 0)), int(f.get("shortener", 0)), int(f.get("host_digits", 0)),
            len(body) / 100.0, body.count("!"),
            (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0,
            sum(c.isdigit() for c in body) / max(len(body), 1),
            int(r["channel"] == "sms"),
        ])
    return np.hstack([base, np.array(extra)])


def full_text(df: pd.DataFrame) -> list[str]:
    return (df["subject_or_title"].fillna("") + " . " + df["body_text"].fillna("")
            + " " + df["sanitized_url_features"].fillna("")).tolist()


# ---------------------------------------------------------------------------
# Models (common interface: fit(df) / proba(df) -> P(phishing))
# ---------------------------------------------------------------------------
class RulesModel:
    """Baseline: weighted keyword/cue score. Mirrors what 'obvious keyword' training teaches."""
    name = "rules"
    threshold = 2.5

    def fit(self, df):
        return self

    def score(self, df):
        counts = cue_counts(df)
        w = np.array([CUES[k][1] for k in CUE_NAMES])
        return (np.minimum(counts, 1) * w).sum(axis=1)

    def proba(self, df):
        return 1 / (1 + np.exp(-(self.score(df) - self.threshold) * 1.2))


class SkModel:
    def __init__(self, name, pipe):
        self.name, self.pipe = name, pipe

    def fit(self, df):
        self.pipe.fit(df, df["label"].values)
        return self

    def proba(self, df):
        return self.pipe.predict_proba(df)[:, 1]


def _lr():
    return SkModel("tfidf_lr", Pipeline([
        ("txt", FunctionTransformer(full_text)),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                  token_pattern=r"(?u)\b\w+\b")),
        ("clf", LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced",
                                   random_state=SEED)),
    ]))


def _rf():
    return SkModel("rf_cues", Pipeline([
        ("feat", FunctionTransformer(feature_matrix)),
        ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                       class_weight="balanced_subsample",
                                       random_state=SEED, n_jobs=-1)),
    ]))


MODEL_FACTORIES = {"rules": RulesModel, "tfidf_lr": _lr, "rf_cues": _rf}


# ---------------------------------------------------------------------------
# Data checks, dedup, splits
# ---------------------------------------------------------------------------
def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not set(df["label"].unique()) <= {0, 1}:
        raise ValueError("label must be 0 (legitimate) or 1 (phishing)")
    if df["message_id"].duplicated().any():
        raise ValueError("message_id must be unique")


def _norm(t: str) -> str:
    t = t.lower()
    t = re.sub(r"hxxps?://\S+|https?://\S+", " url ", t)
    t = re.sub(r"\d+", "0", t)
    t = re.sub(r"[^a-z0 ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _dedup_texts(df):
    return [_norm(f"{s} {b}") for s, b in zip(df["subject_or_title"], df["body_text"])]


def dedup_near_duplicates(df: pd.DataFrame, threshold: float = 0.85):
    """FR-01: greedy near-duplicate removal on normalised TF-IDF cosine similarity."""
    df = df.reset_index(drop=True)
    X = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(_dedup_texts(df))
    S = cosine_similarity(X)
    keep = np.ones(len(df), bool)
    log = []
    for i in range(len(df)):
        if not keep[i]:
            continue
        for j in np.where(S[i, i + 1:] >= threshold)[0] + i + 1:
            if keep[j]:
                keep[j] = False
                log.append({"removed_id": df.loc[j, "message_id"], "kept_id": df.loc[i, "message_id"],
                            "similarity": round(float(S[i, j]), 3),
                            "removed_campaign": df.loc[j, "campaign_group"],
                            "same_label": bool(df.loc[i, "label"] == df.loc[j, "label"])})
    return df[keep].reset_index(drop=True), pd.DataFrame(log)


def cross_split_similarity(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Max cosine similarity of each test message to any train message."""
    tr, te = _dedup_texts(train), _dedup_texts(test)
    vec = TfidfVectorizer(ngram_range=(1, 2)).fit(tr + te)
    return cosine_similarity(vec.transform(te), vec.transform(tr)).max(axis=1)


def is_legit_urgent(df: pd.DataFrame) -> pd.Series:
    """Legitimate messages that *look* risky: urgency cues or password/payroll wording."""
    mask = []
    for _, r in df.iterrows():
        t = _subject_body(r).lower()
        urgent = "urgency" in triggered_cues(r) or bool(re.search(r"password|payroll|payslip|verification code", t))
        mask.append(r["label"] == 0 and urgent)
    return pd.Series(mask, index=df.index)


# ---------------------------------------------------------------------------
# Metrics, abstain
# ---------------------------------------------------------------------------
def binary_metrics(y, p, thr: float = 0.5) -> dict:
    y = np.asarray(y)
    pred = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    out = dict(n=len(y), accuracy=(tp + tn) / len(y), precision=prec, recall=rec, f1=f1,
               fpr=fp / (fp + tn) if fp + tn else float("nan"), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    out["roc_auc"] = float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan")
    return out


def abstain_metrics(y, p, tau: float) -> dict:
    """FR-04: abstain when confidence max(p, 1-p) < tau."""
    y, p = np.asarray(y), np.asarray(p)
    conf = np.maximum(p, 1 - p)
    abst = conf < tau
    pred = (p >= 0.5).astype(int)
    ph, lg = y == 1, y == 0
    return dict(tau=tau, coverage=1 - abst.mean(),
                selective_accuracy=float((pred[~abst] == y[~abst]).mean()) if (~abst).any() else float("nan"),
                phish_caught=float((ph & ~abst & (pred == 1)).sum() / max(ph.sum(), 1)),
                phish_abstained=float((ph & abst).sum() / max(ph.sum(), 1)),
                phish_missed=float((ph & ~abst & (pred == 0)).sum() / max(ph.sum(), 1)),
                legit_false_positive=float((lg & ~abst & (pred == 1)).sum() / max(lg.sum(), 1)),
                legit_abstained=float((lg & abst).sum() / max(lg.sum(), 1)))


def decide(p: float, tau: float) -> str:
    if max(p, 1 - p) < tau:
        return "ABSTAIN"
    return "LIKELY_PHISHING" if p >= 0.5 else "LIKELY_LEGITIMATE"


# ---------------------------------------------------------------------------
# Safe explanation (FR-04)
# ---------------------------------------------------------------------------
def explain(row: pd.Series, model: SkModel | None = None, top_k: int = 5) -> dict:
    cues = triggered_cues(row)
    out = {"cues": [(k, CUES[k][2]) for k in cues], "top_terms": []}
    if model is not None and model.name == "tfidf_lr":
        vec, clf = model.pipe.named_steps["tfidf"], model.pipe.named_steps["clf"]
        X = vec.transform(full_text(row.to_frame().T))
        r = X[0]
        contrib = r.data * clf.coef_[0][r.indices]
        names = vec.get_feature_names_out()
        order = np.argsort(-contrib)[:top_k]
        out["top_terms"] = [names[r.indices[i]] for i in order if contrib[i] > 0]
    return out


# ---------------------------------------------------------------------------
# Perturbations for robustness drift (FR-05, AT-03, CP-5)
# ---------------------------------------------------------------------------
_SYN = {"urgent": "time-sensitive", "immediately": "right away", "verify": "double-check",
        "password": "passcode", "confirm": "re-check", "suspended": "paused",
        "overdue": "past due", "click": "tap", "pay now": "settle today"}


def _p_synonym(s, rng):
    for k, v in _SYN.items():
        s = re.sub(rf"\b{k}\b", v, s, flags=re.I)
    return s


def _p_punct_noise(s, rng):
    return s.replace(".", "...").replace(",", " ,") + " !!"


def _p_no_punct(s, rng):
    return re.sub(r"[^\w\s\[\]/:\-]", "", s)


def _p_upper(s, rng):
    return s.upper()


def _p_typos(s, rng):
    words = s.split()
    for i, w in enumerate(words):
        if len(w) > 4 and rng.random() < 0.10:
            c = list(w)
            k = rng.randrange(len(c) - 1)
            c[k], c[k + 1] = c[k + 1], c[k]
            words[i] = "".join(c)
    return " ".join(words)


def _p_undefang(s, rng):
    return s.replace("hxxp", "http").replace("[.]", ".")


def _p_spaced_dots(s, rng):
    return s.replace("[.]", " dot ")


PERTURBATIONS = {"synonym_swap": _p_synonym, "punctuation_noise": _p_punct_noise,
                 "strip_punctuation": _p_no_punct, "all_caps": _p_upper,
                 "typos": _p_typos, "undefang_url": _p_undefang, "spaced_dots": _p_spaced_dots}


def perturb(df: pd.DataFrame, name: str, seed: int = SEED) -> pd.DataFrame:
    import random
    rng = random.Random(seed)
    f = PERTURBATIONS[name]
    out = df.copy()
    out["subject_or_title"] = [f(s, rng) for s in out["subject_or_title"]]
    out["body_text"] = [f(s, rng) for s in out["body_text"]]
    return out
