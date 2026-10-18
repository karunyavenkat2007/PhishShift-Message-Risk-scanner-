"""PhishShift web front-end (for Render or local use). Standard library only - no extra packages.

Start:  python app.py          (listens on $PORT, default 8000)
Safety: message text is only classified, never fetched/opened; output is HTML-escaped,
        defanged and masked; nothing is stored or logged. Decision support only (NFR-03).
"""
import html
import os
import pathlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

import joblib
import pandas as pd

import phishshift as ps
import scan

ROOT = pathlib.Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "phishshift_v1.joblib"
MAX_BODY = 20000
_BUNDLE = None


def get_bundle():
    """Load the saved model; if it is missing or incompatible with this Python/scikit-learn, retrain quickly."""
    global _BUNDLE
    if _BUNDLE is None:
        try:
            _BUNDLE = joblib.load(MODEL_PATH)
        except Exception:  # noqa: BLE001  (missing file or version mismatch)
            df = pd.read_csv(ROOT / "data" / "clean_messages.csv", keep_default_na=False)
            _BUNDLE = {"model": ps.MODEL_FACTORIES["tfidf_lr"]().fit(df), "model_name": "tfidf_lr",
                       "version": "phishshift-v1-tfidf_lr-retrained", "tau": 0.6}
    return _BUNDLE


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>PhishShift</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem;color:#1a2433}}
textarea,input,select{{width:100%;box-sizing:border-box;padding:.6rem;margin:.3rem 0 1rem;font:inherit}}
button{{padding:.7rem 1.2rem;font:inherit;background:#0b6bcb;color:#fff;border:0;border-radius:6px;cursor:pointer}}
.card{{padding:1rem;border-radius:8px;margin:1.2rem 0}} .high{{background:#fde8e4}} .low{{background:#e3f5ee}}
.unsure{{background:#fff4d6}} small{{color:#566}}
</style></head><body>
<h1>PhishShift Message Risk Scanner</h1>
<p><small>Educational triage tool. Paste a <b>simulated</b> message. Links are never opened. Do not paste real passwords or personal data.</small></p>
<form method="post" action="/scan">
<label>Channel<select name="channel"><option value="email"{e}>Email</option><option value="sms"{s}>SMS</option></select></label>
<label>Subject (optional)<input name="subject" maxlength="200" value="{subject}"></label>
<label>Message text<textarea name="text" rows="7" required>{text}</textarea></label>
<button type="submit">Scan message</button></form>{result}
<p><small>{disclaimer}</small></p></body></html>"""


def render_page(text="", subject="", channel="email", result=""):
    return PAGE.format(text=html.escape(text), subject=html.escape(subject), result=result,
                       e=" selected" if channel == "email" else "", s=" selected" if channel == "sms" else "",
                       disclaimer=html.escape(ps.DISCLAIMER))


def render_result(res):
    css, head = {"LIKELY_PHISHING": ("high", "HIGH RISK - likely phishing"),
                 "LIKELY_LEGITIMATE": ("low", "LOW RISK - likely legitimate"),
                 "ABSTAIN": ("unsure", "UNSURE - send to a human analyst")}[res["decision"]]
    cues = "".join(f"<li><b>{html.escape(k.replace('_', ' '))}</b>: {html.escape(w)}</li>" for k, w in res["cues"])
    terms = html.escape(", ".join(res["top_terms"]))
    return (f'<div class="card {css}"><h2>{head}</h2><p>P(phishing) = {res["p_phish"]} '
            f'<small>({html.escape(res["model_version"])})</small></p>'
            f'{"<p>Risk cues:</p><ul>" + cues + "</ul>" if cues else "<p>No rule-based risk cues found.</p>"}'
            f'{"<p>Words that raised the score: " + terms + "</p>" if terms else ""}'
            f'<p><small>Message shown safely: {html.escape(res["safe_text"][:300])}</small></p></div>')


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, "ok", "text/plain")
        if self.path in ("/", "/index.html"):
            return self._send(200, render_page())
        self._send(404, "Not found", "text/plain")

    def do_POST(self):
        if self.path != "/scan":
            return self._send(404, "Not found", "text/plain")
        try:
            n = min(int(self.headers.get("Content-Length", 0)), MAX_BODY)
            form = parse_qs(self.rfile.read(n).decode("utf-8", errors="replace"))
            text = form.get("text", [""])[0]
            subject = form.get("subject", [""])[0][:200]
            channel = form.get("channel", ["email"])[0]
            try:
                res = scan.scan(text, subject, channel, get_bundle())
                self._send(200, render_page(text, subject, channel, render_result(res)))
            except ValueError as e:
                err = f'<div class="card unsure">Error: {html.escape(str(e))}</div>'
                self._send(400, render_page(text, subject, channel, err))
        except Exception:  # noqa: BLE001  (never leak internals to the browser)
            self._send(500, render_page(result='<div class="card unsure">Something went wrong. Try again.</div>'))

    def log_message(self, *args):  # do not log message content
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    get_bundle()  # load/train before accepting traffic
    print(f"PhishShift web app on port {port}")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
    