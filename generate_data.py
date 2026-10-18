"""Generate a locally-authored, fully synthetic, defanged message corpus.

No real people, credentials, mailboxes or live URLs. Every URL is defanged (hxxp / [.]).
Provenance: locally authored simulation templates (source = 'locally_authored_synthetic_v1').

Usage:  python generate_data.py --per-campaign 75 --seed 7
"""
import argparse
import csv
import datetime
import pathlib
import random
import string

import phishshift as ps

NAMES = ["Priya", "Arjun", "Meera", "Rahul", "Ananya", "Vikram", "Sneha", "Karan", "Isha", "Rohan", "Neha", "Aditya"]
DEPTS = ["Finance", "Sales", "Operations", "Engineering", "Legal", "Marketing"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
MONTHS = ["August", "September", "October"]
SYSTEMS = ["VPN", "HR portal", "email", "timesheet", "expense"]


def rid(n=6):
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def make_ref():
    return random.choice(["INV", "TKT", "REQ", "PO", "PKG"]) + "-" + str(random.randint(10000, 99999))


def make_url(kind):
    if kind == "short":
        return f"hxxps://{random.choice(['bit[.]ly', 'tinyurl[.]com', 't[.]co', 'is[.]gd', 'rb[.]gy'])}/{rid()}"
    if kind == "look":
        return (f"hxxps://{random.choice(['secure', 'login', 'portal', 'verify'])}-"
                f"{random.choice(['corp', 'acme', 'staff'])}{random.randint(1, 99)}"
                f"[.]{random.choice(['xyz', 'top', 'info', 'online'])}/{rid()}")
    if kind == "internal":
        return (f"hxxps://{random.choice(['portal', 'hr', 'it', 'helpdesk'])}[.]examplecorp[.]test/"
                f"{random.choice(['reset', 'payslip', 'mfa', 'benefits', 'tickets'])}/{rid(4)}")
    return None


PHISH_SENDERS = {"lookalike": .4, "external_free": .3, "unknown": .2, "internal": .1}
LEGIT_SENDERS = {"internal": .6, "vendor": .25, "unknown": .1, "external_free": .05}

# Each campaign builds a message from interchangeable parts: open / core / cta / close.
C = {}


def camp(group, label, channel, subjects, opens, cores, ctas, closes, urls, senders, start):
    C[group] = dict(label=label, channel=channel, subjects=subjects, parts=[opens, cores, ctas, closes],
                    urls=urls, senders=senders, start=start)


# ---- phishing campaigns (training) ----
camp("PH_PAYROLL", 1, "email", ["Payroll update required", "Confirm your direct deposit", "Salary hold notice"],
     ["Hello {name},", "Dear employee,", "Hi {name},"],
     ["your payroll profile for {month} needs re-verification.",
      "we could not validate your direct deposit details for this pay cycle.",
      "HR has flagged a mismatch in your salary record."],
     ["Confirm your login details at {url} within {h} hours.",
      "Sign in here {url} and re-enter your password to avoid a delay in salary.",
      "Verify your account now: {url}"],
     ["Payroll Team", "Failure to respond will hold your payment.", "", "Thank you."],
     ["look", "short"], PHISH_SENDERS, "2026-09-01")
camp("PH_QUOTA", 1, "email", ["Mailbox full", "Account suspension warning", "Unusual sign-in detected"],
     ["Dear user,", "Notice for {name}:", "Attention,"],
     ["your mailbox is {pct}% full and new messages will bounce.",
      "your email storage limit has been exceeded.",
      "we detected unusual sign-in activity on your account."],
     ["Sign in at {url} to restore access immediately.", "Click {url} to upgrade storage today.",
      "Verify your identity at {url} or your account will be suspended in {h} hours."],
     ["IT Support", "Do not ignore this message.", "", "Mail Admin"],
     ["look", "short"], PHISH_SENDERS, "2026-09-05")
camp("PH_PARCEL", 1, "sms", [""],
     ["", "[Courier]", "Notice:"],
     ["Your parcel {ref} is on hold due to an unpaid fee of Rs {amt}.",
      "Delivery of {ref} failed. Address needs confirmation.",
      "Customs fee pending for package {ref}."],
     ["Pay now at {url}", "Update details: {url}", "Reschedule here {url} today."],
     ["", "Reply STOP to opt out"], ["short", "look"], PHISH_SENDERS, "2026-09-10")
camp("PH_BOSS", 1, "email", ["Quick favour", "Are you available?", "Urgent request"],
     ["Hi {name},", "{name}, are you at your desk?", "Quick favour,"],
     ["I'm stuck in a meeting and need you to buy {n} gift cards worth Rs {amt} each for a client.",
      "Can you handle a confidential purchase for me? I need gift cards today.",
      "I need you to process an urgent transfer for a vendor, no time to call."],
     ["Send me the card codes when done.", "Reply with the codes as soon as you can.",
      "Don't discuss this with anyone yet."],
     ["Sent from my phone", "Thanks", "- {dept} Head"],
     ["none"], {"external_free": .6, "lookalike": .3, "internal": .1}, "2026-09-12")
camp("PH_INVOICE", 1, "email", ["Overdue invoice", "Payment declined", "Final notice: invoice"],
     ["Hello,", "Dear {name},", "Accounts payable here."],
     ["Invoice {ref} is now overdue and a late penalty applies.", "Your payment for {ref} was declined.",
      "We have not received payment for {ref}."],
     ["View and settle the invoice at {url} immediately.", "Open {url} to avoid service interruption.",
      "Review the statement here: {url}"],
     ["Billing Department", "Regards", "Final notice."],
     ["look", "short"], PHISH_SENDERS, "2026-09-18")
# ---- phishing campaigns (unfamiliar wording - default holdout) ----
camp("PH_VENDOR", 1, "email", ["Updated remittance details", "FYI: change in our accounts", "Re: {ref}"],
     ["Hi {name},", "Hello team,", "Good morning {name},"],
     ["our finance team has moved to a new bank, so the account details for next week's transfer have changed.",
      "we have restructured and our remittance details are different from last quarter.",
      "please note the updated settlement information for {ref} in the shared note."],
     ["The new details are in the document at {url}.", "Kindly use what is listed at {url} for the next payment.",
      "Details here: {url}"],
     ["Thanks for the quick turnaround.", "Appreciate your help.", "Best regards"],
     ["look", "short"], {"vendor": .4, "lookalike": .3, "external_free": .3}, "2026-10-02")
camp("PH_SURVEY", 1, "email", ["Two-minute pulse survey", "Claim your thank-you voucher", "Team feedback"],
     ["Thanks for being part of the {dept} team!", "Hello {name},", "Hi there,"],
     ["Share your thoughts in a two-minute survey and collect a Rs {amt} voucher.",
      "We are rewarding colleagues who give feedback this month.",
      "Complete the pulse check to claim your thank-you gift."],
     ["Start here: {url}", "Link: {url}", "Open {url} when you have a minute."],
     ["People Experience Team", "Cheers", "Offer ends Friday."],
     ["short", "look"], PHISH_SENDERS, "2026-10-06")
# ---- legitimate campaigns (training) ----
camp("LG_PWRESET", 0, "email", ["Password reset request", "Reset your {sys} password", "Your password reset link"],
     ["Hello {name},", "Hi {name},", "Dear {name},"],
     ["We received a request to reset the password for your {sys} account.",
      "A password reset was requested for your {sys} login.",
      "Use the link below to choose a new password."],
     ["Use {url} within {m} minutes.", "The link expires in {m} minutes: {url}", "Reset here: {url}"],
     ["If you did not request this, ignore this message and your password will not change.",
      "IT Helpdesk will never ask for your password by phone or email.", "Regards, IT Helpdesk"],
     ["internal", "internal", "internal", "short"], LEGIT_SENDERS, "2026-09-02")
camp("LG_PAYROLL", 0, "email", ["Payslip available", "Payroll cutoff reminder", "{month} payroll processed"],
     ["Hi {name},", "Hello {name},", "Dear colleague,"],
     ["Your payslip for {month} is now available in the HR portal.",
      "Payroll for {month} has been processed and credited.",
      "Reminder: payroll cutoff for expense claims is {day}."],
     ["Please submit any corrections before {day}: {url}", "View it at {url}.",
      "Raise a query with Payroll if the amount looks wrong: {url}"],
     ["Payroll Team", "Regards, HR Operations", "Thank you."],
     ["internal"], LEGIT_SENDERS, "2026-09-03")
camp("LG_ITMAINT", 0, "email", ["Urgent: maintenance tonight", "Planned downtime", "System outage window"],
     ["Urgent notice:", "Heads up {name},", "Important:"],
     ["scheduled maintenance tonight will make the VPN unavailable from {h2}pm.",
      "the email server will be restarted immediately after business hours.",
      "systems will be offline for a short window this weekend."],
     ["Please save your work and log out before then.", "Plan urgent work accordingly.", "Details: {url}"],
     ["IT Operations", "Thanks for your patience", "Regards"],
     ["internal"], LEGIT_SENDERS, "2026-09-08")
camp("LG_MFA", 0, "email", ["Action required: enroll in MFA", "Security update: two-step verification",
                            "MFA enrollment deadline"],
     ["Action required, {name}:", "Hello {name},", "Reminder:"],
     ["you must enroll in multi-factor authentication by {day} to keep access to {sys}.",
      "your account will require a second verification step from next week.",
      "security policy now needs all staff to set up an authenticator app."],
     ["Follow the steps at {url}.", "Enroll here: {url}", "Need help? Visit the helpdesk desk on floor 2."],
     ["We will never ask for your password or codes.", "Information Security", "Regards, IT"],
     ["internal", "short"], LEGIT_SENDERS, "2026-09-14")
camp("LG_HR", 0, "email", ["Benefits enrollment closing", "Leave balance update", "{dept} offsite details"],
     ["Hi all,", "Hello {name},", "Dear team,"],
     ["benefits enrollment closes on {day}.", "annual leave balances have been updated.",
      "the {dept} offsite details have been finalised."],
     ["Review your options at {url}.", "Please confirm attendance by {day}: {url}", "See the policy page: {url}"],
     ["HR Team", "Thanks", "Regards"],
     ["internal", "short"], LEGIT_SENDERS, "2026-09-16")
camp("LG_TICKET", 0, "email", ["Ticket {ref} resolved", "Update on {ref}", "Information needed for {ref}"],
     ["Hi {name},", "Hello,", "Dear {name},"],
     ["Ticket {ref} has been resolved.", "We have updated ticket {ref} with new notes.",
      "Your request {ref} is waiting for more information."],
     ["Reply within 3 days if the issue persists.", "Please respond today so we can proceed: {url}",
      "Track it at {url}."],
     ["IT Helpdesk", "Thanks", "Regards"],
     ["internal"], LEGIT_SENDERS, "2026-09-20")
camp("LG_EXPENSE", 0, "email", ["Expense approved", "Receipt needed", "Reimbursement update"],
     ["Hi {name},", "Hello {name},"],
     ["Your expense report {ref} was approved.", "Reimbursement of Rs {amt} will arrive with the next payroll run.",
      "Your claim {ref} needs a receipt before approval."],
     ["Upload the receipt at {url} by {day}.", "No action needed.", "Review at {url}."],
     ["Finance Team", "Thanks", "Regards, Accounts"],
     ["internal"], LEGIT_SENDERS, "2026-09-22")
camp("LG_SMS", 0, "sms", [""],
     [""],
     ["Your verification code is {code}. Do not share it with anyone.",
      "{code} is your login code for the {sys}. It is valid for 5 minutes.",
      "Security alert: a new sign-in was detected. If this was you, no action is needed."],
     ["", "Never share this code, our staff will not ask for it."],
     [""], ["none"], LEGIT_SENDERS, "2026-09-25")
# ---- legitimate campaigns (different business message set - default holdout) ----
camp("LG_VENDOR", 0, "email", ["Approved vendor update {ref}", "Vendor record change", "Re: {ref} remittance"],
     ["Hi {name},", "Hello team,"],
     ["Finance has confirmed the new supplier bank details for {ref} after a call-back to the vendor's known number.",
      "Procurement has updated the vendor record for {ref} following our verification process.",
      "Please note the approved change in remittance details for {ref}."],
     ["Details are in the vendor portal: {url}",
      "Reply to this thread if anything looks different from what you were told by phone.",
      "Process payment after the usual two-person approval."],
     ["Thanks for the quick turnaround.", "Regards, Finance", "Best regards"],
     ["internal"], {"internal": .6, "vendor": .4}, "2026-10-03")
camp("LG_COURIER", 0, "email", ["Delivery scheduled", "Your equipment has shipped", "Asset delivery {ref}"],
     ["Hi {name},", "Hello,"],
     ["Your laptop delivery {ref} is scheduled for {day}.",
      "The courier will arrive at reception between 10 and 12; no payment is required.",
      "Your replacement monitor has shipped."],
     ["Track it at {url}.", "Please collect it from reception.", "Sign for it when it arrives."],
     ["IT Asset Management", "Thanks", "Regards"],
     ["internal", "none"], LEGIT_SENDERS, "2026-10-08")


def weighted(d):
    return random.choices(list(d), weights=list(d.values()))[0]


def fill(spec, choice, url_kind):
    url = make_url(url_kind)
    slots = dict(name=random.choice(NAMES), dept=random.choice(DEPTS), day=random.choice(DAYS),
                 month=random.choice(MONTHS), h=random.choice([2, 6, 12, 24, 48]), h2=random.choice([6, 8, 10]),
                 m=random.choice([10, 15, 30, 60]), pct=random.randint(90, 99), amt=random.choice([49, 99, 250, 500]),
                 n=random.randint(2, 6), ref=make_ref(), code=random.randint(100000, 999999),
                 sys=random.choice(SYSTEMS), url=url or "")
    subj = spec["subjects"][choice[-1]].format(**slots)
    parts = [spec["parts"][i][c].format(**slots) for i, c in enumerate(choice[:-1])]
    body = " ".join(p for p in parts if p).strip()
    if url is None:
        body = body.replace("  ", " ")
    return subj, body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-campaign", type=int, default=75)
    ap.add_argument("--dup-rate", type=float, default=0.15, help="share of near-duplicate re-uses of a template")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/raw_messages.csv")
    args = ap.parse_args()
    random.seed(args.seed)
    rows, n = [], 0
    for group, spec in C.items():
        used = []
        for _ in range(args.per_campaign):
            if used and random.random() < args.dup_rate:
                choice, kind = random.choice(used)
            else:
                choice = tuple(random.randrange(len(p)) for p in spec["parts"]) + (random.randrange(len(spec["subjects"])),)
                kind = random.choice(spec["urls"])
                used.append((choice, kind))
            subj, body = fill(spec, choice, kind)
            if kind != "none" and "{url}" not in "".join(spec["parts"][2]):
                pass
            n += 1
            d = datetime.date.fromisoformat(spec["start"]) + datetime.timedelta(days=random.randint(0, 9))
            rows.append(dict(message_id=f"M{n:05d}", channel=spec["channel"], subject_or_title=subj,
                             body_text=body, sanitized_url_features=ps.url_features_from_text(body),
                             sender_domain_type=weighted(spec["senders"]), label=spec["label"],
                             campaign_group=group, collected_date=d.isoformat(),
                             source="locally_authored_synthetic_v1"))
    random.shuffle(rows)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} messages from {len(C)} campaigns to {out}")


if __name__ == "__main__":
    main()
