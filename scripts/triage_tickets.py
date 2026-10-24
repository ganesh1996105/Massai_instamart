"""Triage support tickets with privacy guardrails around the offline classifier."""
import json
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mock_llm
from mock_llm import TransientLLMError, chat, est_tokens, reset

TICKETS = ROOT / "data" / "raw" / "tickets.csv"
SEGMENTS = ROOT / "outputs" / "customer_segments.csv"
BUDGET_PATH = ROOT / "outputs" / "triage_budget_demo.csv"
RESULTS_PATH = ROOT / "outputs" / "triage_results.csv"

CATEGORIES = ("billing", "delivery", "technical", "fraud")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE = re.compile(r"\b[6-9]\d{9}\b")
ORDER_ID = re.compile(r"ORD-\d{6}")
WORD = re.compile(r"[a-z]+")
PII = (EMAIL, PHONE, ORDER_ID)
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
HUMAN_STATUSES = {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"}
CLASSIFIED = {"classified_json", "classified_keyword"}
COLUMNS = [
    "ticket_id", "customer_id", "status", "category", "segment_name",
    "priority", "route", "tokens_used",
]

ZERO_SHOT = (
    "CLASSIFY each support ticket into exactly one category: "
    "billing, delivery, technical, or fraud. "
    'Reply with JSON only, in the form {"category": "...", "reason": "..."}. '
    "The category must be one of billing, delivery, technical, fraud."
)
FEW_SHOT = (
    "CLASSIFY each support ticket into exactly one category: "
    "billing, delivery, technical, or fraud. "
    'Reply with JSON only, in the form {"category": "...", "reason": "..."}. '
    "The category must be one of billing, delivery, technical, fraud.\n"
    "EXAMPLES:\n"
    "Ticket: An unauthorized charge on my account is fraud. Contact [EMAIL].\n"
    '{"category": "fraud", "reason": "unauthorized charge"}\n'
    "Ticket: I was charged twice and the invoice is wrong for [ORDER_ID].\n"
    '{"category": "billing", "reason": "double charge"}\n'
)


def check(name, got, expected):
    if got != expected:
        raise SystemExit(f"FAIL {name}: got {got}, expected {expected}")


def redact(text):
    text, emails = EMAIL.subn("[EMAIL]", text)
    text, phones = PHONE.subn("[PHONE]", text)
    text, orders = ORDER_ID.subn("[ORDER_ID]", text)
    return text, emails, phones, orders


def contains_pii(text):
    return any(pattern.search(text or "") for pattern in PII)


def assert_prompt_clean(messages):
    for message in messages:
        if contains_pii(message.get("content") or ""):
            raise SystemExit("FAIL personal data reached the model prompt")


def call_with_retry(ticket_id, messages):
    errors = 0
    for attempt in range(4):
        assert_prompt_clean(messages)
        try:
            return chat(messages), errors
        except TransientLLMError:
            errors += 1
            if attempt == 3:
                raise SystemExit(f"FAIL permanent error for {ticket_id}")
            wait = BASE_DELAY * (2 ** attempt) * random.uniform(0.9, 1.15)
            low = BASE_DELAY * (2 ** attempt) * 0.9
            high = BASE_DELAY * (2 ** attempt) * 1.15
            if not (low <= wait <= high):
                raise SystemExit(f"FAIL retry wait {wait} outside {low}..{high}")
            print(f"RETRY ticket_id={ticket_id} retry={attempt} wait={wait:.4f}")
            time.sleep(wait)
    raise SystemExit(f"FAIL retry loop ended for {ticket_id}")


def interpret(content):
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        payload = None
    if isinstance(payload, dict) and "category" in payload:
        category = payload.get("category")
        if category in CATEGORIES:
            return "classified_json", category
        return "flagged_invalid_output", ""
    words = WORD.findall((content or "").lower())
    found = [category for category in CATEGORIES if category in words]
    if len(found) == 1:
        return "classified_keyword", found[0]
    return "flagged_invalid_output", ""


def assign_route(status, category, is_premium):
    if status in HUMAN_STATUSES:
        return "human_review", "NORMAL"
    if category == "fraud":
        return "security_team", "HIGH"
    if is_premium:
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"


def prepare(tickets):
    emails = phones = orders = 0
    redacted = []
    for text in tickets["text"]:
        clean, n_email, n_phone, n_order = redact(text)
        emails += n_email
        phones += n_phone
        orders += n_order
        redacted.append(clean)
    tickets = tickets.copy()
    tickets["redacted"] = redacted
    leftover = sum(len(pattern.findall(text)) for text in redacted for pattern in PII)
    check("emails", emails, 55)
    check("phones", phones, 91)
    check("order_ids", orders, 132)
    check("redactions", emails + phones + orders, 278)
    check("pii_remaining", leftover, 0)
    empty = int(tickets["text"].str.strip().eq("").sum())
    long_mask = tickets["text"].map(est_tokens) > MAX_INPUT_TOKENS
    long_tokens = sorted(est_tokens(text) for text in tickets.loc[long_mask, "text"])
    check("empty", empty, 4)
    check("long", int(long_mask.sum()), 4)
    check("long_tokens", long_tokens, [734, 734, 734, 734])
    print(f"REDACT emails={emails} phones={phones} order_ids={orders} total={emails + phones + orders}")
    print(f"GUARD empty={empty} long={int(long_mask.sum())} long_tokens={long_tokens[0]}")
    return tickets


def run_pipeline(tickets, segments, system_prompt, budget):
    reset()
    random.seed(42)
    spent = 0
    errors = 0
    rows = []
    for ticket in tickets.itertuples(index=False):
        text = ticket.text
        if text.strip() == "":
            status, category, used = "skipped_empty", "", 0
        elif est_tokens(text) > MAX_INPUT_TOKENS:
            status, category, used = "flagged_long", "", 0
        elif budget is not None and spent >= budget:
            status, category, used = "budget_exhausted", "", 0
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ticket.redacted},
            ]
            if messages[1]["content"] != ticket.redacted:
                raise SystemExit("FAIL user message is not the redacted text")
            response, call_errors = call_with_retry(ticket.ticket_id, messages)
            errors += call_errors
            used = int(response.usage.total_tokens)
            spent += used
            status, category = interpret(response.choices[0].message.content)
        segment = segments.get(ticket.customer_id)
        if segment is None:
            segment_name, is_premium = "", False
        else:
            segment_name = segment["segment_name"]
            is_premium = bool(segment["is_premium"])
        route, priority = assign_route(status, category, is_premium)
        rows.append({
            "ticket_id": ticket.ticket_id,
            "customer_id": ticket.customer_id,
            "true_category": ticket.true_category,
            "status": status,
            "category": category,
            "segment_name": segment_name,
            "is_premium": is_premium,
            "priority": priority,
            "route": route,
            "tokens_used": used,
        })
    frame = pd.DataFrame(rows)
    return frame, errors, spent


def accuracy_report(frame, label):
    classified = frame.loc[frame["status"].isin(CLASSIFIED)]
    correct = int((classified["category"] == classified["true_category"]).sum())
    total_tokens = int(frame["tokens_used"].sum())
    per_ticket = total_tokens / len(classified)
    print(
        f"{label} classified={len(classified)} correct={correct} "
        f"accuracy={100 * correct / len(classified):.1f}% "
        f"tokens={total_tokens} per_classified={per_ticket:.2f}"
    )
    print(frame["status"].value_counts().to_string())
    return classified, correct, total_tokens, per_ticket


def check_full_run(frame, errors):
    counts = frame["status"].value_counts().to_dict()
    check("skipped_empty", counts.get("skipped_empty", 0), 4)
    check("flagged_long", counts.get("flagged_long", 0), 4)
    check("classified_json", counts.get("classified_json", 0), 135)
    check("classified_keyword", counts.get("classified_keyword", 0), 12)
    check("flagged_invalid_output", counts.get("flagged_invalid_output", 0), 9)
    check("transient_errors", errors, 19)
    check("rows", len(frame), 164)


def check_zero_shot(classified):
    correct = int((classified["category"] == classified["true_category"]).sum())
    check("zero_shot_classified", len(classified), 147)
    check("zero_shot_correct", correct, 126)
    wrong = classified.loc[classified["category"] != classified["true_category"]]
    pairs = (
        wrong.groupby(["true_category", "category"]).size().to_dict()
    )
    check("zero_shot_errors", pairs, {("fraud", "billing"): 11, ("technical", "billing"): 10})
    print("ZERO-SHOT CONFUSION")
    print(pd.crosstab(classified["true_category"], classified["category"], dropna=False).to_string())


def check_routing(frame):
    for row in frame.itertuples(index=False):
        if row.status in HUMAN_STATUSES:
            if row.route != "human_review" or row.priority != "NORMAL" or row.category != "":
                raise SystemExit(f"FAIL human route for {row.ticket_id}")
        elif row.category == "fraud":
            if row.route != "security_team" or row.priority != "HIGH":
                raise SystemExit(f"FAIL fraud route for {row.ticket_id}")
        elif row.is_premium:
            if row.route != "premium_desk" or row.priority != "HIGH":
                raise SystemExit(f"FAIL premium route for {row.ticket_id}")
        elif row.route != "support_queue" or row.priority != "NORMAL":
            raise SystemExit(f"FAIL queue route for {row.ticket_id}")
        if row.customer_id not in frame.attrs.get("known_customers", set()) and row.segment_name != "":
            raise SystemExit(f"FAIL segment name for unknown customer {row.customer_id}")


def write_output(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame[COLUMNS].to_csv(path, index=False)


def main():
    if "EXAMPLES:" in ZERO_SHOT or "CLASSIFY" not in ZERO_SHOT:
        raise SystemExit("FAIL zero-shot prompt")
    if "EXAMPLES:" not in FEW_SHOT or "fraud" not in FEW_SHOT or "CLASSIFY" not in FEW_SHOT:
        raise SystemExit("FAIL few-shot prompt")
    for category in CATEGORIES:
        if category not in ZERO_SHOT or category not in FEW_SHOT:
            raise SystemExit(f"FAIL prompt missing {category}")

    tickets = prepare(pd.read_csv(TICKETS, keep_default_na=False))
    segment_frame = pd.read_csv(SEGMENTS)
    segments = segment_frame.set_index("customer_id")[["segment_name", "is_premium"]].to_dict("index")
    known = set(segment_frame["customer_id"])

    zero, zero_errors, _ = run_pipeline(tickets, segments, ZERO_SHOT, budget=None)
    check_full_run(zero, zero_errors)
    zero_classified, _, zero_tokens, zero_per = accuracy_report(zero, "ZERO-SHOT")
    check_zero_shot(zero_classified)

    few, few_errors, _ = run_pipeline(tickets, segments, FEW_SHOT, budget=None)
    check_full_run(few, few_errors)
    few_classified, few_correct, few_tokens, few_per = accuracy_report(few, "FEW-SHOT")
    check("few_shot_classified", len(few_classified), 147)
    check("few_shot_correct", few_correct, 147)

    budget = 6000
    demo, _, _ = run_pipeline(tickets, segments, ZERO_SHOT, budget=budget)
    exhausted = int(demo["status"].eq("budget_exhausted").sum())
    if exhausted == 0:
        budget = 5000
        demo, _, _ = run_pipeline(tickets, segments, ZERO_SHOT, budget=budget)
        exhausted = int(demo["status"].eq("budget_exhausted").sum())
    check("budget_exhausted_at_least_one", exhausted > 0, True)
    demo.attrs["known_customers"] = known
    check_routing(demo)
    write_output(demo, BUDGET_PATH)
    print(f"BUDGET {budget} exhausted={exhausted} wrote={BUDGET_PATH.name}")

    final, final_errors, _ = run_pipeline(tickets, segments, FEW_SHOT, budget=100000)
    check_full_run(final, final_errors)
    check("final_budget_exhausted", int(final["status"].eq("budget_exhausted").sum()), 0)
    final.attrs["known_customers"] = known
    check_routing(final)
    write_output(final, RESULTS_PATH)
    print("ROUTES")
    print(final.groupby(["route", "priority"]).size().to_string())
    print(
        f"TOKENS zero={zero_tokens} ({zero_per:.2f}/classified) "
        f"few={few_tokens} ({few_per:.2f}/classified)"
    )
    print("TRIAGE PASSED")


if __name__ == "__main__":
    main()
