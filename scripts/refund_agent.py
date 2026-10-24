"""Refund agent: look up an order, search policy, then issue or escalate."""
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from mock_llm import chat
from policy_assistant import retrieve

DB_PATH = ROOT / "data" / "instamart.db"
SCENARIOS = ROOT / "data" / "raw" / "agent_scenarios.csv"
RUNS_PATH = ROOT / "outputs" / "agent_runs.jsonl"
LEDGER_PATH = ROOT / "outputs" / "refund_ledger.jsonl"
STATE_DIR = ROOT / "state"
SYSTEM_PROMPT = "AGENT. Decide the refund with tools. Stop when the outcome is known."

TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Look up one order in the analytical database.",
        "parameters": {"order_id": "string"},
    },
    {
        "name": "search_policy",
        "description": "Retrieve the policy chunks most relevant to a query.",
        "parameters": {"query": "string"},
    },
    {
        "name": "calculate_refund",
        "description": "Compute the late-delivery refund for a completed order.",
        "parameters": {"order_amount": "number", "delivery_minutes": "number"},
    },
    {
        "name": "issue_refund",
        "description": "Record an automatic refund at or below INR 1000.",
        "parameters": {"order_id": "string", "refund_inr": "number"},
    },
    {
        "name": "escalate_to_human",
        "description": "Send a refund above INR 1000 to a human support lead.",
        "parameters": {"order_id": "string", "refund_inr": "number"},
    },
]


class SimulatedCrash(Exception):
    """Raised after a checkpoint so a run can be resumed."""


def check(name, got, expected):
    if got != expected:
        raise SystemExit(f"FAIL {name}: got {got}, expected {expected}")


def get_order(order_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT order_id, status, order_amount, delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,),
        ).fetchone()
    if row is None:
        return {"error": "order not found"}
    minutes = row["delivery_minutes"]
    return {
        "order_id": row["order_id"],
        "status": row["status"],
        "order_amount": round(float(row["order_amount"]), 2),
        "delivery_minutes": None if minutes is None else float(minutes),
    }


def search_policy(query):
    return {"results": retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes):
    amount = round(float(order_amount), 2)
    refund = amount if float(delivery_minutes) > 45 else 0
    return {"refund_inr": refund}


def _ledger(tool_name, order_id, refund_inr):
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool_name,
        "order_id": order_id,
        "refund_inr": refund_inr,
    }
    with LEDGER_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    return {"refund_inr": refund_inr}


def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


def _arguments(raw):
    if isinstance(raw, str):
        return json.loads(raw)
    return dict(raw)


def _fallback(status, steps, llm_calls, tokens_used):
    names = ", ".join(step["tool"] for step in steps) or "none"
    summary = f"Handing this case to a human agent. Status {status}. Tools so far: {names}."
    return {
        "status": status,
        "final_answer": summary,
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
    }


def _checkpoint(run_id, scenario_id, messages, tokens_used, steps):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens_used,
        "step_number": len(steps),
        "tools_executed": [step["tool"] for step in steps],
        "steps": steps,
        "llm_calls": sum(1 for message in messages if message.get("role") == "assistant"),
    }
    (STATE_DIR / f"{run_id}.json").write_text(json.dumps(payload), encoding="utf-8")


def _refund_amount(result):
    if result["status"] not in {"REFUND", "ESCALATE"}:
        return 0.0
    for step in reversed(result["steps"]):
        observation = step["observation"]
        if isinstance(observation, dict) and "refund_inr" in observation:
            return float(observation["refund_inr"])
    return 0.0


def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    crash_after_tools=None,
    resume_state=None,
):
    run_id = run_id or scenario_id
    if resume_state is None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        steps = []
        tokens_used = 0
        llm_calls = 0
    else:
        messages = resume_state["messages"]
        steps = resume_state["steps"]
        tokens_used = resume_state["tokens_used"]
        llm_calls = resume_state["llm_calls"]
    started = time.monotonic()
    while llm_calls < max_iterations:
        if time.monotonic() - started >= time_limit_seconds:
            return _fallback("FALLBACK_TIMEOUT", steps, llm_calls, tokens_used)
        response = chat(messages)
        llm_calls += 1
        tokens_used += int(response.usage.total_tokens)
        message = response.choices[0].message
        if tokens_used > token_limit:
            return _fallback("FALLBACK_TOKEN_LIMIT", steps, llm_calls, tokens_used)
        if not message.tool_calls:
            content = message.content or ""
            return {
                "status": content.split(":", 1)[0].strip(),
                "final_answer": content,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
            }
        tool_call = message.tool_calls[0]
        arguments = _arguments(tool_call.arguments)
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_call.id,
                "name": tool_call.name,
                "arguments": arguments,
            }],
        })
        observation = TOOL_MAP[tool_call.name](**arguments)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.name,
            "content": json.dumps(observation, separators=(",", ":")),
        })
        steps.append({
            "tool": tool_call.name,
            "arguments": arguments,
            "observation": observation,
        })
        _checkpoint(run_id, scenario_id, messages, tokens_used, steps)
        if crash_after_tools is not None and len(steps) == crash_after_tools:
            raise SimulatedCrash(run_id)
    return _fallback("FALLBACK_MAX_ITERATIONS", steps, llm_calls, tokens_used)


def resume_agent(run_id):
    payload = json.loads((STATE_DIR / f"{run_id}.json").read_text(encoding="utf-8"))
    return run_agent(
        user_message=None,
        scenario_id=payload["scenario_id"],
        run_id=run_id,
        resume_state=payload,
    )


def _scenario_rows():
    import csv
    with SCENARIOS.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    if LEDGER_PATH.exists():
        LEDGER_PATH.unlink()
    RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with RUNS_PATH.open("w", encoding="utf-8") as handle:
        for row in _scenario_rows():
            result = run_agent(row["message"], row["scenario_id"])
            amount = _refund_amount(result)
            expected = float(row["expected_refund_inr"])
            if abs(amount - expected) > 0.01:
                raise SystemExit(
                    f"FAIL {row['scenario_id']} amount: got {amount}, expected {expected}"
                )
            if row["expected_outcome"] == "FALLBACK":
                check(f"{row['scenario_id']}_status", result["status"], "FALLBACK_MAX_ITERATIONS")
                if result["llm_calls"] != 6:
                    raise SystemExit(f"FAIL S7 calls: {result['llm_calls']}")
                if result["tokens_used"] > 6000:
                    raise SystemExit(f"FAIL S7 tokens: {result['tokens_used']}")
            else:
                check(f"{row['scenario_id']}_status", result["status"], row["expected_outcome"])
            record = {
                "scenario_id": row["scenario_id"],
                "status": result["status"],
                "final_answer": result["final_answer"],
                "llm_calls": result["llm_calls"],
                "tokens_used": result["tokens_used"],
                "steps": result["steps"],
            }
            handle.write(json.dumps(record) + "\n")
            results.append((row["scenario_id"], result["status"], amount, result["llm_calls"], result["tokens_used"]))
            print(
                f"{row['scenario_id']} {result['status']} refund={amount:.2f} "
                f"calls={result['llm_calls']} tokens={result['tokens_used']}"
            )

    check("S1_calls", results[0][3], 5)
    check("run_lines", sum(1 for _ in RUNS_PATH.open(encoding="utf-8")), 7)

    token_demo = run_agent(results and _scenario_rows()[0]["message"], "S1", token_limit=200, run_id="s1-token")
    check("token_demo", token_demo["status"], "FALLBACK_TOKEN_LIMIT")
    print(f"TOKEN DEMO {token_demo['status']} calls={token_demo['llm_calls']} tokens={token_demo['tokens_used']}")

    timeout_demo = run_agent(_scenario_rows()[0]["message"], "S1", time_limit_seconds=0, run_id="s1-timeout")
    check("timeout_demo", timeout_demo["status"], "FALLBACK_TIMEOUT")
    check("timeout_calls", timeout_demo["llm_calls"], 0)
    print(f"TIMEOUT DEMO {timeout_demo['status']} calls={timeout_demo['llm_calls']}")

    s1 = _scenario_rows()[0]
    try:
        run_agent(s1["message"], "S1", run_id="s1-crash", crash_after_tools=2)
        raise SystemExit("FAIL crash was not raised")
    except SimulatedCrash:
        resumed = resume_agent("s1-crash")
    tools = [step["tool"] for step in resumed["steps"]]
    check("crash_tools", tools, ["get_order", "search_policy", "calculate_refund", "issue_refund"])
    check("crash_status", resumed["status"], "REFUND")
    if abs(_refund_amount(resumed) - 128.25) > 0.01:
        raise SystemExit(f"FAIL crash amount: {_refund_amount(resumed)}")
    print("CRASH TOOLS " + ", ".join(tools))
    print(f"CRASH RESULT {resumed['status']} refund={_refund_amount(resumed):.2f}")
    print("AGENT PASSED")


if __name__ == "__main__":
    main()
