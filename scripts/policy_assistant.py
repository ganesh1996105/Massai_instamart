"""Retrieval-grounded answers over the five policy documents."""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mock_llm import chat

POLICIES = ROOT / "data" / "policies"
EVAL_PATH = ROOT / "data" / "raw" / "policy_eval.csv"
CHUNK_SIZE = 40
DEFAULT_OVERLAP = 10
EXPECTED_COUNTS = {
    "refund_policy.txt": 4,
    "delivery_policy.txt": 3,
    "escalation_policy.txt": 3,
    "payment_policy.txt": 3,
    "privacy_policy.txt": 2,
}
SYSTEM_PROMPT = (
    "POLICY ASSISTANT. Answer only from the supplied context. "
    "If the context does not contain the answer, say you could not find the information."
)

_INDEX = None


def check(name, got, expected):
    if got != expected:
        raise SystemExit(f"FAIL {name}: got {got}, expected {expected}")


def chunk_text(text, overlap):
    words = text.split()
    step = CHUNK_SIZE - overlap
    chunks = []
    start = 0
    while words:
        end = min(start + CHUNK_SIZE, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step
    return chunks


def load_chunks(overlap=DEFAULT_OVERLAP):
    chunks = []
    for path in sorted(POLICIES.glob("*.txt")):
        for text in chunk_text(path.read_text(encoding="utf-8"), overlap):
            chunks.append({"source": path.name, "text": text})
    return chunks


def fit_index(chunks):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(chunk["text"] for chunk in chunks)
    return {"chunks": chunks, "vectorizer": vectorizer, "matrix": matrix}


def get_index():
    global _INDEX
    if _INDEX is None:
        _INDEX = fit_index(load_chunks(DEFAULT_OVERLAP))
    return _INDEX


def retrieve(query, k=3, index=None):
    index = get_index() if index is None else index
    query_vector = index["vectorizer"].transform([query])
    scores = cosine_similarity(query_vector, index["matrix"]).ravel()
    order = scores.argsort()[::-1][:k]
    return [
        {
            "source": index["chunks"][position]["source"],
            "score": float(scores[position]),
            "text": index["chunks"][position]["text"],
        }
        for position in order
    ]


def answer(question, index=None):
    hits = retrieve(question, k=3, index=index)
    context = "\n".join(f"[{hit['source']}]\n{hit['text']}" for hit in hits)
    user = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
    response = chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
    )
    return response.choices[0].message.content, [hit["source"] for hit in hits]


def evaluate(overlap):
    index = fit_index(load_chunks(overlap))
    questions = pd.read_csv(EVAL_PATH, keep_default_na=False)
    doc_hits = fact_hits = fallback_hits = 0
    answerable = 0
    unanswerable = 0
    failures = []
    for row in questions.itertuples(index=False):
        text, sources = answer(row.question, index=index)
        if row.expected_doc == "NONE":
            unanswerable += 1
            if text.startswith("I could not find"):
                fallback_hits += 1
            else:
                failures.append((row.question, "fallback", text))
            continue
        answerable += 1
        doc_ok = row.expected_doc in sources
        fact_ok = row.expected_fact in text
        doc_hits += int(doc_ok)
        fact_hits += int(fact_ok)
        if not doc_ok or not fact_ok:
            failures.append((row.question, row.expected_doc, row.expected_fact, sources, text))
    return {
        "answerable": answerable,
        "doc_hits": doc_hits,
        "fact_hits": fact_hits,
        "unanswerable": unanswerable,
        "fallback_hits": fallback_hits,
        "failures": failures,
    }


def main():
    chunks = load_chunks(DEFAULT_OVERLAP)
    counts = Counter(chunk["source"] for chunk in chunks)
    check("chunk_total", len(chunks), 15)
    check("chunk_counts", dict(counts), EXPECTED_COUNTS)
    print(f"CHUNKS total={len(chunks)} " + " ".join(f"{name}={counts[name]}" for name in sorted(counts)))

    hits = retrieve("refund policy late delivery", k=3)
    check("refund_query_sources", [hit["source"] for hit in hits], ["refund_policy.txt"] * 3)
    print("QUERY refund policy late delivery -> " + ", ".join(hit["source"] for hit in hits))

    overlap_10 = evaluate(DEFAULT_OVERLAP)
    check("overlap10_docs", (overlap_10["doc_hits"], overlap_10["answerable"]), (12, 12))
    check("overlap10_facts", (overlap_10["fact_hits"], overlap_10["answerable"]), (12, 12))
    check("overlap10_fallback", (overlap_10["fallback_hits"], overlap_10["unanswerable"]), (2, 2))
    print(
        f"OVERLAP 10 docs={overlap_10['doc_hits']}/12 "
        f"facts={overlap_10['fact_hits']}/12 fallback={overlap_10['fallback_hits']}/2"
    )

    overlap_0 = evaluate(0)
    print(
        f"OVERLAP 0 docs={overlap_0['doc_hits']}/12 "
        f"facts={overlap_0['fact_hits']}/12 fallback={overlap_0['fallback_hits']}/2"
    )
    if not overlap_0["failures"]:
        raise SystemExit("FAIL overlap 0 was expected to miss at least one answerable question")
    for failure in overlap_0["failures"]:
        print("OVERLAP 0 FAIL", failure[0])
    print("POLICY PASSED")


if __name__ == "__main__":
    main()
