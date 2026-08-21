"""
utils.py — shared helpers used by every script in this pipeline.

WHY THIS FILE EXISTS:
Every step (chunking, embedding, retrieval) needs to load the same dataset
and read/write the same JSONL chunk format. Centralizing that here means
if the HuggingFace dataset schema turns out slightly different from what
we expect once you actually load it, you only fix it in ONE place instead
of in all seven scripts.
"""

import json
import os

SAMPLE_PATH = "sample_data.json"


def load_msmarco_xi(lang="hi", split="train", n=200):
    """
    Load n examples from ai4bharat/MSMARCO-XI.

    WHY n=200 by default: on Day 2 you don't want to wait on a 3+ GB parquet
    download and iterate over the full split every time you test a chunking
    change. Pull a small slice, iterate fast, and only run against the full
    dataset once your pipeline is stable (Day 3+).

    WHY a fallback: this script may be the first thing you run today, before
    you've confirmed your machine has a working HF connection / auth. If the
    real load fails, we fall back to a small synthetic sample with the same
    *shape* we expect (query / answers / passages) so every downstream
    script (chunkers, embedder, retriever) can still be written and tested
    today without blocking on the download.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("ai4bharat/MSMARCO-XI", lang, split=split)
        examples = []
        for i, ex in enumerate(ds):
            if i >= n:
                break
            examples.append(dict(ex))
        print(f"[utils] Loaded {len(examples)} real examples "
              f"(lang={lang}, split={split}) from ai4bharat/MSMARCO-XI.")
        return examples
    except Exception as e:
        print(f"[utils] Could not load the real dataset ({e}).")
        print("[utils] Falling back to a small synthetic sample so the "
              "pipeline is still testable. Fix the load before Day 3.")
        return _synthetic_sample(n=min(n, 20), lang=lang)


def _synthetic_sample(n=20, lang="hi"):
    """
    Small hand-written stand-in with the shape we expect from the real
    dataset: a query, an answer, and a list of candidate passages per query.
    Only used when the real dataset can't be reached — swap out immediately
    once you've confirmed the real load works on your machine.
    """
    base_passages = [
        "The Reserve Bank of India regulates monetary policy and issues currency notes.",
        "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen.",
        "The Indian Premier League is a professional Twenty20 cricket league founded in 2008.",
        "Mount Everest, located in the Himalayas, is the tallest mountain above sea level.",
        "The Constitution of India was adopted on 26 January 1950.",
        "A vector database stores embeddings and supports approximate nearest neighbor search.",
        "The Taj Mahal was built by Mughal emperor Shah Jahan in memory of his wife.",
        "Machine learning models learn patterns from data instead of explicit rules.",
    ]
    examples = []
    for i in range(n):
        examples.append({
            "query": f"sample query {i}",
            "answers": [base_passages[i % len(base_passages)][:40]],
            "passages": [
                {"passage_text": base_passages[(i + j) % len(base_passages)],
                 "is_selected": 1 if j == 0 else 0}
                for j in range(3)
            ],
            "source_lang": "eng_Latn",
            "target_lang": f"{lang}_synthetic",
        })
    return examples


def flatten_passages(examples):
    """
    Turn the nested {query, passages: [...]} structure into a flat list of
    passage records, each tagged with the query it came from.

    WHY WE FLATTEN: chunking operates on individual passages, not on whole
    query objects. Flattening once here means every chunker script just
    receives a simple list of {id, text, query, lang} dicts and doesn't need
    to know anything about the original nested dataset schema.
    """
    records = []
    for qi, ex in enumerate(examples):
        query = ex.get("query", "")
        lang = ex.get("target_lang") or ex.get("source_lang") or "unknown"
        passages = ex.get("passages", [])

        # Defensive handling: passages might be a list of dicts
        # (MS MARCO's usual {"passage_text": ..., "is_selected": ...} shape)
        # or, depending on the exact HF config, a plain list of strings.
        # We normalize both to the same output shape so nothing downstream
        # has to branch on this again.
        for pi, p in enumerate(passages):
            if isinstance(p, dict):
                text = p.get("passage_text") or p.get("text") or ""
            else:
                text = str(p)
            if not text.strip():
                continue
            records.append({
                "doc_id": f"q{qi}_p{pi}",
                "query_id": qi,
                "query": query,
                "lang": lang,
                "text": text.strip(),
            })
    return records


def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[utils] Wrote {len(records)} records to {path}")


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)