"""
02_profile_data.py
STEP 2 of the chunking & retrieval track.

WHY THIS SCRIPT EXISTS:
Chunk size decisions ("200 tokens with 20% overlap") are meaningless if you
picked them without looking at the data. MS MARCO-style passages are often
already short (a few sentences) — if that's true here too, a naive fixed
chunker might barely do anything, and that's worth knowing NOW rather than
after you've built three chunking strategies around the wrong assumption.

This also produces the numbers/plot you can drop straight into your
comparison writeup for judges — "we profiled the data and found X, which is
why we chose Y" reads as real methodology, not guesswork.
"""

import statistics
from utils import load_json, flatten_passages, save_json

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def word_count(text):
    return len(text.split())


def main():
    examples = load_json("sample_data.json")
    passages = flatten_passages(examples)

    lengths = [word_count(p["text"]) for p in passages]
    lengths_sorted = sorted(lengths)

    def pct(p):
        idx = int(len(lengths_sorted) * p / 100)
        idx = min(idx, len(lengths_sorted) - 1)
        return lengths_sorted[idx]

    report = {
        "num_passages": len(passages),
        "num_queries": len(examples),
        "avg_passages_per_query": round(len(passages) / max(1, len(examples)), 2),
        "word_count_min": min(lengths),
        "word_count_max": max(lengths),
        "word_count_mean": round(statistics.mean(lengths), 1),
        "word_count_median": statistics.median(lengths),
        "word_count_p90": pct(90),
        "empty_or_near_empty": sum(1 for l in lengths if l < 3),
    }

    print("--- Passage length profile (word count) ---")
    for k, v in report.items():
        print(f"{k}: {v}")

    # WHY THIS MATTERS: if word_count_p90 is small (say, under 60-80 words),
    # a 200-token fixed chunk is oversized for this dataset — most passages
    # already fit in one chunk, so "fixed-size chunking" and "one passage
    # per chunk" would look nearly identical. That's a real, useful finding
    # to report rather than something to hide.
    if report["word_count_p90"] < 80:
        print("\nNOTE: 90% of passages are under 80 words. A large fixed "
              "chunk size will barely split anything — consider a smaller "
              "chunk size, or lean harder on the semantic/metadata "
              "strategies to show meaningful differentiation.")

    save_json(report, "profile_report.json")

    if HAS_MPL:
        plt.figure(figsize=(6, 4))
        plt.hist(lengths, bins=20)
        plt.title("Passage length distribution (words)")
        plt.xlabel("Word count")
        plt.ylabel("Number of passages")
        plt.tight_layout()
        plt.savefig("passage_length_histogram.png")
        print("\nSaved passage_length_histogram.png "
              "(useful for your chunking-strategy comparison artifact).")
    else:
        print("\n(matplotlib not installed — skipping histogram image. "
              "pip install matplotlib to get the chart.)")


if __name__ == "__main__":
    main()