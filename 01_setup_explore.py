"""
01_setup_explore.py
STEP 1 of the chunking & retrieval track.

WHY THIS SCRIPT EXISTS:
Before writing a single chunker, you need to know the EXACT shape of the
data you're chunking — field names, nesting, and whether "passages" is a
list of strings or list of dicts. Guessing this wrong is the #1 way to burn
an hour on Day 2 debugging a KeyError instead of building. This script's
only job is to load a small sample and print/save that shape so every
later script can rely on it instead of re-discovering it.

RUN THIS FIRST. It also writes sample_data.json, which every other script
in this folder reads from — so later scripts don't need internet access
or re-download the dataset every time you test a change.
"""

from utils import load_msmarco_xi, save_json

LANG = "hi"     # change to "en"/another subset if your team picks a different language
SPLIT = "train"
N = 200         # small slice for fast iteration today; scale up later


def main():
    examples = load_msmarco_xi(lang=LANG, split=SPLIT, n=N)

    print("\n--- Top-level fields on one example ---")
    print(list(examples[0].keys()))

    print("\n--- First example (trimmed) ---")
    first = examples[0]
    for k, v in first.items():
        preview = str(v)
        if len(preview) > 200:
            preview = preview[:200] + " ...(truncated)"
        print(f"{k}: {preview}")

    print("\n--- Passages field structure ---")
    passages = first.get("passages", [])
    print(f"type: {type(passages)}, count: {len(passages)}")
    if passages:
        print(f"first passage item: {passages[0]}")

    # Persist the sample so 02-07 can run offline against the same data
    save_json(examples, "sample_data.json")
    print(f"\nSaved {len(examples)} examples to sample_data.json for reuse "
          f"by the rest of the pipeline.")


if __name__ == "__main__":
    main()