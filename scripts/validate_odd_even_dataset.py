import os
import sys
import json
import argparse
import numpy as np


ODD_ID = 11   # after +1 offset (ODD_TOKEN=10 -> +1)
EVEN_ID = 12  # after +1 offset (EVEN_TOKEN=11 -> +1)


def validate_split(split_dir: str, max_samples: int | None) -> dict:
    inputs = np.load(os.path.join(split_dir, "all__inputs.npy"))
    labels = np.load(os.path.join(split_dir, "all__labels.npy"))

    if max_samples is not None:
        inputs = inputs[:max_samples]
        labels = labels[:max_samples]

    # Range checks
    stats = {
        "inputs_min": int(inputs.min()),
        "inputs_max": int(inputs.max()),
        "labels_min": int(labels.min()),
        "labels_max": int(labels.max()),
        "samples": int(inputs.shape[0]),
        "seq_len": int(inputs.shape[1]),
    }

    errors = []
    if stats["inputs_min"] < 1 or stats["inputs_max"] > 12:
        errors.append(f"Inputs out of range [1,12]: {stats['inputs_min']}..{stats['inputs_max']}")
    if stats["labels_min"] < 1 or stats["labels_max"] > 10:
        errors.append(f"Labels out of range [1,10]: {stats['labels_min']}..{stats['labels_max']}")

    # Parity consistency checks
    inputs_0 = inputs - 1   # 0..11
    labels_0 = labels - 1   # 0..9

    odd_mask = inputs == ODD_ID
    even_mask = inputs == EVEN_ID

    odd_ok = (labels_0 % 2 == 1)
    even_ok = (labels_0 % 2 == 0)

    odd_mismatch = np.logical_and(odd_mask, np.logical_not(odd_ok)).sum()
    even_mismatch = np.logical_and(even_mask, np.logical_not(even_ok)).sum()

    stats.update({
        "odd_tokens": int(odd_mask.sum()),
        "even_tokens": int(even_mask.sum()),
        "odd_mismatch": int(odd_mismatch),
        "even_mismatch": int(even_mismatch),
    })

    if odd_mismatch > 0 or even_mismatch > 0:
        errors.append(f"Parity mismatches: odd={odd_mismatch}, even={even_mismatch}")

    # Hint coverage vs blanks (approximate)
    # Consider original blanks as positions that are either PAD-digit 1 (zero before +1) or parity tokens 11/12
    blank_mask = (inputs == 1) | odd_mask | even_mask
    hints_mask = odd_mask | even_mask

    blanks_per_sample = blank_mask.sum(axis=1)
    hints_per_sample = hints_mask.sum(axis=1)

    # Avoid division by zero
    valid = blanks_per_sample > 0
    coverage = np.zeros_like(hints_per_sample, dtype=np.float64)
    coverage[valid] = hints_per_sample[valid] / blanks_per_sample[valid]

    stats.update({
        "mean_hint_coverage": float(coverage.mean()),
        "median_hint_coverage": float(np.median(coverage)),
    })

    return {"stats": stats, "errors": errors}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/sudoku-odd-even-1k")
    parser.add_argument("--max-samples", type=int, default=20000)
    args = parser.parse_args()

    # Metadata sanity
    with open(os.path.join(args.data_path, "train", "dataset.json"), "r") as f:
        meta = json.load(f)
    if meta.get("seq_len") != 81 or meta.get("vocab_size") != 13:
        print(f"Warning: unexpected metadata: {meta}")

    results = {}
    all_errors = []
    for split in ["train", "test"]:
        split_dir = os.path.join(args.data_path, split)
        res = validate_split(split_dir, args.max_samples)
        results[split] = res
        all_errors.extend(res["errors"])

    print(json.dumps(results, indent=2))

    if all_errors:
        print("\nERRORS:")
        for e in all_errors:
            print("-", e)
        sys.exit(1)


if __name__ == "__main__":
    main()



