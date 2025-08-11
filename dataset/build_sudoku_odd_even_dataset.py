from typing import Optional, Tuple
import os
import csv
import json
import numpy as np  # type: ignore[import]

from argdantic import ArgParser  # type: ignore[import]
from pydantic import BaseModel  # type: ignore[import]
from tqdm import tqdm  # type: ignore[import]
from huggingface_hub import hf_hub_download  # type: ignore[import]

from common import PuzzleDatasetMetadata


cli = ArgParser()


class DataProcessConfig(BaseModel):
    # Source classic Sudoku dataset (questions + solutions)
    source_repo: str = "sapientinc/sudoku-extreme"
    output_dir: str = "data/sudoku-odd-even-1k"

    subsample_size: Optional[int] = 1000
    min_difficulty: Optional[int] = None
    num_aug: int = 0

    # Odd/Even constraints
    # If fixed_num_hints is provided, it overrides hint_fraction per puzzle
    hint_fraction: float = 0.35
    fixed_num_hints: Optional[int] = None


# Internal encoding (pre-offset by +1 at the end):
# 0..9 -> digits '0'..'9'
# 10 -> ODD hint, 11 -> EVEN hint
ODD_TOKEN = 10
EVEN_TOKEN = 11


def _load_subset_csv(source_repo: str, set_name: str, min_difficulty: Optional[int]) -> Tuple[list[np.ndarray], list[np.ndarray]]:
    inputs: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    with open(hf_hub_download(source_repo, f"{set_name}.csv", repo_type="dataset"), newline="") as csvfile:  # type: ignore
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for _source, q, a, rating in reader:
            if (min_difficulty is None) or (int(rating) >= min_difficulty):
                assert len(q) == 81 and len(a) == 81

                # q: '.' means blank → map to 0
                q_arr = np.frombuffer(q.replace('.', '0').encode(), dtype=np.uint8).reshape(9, 9) - ord('0')
                a_arr = np.frombuffer(a.encode(), dtype=np.uint8).reshape(9, 9) - ord('0')
                inputs.append(q_arr)
                labels.append(a_arr)

    return inputs, labels


def _permute_parity_preserving_digit_map() -> np.ndarray:
    # Permute odds among odds and evens among evens to preserve odd/even constraints
    odds = np.array([1, 3, 5, 7, 9], dtype=np.uint8)
    evens = np.array([2, 4, 6, 8], dtype=np.uint8)
    odds_perm = np.random.permutation(odds)
    evens_perm = np.random.permutation(evens)

    digit_map = np.zeros(10, dtype=np.uint8)
    digit_map[0] = 0
    for src, dst in zip(odds, odds_perm):
        digit_map[src] = dst
    for src, dst in zip(evens, evens_perm):
        digit_map[src] = dst
    return digit_map


def shuffle_sudoku_odd_even(board: np.ndarray, solution: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Parity-preserving digit map
    digit_map = _permute_parity_preserving_digit_map()

    # Random transpose flag
    transpose_flag = np.random.rand() < 0.5

    # Valid row/col permutations via bands and stacks
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])

    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])

    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply(x: np.ndarray) -> np.ndarray:
        y = x.T if transpose_flag else x
        y = y.flatten()[mapping].reshape(9, 9).copy()
        return digit_map[y]

    return apply(board), apply(solution)


def make_odd_even_input(base_board: np.ndarray, solution: np.ndarray, hint_fraction: float, fixed_num_hints: Optional[int]) -> np.ndarray:
    # Copy to start from original givens
    inputs = base_board.copy()
    blanks = (base_board == 0)

    # Candidate indices for parity hints: only where puzzle is blank
    blank_indices = np.where(blanks.reshape(-1))[0]
    num_blanks = blank_indices.size

    if fixed_num_hints is not None:
        k = min(fixed_num_hints, num_blanks)
    else:
        k = int(round(hint_fraction * num_blanks))

    if k > 0:
        chosen = np.random.choice(blank_indices, size=k, replace=False)
        chosen_r = chosen // 9
        chosen_c = chosen % 9

        # Apply parity tokens based on solution parity
        sol_vals = solution[chosen_r, chosen_c]
        is_odd = (sol_vals % 2 == 1)
        inputs[chosen_r, chosen_c] = np.where(is_odd, ODD_TOKEN, EVEN_TOKEN)

    return inputs


def convert_subset(set_name: str, config: DataProcessConfig):
    # Load base classic puzzles
    base_inputs, base_labels = _load_subset_csv(config.source_repo, set_name, config.min_difficulty)

    # Subsample train set if requested
    if set_name == "train" and config.subsample_size is not None:
        total_samples = len(base_inputs)
        if config.subsample_size < total_samples:
            indices = np.random.choice(total_samples, size=config.subsample_size, replace=False)
            base_inputs = [base_inputs[i] for i in indices]
            base_labels = [base_labels[i] for i in indices]

    # Generate dataset
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0

    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    for orig_inp, orig_out in zip(tqdm(base_inputs), base_labels):
        aug_rounds = 1 + (config.num_aug if set_name == "train" else 0)
        for aug_idx in range(aug_rounds):
            if aug_idx == 0:
                inp, out = orig_inp, orig_out
            else:
                inp, out = shuffle_sudoku_odd_even(orig_inp, orig_out)

            # Build odd/even input with hints on blank cells
            odd_even_input = make_odd_even_input(
                base_board=inp,
                solution=out,
                hint_fraction=config.hint_fraction,
                fixed_num_hints=config.fixed_num_hints,
            )

            # Append single-example puzzle
            results["inputs"].append(odd_even_input)
            results["labels"].append(out)
            example_id += 1
            puzzle_id += 1

            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)

        # Push group delimiter
        results["group_indices"].append(puzzle_id)

    # To numpy with +1 offset (reserve 0 for PAD)
    def _seq_to_numpy_inputs(seq):
        arr = np.concatenate(seq).reshape(len(seq), -1)
        # Check token range before +1: digits 0..9, ODD 10, EVEN 11
        assert np.all((arr >= 0) & (arr <= EVEN_TOKEN)), "Input token out of range"
        return arr + 1

    def _seq_to_numpy_labels(seq):
        arr = np.concatenate(seq).reshape(len(seq), -1)
        assert np.all((arr >= 0) & (arr <= 9)), "Label digit out of range"
        return arr + 1

    results_np = {
        "inputs": _seq_to_numpy_inputs(results["inputs"]),
        "labels": _seq_to_numpy_labels(results["labels"]),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Metadata: vocab = PAD(0) + digits(10) + parity(2) = 13
    metadata = PuzzleDatasetMetadata(
        seq_len=81,
        vocab_size=13,
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results_np["group_indices"]) - 1,
        mean_puzzle_examples=1,
        sets=["all"],
    )

    # Save
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    for k, v in results_np.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    # IDs mapping (for visualization only)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    convert_subset("train", config)
    convert_subset("test", config)


if __name__ == "__main__":
    cli()


