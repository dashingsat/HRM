import os
import json
import argparse
import numpy as np


def decode_input_cell(token_id: int) -> str:
    # Stored IDs are offset by +1: 1..10 digits 0..9, 11=ODD, 12=EVEN
    if token_id == 11:
        return 'O'
    if token_id == 12:
        return 'E'
    # 1..10 -> digits 0..9
    if 1 <= token_id <= 10:
        return str(token_id - 1)
    if token_id == 0:
        return '#'
    return '?'


def decode_label_cell(token_id: int) -> str:
    # Labels: 1..10 -> digits 0..9 (solutions typically 1..9 -> 2..10)
    if 1 <= token_id <= 10:
        return str(token_id - 1)
    return '?'


def print_grid(arr: np.ndarray, decoder) -> None:
    for r in range(9):
        row = ''.join(decoder(int(arr[r, c])) for c in range(9))
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='data/sudoku-odd-even-1k')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()

    # Load metadata
    with open(os.path.join(args.data_path, args.split, 'dataset.json'), 'r') as f:
        meta = json.load(f)
    print('Metadata:')
    print(json.dumps(meta, indent=2))

    # Load arrays
    inputs = np.load(os.path.join(args.data_path, args.split, 'all__inputs.npy'))
    labels = np.load(os.path.join(args.data_path, args.split, 'all__labels.npy'))

    idx = max(0, min(args.index, inputs.shape[0] - 1))
    inp = inputs[idx].reshape(9, 9)
    lab = labels[idx].reshape(9, 9)

    # Compute simple stats
    odd_mask = (inp == 11)
    even_mask = (inp == 12)
    blank_mask = (inp == 1) | odd_mask | even_mask
    hints_mask = odd_mask | even_mask
    blanks = int(blank_mask.sum())
    hints = int(hints_mask.sum())
    coverage = (hints / blanks) if blanks > 0 else 0.0

    print('\nExample index:', idx)
    print('Input grid (O=odd hint, E=even hint, digits 0-9):')
    print_grid(inp, decode_input_cell)

    print('\nLabel grid (digits 0-9):')
    print_grid(lab, decode_label_cell)

    print('\nStats:')
    print('  odd hints:', int(odd_mask.sum()))
    print('  even hints:', int(even_mask.sum()))
    print('  blanks (incl. hinted):', blanks)
    print('  hint coverage:', round(coverage, 4))


if __name__ == '__main__':
    main()



