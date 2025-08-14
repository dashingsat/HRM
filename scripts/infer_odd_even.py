import os
import sys
import json
import argparse
import torch
import yaml
import numpy as np

from pretrain import PretrainConfig
from dataset.common import PuzzleDatasetMetadata
from utils.functions import load_model_class


def decode_digit_id(tid: int) -> str:
    # 1..10 -> digits 0..9
    if 1 <= tid <= 10:
        return str(tid - 1)
    return '?'  # other tokens shouldn't appear in final output


def encode_input_ids(grid: str) -> np.ndarray:
    """Encode a 9x9 grid string into input IDs (length 81).

    Allowed chars per cell:
      - '1'..'9' = given digit
      - '0' or '.' = blank
      - 'O' = odd hint, 'E' = even hint
    Mapping to IDs:
      - digits d -> (d + 1) for d in 0..9
      - O -> 11, E -> 12
    """
    s = ''.join(ch for ch in grid if not ch.isspace())
    assert len(s) == 81, f"Grid must have 81 cells, got {len(s)}"

    ids = np.zeros(81, dtype=np.int32)
    for i, ch in enumerate(s):
        if ch in '.0':
            ids[i] = 1  # digit 0 -> ID 1
        elif '1' <= ch <= '9':
            ids[i] = (ord(ch) - ord('0')) + 1
        elif ch.upper() == 'O':
            ids[i] = 11
        elif ch.upper() == 'E':
            ids[i] = 12
        else:
            raise ValueError(f"Unsupported cell char '{ch}' at pos {i}")
    return ids


def pretty_print_grid(ids: np.ndarray) -> None:
    assert ids.size == 81
    arr = ids.reshape(9, 9)
    for r in range(9):
        print(''.join(decode_digit_id(int(arr[r, c])) for c in range(9)))


def load_config_from_checkpoint(ckpt_path: str) -> PretrainConfig:
    cfg_dir = os.path.dirname(ckpt_path)
    with open(os.path.join(cfg_dir, 'all_config.yaml'), 'r') as f:
        return PretrainConfig(**yaml.safe_load(f))


def build_model_from_config(config: PretrainConfig, device: torch.device) -> torch.nn.Module:
    # Minimal metadata for constructing the model
    vocab_size = 13
    seq_len = 81
    num_puzzle_identifiers = 1

    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=1,
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_puzzle_identifiers=num_puzzle_identifiers,
        causal=False,
    )

    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device(device.type):
        model = model_cls(model_cfg)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        if 'DISABLE_COMPILE' not in os.environ:
            model = torch.compile(model, dynamic=False)  # type: ignore
    return model


def load_weights(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    state = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(state, assign=True)
    except Exception:
        # Handle torch.compile wrapped checkpoints
        model.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in state.items()}, assign=True)


def run_inference(ckpt_path: str, grid: str, override_L_cycles: int | None, override_halt_max_steps: int | None, avg_logits: bool) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config_from_checkpoint(ckpt_path)

    # Optional overrides for extra test-time compute
    if override_L_cycles is not None:
        config.arch.__pydantic_extra__['L_cycles'] = int(override_L_cycles)  # type: ignore
    if override_halt_max_steps is not None:
        config.arch.__pydantic_extra__['halt_max_steps'] = int(override_halt_max_steps)  # type: ignore

    model = build_model_from_config(config, device)
    load_weights(model, ckpt_path, device)

    model.eval()

    # Encode batch
    inputs = torch.from_numpy(encode_input_ids(grid)).view(1, -1)
    labels = torch.full_like(inputs, fill_value=-100)
    puzzle_identifiers = torch.zeros((1,), dtype=torch.int32)

    batch = {
        'inputs': inputs.to(device),
        'labels': labels.to(device),
        'puzzle_identifiers': puzzle_identifiers.to(device),
    }

    # Initialize carry
    with torch.device(device.type):
        carry = model.initial_carry(batch)  # type: ignore

    # Forward until halted
    preds = None
    logits_acc = None
    steps = 0
    while True:
        carry, _, _metrics, out, all_finish = model(carry=carry, batch=batch, return_keys=['logits'])  # type: ignore
        preds = out
        if avg_logits:
            step_logits = out['logits'].detach().to(torch.float32)
            logits_acc = step_logits if logits_acc is None else (logits_acc + step_logits)
            steps += 1
        if all_finish:
            break

    final_logits = preds['logits'] if not avg_logits else (logits_acc / max(steps, 1))
    logits = final_logits.argmax(dim=-1).detach().cpu().numpy()[0]
    print('Prediction:')
    pretty_print_grid(logits)


def main():
    parser = argparse.ArgumentParser(description='Odd/Even Sudoku inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file (checkpoints/.../step_*)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--grid', type=str, help='81-char string or 9 lines with digits/O/E/. for blanks')
    group.add_argument('--grid-file', type=str, help='Path to text file containing the grid')
    parser.add_argument('--L-cycles', type=int, default=None, help='Override L_cycles at inference (test-time compute)')
    parser.add_argument('--halt-max-steps', type=int, default=None, help='Override halt_max_steps at inference')
    parser.add_argument('--avg-logits', action='store_true', help='Average logits across ACT steps instead of using last step')
    args = parser.parse_args()

    if args.grid_file:
        with open(args.grid_file, 'r') as f:
            grid = f.read()
    else:
        grid = args.grid

    run_inference(args.checkpoint, grid, args.L_cycles, args.halt_max_steps, args.avg_logits)


if __name__ == '__main__':
    main()


