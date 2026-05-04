import argparse
import json

import torch

from experiments.conv_logic_binary_example import run_check


DEFAULT_SEEDS = [1234, 2026, 3407, 4512, 9001]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available.')

    rows = [run_check(seed, args.device) for seed in args.seeds]
    print(json.dumps({'device': args.device, 'seeds': args.seeds, 'rows': rows}, indent=2))


if __name__ == '__main__':
    main()
