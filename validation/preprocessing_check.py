import argparse
import json

import torch

import difflogic


DEFAULT_SEEDS = [1234, 2026, 3407, 4512, 9001]


def run_seed(seed, device):
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    train = torch.randn(12, 3, 4, 5, device=device) * 2 + 3
    test = torch.randn(7, 3, 4, 5, device=device) * 2 + 3

    minmax = difflogic.MinMaxScaler(feature_dims=(1, 2, 3))
    train_minmax = minmax.fit_transform(train)
    test_minmax = minmax.transform(test)
    assert minmax.data_min.shape == (1, 3, 4, 5), minmax.data_min.shape
    assert minmax.data_max.shape == (1, 3, 4, 5), minmax.data_max.shape
    assert float(train_minmax.min().cpu()) >= 0.
    assert float(train_minmax.max().cpu()) <= 1.
    assert float(test_minmax.min().cpu()) >= 0.
    assert float(test_minmax.max().cpu()) <= 1.

    standard = difflogic.StandardScaler(feature_dims=(1, 2, 3))
    train_standard = standard.fit_transform(train)
    assert standard.mean.shape == (1, 3, 4, 5), standard.mean.shape
    assert standard.std.shape == (1, 3, 4, 5), standard.std.shape
    train_mean_max_abs = float(train_standard.mean(dim=0).abs().max().cpu())
    train_std_max_abs_diff = float((train_standard.std(dim=0, unbiased=False) - 1).abs().max().cpu())
    assert train_mean_max_abs < 1e-5, train_mean_max_abs
    assert train_std_max_abs_diff < 1e-5, train_std_max_abs_diff

    minmax_encoder = difflogic.ThermometerEncoding(num_bits=4, value_range=(0., 1.)).to(device)
    minmax_encoded = minmax_encoder(train_minmax)
    assert minmax_encoded.shape == (12, 12, 4, 5), minmax_encoded.shape

    standard_encoder = difflogic.ThermometerEncoding(num_bits=4, value_range=(-2., 2.)).to(device)
    standard_encoded = standard_encoder(train_standard)
    assert standard_encoded.shape == (12, 12, 4, 5), standard_encoded.shape

    manual = torch.stack([(train_standard > t).float() for t in standard_encoder.thresholds], dim=2)
    manual = manual.reshape(12, 12, 4, 5)
    manual_max_abs_diff = float((standard_encoded - manual).abs().max().cpu())
    assert manual_max_abs_diff == 0., manual_max_abs_diff

    if device == 'cuda':
        torch.cuda.synchronize()

    return {
        'seed': seed,
        'minmax_train_min': float(train_minmax.min().cpu()),
        'minmax_train_max': float(train_minmax.max().cpu()),
        'standard_train_mean_max_abs': train_mean_max_abs,
        'standard_train_std_max_abs_diff': train_std_max_abs_diff,
        'minmax_encoded_sum': float(minmax_encoded.sum().cpu()),
        'standard_encoded_sum': float(standard_encoded.sum().cpu()),
        'manual_max_abs_diff': manual_max_abs_diff,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available.')

    rows = [run_seed(seed, args.device) for seed in args.seeds]
    print(json.dumps({'device': args.device, 'seeds': args.seeds, 'rows': rows}, indent=2))


if __name__ == '__main__':
    main()
