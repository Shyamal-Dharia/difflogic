import argparse
import json
import statistics
import time
import math

import torch

import difflogic
from residual_init_eeg_shape_experiment import DEFAULT_SEEDS, EEGConvLogicStack


PRESETS = {
    'alexnet_like_1m': {
        'stack_depth': 5,
        'pool_every': 1,
        'convlogic_channels': 833,
        'convlogic_tree_depth': 4,
        'convnet_channels': 95,
    },
    'resnet18_like_1m': {
        'stack_depth': 18,
        'pool_every': 3,
        'convlogic_channels': 496,
        'convlogic_tree_depth': 3,
        'convnet_channels': 47,
    },
}


class EEGConvNetStack(torch.nn.Module):
    """
    Small baseline ConvNet with the same EEG input shape and pooling schedule.
    """
    def __init__(
            self,
            in_channels=15,
            conv_channels=64,
            stack_depth=5,
            kernel_size=(3, 9),
            pool_every=1,
    ):
        super().__init__()
        layers = []
        cur_channels = in_channels
        for layer_idx in range(stack_depth):
            layers.append(torch.nn.Conv2d(
                cur_channels,
                conv_channels,
                kernel_size=kernel_size,
                padding=(kernel_size[0] // 2, kernel_size[1] // 2),
                bias=False,
            ))
            layers.append(torch.nn.SiLU())
            cur_channels = conv_channels
            should_pool = pool_every > 0 and (layer_idx + 1) % pool_every == 0 and layer_idx < stack_depth - 1
            if should_pool:
                layers.append(torch.nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)))
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class EEGResidualConvNetStack(torch.nn.Module):
    """
    Residual ConvNet baseline for the deeper comparison.
    """
    def __init__(
            self,
            in_channels=15,
            conv_channels=64,
            stack_depth=18,
            kernel_size=(3, 9),
            pool_every=3,
    ):
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.input_conv = torch.nn.Conv2d(
            in_channels,
            conv_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.blocks = torch.nn.ModuleList([
            torch.nn.Conv2d(
                conv_channels,
                conv_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
            for _ in range(stack_depth - 1)
        ])
        self.activation = torch.nn.SiLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.pool_every = pool_every
        self.stack_depth = stack_depth
        self.residual_scale = 1.0 / math.sqrt(2.0)

    def forward(self, x):
        x = self.activation(self.input_conv(x))
        if self.pool_every > 0 and 1 % self.pool_every == 0 and self.stack_depth > 1:
            x = self.pool(x)
        for block_idx, block in enumerate(self.blocks):
            x = (x + self.activation(block(x))) * self.residual_scale
            layer_idx = block_idx + 2
            should_pool = self.pool_every > 0 and layer_idx % self.pool_every == 0 and block_idx < len(self.blocks) - 1
            if should_pool:
                x = self.pool(x)
        return x


def make_input(seed, shape, device):
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    return torch.rand(*shape, device=device)


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def conv_weight_layers(model):
    return [
        layer for layer in model.modules()
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, difflogic.ConvLogicTreeLayer)
    ]


def layer_grad_norm(layer):
    if hasattr(layer, 'weight'):
        grad = layer.weight.grad
    else:
        grad = layer.weights.grad
    return float(grad.detach().norm().cpu())


def layer_grad_max_abs(layer):
    if hasattr(layer, 'weight'):
        grad = layer.weight.grad
    else:
        grad = layer.weights.grad
    return float(grad.detach().abs().max().cpu())


def build_model(args, preset, model_type):
    kernel_size = (args.kernel_height, args.kernel_width)
    if model_type == 'convlogic':
        return EEGConvLogicStack(
            in_channels=args.in_channels,
            conv_channels=preset['convlogic_channels'],
            stack_depth=preset['stack_depth'],
            tree_depth=preset['convlogic_tree_depth'],
            kernel_size=kernel_size,
            pool_every=preset['pool_every'],
            device=args.device,
            residual_init=True,
        )
    if model_type == 'convnet':
        return EEGConvNetStack(
            in_channels=args.in_channels,
            conv_channels=preset['convnet_channels'],
            stack_depth=preset['stack_depth'],
            kernel_size=kernel_size,
            pool_every=preset['pool_every'],
        )
    if model_type == 'convnet_residual':
        return EEGResidualConvNetStack(
            in_channels=args.in_channels,
            conv_channels=preset['convnet_channels'],
            stack_depth=preset['stack_depth'],
            kernel_size=kernel_size,
            pool_every=preset['pool_every'],
        )
    raise ValueError(f'Unknown model type: {model_type}')


def run_one(seed, args, preset_name, model_type):
    preset = PRESETS[preset_name]
    torch.manual_seed(seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    model = build_model(args, preset, model_type).to(args.device).train()
    x = make_input(seed, (args.batch_size, args.in_channels, args.height, args.width), args.device)
    x.requires_grad_(True)

    if args.device == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    y = model(x)
    torch.manual_seed(seed + 100000)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(seed + 100000)
    target = torch.rand_like(y)
    loss = (y - target).square().mean()
    loss.backward()
    if args.device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    weight_layers = conv_weight_layers(model)
    grad_norms = [layer_grad_norm(layer) for layer in weight_layers]
    grad_max_abs = [layer_grad_max_abs(layer) for layer in weight_layers]
    first_grad = grad_norms[0]
    last_grad = grad_norms[-1]

    return {
        'seed': seed,
        'preset': preset_name,
        'model_type': model_type,
        'stack_depth': preset['stack_depth'],
        'pool_every': preset['pool_every'],
        'trainable_parameters': count_trainable_parameters(model),
        'input_shape': list(x.shape),
        'output_shape': list(y.shape),
        'loss': float(loss.detach().cpu()),
        'output_mean': float(y.detach().mean().cpu()),
        'output_std': float(y.detach().std(unbiased=False).cpu()),
        'input_grad_norm': float(x.grad.detach().norm().cpu()),
        'input_grad_max_abs': float(x.grad.detach().abs().max().cpu()),
        'first_weight_grad_norm': first_grad,
        'last_weight_grad_norm': last_grad,
        'first_to_last_grad_ratio': first_grad / last_grad if last_grad != 0.0 else None,
        'weight_grad_norm_mean': statistics.mean(grad_norms),
        'weight_grad_norm_min': min(grad_norms),
        'weight_grad_norm_max': max(grad_norms),
        'weight_grad_max_abs_max': max(grad_max_abs),
        'finite': bool(
            torch.isfinite(y).all()
            and torch.isfinite(loss)
            and torch.isfinite(x.grad).all()
            and all(torch.isfinite(layer.weight.grad if hasattr(layer, 'weight') else layer.weights.grad).all()
                    for layer in weight_layers)
        ),
        'ms': elapsed * 1000,
    }


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row['preset'], row['model_type'])
        groups.setdefault(key, []).append(row)

    summary = []
    for (preset, model_type), group_rows in sorted(groups.items()):
        ms_values = [row['ms'] for row in group_rows]
        summary.append({
            'preset': preset,
            'model_type': model_type,
            'stack_depth': group_rows[0]['stack_depth'],
            'pool_every': group_rows[0]['pool_every'],
            'trainable_parameters': group_rows[0]['trainable_parameters'],
            'input_shape': group_rows[0]['input_shape'],
            'output_shape': group_rows[0]['output_shape'],
            'all_finite': all(row['finite'] for row in group_rows),
            'loss_mean': statistics.mean(row['loss'] for row in group_rows),
            'output_std_mean': statistics.mean(row['output_std'] for row in group_rows),
            'input_grad_norm_mean': statistics.mean(row['input_grad_norm'] for row in group_rows),
            'first_weight_grad_norm_mean': statistics.mean(row['first_weight_grad_norm'] for row in group_rows),
            'last_weight_grad_norm_mean': statistics.mean(row['last_weight_grad_norm'] for row in group_rows),
            'first_to_last_grad_ratio_mean': statistics.mean(
                row['first_to_last_grad_ratio'] for row in group_rows
                if row['first_to_last_grad_ratio'] is not None
            ),
            'weight_grad_norm_mean': statistics.mean(row['weight_grad_norm_mean'] for row in group_rows),
            'weight_grad_max_abs_max': max(row['weight_grad_max_abs_max'] for row in group_rows),
            'ms_mean': statistics.mean(ms_values),
            'ms_median': statistics.median(ms_values),
            'ms_mean_without_first_seed': statistics.mean(ms_values[1:]) if len(ms_values) > 1 else ms_values[0],
        })
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    parser.add_argument('--presets', nargs='+', choices=sorted(PRESETS), default=sorted(PRESETS))
    parser.add_argument(
        '--model-types',
        nargs='+',
        choices=['convlogic', 'convnet', 'convnet_residual'],
        default=['convlogic', 'convnet', 'convnet_residual'],
    )
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--in-channels', type=int, default=15)
    parser.add_argument('--height', type=int, default=19)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--kernel-height', type=int, default=3)
    parser.add_argument('--kernel-width', type=int, default=9)
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available.')

    rows = []
    for preset_name in args.presets:
        for model_type in args.model_types:
            for seed in args.seeds:
                rows.append(run_one(seed, args, preset_name, model_type))

    print(json.dumps({
        'device': args.device,
        'seeds': args.seeds,
        'summary': summarize(rows),
        'rows': rows,
    }, indent=2))


if __name__ == '__main__':
    main()
