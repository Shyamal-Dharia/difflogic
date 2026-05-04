import argparse
import json
import statistics
import time

import torch

import difflogic


DEFAULT_SEEDS = [1234, 2026, 3407, 4512, 9001]


class EEGConvLogicStack(torch.nn.Module):
    """
    Convolutional logic stack for already-thermometer-encoded EEG-like inputs.
    """
    def __init__(
            self,
            in_channels=15,
            conv_channels=16,
            stack_depth=3,
            tree_depth=3,
            kernel_size=(3, 9),
            pool_every=1,
            device='cuda',
            residual_init=False,
    ):
        super().__init__()
        implementation = 'cuda' if device == 'cuda' else 'python'
        layers = []
        cur_channels = in_channels
        for layer_idx in range(stack_depth):
            layers.append(difflogic.ConvLogicTreeLayer(
                cur_channels,
                conv_channels,
                kernel_size=kernel_size,
                tree_depth=tree_depth,
                padding=(kernel_size[0] // 2, kernel_size[1] // 2),
                device=device,
                implementation=implementation,
                residual_init=residual_init,
            ))
            cur_channels = conv_channels
            should_pool = pool_every > 0 and (layer_idx + 1) % pool_every == 0 and layer_idx < stack_depth - 1
            if should_pool:
                layers.append(difflogic.LogicORPool2d(kernel_size=(1, 2), stride=(1, 2)))
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def count_model(model):
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    conv_layers = [layer for layer in model.layers if isinstance(layer, difflogic.ConvLogicTreeLayer)]
    conv_gates = sum(layer.out_channels * layer.num_tree_gates for layer in conv_layers)
    conv_weight_parameters = sum(layer.weights.numel() for layer in conv_layers)
    return {
        'trainable_parameters': trainable_parameters,
        'conv_weight_parameters': conv_weight_parameters,
        'conv_gates': conv_gates,
        'conv_layers': len(conv_layers),
    }


def make_input(seed, shape, device):
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    return torch.rand(*shape, device=device)


def run_one(seed, args, stack_depth, residual_init):
    torch.manual_seed(seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    model = EEGConvLogicStack(
        in_channels=args.in_channels,
        conv_channels=args.conv_channels,
        stack_depth=stack_depth,
        tree_depth=args.tree_depth,
        kernel_size=(args.kernel_height, args.kernel_width),
        pool_every=args.pool_every,
        device=args.device,
        residual_init=residual_init,
    ).to(args.device).train()

    x = make_input(seed, (args.batch_size, args.in_channels, args.height, args.width), args.device)
    x.requires_grad_(True)

    if args.device == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    y = model(x)
    loss = y.square().mean()
    loss.backward()
    if args.device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    grad_norms = []
    grad_max_abs = []
    for layer in model.layers:
        if isinstance(layer, difflogic.ConvLogicTreeLayer):
            grad = layer.weights.grad
            grad_norms.append(float(grad.norm().detach().cpu()))
            grad_max_abs.append(float(grad.abs().max().detach().cpu()))

    counts = count_model(model)
    return {
        'seed': seed,
        'stack_depth': stack_depth,
        'residual_init': residual_init,
        **counts,
        'input_shape': list(x.shape),
        'output_shape': list(y.shape),
        'loss': float(loss.detach().cpu()),
        'output_mean': float(y.detach().mean().cpu()),
        'output_std': float(y.detach().std(unbiased=False).cpu()),
        'input_grad_norm': float(x.grad.detach().norm().cpu()),
        'input_grad_max_abs': float(x.grad.detach().abs().max().cpu()),
        'weight_grad_norm_mean': statistics.mean(grad_norms),
        'weight_grad_norm_min': min(grad_norms),
        'weight_grad_norm_max': max(grad_norms),
        'weight_grad_max_abs_max': max(grad_max_abs),
        'finite': bool(
            torch.isfinite(y).all()
            and torch.isfinite(loss)
            and torch.isfinite(x.grad).all()
            and all(torch.isfinite(layer.weights.grad).all() for layer in model.layers
                    if isinstance(layer, difflogic.ConvLogicTreeLayer))
        ),
        'ms': elapsed * 1000,
    }


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row['stack_depth'], row['residual_init'])
        groups.setdefault(key, []).append(row)

    summary = []
    for (stack_depth, residual_init), group_rows in sorted(groups.items()):
        ms_values = [row['ms'] for row in group_rows]
        summary.append({
            'stack_depth': stack_depth,
            'residual_init': residual_init,
            'trainable_parameters': group_rows[0]['trainable_parameters'],
            'conv_weight_parameters': group_rows[0]['conv_weight_parameters'],
            'conv_gates': group_rows[0]['conv_gates'],
            'input_shape': group_rows[0]['input_shape'],
            'output_shape': group_rows[0]['output_shape'],
            'all_finite': all(row['finite'] for row in group_rows),
            'loss_mean': statistics.mean(row['loss'] for row in group_rows),
            'output_std_mean': statistics.mean(row['output_std'] for row in group_rows),
            'input_grad_norm_mean': statistics.mean(row['input_grad_norm'] for row in group_rows),
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
    parser.add_argument('--stack-depths', nargs='+', type=int, default=[2, 3, 4])
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--in-channels', type=int, default=15)
    parser.add_argument('--height', type=int, default=19)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--conv-channels', type=int, default=16)
    parser.add_argument('--tree-depth', type=int, default=3)
    parser.add_argument('--kernel-height', type=int, default=3)
    parser.add_argument('--kernel-width', type=int, default=9)
    parser.add_argument('--pool-every', type=int, default=1)
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available.')

    rows = []
    for stack_depth in args.stack_depths:
        for residual_init in [False, True]:
            for seed in args.seeds:
                rows.append(run_one(seed, args, stack_depth, residual_init))

    print(json.dumps({
        'device': args.device,
        'seeds': args.seeds,
        'summary': summarize(rows),
        'rows': rows,
    }, indent=2))


if __name__ == '__main__':
    main()
