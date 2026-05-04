import argparse
import json
import statistics
import time

import torch

import difflogic
from residual_init_eeg_shape_experiment import DEFAULT_SEEDS, EEGConvLogicStack


PRESETS = {
    'alexnet_like_1m': {
        'stack_depth': 5,
        'pool_every': 1,
        'conv_channels': 833,
        'tree_depth': 4,
    },
    'resnet18_like_1m': {
        'stack_depth': 18,
        'pool_every': 3,
        'conv_channels': 496,
        'tree_depth': 3,
    },
}


class EEGConcatResidualConvLogicStack(torch.nn.Module):
    """
    ConvLogic stack where deeper layers can concatenate an earlier layer output.
    """
    def __init__(
            self,
            in_channels=15,
            conv_channels=16,
            stack_depth=3,
            tree_depth=3,
            kernel_size=(3, 9),
            pool_every=1,
            residual_distance=2,
            structured_connections=False,
            structured_gate_init='current',
            device='cuda',
            residual_init=True,
    ):
        super().__init__()
        implementation = 'cuda' if device == 'cuda' else 'python'
        self.pool_every = pool_every
        self.residual_distance = residual_distance
        self.structured_connections = structured_connections
        self.structured_gate_init = structured_gate_init
        self.skip_from = []
        self.layers = torch.nn.ModuleList()

        cur_channels = in_channels
        for layer_idx in range(stack_depth):
            skip_idx = layer_idx - residual_distance
            use_skip = skip_idx >= 0
            self.skip_from.append(skip_idx if use_skip else None)
            layer_in_channels = cur_channels + conv_channels if use_skip else cur_channels
            self.layers.append(difflogic.ConvLogicTreeLayer(
                layer_in_channels,
                conv_channels,
                kernel_size=kernel_size,
                tree_depth=tree_depth,
                padding=(kernel_size[0] // 2, kernel_size[1] // 2),
                device=device,
                implementation=implementation,
                residual_init=residual_init,
            ))
            if use_skip and structured_connections:
                self._set_structured_leaf_connections(
                    self.layers[-1],
                    current_channels=cur_channels,
                    skip_channels=conv_channels,
                    kernel_size=kernel_size,
                    gate_init=structured_gate_init,
                )
            cur_channels = conv_channels

        self.pool = difflogic.LogicORPool2d(kernel_size=(1, 2), stride=(1, 2))

    def forward(self, x):
        history = []
        for layer_idx, layer in enumerate(self.layers):
            skip_idx = self.skip_from[layer_idx]
            if skip_idx is not None:
                skip = self._match_spatial(history[skip_idx], x)
                x = torch.cat([x, skip], dim=1)
            x = layer(x)
            should_pool = self.pool_every > 0 and (layer_idx + 1) % self.pool_every == 0 and layer_idx < len(self.layers) - 1
            if should_pool:
                x = self.pool(x)
            history.append(x)
        return x

    @staticmethod
    def _match_spatial(skip, current):
        if skip.shape[-2:] == current.shape[-2:]:
            return skip
        skip_h, skip_w = skip.shape[-2:]
        cur_h, cur_w = current.shape[-2:]
        if skip_h >= cur_h and skip_w >= cur_w and skip_h % cur_h == 0 and skip_w % cur_w == 0:
            return torch.nn.functional.max_pool2d(
                skip,
                kernel_size=(skip_h // cur_h, skip_w // cur_w),
                stride=(skip_h // cur_h, skip_w // cur_w),
            )
        return torch.nn.functional.adaptive_max_pool2d(skip, current.shape[-2:])

    @staticmethod
    def _sample_patch_indices(out_channels, patch_dim, count, device):
        if count <= patch_dim:
            rows = [torch.randperm(patch_dim, device=device)[:count] for _ in range(out_channels)]
            return torch.stack(rows, dim=0)
        return torch.randint(0, patch_dim, (out_channels, count), device=device)

    @classmethod
    def _set_structured_leaf_connections(cls, layer, current_channels, skip_channels, kernel_size, gate_init):
        device = layer.leaf_indices.device
        num_leaves = layer.num_leaves
        even_positions = torch.arange(0, num_leaves, 2, device=device)
        odd_positions = torch.arange(1, num_leaves, 2, device=device)

        kernel_area = kernel_size[0] * kernel_size[1]
        current_patch_dim = current_channels * kernel_area
        skip_patch_dim = skip_channels * kernel_area

        current_indices = cls._sample_patch_indices(
            layer.out_channels,
            current_patch_dim,
            len(even_positions),
            device,
        )
        skip_indices = cls._sample_patch_indices(
            layer.out_channels,
            skip_patch_dim,
            len(odd_positions),
            device,
        ) + current_patch_dim

        leaf_indices = torch.empty_like(layer.leaf_indices)
        leaf_indices[:, even_positions] = current_indices.to(torch.int64)
        leaf_indices[:, odd_positions] = skip_indices.to(torch.int64)
        layer.leaf_indices.copy_(leaf_indices)

        if gate_init != 'current':
            gate_op = {
                'skip': 5,
                'or': 7,
            }[gate_init]
            first_level_gates = num_leaves // 2
            with torch.no_grad():
                layer.weights[:, :first_level_gates, :] = 0.
                layer.weights[:, :first_level_gates, gate_op] = 5.


def make_input(seed, shape, device):
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    return torch.rand(*shape, device=device)


def count_model(model):
    conv_layers = [
        layer for layer in model.modules()
        if isinstance(layer, difflogic.ConvLogicTreeLayer)
    ]
    return {
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'conv_weight_parameters': sum(layer.weights.numel() for layer in conv_layers),
        'conv_gates': sum(layer.out_channels * layer.num_tree_gates for layer in conv_layers),
        'conv_layers': len(conv_layers),
        'concat_skips': sum(1 for skip_idx in getattr(model, 'skip_from', []) if skip_idx is not None),
        'structured_skips': (
            sum(1 for skip_idx in getattr(model, 'skip_from', []) if skip_idx is not None)
            if getattr(model, 'structured_connections', False)
            else 0
        ),
    }


def conv_layers(model):
    return [
        layer for layer in model.modules()
        if isinstance(layer, difflogic.ConvLogicTreeLayer)
    ]


def build_model(args, preset, model_type):
    common = {
        'in_channels': args.in_channels,
        'conv_channels': preset['conv_channels'],
        'stack_depth': preset['stack_depth'],
        'tree_depth': preset['tree_depth'],
        'kernel_size': (args.kernel_height, args.kernel_width),
        'pool_every': preset['pool_every'],
        'device': args.device,
        'residual_init': True,
    }
    if model_type == 'sequential':
        return EEGConvLogicStack(**common)
    if model_type == 'concat_residual':
        return EEGConcatResidualConvLogicStack(
            **common,
            residual_distance=args.residual_distance,
        )
    if model_type == 'structured_residual':
        return EEGConcatResidualConvLogicStack(
            **common,
            residual_distance=args.residual_distance,
            structured_connections=True,
        )
    if model_type == 'structured_or_residual':
        return EEGConcatResidualConvLogicStack(
            **common,
            residual_distance=args.residual_distance,
            structured_connections=True,
            structured_gate_init='or',
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

    layers = conv_layers(model)
    grad_norms = [float(layer.weights.grad.detach().norm().cpu()) for layer in layers]
    grad_max_abs = [float(layer.weights.grad.detach().abs().max().cpu()) for layer in layers]
    first_grad = grad_norms[0]
    last_grad = grad_norms[-1]

    return {
        'seed': seed,
        'preset': preset_name,
        'model_type': model_type,
        'stack_depth': preset['stack_depth'],
        'pool_every': preset['pool_every'],
        'residual_distance': (
            args.residual_distance
            if model_type in ['concat_residual', 'structured_residual', 'structured_or_residual']
            else None
        ),
        **count_model(model),
        'input_shape': list(x.shape),
        'output_shape': list(y.shape),
        'structured_gate_init': (
            model.structured_gate_init
            if getattr(model, 'structured_connections', False)
            else None
        ),
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
            and all(torch.isfinite(layer.weights.grad).all() for layer in layers)
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
        ratios = [
            row['first_to_last_grad_ratio'] for row in group_rows
            if row['first_to_last_grad_ratio'] is not None
        ]
        summary.append({
            'preset': preset,
            'model_type': model_type,
            'stack_depth': group_rows[0]['stack_depth'],
            'pool_every': group_rows[0]['pool_every'],
            'residual_distance': group_rows[0]['residual_distance'],
            'structured_gate_init': group_rows[0]['structured_gate_init'],
            'trainable_parameters': group_rows[0]['trainable_parameters'],
            'conv_gates': group_rows[0]['conv_gates'],
            'concat_skips': group_rows[0]['concat_skips'],
            'structured_skips': group_rows[0]['structured_skips'],
            'input_shape': group_rows[0]['input_shape'],
            'output_shape': group_rows[0]['output_shape'],
            'all_finite': all(row['finite'] for row in group_rows),
            'loss_mean': statistics.mean(row['loss'] for row in group_rows),
            'output_std_mean': statistics.mean(row['output_std'] for row in group_rows),
            'input_grad_norm_mean': statistics.mean(row['input_grad_norm'] for row in group_rows),
            'first_weight_grad_norm_mean': statistics.mean(row['first_weight_grad_norm'] for row in group_rows),
            'last_weight_grad_norm_mean': statistics.mean(row['last_weight_grad_norm'] for row in group_rows),
            'first_to_last_grad_ratio_mean': statistics.mean(ratios) if ratios else None,
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
        choices=['sequential', 'concat_residual', 'structured_residual', 'structured_or_residual'],
        default=['sequential', 'concat_residual', 'structured_residual', 'structured_or_residual'],
    )
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--in-channels', type=int, default=15)
    parser.add_argument('--height', type=int, default=19)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--kernel-height', type=int, default=3)
    parser.add_argument('--kernel-width', type=int, default=9)
    parser.add_argument('--residual-distance', type=int, default=2)
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available.')
    if args.residual_distance < 1:
        raise ValueError('--residual-distance must be >= 1')

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
