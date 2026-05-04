import argparse
import json
import statistics

import torch

import difflogic


DEFAULT_SEEDS = [1234, 2026, 3407, 4512, 9001]


def _copy_stack_state(dst, src):
    for dst_layer, src_layer in zip(dst.layers, src.layers):
        dst_layer.leaf_indices.copy_(src_layer.leaf_indices)
        dst_layer.weights.data.copy_(src_layer.weights.data)


def run_seed(seed, device):
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    stack = difflogic.ConvLogicConcatResidualStack(
        in_channels=3,
        out_channels=4,
        num_layers=4,
        kernel_size=3,
        tree_depth=2,
        padding=1,
        pool_every=2,
        pool_kernel_size=(1, 2),
        pool_stride=(1, 2),
        residual_distance=2,
        device=device,
        implementation='cuda' if device == 'cuda' else 'python',
        residual_init=True,
    ).to(device).train()

    x = torch.rand(2, 3, 6, 8, device=device, requires_grad=True)
    y = stack(x)
    assert tuple(y.shape) == (2, 4, 6, 4), y.shape

    loss = y.square().mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(y).all()
    assert torch.isfinite(loss)
    assert torch.isfinite(x.grad).all()
    assert all(torch.isfinite(layer.weights.grad).all() for layer in stack.layers)

    row = {
        'seed': seed,
        'output_shape': list(y.shape),
        'loss': float(loss.detach().cpu()),
        'output_sum': float(y.detach().sum().cpu()),
        'input_grad_norm': float(x.grad.detach().norm().cpu()),
        'first_weight_grad_norm': float(stack.layers[0].weights.grad.detach().norm().cpu()),
        'last_weight_grad_norm': float(stack.layers[-1].weights.grad.detach().norm().cpu()),
        'finite': True,
    }

    if device == 'cuda':
        row.update(run_cuda_compare_seed(seed))
    return row


def run_cuda_compare_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    stack_python = difflogic.ConvLogicConcatResidualStack(
        in_channels=3,
        out_channels=4,
        num_layers=4,
        kernel_size=3,
        tree_depth=2,
        padding=1,
        pool_every=0,
        residual_distance=2,
        device='cuda',
        implementation='python',
        residual_init=True,
    ).train()
    stack_cuda = difflogic.ConvLogicConcatResidualStack(
        in_channels=3,
        out_channels=4,
        num_layers=4,
        kernel_size=3,
        tree_depth=2,
        padding=1,
        pool_every=0,
        residual_distance=2,
        device='cuda',
        implementation='cuda',
        residual_init=True,
    ).train()
    _copy_stack_state(stack_cuda, stack_python)

    x_python = torch.rand(2, 3, 6, 8, device='cuda', requires_grad=True)
    x_cuda = x_python.detach().clone().requires_grad_(True)

    y_python = stack_python(x_python)
    y_cuda = stack_cuda(x_cuda)
    grad_out = torch.randn_like(y_python)

    (y_python * grad_out).sum().backward()
    (y_cuda * grad_out).sum().backward()
    torch.cuda.synchronize()

    output_max_abs_diff = float((y_python - y_cuda).abs().max().detach().cpu())
    grad_x_max_abs_diff = float((x_python.grad - x_cuda.grad).abs().max().detach().cpu())
    grad_weights_max_abs_diff = max(
        float((python_layer.weights.grad - cuda_layer.weights.grad).abs().max().detach().cpu())
        for python_layer, cuda_layer in zip(stack_python.layers, stack_cuda.layers)
    )

    assert output_max_abs_diff < 1e-6, output_max_abs_diff
    assert grad_x_max_abs_diff < 1e-6, grad_x_max_abs_diff
    assert grad_weights_max_abs_diff < 1e-5, grad_weights_max_abs_diff

    return {
        'cuda_output_max_abs_diff_vs_python': output_max_abs_diff,
        'cuda_grad_x_max_abs_diff_vs_python': grad_x_max_abs_diff,
        'cuda_grad_weights_max_abs_diff_vs_python': grad_weights_max_abs_diff,
    }


def summarize(rows):
    return {
        'all_finite': all(row['finite'] for row in rows),
        'output_shape': rows[0]['output_shape'],
        'loss_mean': statistics.mean(row['loss'] for row in rows),
        'input_grad_norm_mean': statistics.mean(row['input_grad_norm'] for row in rows),
        'first_weight_grad_norm_mean': statistics.mean(row['first_weight_grad_norm'] for row in rows),
        'last_weight_grad_norm_mean': statistics.mean(row['last_weight_grad_norm'] for row in rows),
        'cuda_output_max_abs_diff_vs_python': max(
            row.get('cuda_output_max_abs_diff_vs_python', 0.) for row in rows
        ),
        'cuda_grad_x_max_abs_diff_vs_python': max(
            row.get('cuda_grad_x_max_abs_diff_vs_python', 0.) for row in rows
        ),
        'cuda_grad_weights_max_abs_diff_vs_python': max(
            row.get('cuda_grad_weights_max_abs_diff_vs_python', 0.) for row in rows
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available.')

    rows = [run_seed(seed, args.device) for seed in args.seeds]
    print(json.dumps({
        'device': args.device,
        'seeds': args.seeds,
        'summary': summarize(rows),
        'rows': rows,
    }, indent=2))


if __name__ == '__main__':
    main()
