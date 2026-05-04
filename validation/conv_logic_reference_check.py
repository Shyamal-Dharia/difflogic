import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F

import difflogic
from difflogic.functional import bin_op_s


DEFAULT_SEEDS = [1234, 2026, 3407, 4512, 9001]


def _slow_gate(a, b, w):
    return bin_op_s(a.reshape(1), b.reshape(1), w).reshape(())


def slow_conv_logic_tree(layer, x):
    batch_size, _, height, width = x.shape
    out_h = layer._conv_out_size(height, layer.kernel_size[0], layer.stride[0], layer.padding[0], layer.dilation[0])
    out_w = layer._conv_out_size(width, layer.kernel_size[1], layer.stride[1], layer.padding[1], layer.dilation[1])

    patches = F.unfold(
        x,
        kernel_size=layer.kernel_size,
        dilation=layer.dilation,
        padding=layer.padding,
        stride=layer.stride,
    ).transpose(1, 2)

    if layer.training:
        weights = F.softmax(layer.weights, dim=-1).to(x.dtype)
    else:
        weights = F.one_hot(layer.weights.argmax(-1), 16).to(x.dtype)

    outputs = []
    for batch in range(batch_size):
        position_outputs = []
        for position in range(out_h * out_w):
            channel_outputs = []
            for out_channel in range(layer.out_channels):
                values = [patches[batch, position, idx] for idx in layer.leaf_indices[out_channel]]
                gate_offset = 0
                for _ in range(layer.tree_depth):
                    next_values = []
                    for gate in range(len(values) // 2):
                        a = values[2 * gate]
                        b = values[2 * gate + 1]
                        w = weights[out_channel, gate_offset + gate]
                        next_values.append(_slow_gate(a, b, w))
                    gate_offset += len(values) // 2
                    values = next_values
                channel_outputs.append(values[0])
            position_outputs.append(torch.stack(channel_outputs))
        outputs.append(torch.stack(position_outputs))

    return torch.stack(outputs).reshape(batch_size, out_h, out_w, layer.out_channels).permute(0, 3, 1, 2).contiguous()


def run_seed(seed, device):
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    layer = difflogic.ConvLogicTreeLayer(
        in_channels=2,
        out_channels=3,
        kernel_size=3,
        tree_depth=2,
        padding=1,
        device=device,
    ).train()
    x = torch.rand(2, 2, 4, 5, device=device, requires_grad=True)

    y = layer(x)
    y_slow = slow_conv_logic_tree(layer, x)
    train_max_abs_diff = float((y - y_slow).abs().max().detach().cpu())
    assert train_max_abs_diff < 1e-6, train_max_abs_diff

    loss = y.mean()
    loss.backward()
    assert x.grad is not None
    assert layer.weights.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.weights.grad).all()

    layer.eval()
    with torch.no_grad():
        y_eval = layer(x.detach())
        y_eval_slow = slow_conv_logic_tree(layer, x.detach())
    eval_max_abs_diff = float((y_eval - y_eval_slow).abs().max().detach().cpu())
    assert eval_max_abs_diff < 1e-6, eval_max_abs_diff

    pool = difflogic.LogicORPool2d(kernel_size=2, stride=2)
    pool_x = torch.rand(2, 3, 8, 8, device=device)
    pool_y = pool(pool_x)
    pool_expected = F.max_pool2d(pool_x, kernel_size=2, stride=2)
    pool_max_abs_diff = float((pool_y - pool_expected).abs().max().detach().cpu())
    assert pool_max_abs_diff == 0., pool_max_abs_diff

    if device == 'cuda':
        torch.cuda.synchronize()

    row = {
        'seed': seed,
        'train_max_abs_diff': train_max_abs_diff,
        'eval_max_abs_diff': eval_max_abs_diff,
        'pool_max_abs_diff': pool_max_abs_diff,
        'train_output_sum': float(y.detach().sum().cpu()),
        'eval_output_sum': float(y_eval.detach().sum().cpu()),
        'weight_grad_sum': float(layer.weights.grad.detach().sum().cpu()),
    }
    if device == 'cuda':
        row.update(run_cuda_forward_seed(seed))
    return row


def run_cuda_forward_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    layer_python = difflogic.ConvLogicTreeLayer(
        in_channels=2,
        out_channels=3,
        kernel_size=3,
        tree_depth=2,
        padding=1,
        device='cuda',
        implementation='python',
    )
    layer_cuda = difflogic.ConvLogicTreeLayer(
        in_channels=2,
        out_channels=3,
        kernel_size=3,
        tree_depth=2,
        padding=1,
        device='cuda',
        implementation='cuda',
    )
    layer_cuda.leaf_indices.copy_(layer_python.leaf_indices)
    layer_cuda.weights.data.copy_(layer_python.weights.data)

    x = torch.rand(2, 2, 4, 5, device='cuda')

    layer_python.train()
    layer_cuda.train()
    with torch.no_grad():
        y_python_train = layer_python(x)
        y_cuda_train = layer_cuda(x)
    cuda_train_max_abs_diff = float((y_python_train - y_cuda_train).abs().max().cpu())
    assert cuda_train_max_abs_diff < 1e-6, cuda_train_max_abs_diff

    x_requires_grad = x.detach().requires_grad_(True)
    try:
        layer_cuda(x_requires_grad)
        raise AssertionError('CUDA ConvLogicTreeLayer training should fail while backward is not implemented.')
    except RuntimeError as err:
        assert 'forward-only' in str(err), str(err)

    layer_python.eval()
    layer_cuda.eval()
    with torch.no_grad():
        y_python_eval = layer_python(x)
        y_cuda_eval = layer_cuda(x)
    cuda_eval_max_abs_diff = float((y_python_eval - y_cuda_eval).abs().max().cpu())
    assert cuda_eval_max_abs_diff < 1e-6, cuda_eval_max_abs_diff

    torch.cuda.synchronize()

    return {
        'cuda_train_max_abs_diff_vs_python': cuda_train_max_abs_diff,
        'cuda_eval_max_abs_diff_vs_python': cuda_eval_max_abs_diff,
        'cuda_train_output_sum': float(y_cuda_train.sum().cpu()),
        'cuda_eval_output_sum': float(y_cuda_eval.sum().cpu()),
    }


def run_timing(seeds, device):
    rows = []
    for seed in seeds:
        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed_all(seed)
        layer = difflogic.ConvLogicTreeLayer(3, 8, 3, tree_depth=3, padding=1, device=device).train()
        x = torch.rand(8, 3, 16, 16, device=device, requires_grad=True)

        warmup = 5
        iterations = 20
        for _ in range(warmup):
            y = layer(x)
            loss = y.mean()
            loss.backward()
            layer.zero_grad(set_to_none=True)
            x.grad = None

        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            y = layer(x)
            loss = y.mean()
            loss.backward()
            layer.zero_grad(set_to_none=True)
            x.grad = None
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        rows.append(elapsed / iterations * 1000)

    return {
        'warmup': warmup,
        'iterations': iterations,
        'mean_ms_per_iter': statistics.mean(rows),
        'per_seed_ms_per_iter': rows,
    }


def run_cuda_forward_timing(seeds):
    rows = []
    warmup = 10
    iterations = 50
    for seed in seeds:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        layer_python = difflogic.ConvLogicTreeLayer(
            3,
            16,
            3,
            tree_depth=3,
            padding=1,
            device='cuda',
            implementation='python',
        ).eval()
        layer_cuda = difflogic.ConvLogicTreeLayer(
            3,
            16,
            3,
            tree_depth=3,
            padding=1,
            device='cuda',
            implementation='cuda',
        ).eval()
        layer_cuda.leaf_indices.copy_(layer_python.leaf_indices)
        layer_cuda.weights.data.copy_(layer_python.weights.data)
        x = torch.rand(8, 3, 32, 32, device='cuda')

        with torch.no_grad():
            for _ in range(warmup):
                y_python = layer_python(x)
                y_cuda = layer_cuda(x)
        torch.cuda.synchronize()

        with torch.no_grad():
            start = time.perf_counter()
            for _ in range(iterations):
                y_python = layer_python(x)
            torch.cuda.synchronize()
            python_elapsed = time.perf_counter() - start

            start = time.perf_counter()
            for _ in range(iterations):
                y_cuda = layer_cuda(x)
            torch.cuda.synchronize()
            cuda_elapsed = time.perf_counter() - start

        max_abs_diff = float((y_python - y_cuda).abs().max().cpu())
        assert max_abs_diff < 1e-6, max_abs_diff

        python_ms = python_elapsed / iterations * 1000
        cuda_ms = cuda_elapsed / iterations * 1000
        rows.append({
            'seed': seed,
            'python_ms_per_iter': python_ms,
            'cuda_ms_per_iter': cuda_ms,
            'speedup': python_ms / cuda_ms,
            'max_abs_diff': max_abs_diff,
        })

    return {
        'warmup': warmup,
        'iterations': iterations,
        'mean_python_ms_per_iter': statistics.mean(row['python_ms_per_iter'] for row in rows),
        'mean_cuda_ms_per_iter': statistics.mean(row['cuda_ms_per_iter'] for row in rows),
        'mean_speedup': statistics.mean(row['speedup'] for row in rows),
        'rows': rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    parser.add_argument('--skip-timing', action='store_true')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available.')

    rows = [run_seed(seed, args.device) for seed in args.seeds]
    result = {
        'device': args.device,
        'seeds': args.seeds,
        'reference_checks': rows,
    }
    if not args.skip_timing:
        result['timing'] = run_timing(args.seeds, args.device)
        if args.device == 'cuda':
            result['cuda_forward_timing'] = run_cuda_forward_timing(args.seeds)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
