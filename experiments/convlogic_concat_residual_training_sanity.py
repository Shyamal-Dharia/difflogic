import argparse
import json
import statistics
import time

import torch

import difflogic


DEFAULT_SEEDS = [1234, 2026, 3407, 4512, 9001]


class SyntheticEEGConvLogicClassifier(torch.nn.Module):
    def __init__(
            self,
            device='cuda',
            in_channels=15,
            height=19,
            width=512,
            conv_channels=32,
            num_layers=4,
            tree_depth=3,
            pooled_width=8,
            head_out_dim=128,
    ):
        super().__init__()
        implementation = 'cuda' if device == 'cuda' else 'python'
        head_in_dim = conv_channels * pooled_width
        assert head_out_dim * 2 >= head_in_dim, (head_out_dim, head_in_dim)

        self.features = difflogic.ConvLogicConcatResidualStack(
            in_channels=in_channels,
            out_channels=conv_channels,
            num_layers=num_layers,
            kernel_size=(3, 9),
            tree_depth=tree_depth,
            padding=(1, 4),
            pool_every=2,
            pool_kernel_size=(1, 2),
            pool_stride=(1, 2),
            residual_distance=2,
            device=device,
            implementation=implementation,
            residual_init=True,
        )
        self.head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, pooled_width)),
            torch.nn.Flatten(),
            difflogic.LogicLayer(
                in_dim=head_in_dim,
                out_dim=head_out_dim,
                device=device,
                implementation=implementation,
            ),
            difflogic.GroupSum(k=2, device=device),
        )
        self.input_shape = (in_channels, height, width)

    def forward(self, x):
        return self.head(self.features(x))


def make_synthetic_eeg_batch(batch_size, in_channels, height, width, device):
    y = torch.arange(batch_size, device=device) % 2
    x = torch.rand(batch_size, in_channels, height, width, device=device) * 0.4 + 0.3

    h0 = max(0, height // 2 - 2)
    h1 = min(height, height // 2 + 3)
    w0 = width // 3
    w1 = min(width, w0 + max(8, width // 4))
    c1 = min(in_channels, 5)

    pos = y == 1
    neg = ~pos
    x[pos, :c1, h0:h1, w0:w1] = torch.rand(
        int(pos.sum()), c1, h1 - h0, w1 - w0, device=device
    ) * 0.05 + 0.9
    x[neg, :c1, h0:h1, w0:w1] = torch.rand(
        int(neg.sum()), c1, h1 - h0, w1 - w0, device=device
    ) * 0.05 + 0.02

    return x, y


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def grad_norm(model):
    total = 0.
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum().cpu())
    return total ** 0.5


def run_seed(seed, args):
    torch.manual_seed(seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    model = SyntheticEEGConvLogicClassifier(
        device=args.device,
        in_channels=args.in_channels,
        height=args.height,
        width=args.width,
        conv_channels=args.conv_channels,
        num_layers=args.num_layers,
        tree_depth=args.tree_depth,
        pooled_width=args.pooled_width,
        head_out_dim=args.head_out_dim,
    ).to(args.device).train()

    x, y = make_synthetic_eeg_batch(
        args.batch_size,
        args.in_channels,
        args.height,
        args.width,
        args.device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    losses = []
    accuracies = []
    grad_norms = []

    if args.device == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        grad_norms.append(grad_norm(model))
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        accuracies.append(float((logits.argmax(dim=1) == y).float().mean().detach().cpu()))

    if args.device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    with torch.no_grad():
        final_logits = model(x)
        final_loss = torch.nn.functional.cross_entropy(final_logits, y)
        final_accuracy = float((final_logits.argmax(dim=1) == y).float().mean().cpu())

    first_window = losses[:min(5, len(losses))]
    last_window = losses[-min(5, len(losses)):]

    return {
        'seed': seed,
        'trainable_parameters': count_parameters(model),
        'input_shape': [args.batch_size, args.in_channels, args.height, args.width],
        'logits_shape': list(final_logits.shape),
        'initial_loss': losses[0],
        'final_step_loss': losses[-1],
        'final_eval_loss': float(final_loss.detach().cpu()),
        'best_loss': min(losses),
        'first_window_loss_mean': statistics.mean(first_window),
        'last_window_loss_mean': statistics.mean(last_window),
        'loss_drop': losses[0] - losses[-1],
        'window_loss_drop': statistics.mean(first_window) - statistics.mean(last_window),
        'initial_accuracy': accuracies[0],
        'final_step_accuracy': accuracies[-1],
        'final_eval_accuracy': final_accuracy,
        'grad_norm_mean': statistics.mean(grad_norms),
        'grad_norm_max': max(grad_norms),
        'ms_per_step': elapsed * 1000 / args.steps,
    }


def summarize(rows):
    return {
        'all_final_loss_lower_than_initial': all(row['final_step_loss'] < row['initial_loss'] for row in rows),
        'all_last_window_lower_than_first_window': all(
            row['last_window_loss_mean'] < row['first_window_loss_mean'] for row in rows
        ),
        'all_final_accuracy_one': all(row['final_eval_accuracy'] == 1. for row in rows),
        'trainable_parameters': rows[0]['trainable_parameters'],
        'input_shape': rows[0]['input_shape'],
        'logits_shape': rows[0]['logits_shape'],
        'initial_loss_mean': statistics.mean(row['initial_loss'] for row in rows),
        'final_step_loss_mean': statistics.mean(row['final_step_loss'] for row in rows),
        'final_eval_loss_mean': statistics.mean(row['final_eval_loss'] for row in rows),
        'first_window_loss_mean': statistics.mean(row['first_window_loss_mean'] for row in rows),
        'last_window_loss_mean': statistics.mean(row['last_window_loss_mean'] for row in rows),
        'loss_drop_mean': statistics.mean(row['loss_drop'] for row in rows),
        'window_loss_drop_mean': statistics.mean(row['window_loss_drop'] for row in rows),
        'final_eval_accuracy_mean': statistics.mean(row['final_eval_accuracy'] for row in rows),
        'grad_norm_mean': statistics.mean(row['grad_norm_mean'] for row in rows),
        'ms_per_step_mean': statistics.mean(row['ms_per_step'] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    parser.add_argument('--steps', type=int, default=80)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--in-channels', type=int, default=15)
    parser.add_argument('--height', type=int, default=19)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--conv-channels', type=int, default=32)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--tree-depth', type=int, default=3)
    parser.add_argument('--pooled-width', type=int, default=8)
    parser.add_argument('--head-out-dim', type=int, default=128)
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available.')

    rows = [run_seed(seed, args) for seed in args.seeds]
    print(json.dumps({
        'device': args.device,
        'seeds': args.seeds,
        'steps': args.steps,
        'lr': args.lr,
        'summary': summarize(rows),
        'rows': rows,
    }, indent=2))


if __name__ == '__main__':
    main()
