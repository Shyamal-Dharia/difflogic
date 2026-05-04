import argparse
import json
import time

import torch

import difflogic


class BinaryConvLogicNet(torch.nn.Module):
    """
    Small binary-classification model using convolutional logic layers.
    """
    def __init__(
            self,
            input_channels=1,
            image_size=16,
            thermometer_bits=4,
            conv_channels=(8, 16),
            tree_depth=3,
            hidden_dim=128,
            device='cpu',
    ):
        super().__init__()
        self.encoder = difflogic.ThermometerEncoding(
            num_bits=thermometer_bits,
            value_range=(0., 1.),
        )
        self.conv1 = difflogic.ConvLogicTreeLayer(
            input_channels * thermometer_bits,
            conv_channels[0],
            kernel_size=3,
            tree_depth=tree_depth,
            padding=1,
            device=device,
        )
        self.pool1 = difflogic.LogicORPool2d(2)
        self.conv2 = difflogic.ConvLogicTreeLayer(
            conv_channels[0],
            conv_channels[1],
            kernel_size=3,
            tree_depth=tree_depth,
            padding=1,
            device=device,
        )
        self.pool2 = difflogic.LogicORPool2d(2)
        pooled_size = image_size // 4
        self.flatten = torch.nn.Flatten()
        self.head = difflogic.LogicLayer(
            conv_channels[1] * pooled_size * pooled_size,
            hidden_dim,
            device=device,
            implementation='python',
        )
        self.group_sum = difflogic.GroupSum(k=2)

    def forward(self, x):
        x = self.encoder(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.head(x)
        return self.group_sum(x)


def make_synthetic_binary_images(batch_size, image_size, device):
    x = torch.rand(batch_size, 1, image_size, image_size, device=device)
    top_left = x[:, :, :image_size // 2, :image_size // 2].mean(dim=(1, 2, 3))
    bottom_right = x[:, :, image_size // 2:, image_size // 2:].mean(dim=(1, 2, 3))
    y = (top_left > bottom_right).long()
    return x, y


def run_check(seed, device):
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    batch_size = 16
    image_size = 16
    model = BinaryConvLogicNet(image_size=image_size, device=device).to(device).train()
    x, y = make_synthetic_binary_images(batch_size, image_size, device)

    start = time.perf_counter()
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    assert logits.shape == (batch_size, 2), logits.shape
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    assert model.conv1.weights.grad is not None
    assert model.conv2.weights.grad is not None
    assert model.head.weights.grad is not None
    assert torch.isfinite(model.conv1.weights.grad).all()
    assert torch.isfinite(model.conv2.weights.grad).all()
    assert torch.isfinite(model.head.weights.grad).all()

    model.eval()
    with torch.no_grad():
        eval_logits = model(x)
    assert eval_logits.shape == (batch_size, 2), eval_logits.shape
    assert torch.isfinite(eval_logits).all()

    return {
        'seed': seed,
        'loss': float(loss.detach().cpu()),
        'logits_sum': float(logits.detach().sum().cpu()),
        'eval_logits_sum': float(eval_logits.detach().sum().cpu()),
        'ms': elapsed * 1000,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1234, 2026, 3407, 4512, 9001])
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available.')

    rows = [run_check(seed, args.device) for seed in args.seeds]
    print(json.dumps({'device': args.device, 'rows': rows}, indent=2))


if __name__ == '__main__':
    main()
