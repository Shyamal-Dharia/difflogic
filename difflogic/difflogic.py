import torch
import difflogic_cuda
import numpy as np
from .functional import bin_op_s, get_unique_connections, GradFactor
from .packbitstensor import PackBitsTensor


########################################################################################################################


def _pair(v):
    if isinstance(v, tuple):
        assert len(v) == 2, v
        return v
    return v, v


class ConvLogicTreeLayer(torch.nn.Module):
    """
    Convolutional differentiable logic layer using a shared local logic gate tree.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            tree_depth: int = 3,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            device: str = 'cuda',
            grad_factor: float = 1.,
            implementation: str = None,
            connections: str = 'random',
            residual_init: bool = False,
    ):
        """
        :param in_channels:   number of input channels
        :param out_channels:  number of output channels
        :param kernel_size:   local spatial kernel size
        :param tree_depth:    depth of the binary logic tree; uses 2 ** tree_depth leaves
        :param stride:        convolution stride
        :param padding:       convolution padding
        :param dilation:      convolution dilation
        :param device:        device for parameters and connection indices
        :param grad_factor:   gradient multiplier applied to the input
        :param implementation: currently only 'python'
        :param connections:   currently only 'random'
        :param residual_init: initialize gates toward the identity operation A
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.tree_depth = tree_depth
        self.num_leaves = 2 ** tree_depth
        self.num_tree_gates = self.num_leaves - 1
        self.device = device
        self.grad_factor = grad_factor
        self.connections = connections

        if implementation is None:
            implementation = 'python'
        self.implementation = implementation
        assert self.implementation == 'python', 'ConvLogicTreeLayer currently only supports implementation="python".'
        assert self.connections == 'random', 'ConvLogicTreeLayer currently only supports connections="random".'
        assert self.tree_depth >= 1, self.tree_depth

        patch_dim = in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.patch_dim = patch_dim
        self.register_buffer('leaf_indices', self.get_leaf_connections(patch_dim, device))

        if residual_init:
            weights = torch.zeros(out_channels, self.num_tree_gates, 16, device=device)
            weights[..., 3] = 5.
        else:
            weights = torch.randn(out_channels, self.num_tree_gates, 16, device=device)
        self.weights = torch.nn.parameter.Parameter(weights)

        self.num_neurons = out_channels
        self.num_weights = out_channels * self.num_tree_gates

    def forward(self, x):
        assert self.implementation == 'python', self.implementation
        assert x.ndim == 4, x.ndim
        assert x.shape[1] == self.in_channels, (x.shape, self.in_channels)

        if self.grad_factor != 1.:
            x = GradFactor.apply(x, self.grad_factor)

        return self.forward_python(x)

    def forward_python(self, x):
        batch_size, _, height, width = x.shape
        out_h = self._conv_out_size(height, self.kernel_size[0], self.stride[0], self.padding[0], self.dilation[0])
        out_w = self._conv_out_size(width, self.kernel_size[1], self.stride[1], self.padding[1], self.dilation[1])
        assert out_h > 0 and out_w > 0, (x.shape, self.kernel_size, self.stride, self.padding, self.dilation)

        patches = torch.nn.functional.unfold(
            x,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        ).transpose(1, 2)

        current = patches[:, :, self.leaf_indices]
        if self.training:
            weights = torch.nn.functional.softmax(self.weights, dim=-1).to(x.dtype)
        else:
            weights = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(x.dtype)

        gate_offset = 0
        for _ in range(self.tree_depth):
            gates_at_level = current.shape[-1] // 2
            a = current[..., 0::2]
            b = current[..., 1::2]
            w = weights[:, gate_offset: gate_offset + gates_at_level]
            current = bin_op_s(a, b, w)
            gate_offset += gates_at_level

        y = current.squeeze(-1).reshape(batch_size, out_h, out_w, self.out_channels)
        return y.permute(0, 3, 1, 2).contiguous()

    def get_leaf_connections(self, patch_dim, device='cuda'):
        if self.num_leaves <= patch_dim:
            indices = [torch.randperm(patch_dim)[:self.num_leaves] for _ in range(self.out_channels)]
            indices = torch.stack(indices, dim=0)
        else:
            indices = torch.randint(0, patch_dim, (self.out_channels, self.num_leaves))
        return indices.to(torch.int64).to(device)

    @staticmethod
    def _conv_out_size(size, kernel_size, stride, padding, dilation):
        return (size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    def extra_repr(self):
        return 'in_channels={}, out_channels={}, kernel_size={}, tree_depth={}, stride={}, padding={}'.format(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.tree_depth,
            self.stride,
            self.padding,
        )


########################################################################################################################


class LogicORPool2d(torch.nn.Module):
    """
    Differentiable OR-style spatial pooling.
    """
    def __init__(self, kernel_size: int = 2, stride: int = None, padding: int = 0, dilation: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation

    def forward(self, x):
        assert not isinstance(x, PackBitsTensor), 'LogicORPool2d does not support PackBitsTensor yet.'
        return torch.nn.functional.max_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

    def extra_repr(self):
        return 'kernel_size={}, stride={}, padding={}'.format(self.kernel_size, self.stride, self.padding)


########################################################################################################################


class LogicLayer(torch.nn.Module):
    """
    The core module for differentiable logic gate networks. Provides a differentiable logic gate layer.
    """
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            device: str = 'cuda',
            grad_factor: float = 1.,
            implementation: str = None,
            connections: str = 'random',
    ):
        """
        :param in_dim:      input dimensionality of the layer
        :param out_dim:     output dimensionality of the layer
        :param device:      device (options: 'cuda' / 'cpu')
        :param grad_factor: for deep models (>6 layers), the grad_factor should be increased (e.g., 2) to avoid vanishing gradients
        :param implementation: implementation to use (options: 'cuda' / 'python'). cuda is around 100x faster than python
        :param connections: method for initializing the connectivity of the logic gate net
        """
        super().__init__()
        self.weights = torch.nn.parameter.Parameter(torch.randn(out_dim, 16, device=device))
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.grad_factor = grad_factor

        """
        The CUDA implementation is the fast implementation. As the name implies, the cuda implementation is only 
        available for device='cuda'. The `python` implementation exists for 2 reasons:
        1. To provide an easy-to-understand implementation of differentiable logic gate networks 
        2. To provide a CPU implementation of differentiable logic gate networks 
        """
        self.implementation = implementation
        if self.implementation is None and device == 'cuda':
            self.implementation = 'cuda'
        elif self.implementation is None and device == 'cpu':
            self.implementation = 'python'
        assert self.implementation in ['cuda', 'python'], self.implementation

        self.connections = connections
        assert self.connections in ['random', 'unique'], self.connections
        self.indices = self.get_connections(self.connections, device)

        if self.implementation == 'cuda':
            """
            Defining additional indices for improving the efficiency of the backward of the CUDA implementation.
            """
            given_x_indices_of_y = [[] for _ in range(in_dim)]
            indices_0_np = self.indices[0].cpu().numpy()
            indices_1_np = self.indices[1].cpu().numpy()
            for y in range(out_dim):
                given_x_indices_of_y[indices_0_np[y]].append(y)
                given_x_indices_of_y[indices_1_np[y]].append(y)
            self.given_x_indices_of_y_start = torch.tensor(
                np.array([0] + [len(g) for g in given_x_indices_of_y]).cumsum(), device=device, dtype=torch.int64)
            self.given_x_indices_of_y = torch.tensor(
                [item for sublist in given_x_indices_of_y for item in sublist], dtype=torch.int64, device=device)

        self.num_neurons = out_dim
        self.num_weights = out_dim

    def forward(self, x):
        if isinstance(x, PackBitsTensor):
            assert not self.training, 'PackBitsTensor is not supported for the differentiable training mode.'
            assert self.device == 'cuda', 'PackBitsTensor is only supported for CUDA, not for {}. ' \
                                          'If you want fast inference on CPU, please use CompiledDiffLogicModel.' \
                                          ''.format(self.device)

        else:
            if self.grad_factor != 1.:
                x = GradFactor.apply(x, self.grad_factor)

        if self.implementation == 'cuda':
            if isinstance(x, PackBitsTensor):
                return self.forward_cuda_eval(x)
            return self.forward_cuda(x)
        elif self.implementation == 'python':
            return self.forward_python(x)
        else:
            raise ValueError(self.implementation)

    def forward_python(self, x):
        assert x.shape[-1] == self.in_dim, (x[0].shape[-1], self.in_dim)

        if self.indices[0].dtype != torch.int64 or self.indices[1].dtype != torch.int64:
            self.indices = self.indices[0].long(), self.indices[1].long()

        a, b = x[..., self.indices[0]], x[..., self.indices[1]]
        if self.training:
            x = bin_op_s(a, b, torch.nn.functional.softmax(self.weights, dim=-1))
        else:
            weights = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(torch.float32)
            x = bin_op_s(a, b, weights)
        return x

    def forward_cuda(self, x):
        if self.training:
            assert x.device.type == 'cuda', x.device
        assert x.ndim == 2, x.ndim

        x = x.transpose(0, 1)
        x = x.contiguous()

        assert x.shape[0] == self.in_dim, (x.shape, self.in_dim)

        a, b = self.indices

        if self.training:
            w = torch.nn.functional.softmax(self.weights, dim=-1).to(x.dtype)
            return LogicLayerCudaFunction.apply(
                x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
            ).transpose(0, 1)
        else:
            w = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(x.dtype)
            with torch.no_grad():
                return LogicLayerCudaFunction.apply(
                    x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
                ).transpose(0, 1)

    def forward_cuda_eval(self, x: PackBitsTensor):
        """
        WARNING: this is an in-place operation.

        :param x:
        :return:
        """
        assert not self.training
        assert isinstance(x, PackBitsTensor)
        assert x.t.shape[0] == self.in_dim, (x.t.shape, self.in_dim)

        a, b = self.indices
        w = self.weights.argmax(-1).to(torch.uint8)
        x.t = difflogic_cuda.eval(x.t, a, b, w)

        return x

    def extra_repr(self):
        return '{}, {}, {}'.format(self.in_dim, self.out_dim, 'train' if self.training else 'eval')

    def get_connections(self, connections, device='cuda'):
        assert self.out_dim * 2 >= self.in_dim, 'The number of neurons ({}) must not be smaller than half of the ' \
                                                'number of inputs ({}) because otherwise not all inputs could be ' \
                                                'used or considered.'.format(self.out_dim, self.in_dim)
        if connections == 'random':
            c = torch.randperm(2 * self.out_dim) % self.in_dim
            c = torch.randperm(self.in_dim)[c]
            c = c.reshape(2, self.out_dim)
            a, b = c[0], c[1]
            a, b = a.to(torch.int64), b.to(torch.int64)
            a, b = a.to(device), b.to(device)
            return a, b
        elif connections == 'unique':
            return get_unique_connections(self.in_dim, self.out_dim, device)
        else:
            raise ValueError(connections)


########################################################################################################################


class GroupSum(torch.nn.Module):
    """
    The GroupSum module.
    """
    def __init__(self, k: int, tau: float = 1., device='cuda'):
        """

        :param k: number of intended real valued outputs, e.g., number of classes
        :param tau: the (softmax) temperature tau. The summed outputs are divided by tau.
        :param device:
        """
        super().__init__()
        self.k = k
        self.tau = tau
        self.device = device

    def forward(self, x):
        if isinstance(x, PackBitsTensor):
            return x.group_sum(self.k)

        assert x.shape[-1] % self.k == 0, (x.shape, self.k)
        return x.reshape(*x.shape[:-1], self.k, x.shape[-1] // self.k).sum(-1) / self.tau

    def extra_repr(self):
        return 'k={}, tau={}'.format(self.k, self.tau)


########################################################################################################################


class LogicLayerCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y):
        ctx.save_for_backward(x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y)
        return difflogic_cuda.forward(x, a, b, w)

    @staticmethod
    def backward(ctx, grad_y):
        x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y = ctx.saved_tensors
        grad_y = grad_y.contiguous()

        grad_w = grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = difflogic_cuda.backward_x(x, a, b, w, grad_y, given_x_indices_of_y_start, given_x_indices_of_y)
        if ctx.needs_input_grad[3]:
            grad_w = difflogic_cuda.backward_w(x, a, b, grad_y)
        return grad_x, None, None, grad_w, None, None, None


########################################################################################################################
