#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <vector>


namespace py = pybind11;

torch::Tensor logic_layer_cuda_forward(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w
);
torch::Tensor logic_layer_cuda_backward_w(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor grad_y
);
torch::Tensor logic_layer_cuda_backward_x(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w,
    torch::Tensor grad_y,
    torch::Tensor given_x_indices_of_y_start,
    torch::Tensor given_x_indices_of_y
);
torch::Tensor logic_layer_cuda_eval(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w
);
std::tuple<torch::Tensor, int> tensor_packbits_cuda(
    torch::Tensor t,
    const int bit_count
);
torch::Tensor groupbitsum(
    torch::Tensor b,
    const int pad_len,
    const int k
);
torch::Tensor conv_logic_tree_cuda_forward(
    torch::Tensor x,
    torch::Tensor leaf_indices,
    torch::Tensor w,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t padding_h,
    const int64_t padding_w,
    const int64_t dilation_h,
    const int64_t dilation_w,
    const int64_t tree_depth
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        [](torch::Tensor x, torch::Tensor a, torch::Tensor b, torch::Tensor w) {
            return logic_layer_cuda_forward(x, a, b, w);
        },
        "logic layer forward (CUDA)");
    m.def(
        "backward_w", [](torch::Tensor x, torch::Tensor a, torch::Tensor b, torch::Tensor grad_y) {
            return logic_layer_cuda_backward_w(x, a, b, grad_y);
        },
        "logic layer backward w (CUDA)");
    m.def(
        "backward_x",
        [](torch::Tensor x, torch::Tensor a, torch::Tensor b, torch::Tensor w, torch::Tensor grad_y, torch::Tensor given_x_indices_of_y_start, torch::Tensor given_x_indices_of_y) {
            return logic_layer_cuda_backward_x(x, a, b, w, grad_y, given_x_indices_of_y_start, given_x_indices_of_y);
        },
        "logic layer backward x (CUDA)");
    m.def(
        "eval",
        [](torch::Tensor x, torch::Tensor a, torch::Tensor b, torch::Tensor w) {
            return logic_layer_cuda_eval(x, a, b, w);
        },
        "logic layer eval (CUDA)");
    m.def(
        "tensor_packbits_cuda",
        [](torch::Tensor t, const int bit_count) {
            return tensor_packbits_cuda(t, bit_count);
        },
        "ltensor_packbits_cuda (CUDA)");
    m.def(
        "groupbitsum",
        [](torch::Tensor b, const int pad_len, const unsigned int k) {
            if (b.size(0) % k != 0) {
                throw py::value_error("in_dim (" + std::to_string(b.size(0)) + ") has to be divisible by k (" + std::to_string(k) + ") but it is not");
            }
            return groupbitsum(b, pad_len, k);
        },
        "groupbitsum (CUDA)");
    m.def(
        "conv_logic_tree_forward",
        [](torch::Tensor x, torch::Tensor leaf_indices, torch::Tensor w, const int64_t kernel_h,
           const int64_t kernel_w, const int64_t stride_h, const int64_t stride_w, const int64_t padding_h,
           const int64_t padding_w, const int64_t dilation_h, const int64_t dilation_w, const int64_t tree_depth) {
            return conv_logic_tree_cuda_forward(
                x,
                leaf_indices,
                w,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                padding_h,
                padding_w,
                dilation_h,
                dilation_w,
                tree_depth);
        },
        "convolutional logic tree forward (CUDA)");
}
