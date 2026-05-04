#include <torch/extension.h>

#include <c10/util/Half.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <vector>

#define BACKWARD_W_BATCH_THREADS 32

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                                                 \
    CHECK_CUDA(x);                                                                                                     \
    CHECK_CONTIGUOUS(x)

// adapted from https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans)                                                                                                 \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(const cudaError_t code, const char *const file, const int line, const bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#ifdef DIFFLOGIC_CUDA_DEBUG_SYNC
#define DIFFLOGIC_CUDA_DEBUG_SYNC_IF_ENABLED() gpuErrchk(cudaDeviceSynchronize())
#else
#define DIFFLOGIC_CUDA_DEBUG_SYNC_IF_ENABLED()
#endif

#define CUDA_KERNEL_CHECK()                                                                                            \
    do {                                                                                                               \
        gpuErrchk(cudaPeekAtLastError());                                                                              \
        DIFFLOGIC_CUDA_DEBUG_SYNC_IF_ENABLED();                                                                        \
    } while (0)

template <typename T> T ceil_div(const T x, const T y) { return x / y + !!(x % y); }


/**********************************************************************************************************************/


template <typename T> struct AtomicFPOp;

template <> struct AtomicFPOp<at::Half> {
    template <typename func_t> inline __device__ at::Half operator()(at::Half *address, at::Half val, const func_t &func) {
        unsigned int *address_as_ui = (unsigned int *)((char *)address - ((size_t)address & 2));
        unsigned int old = *address_as_ui;
        unsigned int assumed;

        at::Half hsum;
        do {
            assumed = old;
            hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
            hsum = func(hsum, val);
            old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16) : (old & 0xffff0000) | hsum.x;
            old = atomicCAS(address_as_ui, assumed, old);
        } while (assumed != old);
        hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
        return hsum;
    }
};

static inline __device__ at::Half gpuAtomicAdd(at::Half *address, at::Half val) {
#if defined(USE_ROCM) || ((defined(CUDA_VERSION) && CUDA_VERSION < 10000) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)))

    unsigned int *aligned = (unsigned int *)((size_t)address - ((size_t)address & 2));
    unsigned int old = *aligned;
    unsigned int assumed;
    do {
        assumed = old;
        unsigned short old_as_us = (unsigned short)((size_t)address & 2 ? old >> 16 : old & 0xffff);
        __half sum = c10::Half(__ushort_as_half(old_as_us)) + c10::Half(__float2half((float)val));
        unsigned short sum_as_us = __half_as_ushort(sum);
        unsigned int sum_as_ui = (size_t)address & 2 ? (sum_as_us << 16) | (old & 0xffff) : (old & 0xffff0000) | sum_as_us;
        old = atomicCAS(aligned, assumed, sum_as_ui);
    } while (assumed != old);
    unsigned short old_as_us = (unsigned short)((size_t)address & 2 ? old >> 16 : old & 0xffff);
    return c10::Half((__half_raw)__ushort_as_half(old_as_us));
#else
    return atomicAdd(reinterpret_cast<__half *>(address), val);
#endif
}

static inline __device__ float gpuAtomicAdd(float *address, float val) { return atomicAdd(address, val); }

static inline __device__ double gpuAtomicAdd(double *address, double val) { return atomicAdd(address, val); }


/**********************************************************************************************************************/
/**  CONVOLUTIONAL LOGIC TREE FORWARD  *******************************************************************************/
/**********************************************************************************************************************/


constexpr int MAX_CONV_LOGIC_TREE_LEAVES = 64;
constexpr int MAX_CONV_LOGIC_TREE_NODES = 2 * MAX_CONV_LOGIC_TREE_LEAVES - 1;
constexpr int MAX_CONV_LOGIC_TREE_LEVELS = 7;

template <typename scalar_t>
__device__ __forceinline__ scalar_t bin_op_soft(const scalar_t a_, const scalar_t b_, const scalar_t *w_) {
    return (
         ((w_[1] * (a_ * b_)
         + w_[2] * (a_ - a_ * b_))
        + (w_[3] * a_
         + w_[4] * (b_ - a_ * b_)))
       + ((w_[5] * b_
         + w_[6] * (a_ + b_ - static_cast<scalar_t>(2) * a_ * b_))
        + (w_[7] * (a_ + b_ - a_ * b_)
         + w_[8] * (static_cast<scalar_t>(1) - (a_ + b_ - a_ * b_)))))
      + (((w_[9] * (static_cast<scalar_t>(1) - (a_ + b_ - static_cast<scalar_t>(2) * a_ * b_))
         + w_[10] * (static_cast<scalar_t>(1) - b_)) +
          (w_[11] * (static_cast<scalar_t>(1) - b_ + a_ * b_)
         + w_[12] * (static_cast<scalar_t>(1) - a_))) +
          (w_[13] * (static_cast<scalar_t>(1) - a_ + a_ * b_)
         + w_[14] * (static_cast<scalar_t>(1) - a_ * b_)
         + w_[15])
    );
}

template <typename scalar_t>
__device__ __forceinline__ void bin_op_partials(
    const scalar_t a_,
    const scalar_t b_,
    const scalar_t *w_,
    scalar_t &dy_da,
    scalar_t &dy_db
) {
    dy_da = (
        (w_[1] * b_
       + w_[2] * (static_cast<scalar_t>(1) - b_)
       + w_[3]) +
        (w_[4] * -b_
       + w_[6] * (static_cast<scalar_t>(1) - static_cast<scalar_t>(2) * b_)
       + w_[7] * (static_cast<scalar_t>(1) - b_)))
     + ((w_[8] * (b_ - static_cast<scalar_t>(1))
       + w_[9] * (static_cast<scalar_t>(2) * b_ - static_cast<scalar_t>(1))
       + w_[11] * b_)
     + (-w_[12]
       + w_[13] * (b_ - static_cast<scalar_t>(1))
       + w_[14] * -b_)
    );

    dy_db = (
         (w_[1] * a_
        + w_[2] * -a_
        + w_[4] * (static_cast<scalar_t>(1) - a_))
       + (w_[5]
        + w_[6] * (static_cast<scalar_t>(1) - static_cast<scalar_t>(2) * a_)
        + w_[7] * (static_cast<scalar_t>(1) - a_)))
      + ((w_[8] * (a_ - static_cast<scalar_t>(1))
        + w_[9] * (static_cast<scalar_t>(2) * a_ - static_cast<scalar_t>(1))
        - w_[10])
       + (w_[11] * (a_ - static_cast<scalar_t>(1))
        + w_[13] * a_
        + w_[14] * -a_)
    );
}

template <typename scalar_t>
__device__ __forceinline__ void bin_op_values(const scalar_t a_, const scalar_t b_, scalar_t *op_values) {
    const scalar_t ab = a_ * b_;
    op_values[0] = static_cast<scalar_t>(0);
    op_values[1] = ab;
    op_values[2] = a_ - ab;
    op_values[3] = a_;
    op_values[4] = b_ - ab;
    op_values[5] = b_;
    op_values[6] = a_ + b_ - static_cast<scalar_t>(2) * ab;
    op_values[7] = a_ + b_ - ab;
    op_values[8] = static_cast<scalar_t>(1) - (a_ + b_ - ab);
    op_values[9] = static_cast<scalar_t>(1) - (a_ + b_ - static_cast<scalar_t>(2) * ab);
    op_values[10] = static_cast<scalar_t>(1) - b_;
    op_values[11] = static_cast<scalar_t>(1) - b_ + ab;
    op_values[12] = static_cast<scalar_t>(1) - a_;
    op_values[13] = static_cast<scalar_t>(1) - a_ + ab;
    op_values[14] = static_cast<scalar_t>(1) - ab;
    op_values[15] = static_cast<scalar_t>(1);
}

template <typename scalar_t>
__global__ void conv_logic_tree_cuda_forward_kernel(
    torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> x,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> leaf_indices,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> w,
    torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> y,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t padding_h,
    const int64_t padding_w,
    const int64_t dilation_h,
    const int64_t dilation_w,
    const int64_t tree_depth
) {
    const int64_t total = y.size(0) * y.size(1) * y.size(2) * y.size(3);
    const int64_t out_w = y.size(3);
    const int64_t out_h = y.size(2);
    const int64_t out_channels = y.size(1);
    const int64_t in_h = x.size(2);
    const int64_t in_w = x.size(3);

    for (int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
         linear_idx < total;
         linear_idx += blockDim.x * gridDim.x) {
        const int64_t ow = linear_idx % out_w;
        const int64_t oh = (linear_idx / out_w) % out_h;
        const int64_t oc = (linear_idx / (out_w * out_h)) % out_channels;
        const int64_t batch = linear_idx / (out_w * out_h * out_channels);

        scalar_t values[MAX_CONV_LOGIC_TREE_LEAVES];
        const int64_t num_leaves = static_cast<int64_t>(1) << tree_depth;

        for (int64_t leaf = 0; leaf < num_leaves; ++leaf) {
            const int64_t patch_idx = leaf_indices[oc][leaf];
            const int64_t kw = patch_idx % kernel_w;
            const int64_t kh = (patch_idx / kernel_w) % kernel_h;
            const int64_t ic = patch_idx / (kernel_h * kernel_w);
            const int64_t ih = oh * stride_h + kh * dilation_h - padding_h;
            const int64_t iw = ow * stride_w + kw * dilation_w - padding_w;

            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                values[leaf] = x[batch][ic][ih][iw];
            } else {
                values[leaf] = static_cast<scalar_t>(0);
            }
        }

        int64_t gate_offset = 0;
        int64_t current_count = num_leaves;
        for (int64_t level = 0; level < tree_depth; ++level) {
            const int64_t gates_at_level = current_count / 2;
            for (int64_t gate = 0; gate < gates_at_level; ++gate) {
                const scalar_t a_ = values[2 * gate];
                const scalar_t b_ = values[2 * gate + 1];
                const auto w_ = w[oc][gate_offset + gate];
                values[gate] = bin_op_soft(a_, b_, &w_[0]);
            }
            gate_offset += gates_at_level;
            current_count = gates_at_level;
        }

        y[batch][oc][oh][ow] = values[0];
    }
}

template <typename scalar_t>
__global__ void conv_logic_tree_cuda_backward_kernel(
    torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> x,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> leaf_indices,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> w,
    torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> grad_y,
    torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> grad_x,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> grad_w,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t padding_h,
    const int64_t padding_w,
    const int64_t dilation_h,
    const int64_t dilation_w,
    const int64_t tree_depth
) {
    const int64_t total = grad_y.size(0) * grad_y.size(1) * grad_y.size(2) * grad_y.size(3);
    const int64_t out_w = grad_y.size(3);
    const int64_t out_h = grad_y.size(2);
    const int64_t out_channels = grad_y.size(1);
    const int64_t in_h = x.size(2);
    const int64_t in_w = x.size(3);

    for (int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
         linear_idx < total;
         linear_idx += blockDim.x * gridDim.x) {
        const int64_t ow = linear_idx % out_w;
        const int64_t oh = (linear_idx / out_w) % out_h;
        const int64_t oc = (linear_idx / (out_w * out_h)) % out_channels;
        const int64_t batch = linear_idx / (out_w * out_h * out_channels);

        scalar_t node_values[MAX_CONV_LOGIC_TREE_NODES];
        scalar_t node_grads[MAX_CONV_LOGIC_TREE_NODES];
        int64_t level_starts[MAX_CONV_LOGIC_TREE_LEVELS];
        int64_t level_counts[MAX_CONV_LOGIC_TREE_LEVELS];

        for (int idx = 0; idx < MAX_CONV_LOGIC_TREE_NODES; ++idx) {
            node_grads[idx] = static_cast<scalar_t>(0);
        }

        const int64_t num_leaves = static_cast<int64_t>(1) << tree_depth;
        int64_t offset = 0;
        int64_t count = num_leaves;
        for (int64_t level = 0; level <= tree_depth; ++level) {
            level_starts[level] = offset;
            level_counts[level] = count;
            offset += count;
            count /= 2;
        }

        for (int64_t leaf = 0; leaf < num_leaves; ++leaf) {
            const int64_t patch_idx = leaf_indices[oc][leaf];
            const int64_t kw = patch_idx % kernel_w;
            const int64_t kh = (patch_idx / kernel_w) % kernel_h;
            const int64_t ic = patch_idx / (kernel_h * kernel_w);
            const int64_t ih = oh * stride_h + kh * dilation_h - padding_h;
            const int64_t iw = ow * stride_w + kw * dilation_w - padding_w;

            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                node_values[leaf] = x[batch][ic][ih][iw];
            } else {
                node_values[leaf] = static_cast<scalar_t>(0);
            }
        }

        int64_t gate_offset = 0;
        for (int64_t level = 0; level < tree_depth; ++level) {
            const int64_t in_start = level_starts[level];
            const int64_t out_start = level_starts[level + 1];
            const int64_t gates_at_level = level_counts[level + 1];
            for (int64_t gate = 0; gate < gates_at_level; ++gate) {
                const scalar_t a_ = node_values[in_start + 2 * gate];
                const scalar_t b_ = node_values[in_start + 2 * gate + 1];
                const auto w_ = w[oc][gate_offset + gate];
                node_values[out_start + gate] = bin_op_soft(a_, b_, &w_[0]);
            }
            gate_offset += gates_at_level;
        }

        node_grads[level_starts[tree_depth]] = grad_y[batch][oc][oh][ow];

        int64_t gates_before_level = 0;
        for (int64_t level = 0; level < tree_depth - 1; ++level) {
            gates_before_level += level_counts[level + 1];
        }
        for (int64_t level = tree_depth - 1; level >= 0; --level) {
            const int64_t in_start = level_starts[level];
            const int64_t out_start = level_starts[level + 1];
            const int64_t gates_at_level = level_counts[level + 1];
            const int64_t gate_offset_level = gates_before_level;

            for (int64_t gate = 0; gate < gates_at_level; ++gate) {
                const int64_t gate_idx = gate_offset_level + gate;
                const scalar_t a_ = node_values[in_start + 2 * gate];
                const scalar_t b_ = node_values[in_start + 2 * gate + 1];
                const auto w_ = w[oc][gate_idx];
                const scalar_t grad_out = node_grads[out_start + gate];

                scalar_t op_values[16];
                bin_op_values(a_, b_, op_values);
                for (int op = 0; op < 16; ++op) {
                    gpuAtomicAdd(&grad_w[oc][gate_idx][op], grad_out * op_values[op]);
                }

                scalar_t dy_da;
                scalar_t dy_db;
                bin_op_partials(a_, b_, &w_[0], dy_da, dy_db);
                node_grads[in_start + 2 * gate] += grad_out * dy_da;
                node_grads[in_start + 2 * gate + 1] += grad_out * dy_db;
            }

            if (level > 0) {
                gates_before_level -= level_counts[level];
            }
        }

        for (int64_t leaf = 0; leaf < num_leaves; ++leaf) {
            const int64_t patch_idx = leaf_indices[oc][leaf];
            const int64_t kw = patch_idx % kernel_w;
            const int64_t kh = (patch_idx / kernel_w) % kernel_h;
            const int64_t ic = patch_idx / (kernel_h * kernel_w);
            const int64_t ih = oh * stride_h + kh * dilation_h - padding_h;
            const int64_t iw = ow * stride_w + kw * dilation_w - padding_w;

            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                gpuAtomicAdd(&grad_x[batch][ic][ih][iw], node_grads[leaf]);
            }
        }
    }
}

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
) {
    CHECK_INPUT(x);
    CHECK_INPUT(leaf_indices);
    CHECK_INPUT(w);

    TORCH_CHECK(x.dim() == 4, "x must be a 4D NCHW tensor");
    TORCH_CHECK(leaf_indices.dim() == 2, "leaf_indices must be a 2D tensor");
    TORCH_CHECK(w.dim() == 3, "w must be a 3D tensor");
    TORCH_CHECK(tree_depth >= 1, "tree_depth must be >= 1");

    const int64_t num_leaves = static_cast<int64_t>(1) << tree_depth;
    TORCH_CHECK(
        num_leaves <= MAX_CONV_LOGIC_TREE_LEAVES,
        "CUDA ConvLogicTreeLayer currently supports tree_depth <= 6");
    TORCH_CHECK(leaf_indices.size(1) == num_leaves, "leaf_indices second dim must equal 2 ** tree_depth");
    TORCH_CHECK(w.size(0) == leaf_indices.size(0), "w and leaf_indices must agree on out_channels");
    TORCH_CHECK(w.size(1) == num_leaves - 1, "w second dim must equal 2 ** tree_depth - 1");
    TORCH_CHECK(w.size(2) == 16, "w last dim must be 16");

    const int64_t batch_size = x.size(0);
    const int64_t out_channels = leaf_indices.size(0);
    const int64_t in_h = x.size(2);
    const int64_t in_w = x.size(3);
    const int64_t out_h = (in_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int64_t out_w = (in_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    TORCH_CHECK(out_h > 0 && out_w > 0, "convolution output size must be positive");

    auto y = torch::empty({batch_size, out_channels, out_h, out_w}, torch::dtype(x.dtype()).device(x.device()));

    const int threads_per_block = 256;
    const int64_t total = batch_size * out_channels * out_h * out_w;
    const int blocks_per_grid = min(static_cast<int64_t>(65535), ceil_div(total, static_cast<int64_t>(threads_per_block)));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.type(), "conv_logic_tree_cuda_forward", ([&] {
                           conv_logic_tree_cuda_forward_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                               x.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                               leaf_indices.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
                               w.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                               y.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                               kernel_h,
                               kernel_w,
                               stride_h,
                               stride_w,
                               padding_h,
                               padding_w,
                               dilation_h,
                               dilation_w,
                               tree_depth);
                       }));

    CUDA_KERNEL_CHECK();

    return y;
}

std::tuple<torch::Tensor, torch::Tensor> conv_logic_tree_cuda_backward(
    torch::Tensor x,
    torch::Tensor leaf_indices,
    torch::Tensor w,
    torch::Tensor grad_y,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t padding_h,
    const int64_t padding_w,
    const int64_t dilation_h,
    const int64_t dilation_w,
    const int64_t tree_depth
) {
    CHECK_INPUT(x);
    CHECK_INPUT(leaf_indices);
    CHECK_INPUT(w);
    CHECK_INPUT(grad_y);

    TORCH_CHECK(x.dim() == 4, "x must be a 4D NCHW tensor");
    TORCH_CHECK(leaf_indices.dim() == 2, "leaf_indices must be a 2D tensor");
    TORCH_CHECK(w.dim() == 3, "w must be a 3D tensor");
    TORCH_CHECK(grad_y.dim() == 4, "grad_y must be a 4D NCHW tensor");
    TORCH_CHECK(tree_depth >= 1, "tree_depth must be >= 1");

    const int64_t num_leaves = static_cast<int64_t>(1) << tree_depth;
    TORCH_CHECK(
        num_leaves <= MAX_CONV_LOGIC_TREE_LEAVES,
        "CUDA ConvLogicTreeLayer currently supports tree_depth <= 6");
    TORCH_CHECK(tree_depth < MAX_CONV_LOGIC_TREE_LEVELS, "tree_depth exceeds local level buffer");
    TORCH_CHECK(leaf_indices.size(1) == num_leaves, "leaf_indices second dim must equal 2 ** tree_depth");
    TORCH_CHECK(w.size(0) == leaf_indices.size(0), "w and leaf_indices must agree on out_channels");
    TORCH_CHECK(w.size(1) == num_leaves - 1, "w second dim must equal 2 ** tree_depth - 1");
    TORCH_CHECK(w.size(2) == 16, "w last dim must be 16");

    const int64_t batch_size = x.size(0);
    const int64_t out_channels = leaf_indices.size(0);
    const int64_t in_h = x.size(2);
    const int64_t in_w = x.size(3);
    const int64_t out_h = (in_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int64_t out_w = (in_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    TORCH_CHECK(grad_y.size(0) == batch_size, "grad_y batch size must match x");
    TORCH_CHECK(grad_y.size(1) == out_channels, "grad_y channels must match leaf_indices");
    TORCH_CHECK(grad_y.size(2) == out_h, "grad_y height is wrong");
    TORCH_CHECK(grad_y.size(3) == out_w, "grad_y width is wrong");

    auto grad_x = torch::zeros_like(x);
    auto grad_w = torch::zeros_like(w);

    const int threads_per_block = 256;
    const int64_t total = batch_size * out_channels * out_h * out_w;
    const int blocks_per_grid = min(static_cast<int64_t>(65535), ceil_div(total, static_cast<int64_t>(threads_per_block)));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.type(), "conv_logic_tree_cuda_backward", ([&] {
                           conv_logic_tree_cuda_backward_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                               x.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                               leaf_indices.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
                               w.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                               grad_y.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                               grad_x.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                               grad_w.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                               kernel_h,
                               kernel_w,
                               stride_h,
                               stride_w,
                               padding_h,
                               padding_w,
                               dilation_h,
                               dilation_w,
                               tree_depth);
                       }));

    CUDA_KERNEL_CHECK();

    return {grad_x, grad_w};
}




/**********************************************************************************************************************/
/**  TRAINING MODE  ***************************************************************************************************/
/**********************************************************************************************************************/


template <typename scalar_t>
__global__ void logic_layer_cuda_forward_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> x,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> a,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> w,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> y
) {

    for (  // batch dim
        auto row = blockIdx.x * blockDim.x + threadIdx.x;
        row < y.size(1);
        row += blockDim.x * gridDim.x
    ) {
        for (  // neuron dim
            auto col = blockIdx.y * blockDim.y + threadIdx.y;
            col < y.size(0);
            col += blockDim.y * gridDim.y
        ) {

            const auto idx_a = a[col];
            const auto idx_b = b[col];
            const auto a_ = x[idx_a][row];
            const auto b_ = x[idx_b][row];

            const auto w_ = w[col];

            y[col][row] = (
                 ((w_[1] * (a_ * b_)
                 + w_[2] * (a_ - a_ * b_))
                + (w_[3] * a_
                 + w_[4] * (b_ - a_ * b_)))
               + ((w_[5] * b_
                 + w_[6] * (a_ + b_ - static_cast<scalar_t>(2) * a_ * b_))
                + (w_[7] * (a_ + b_ - a_ * b_)
                 + w_[8] * (static_cast<scalar_t>(1) - (a_ + b_ - a_ * b_)))))
              + (((w_[9] * (static_cast<scalar_t>(1) - (a_ + b_ - static_cast<scalar_t>(2) * a_ * b_))
                 + w_[10] * (static_cast<scalar_t>(1) - b_)) +
                  (w_[11] * (static_cast<scalar_t>(1) - b_ + a_ * b_)
                 + w_[12] * (static_cast<scalar_t>(1) - a_))) +
                  (w_[13] * (static_cast<scalar_t>(1) - a_ + a_ * b_)
                 + w_[14] * (static_cast<scalar_t>(1) - a_ * b_)
                 + w_[15])
            );
    }}
}


template <typename scalar_t>
__global__ void
logic_layer_cuda_backward_w_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> x,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> a,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_y,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> grad_w_
) {

    const auto row_ = blockIdx.x * blockDim.x + threadIdx.x;

    for (  // neuron dim
        auto col = blockIdx.y * blockDim.y + threadIdx.y;
        col < grad_y.size(0);
        col += blockDim.y * gridDim.y
    ) {
        const auto idx_a = a[col];
        const auto idx_b = b[col];
        scalar_t grad_w_local_1 = 0;
        scalar_t grad_w_local_3 = 0;
        scalar_t grad_w_local_5 = 0;
        scalar_t grad_w_local_15 = 0;
        for (int row = row_; row < grad_y.size(1); row += BACKWARD_W_BATCH_THREADS) {  // batch dim
            const auto a_ = x[idx_a][row];
            const auto b_ = x[idx_b][row];
            const auto grad_y_ = grad_y[col][row];

            // compute grad_w
            grad_w_local_1 += (a_ * b_) * grad_y_;
            grad_w_local_3 += a_ * grad_y_;
            grad_w_local_5 += b_ * grad_y_;
            grad_w_local_15 += grad_y_;
        }

        grad_w_[col][row_][0] = grad_w_local_1;
        grad_w_[col][row_][1] = grad_w_local_3;
        grad_w_[col][row_][2] = grad_w_local_5;
        grad_w_[col][row_][3] = grad_w_local_15;
    }
}


template <typename scalar_t>
__global__ void
logic_layer_cuda_backward_x_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> x,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> a,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> w,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_y,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_x,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> given_x_indices_of_y_start,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> given_x_indices_of_y
) {

    for (  // batch dim
        auto row = blockIdx.x * blockDim.x + threadIdx.x;
        row < grad_x.size(1);
        row += blockDim.x * gridDim.x
    ) {
        for (  // neuron dim
            auto col = blockIdx.y * blockDim.y + threadIdx.y;
            col < grad_x.size(0);
            col += blockDim.y * gridDim.y
        ) {

            scalar_t grad_x_ = 0;

            const auto start = given_x_indices_of_y_start[col];
            const auto end = given_x_indices_of_y_start[col + 1];

            for (int cur = start; cur < end; ++cur) {
                const auto idx_y = given_x_indices_of_y[cur];
                const auto idx_a = a[idx_y];
                const auto idx_b = b[idx_y];
                const auto grad_y_ = grad_y[idx_y][row];
                const auto idx_is_a = idx_a == col;

                // compute grad_x
                if (idx_is_a) {
                    const auto b_ = x[idx_b][row];
                    const auto dy_dx = (
                        (w[idx_y][1] * b_
                       + w[idx_y][2] * (static_cast<scalar_t>(1) - b_)
                       + w[idx_y][3]) +
                        (w[idx_y][4] * -b_
                       + w[idx_y][6] * (static_cast<scalar_t>(1) - static_cast<scalar_t>(2) * b_)
                       + w[idx_y][7] * (static_cast<scalar_t>(1) - b_)))
                     + ((w[idx_y][8] * (b_ - static_cast<scalar_t>(1))
                       + w[idx_y][9] * (static_cast<scalar_t>(2) * b_ - static_cast<scalar_t>(1))
                       + w[idx_y][11] * b_)
                     + (-w[idx_y][12]
                       + w[idx_y][13] * (b_ - static_cast<scalar_t>(1))
                       + w[idx_y][14] * -b_)
                    );
                    grad_x_ += dy_dx * grad_y_;
                } else {
                    const auto a_ = x[idx_a][row];
                    const auto dy_dx = (
                         (w[idx_y][1] * a_
                        + w[idx_y][2] * -a_
                        + w[idx_y][4] * (static_cast<scalar_t>(1) - a_))
                       + (w[idx_y][5]
                        + w[idx_y][6] * (static_cast<scalar_t>(1) - static_cast<scalar_t>(2) * a_)
                        + w[idx_y][7] * (static_cast<scalar_t>(1) - a_)))
                      + ((w[idx_y][8] * (a_ - static_cast<scalar_t>(1))
                        + w[idx_y][9] * (static_cast<scalar_t>(2) * a_ - static_cast<scalar_t>(1))
                        - w[idx_y][10])
                       + (w[idx_y][11] * (a_ - static_cast<scalar_t>(1))
                        + w[idx_y][13] * a_
                        + w[idx_y][14] * -a_)
                    );
                    grad_x_ += dy_dx * grad_y_;
                }
            }
            grad_x[col][row] = grad_x_;
    }}
}


torch::Tensor logic_layer_cuda_forward(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w
) {
    CHECK_INPUT(x);
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(w);

    const auto batch_size = x.size(1);
    const auto in_size = x.size(0);
    const auto out_size = w.size(0);

    auto y = torch::empty({out_size, batch_size}, torch::dtype(x.dtype()).device(x.device()));

    dim3 threads_per_block(32, 32);

    const dim3 blocks_per_grid(
        min(static_cast<int64_t>(65535), ceil_div(batch_size, static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(out_size, static_cast<int64_t>(threads_per_block.y)))
    );

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.type(), "logic_layer_cuda_forward", ([&] {
                           logic_layer_cuda_forward_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                               x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               a.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                               b.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                               w.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>()
                           );
                       }));

    CUDA_KERNEL_CHECK();

    return y;
}


torch::Tensor logic_layer_cuda_backward_w(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor grad_y
) {
    CHECK_INPUT(x);
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(grad_y);


    const auto batch_size = x.size(1);
    const auto in_size = x.size(0);
    const auto out_size = grad_y.size(0);

    auto grad_w_4 = torch::empty({out_size, BACKWARD_W_BATCH_THREADS, 4}, torch::dtype(x.dtype()).device(x.device()));

    dim3 threads_per_block(BACKWARD_W_BATCH_THREADS, 1024 / BACKWARD_W_BATCH_THREADS);

    const dim3 blocks_per_grid(
        1,
        min(static_cast<int64_t>(65535), ceil_div(out_size, static_cast<int64_t>(threads_per_block.y)))
    );

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.type(), "logic_layer_cuda_backward_w", ([&] {
                           logic_layer_cuda_backward_w_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                               x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               a.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                               b.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                               grad_y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               grad_w_4.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());
                       }));

    CUDA_KERNEL_CHECK();

    const auto grad_w_components = grad_w_4.sum(1);
    const auto grad_w_ab = grad_w_components.index({torch::indexing::Slice(), 0});
    const auto grad_w_a = grad_w_components.index({torch::indexing::Slice(), 1});
    const auto grad_w_b = grad_w_components.index({torch::indexing::Slice(), 2});
    const auto grad_w_ = grad_w_components.index({torch::indexing::Slice(), 3});

    const auto grad_w = torch::stack({
        torch::zeros({out_size}, torch::dtype(x.dtype()).device(x.device())),
        grad_w_ab,
        grad_w_a - grad_w_ab,
        grad_w_a,
        grad_w_b - grad_w_ab,
        grad_w_b,
        grad_w_a + grad_w_b - grad_w_ab - grad_w_ab,
        grad_w_a + grad_w_b - grad_w_ab,
        grad_w_ - grad_w_a - grad_w_b + grad_w_ab,
        grad_w_ - grad_w_a - grad_w_b + grad_w_ab + grad_w_ab,
        grad_w_ - grad_w_b,
        grad_w_ - grad_w_b + grad_w_ab,
        grad_w_ - grad_w_a,
        grad_w_ - grad_w_a + grad_w_ab,
        grad_w_ - grad_w_ab,
        grad_w_,
    }, 1);


    return grad_w;
}


torch::Tensor logic_layer_cuda_backward_x(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w,
    torch::Tensor grad_y,
    torch::Tensor given_x_indices_of_y_start,
    torch::Tensor given_x_indices_of_y
) {
    CHECK_INPUT(x);
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(w);
    CHECK_INPUT(grad_y);
    CHECK_INPUT(given_x_indices_of_y_start);
    CHECK_INPUT(given_x_indices_of_y);

    auto grad_x = torch::empty_like(x);

    dim3 threads_per_block(32, 32);

    const dim3 blocks_per_grid(
        min(static_cast<int64_t>(65535), ceil_div(x.size(1), static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(x.size(0), static_cast<int64_t>(threads_per_block.y)))
    );

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.type(), "logic_layer_cuda_backward_x", ([&] {
                           logic_layer_cuda_backward_x_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                               x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               a.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                               b.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                               w.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               grad_y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               grad_x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                               given_x_indices_of_y_start.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                               given_x_indices_of_y.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>()
                           );
                       }));

    CUDA_KERNEL_CHECK();

    return grad_x;
}


/**********************************************************************************************************************/
/**  INFERENCE MODE  **************************************************************************************************/
/**********************************************************************************************************************/


// | id | Operator             | AB=00 | AB=01 | AB=10 | AB=11 |
// |----|----------------------|-------|-------|-------|-------|
// | 0  | 0                    | 0     | 0     | 0     | 0     |
// | 1  | A and B              | 0     | 0     | 0     | 1     |
// | 2  | not(A implies B)     | 0     | 0     | 1     | 0     |
// | 3  | A                    | 0     | 0     | 1     | 1     |
// | 4  | not(B implies A)     | 0     | 1     | 0     | 0     |
// | 5  | B                    | 0     | 1     | 0     | 1     |
// | 6  | A xor B              | 0     | 1     | 1     | 0     |
// | 7  | A or B               | 0     | 1     | 1     | 1     |
// | 8  | not(A or B)          | 1     | 0     | 0     | 0     |
// | 9  | not(A xor B)         | 1     | 0     | 0     | 1     |
// | 10 | not(B)               | 1     | 0     | 1     | 0     |
// | 11 | B implies A          | 1     | 0     | 1     | 1     |
// | 12 | not(A)               | 1     | 1     | 0     | 0     |
// | 13 | A implies B          | 1     | 1     | 0     | 1     |
// | 14 | not(A and B)         | 1     | 1     | 1     | 0     |
// | 15 | 1                    | 1     | 1     | 1     | 1     |

template <typename T> __device__ __forceinline__ T bin_op_eval(const T a_, const T b_, const int op_idx) {
    switch (op_idx) {
    case 0:
        return static_cast<T>(0);
    case 1:
        return a_ & b_;
    case 2:
        return a_ & ~b_;
    case 3:
        return a_;
    case 4:
        return b_ & ~a_;
    case 5:
        return b_;
    case 6:
        return a_ ^ b_;
    case 7:
        return a_ | b_;
    case 8:
        return ~(a_ | b_);
    case 9:
        return ~(a_ ^ b_);
    case 10:
        return ~b_;
    case 11:
        return ~b_ | a_;
    case 12:
        return ~a_;
    case 13:
        return ~a_ | b_;
    case 14:
        return ~(a_ & b_);
    default:
        return ~static_cast<T>(0);
    }
}

template <typename scalar_t>
__global__ void logic_layer_cuda_eval_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> x,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> a,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor64<uint8_t, 1, torch::RestrictPtrTraits> w,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> y
) {
    for (  // batch dim
        auto row = blockIdx.x * blockDim.x + threadIdx.x;
        row < y.size(1);
        row += blockDim.x * gridDim.x
    ) {
        for (  // neuron dim
            auto col = blockIdx.y * blockDim.y + threadIdx.y;
            col < y.size(0);
            col += blockDim.y * gridDim.y
        ) {

            const auto idx_a = a[col];
            const auto idx_b = b[col];
            const auto a_ = x[idx_a][row];
            const auto b_ = x[idx_b][row];
            const auto w_ = w[col];
            y[col][row] = bin_op_eval(a_, b_, w_);
        }
    }
}

torch::Tensor logic_layer_cuda_eval(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w
) {
    CHECK_INPUT(x);
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(w);

    const auto batch_size = x.size(1);
    const auto in_size = x.size(0);
    const auto out_size = w.size(0);

    auto y = torch::empty({out_size, batch_size}, torch::dtype(x.dtype()).device(x.device()));

    dim3 threads_per_block(32, 32);

    const dim3 blocks_per_grid(
        min(static_cast<int64_t>(65535), ceil_div(x.size(1), static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(x.size(0), static_cast<int64_t>(threads_per_block.y)))
    );

    AT_DISPATCH_INTEGRAL_TYPES(x.type(), "logic_layer_cuda_eval_kernel", ([&] {
                                   logic_layer_cuda_eval_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                                       x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       a.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                                       b.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
                                       w.packed_accessor64<uint8_t, 1, torch::RestrictPtrTraits>(),
                                       y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>()
                                   );
                               }));

    CUDA_KERNEL_CHECK();

    return y;
}


/**********************************************************************************************************************/


template <typename scalar_t>
__global__ void tensor_packbits_cuda_kernel(
    torch::PackedTensorAccessor32<bool, 2, torch::RestrictPtrTraits> t,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> b
) {

    for (  // neuron in b and t
        auto row = blockIdx.y * blockDim.y + threadIdx.y;
        row < t.size(0);
        row += blockDim.y * gridDim.y
    ) {
        for (  // batch in b
            auto col = blockIdx.x * blockDim.x + threadIdx.x;
            col < b.size(1);
            col += blockDim.x * gridDim.x
        ) {

            typedef typename std::make_unsigned<scalar_t>::type unsigned_scalar_t;
            union {
                unsigned_scalar_t unsigned_scalar;
                scalar_t signed_scalar;
            } val;
            constexpr int bit_count = std::numeric_limits<unsigned_scalar_t>::digits;
            val.signed_scalar = b[row][col];
            for (unsigned int i = 0; i < bit_count; ++i) {
                const auto t_col = bit_count * col + i;
                if (t_col < t.size(1)) {    
                    const unsigned_scalar_t bit_mask = static_cast<unsigned_scalar_t>(t[row][t_col]) << i;
                    val.unsigned_scalar = val.unsigned_scalar | bit_mask;
                }
            }
            b[row][col] = val.signed_scalar;
        }
    }
}

std::tuple<torch::Tensor, int> tensor_packbits_cuda(
    torch::Tensor t,
    const int bit_count
) {
    CHECK_INPUT(t);

    const auto batch_in_size = t.size(1);
    const auto batch_out_size = ceil_div(batch_in_size, static_cast<int64_t>(bit_count));
    const auto out_size = t.size(0);
    const auto pad_len = (bit_count - batch_in_size % bit_count) % bit_count;

    dim3 threads_per_block(32, 32);

    const dim3 blocks_per_grid(
        min(static_cast<int64_t>(65535), ceil_div(batch_out_size, static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(out_size, static_cast<int64_t>(threads_per_block.y)))
    );

    auto dispatch_type = [bit_count]() {
        switch (bit_count) {
        case 8:
            return torch::kInt8;
        case 16:
            return torch::kInt16;
        case 32:
            return torch::kInt32;
        case 64:
            return torch::kInt64;
        default:
            throw std::invalid_argument("`bit_count` has to be in { 8, 16, 32, 64 }");
        }
    }();
    auto b = torch::zeros({out_size, batch_out_size}, torch::dtype(dispatch_type).device(t.device()));

    AT_DISPATCH_INTEGRAL_TYPES(b.type(), "tensor_packbits_cuda_kernel", ([&] {
                                   tensor_packbits_cuda_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(t.packed_accessor32<bool, 2, torch::RestrictPtrTraits>(),
                                                                                                                            b.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
                               }));
    CUDA_KERNEL_CHECK();

    return {b, pad_len};
}


/**********************************************************************************************************************/


template <typename scalar_t>
__global__ void groupbitsum_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> t
) {

    for (  // class in t
        auto row = blockIdx.y * blockDim.y + threadIdx.y;
        row < t.size(0);
        row += blockDim.y * gridDim.y
    ) {
        for (  // batch in t
            auto col = blockIdx.x * blockDim.x + threadIdx.x;
            col < t.size(1);
            col += blockDim.x * gridDim.x
        ) {

            typedef typename std::make_unsigned<scalar_t>::type unsigned_scalar_t;
            union scalar_t_ {
                unsigned_scalar_t unsigned_scalar;
                scalar_t signed_scalar;
            };
            constexpr int bit_count = std::numeric_limits<unsigned_scalar_t>::digits;
            int res = 0;
            const auto class_size = b.size(0) / t.size(0);
            for (int i = 0; i < class_size; ++i) {
                const scalar_t_ val = {.signed_scalar = b[row * class_size + i][col / bit_count]};
                const unsigned_scalar_t bit_mask = static_cast<unsigned_scalar_t>(1) << static_cast<uint32_t>(col % bit_count);
                res += !!(val.unsigned_scalar & bit_mask);
            }
            t[row][col] = res;
        }
    }
}

torch::Tensor groupbitsum(
    torch::Tensor b,
    const int pad_len,
    const int k
) {
    CHECK_INPUT(b);

    const int bit_count = 8 * b.element_size();

    const auto batch_in_size = b.size(1);
    const auto in_size = b.size(0);
    const auto batch_out_size = batch_in_size * bit_count - pad_len;
    const auto out_size = static_cast<int64_t>(k);
    assert(in_size % k == 0);

    dim3 threads_per_block(32, 32);

    const dim3 blocks_per_grid(
        min(static_cast<int64_t>(65535), ceil_div(batch_out_size, static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(out_size, static_cast<int64_t>(threads_per_block.y)))
    );

    auto t = torch::empty({out_size, batch_out_size}, torch::dtype(torch::kInt32).device(b.device()));

    AT_DISPATCH_INTEGRAL_TYPES(b.type(), "groupbitsum_kernel", ([&] {
                                   groupbitsum_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                                        b.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                        t.packed_accessor32<int, 2, torch::RestrictPtrTraits>()
                                        );
                               }));
    CUDA_KERNEL_CHECK();

    return t.transpose(0, 1).contiguous();
}


/**********************************************************************************************************************/
