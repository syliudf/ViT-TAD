#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/DeviceGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Custom atomicAdd implementations if needed
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
// atomicAdd for double is already provided in CUDA for compute capability >= 6.0
#else
__device__ double atomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

// For half precision support
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
// atomicAdd for half is already provided in CUDA for compute capability >= 7.0
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)
// Use intrinsics for compute capability 6.x
__device__ __half atomicAdd(__half* address, __half val) {
  unsigned int* address_as_ui = (unsigned int*)((char*)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;
  
  do {
    assumed = old;
    unsigned int old_as_ui = assumed;
    __half_raw assumed_bits;
    
    if ((size_t)address & 2) {
      assumed_bits.x = (old_as_ui >> 16);
      old_as_ui = (old_as_ui & 0xFFFF) | (((unsigned int)__half_as_short(__half_raw{__half_as_short(__half_raw{assumed_bits.x}) + __half_as_short(__half_raw{__half_as_short(val)})})) << 16);
    } else {
      assumed_bits.x = (old_as_ui & 0xFFFF);
      old_as_ui = (old_as_ui & 0xFFFF0000) | ((unsigned int)__half_as_short(__half_raw{__half_as_short(__half_raw{assumed_bits.x}) + __half_as_short(__half_raw{__half_as_short(val)})}));
    }
    
    old = atomicCAS(address_as_ui, assumed, old_as_ui);
  } while (assumed != old);
  
  __half_raw old_bits;
  
  if ((size_t)address & 2) {
    old_bits.x = (old >> 16);
  } else {
    old_bits.x = (old & 0xFFFF);
  }
  
  return __half_raw{old_bits.x};
}
#endif

// Add c10::Half atomicAdd support (works with PyTorch's Half type)
#ifdef __CUDA_ARCH__
__device__ void atomicAdd(c10::Half* address, c10::Half val) {
  unsigned int* base_address = reinterpret_cast<unsigned int*>(
      reinterpret_cast<char*>(address) - (reinterpret_cast<size_t>(address) & 2));
  unsigned int old = *base_address;
  unsigned int assumed;
  unsigned short new_value;
  unsigned short old_value;
  do {
    assumed = old;
    old_value = static_cast<unsigned short>(
        reinterpret_cast<size_t>(address) & 2 ? old >> 16 : old & 0xffff);
    // Convert c10::Half to bits
    unsigned short val_bits;
    memcpy(&val_bits, &val, sizeof(val_bits));
    // Convert bits to float for addition
    float old_float, val_float;
    old_float = __half2float(*reinterpret_cast<__half*>(&old_value));
    val_float = __half2float(*reinterpret_cast<__half*>(&val_bits));
    // Do addition in float and convert back to half
    __half sum_half = __float2half(old_float + val_float);
    memcpy(&new_value, &sum_half, sizeof(new_value));
    // Update with new value
    unsigned int new_assumed_base = reinterpret_cast<size_t>(address) & 2
        ? (assumed & 0xffff) | (static_cast<unsigned int>(new_value) << 16)
        : (assumed & 0xffff0000) | static_cast<unsigned int>(new_value);
    old = atomicCAS(base_address, assumed, new_assumed_base);
  } while (assumed != old);
}
#endif

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = 65000;
  return min(optimal_block_num, max_block_num);
}


template <typename scalar_t>
__global__ void BoundaryPoolingForward(
        const int nthreads,
        const scalar_t* input,
        const scalar_t* segments,
        scalar_t* output,
        const int channels,
        const int tscale,
        const int seg_num) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        const int k = index % seg_num;
        const int c = (index / seg_num) % channels;
        const int n = index / seg_num / channels;
        const int seg_type = c / (channels / 2);
        const int seg_index = n * seg_num * 4 + k * 4 + seg_type * 2;
        scalar_t maxn, val;
        int l = static_cast<int>(segments[seg_index]);
        int r = static_cast<int>(segments[seg_index + 1]);
        l = min(max(0, l), tscale - 1);
        r = min(max(0, r), tscale - 1);
        maxn = input[n * channels * tscale + c * tscale + l];
        for (int i = l + 1; i <= r; i++) {
            val = input[n * channels * tscale + c * tscale + i];
            if (val > maxn) {
                maxn = val;
            }
        }
        output[index] = maxn;
    }
}

template <typename scalar_t>
__global__ void BoundaryPoolingBackward(
        const int nthreads,
        const scalar_t* grad_output,
        const scalar_t* input,
        const scalar_t* segments,
        scalar_t* grad_input,
        const int channels,
        const int tscale,
        const int seg_num) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        const int k = index % seg_num;
        const int c = (index / seg_num) % channels;
        const int n = index / seg_num / channels;
        const int seg_type = c / (channels / 2);
        const int seg_index = n * seg_num * 4 + k * 4 + seg_type * 2;
        scalar_t maxn, val;
        int argmax;
        int l = static_cast<int>(segments[seg_index]);
        int r = static_cast<int>(segments[seg_index + 1]);
        l = min(max(0, l), tscale - 1);
        r = min(max(0, r), tscale - 1);
        maxn = input[n * channels * tscale + c * tscale + l];
        argmax = l;
        for (int i = l + 1; i <= r; i++) {
            val = input[n * channels * tscale + c * tscale + i];
            if (val > maxn) {
                maxn = val;
                argmax = i;
            }
        }
        scalar_t grad = grad_output[index];
        atomicAdd(grad_input + n * channels * tscale + c * tscale + argmax, grad);
    }
}

int boundary_max_pooling_cuda_forward(
        const at::Tensor& input,
        const at::Tensor& segments,
        const at::Tensor& output) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int tscale = input.size(2);
    const int seg_num = segments.size(1);
    const int output_size = batch_size * channels * seg_num;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "BoundaryMaxPoolingForward", ([&] {

        BoundaryPoolingForward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
            output_size,
            input.data_ptr<scalar_t>(),
            segments.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            channels,
            tscale,
            seg_num);
    }));

    C10_CUDA_CHECK(cudaGetLastError());
    return 1;
}

int boundary_max_pooling_cuda_backward(
        const at::Tensor& grad_output,
        const at::Tensor& input,
        const at::Tensor& segments,
        const at::Tensor& grad_input) {
    const int batch_size = grad_output.size(0);
    const int channels = grad_output.size(1);
    const int tscale = grad_output.size(2);
    const int seg_num = segments.size(1);

    const int output_size = batch_size * channels * seg_num;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "BoundaryMaxPoolingBackward", ([&] {

        BoundaryPoolingBackward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
            output_size,
            grad_output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            segments.data_ptr<scalar_t>(),
            grad_input.data_ptr<scalar_t>(),
            channels,
            tscale,
            seg_num);
    }));

    C10_CUDA_CHECK(cudaGetLastError());
    return 1;
}
