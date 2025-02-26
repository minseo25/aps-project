#pragma once

#include <vector>
#include <cstdio>

#include "half.hpp" /* for half on CPU ('half_cpu') */
#include "cuda_fp16.h" /* for half on GPU ('half') */

using std::vector;

/* Namespace for half on CPU ('half_cpu') */
typedef half_float::half half_cpu;
using namespace half_float::literal; 

/* Macro for checking CUDA errors */
#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)


// #define FP16 /* [Advanced] Uncomment this line only for FP16 */


/* [Tensor Structure] */
struct Tensor {
  size_t ndim = 0;
  size_t shape[4] = {1, 1, 1, 1};
  float *buf = nullptr; // host buffer
  float *d_buf = nullptr; // device buffer

  Tensor(const vector<size_t> &shape_);
  Tensor(const vector<size_t> &shape_, float *buf_);
  ~Tensor();

  void to_device();
  void to_host();
  size_t num_elem();
  void to_device_with_shape(float *buf_, size_t shape1, size_t shape2, size_t shape3, size_t shape4);
};

typedef Tensor Parameter;
typedef Tensor Activation;