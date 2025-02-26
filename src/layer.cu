#include "layer.h"

#define THREADS_PER_BLOCK 256
#define BLOCK_SIZE 16

/* Conv1D 
 * @param [in1]  in: [BS, os, C * K] => [BS * os, C * K] 로 해석 가능
 * @param [in2]   w: [OC, C * K]
 * @param [in3]   b: [OC]
 * @param [out] out: [BS, os, OC]
 *    
 *    In this model, K is 3, 5, 7, or 9, 
 *    with stride = 1, pad = 0, dilation = 1.
 *    The formula for the output sequence length:
 *      os = (in - K + 2 * pad) / stride + 1
 *         = (s - K + 2 * 0) / 1 + 1
 *         = s - K + 1
 *
 * 'BS' is the batch size
 * 'C' is the input channel size
 * 's' is the input sequence length
 * 'OC' is the output channel size
 * 'os' is the output sequence length
 * 'K' is the kernel (or filter) size
 */
__global__ void __launch_bounds__(BLOCK_SIZE * BLOCK_SIZE) Conv1DKernel(float *in, float *w, float *b, float *out,
                            size_t BSOS, size_t CK, size_t OC) {
  // out idx
  int row = blockIdx.x * BLOCK_SIZE + threadIdx.x / BLOCK_SIZE;
  int col = blockIdx.y * BLOCK_SIZE + threadIdx.x % BLOCK_SIZE;
  // block idx
  int cRow = blockIdx.x;
  int cCol = blockIdx.y;
  // thread idx in block
  int threadRow = threadIdx.x / BLOCK_SIZE;
  int threadCol = threadIdx.x % BLOCK_SIZE;

  in += cRow * BLOCK_SIZE * CK;
  w += cCol * BLOCK_SIZE * CK;
  out += cRow * BLOCK_SIZE * OC + cCol * BLOCK_SIZE;

  float val = 0.f;
  
  __shared__ float in_shared[BLOCK_SIZE * BLOCK_SIZE];
  __shared__ float w_shared[BLOCK_SIZE * BLOCK_SIZE];

  for (size_t blk_i = 0; blk_i < CK; blk_i += BLOCK_SIZE) {
    in_shared[threadRow * BLOCK_SIZE + threadCol] = 
        (cRow * BLOCK_SIZE + threadRow >= BSOS || blk_i + threadCol >= CK) ? 
        0.0f : in[threadRow * CK + threadCol];
    w_shared[threadRow * BLOCK_SIZE + threadCol] = 
        (cCol * BLOCK_SIZE + threadRow >= OC || blk_i + threadCol >= CK) ? 
        0.0f : w[threadRow * CK + threadCol];
    __syncthreads();

    in += BLOCK_SIZE;
    w += BLOCK_SIZE;

    for (size_t dot_i = 0; dot_i < BLOCK_SIZE; dot_i++) {
      val += in_shared[threadRow * BLOCK_SIZE + dot_i] * 
             w_shared[threadCol * BLOCK_SIZE + dot_i];
    }
    __syncthreads();
  }

  if (row < BSOS && col < OC) {
    out[threadRow * OC + threadCol] = val + b[col];
  }
}
void Conv1D_CUDA(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t BS = in->shape[0];
  size_t os = in->shape[1];
  size_t CK = in->shape[2];
  size_t OC = w->shape[0];

  dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);
  dim3 gridDim((BS * os + BLOCK_SIZE - 1) / BLOCK_SIZE, (OC + BLOCK_SIZE - 1) / BLOCK_SIZE);
  Conv1DKernel<<<gridDim, blockDim>>>(in->d_buf, w->d_buf, b->d_buf, out->d_buf, BS * os, CK, OC);
  CHECK_CUDA(cudaDeviceSynchronize());
}

/* ReLU
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
/* ReLU CUDA kernel */
__global__ void ReLUKernel(float *inout, size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    inout[i] = inout[i] > 0 ? inout[i] : 0;
  }
}
/* ReLU using CUDA */
void ReLU_CUDA(Tensor *inout) {
  size_t N = inout->num_elem();

  dim3 blockDim(THREADS_PER_BLOCK);
  dim3 gridDim((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1);
  ReLUKernel<<<gridDim, blockDim>>>(inout->d_buf, N);
  CHECK_CUDA(cudaDeviceSynchronize());
}

/* GetMax
 * @param [in]   in: [BS, s, C]
 * @param [out] out: [BS, C]
 *    
 *    This layer is to get the max value along the sequence dim.
 *    The formula for this layer: out = max(in, dim=-1)
 * 
 * 'BS' is the batch size
 * 's' is the sequence length
 * 'C' is the channel size
 */
__global__ void GetMax_Kernel(float *in, float *out, size_t BS, size_t s, size_t C) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= BS * C) return;
  
  size_t bs = idx / C;
  size_t c = idx % C;

  float max_val = in[bs * C * s + c];
  for (size_t j = 1; j < s; j++) {
    float val = in[bs * C * s + j * C + c];
    max_val = val > max_val ? val : max_val;
  }
  out[bs * C + c] = max_val;
}
void GetMax_CUDA(Tensor *in, Tensor *out) {
  size_t BS = in->shape[0];
  size_t s = in->shape[1];
  size_t C = in->shape[2];

  dim3 blockDim(THREADS_PER_BLOCK);
  dim3 gridDim((BS * C + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1);
  GetMax_Kernel<<<gridDim, blockDim>>>(in->d_buf, out->d_buf, BS, s, C);
  CHECK_CUDA(cudaDeviceSynchronize());
}

/* Concat
 * @param [in1] in1: [BS, N1]
 * @param [in2] in2: [BS, N2]
 * @param [in3] in3: [BS, N3]
 * @param [in4] in4: [BS, N4]
 * @param [out] out: [BS, N1 + N2 + N3 + N4]
 * 'N1', 'N2', 'N3', and 'N4' are the num of elems in the tensors.
 */
void Concat(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, 
            Tensor *out) {
  size_t BS = in1->shape[0];
  size_t N1 = in1->shape[1];
  size_t N2 = in2->shape[1];
  size_t N3 = in3->shape[1];
  size_t N4 = in4->shape[1];

  for (size_t bs = 0; bs < BS; bs++) {
    for (size_t i = 0; i < N1; i++) {
      out->buf[bs * (N1 + N2 + N3 + N4) + i] = in1->buf[bs * N1 + i];
    }
    for (size_t i = 0; i < N2; i++) {
      out->buf[bs * (N1 + N2 + N3 + N4) + N1 + i] = in2->buf[bs * N2 + i];
    }
    for (size_t i = 0; i < N3; i++) {
      out->buf[bs * (N1 + N2 + N3 + N4) + N1 + N2 + i] = 
        in3->buf[bs * N3 + i];
    }
    for (size_t i = 0; i < N4; i++) {
      out->buf[bs * (N1 + N2 + N3 + N4) + N1 + N2 + N3 + i] = 
        in4->buf[bs * N4 + i];
    }
  }
}

/* Concat CUDA kernel */
__global__ void ConcatKernel(float *in1, float *in2, float *in3, float *in4, float *out,
                            size_t BS, size_t N1, size_t N2, size_t N3, size_t N4) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_N = N1 + N2 + N3 + N4;
  
  for (size_t i = idx; i < BS * total_N; i += blockDim.x * gridDim.x) {
    size_t bs = i / total_N;
    size_t offset = i % total_N;
    
    if (offset < N1) {
        out[i] = in1[bs * N1 + offset];
    } else if (offset < N1 + N2) {
        out[i] = in2[bs * N2 + (offset - N1)];
    } else if (offset < N1 + N2 + N3) {
        out[i] = in3[bs * N3 + (offset - N1 - N2)];
    } else {
        out[i] = in4[bs * N4 + (offset - N1 - N2 - N3)];
    }
  }
}
/* Concat using CUDA */
void Concat_CUDA(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, 
                  Tensor *out) {
  size_t BS = in1->shape[0];
  size_t N1 = in1->shape[1];
  size_t N2 = in2->shape[1];
  size_t N3 = in3->shape[1];
  size_t N4 = in4->shape[1];

  dim3 blockDim(THREADS_PER_BLOCK);
  dim3 gridDim((BS * (N1 + N2 + N3 + N4) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1);
  ConcatKernel<<<gridDim, blockDim>>>(in1->d_buf, in2->d_buf, in3->d_buf, in4->d_buf, out->d_buf, BS, N1, N2, N3, N4);
  CHECK_CUDA(cudaDeviceSynchronize());
}

/* Linear 
 * @param [in1]  in: [BS, N]
 * @param [in2]   w: [M, N]
 * @param [in3]   b: [M]
 * @param [out] out: [BS, M]
 * 'N' is the input feature size
 * 'M' is the output feature size
 */
__global__ void __launch_bounds__(BLOCK_SIZE * BLOCK_SIZE) LinearKernel(float *in, float *w, float *b, float *out,
                            size_t BS, size_t N, size_t M) {
    // C idx
    int row = blockIdx.x * BLOCK_SIZE + threadIdx.x / BLOCK_SIZE;
    int col = blockIdx.y * BLOCK_SIZE + threadIdx.x % BLOCK_SIZE;
    // block idx
    int cRow = blockIdx.x;
    int cCol = blockIdx.y;
    // thread idx in block
    int threadRow = threadIdx.x / BLOCK_SIZE;
    int threadCol = threadIdx.x % BLOCK_SIZE;

    in += cRow * BLOCK_SIZE * N;
    w += cCol * BLOCK_SIZE * N;
    out += cRow * BLOCK_SIZE * M + cCol * BLOCK_SIZE;

    float val = 0.f;

    __shared__ float in_shared[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float w_shared[BLOCK_SIZE * BLOCK_SIZE];

    for (size_t blk_i = 0; blk_i < N; blk_i += BLOCK_SIZE) {
        in_shared[threadRow * BLOCK_SIZE + threadCol] = 
            (cRow * BLOCK_SIZE + threadRow >= BS || blk_i + threadCol >= N) ? 
            0.0f : in[threadRow * N + threadCol];
        w_shared[threadRow * BLOCK_SIZE + threadCol] = 
            (cCol * BLOCK_SIZE + threadRow >= M || blk_i + threadCol >= N) ? 
            0.0f : w[threadRow * N + threadCol];
        __syncthreads();

        in += BLOCK_SIZE;
        w += BLOCK_SIZE;

        for (size_t dot_i = 0; dot_i < BLOCK_SIZE; dot_i++) {
            val += in_shared[threadRow * BLOCK_SIZE + dot_i] * 
                   w_shared[threadCol * BLOCK_SIZE + dot_i];
        }
        __syncthreads();
    }
    
    if (row < BS && col < M) {
        out[threadRow * M + threadCol] = val + b[col];
    }
}
void Linear_CUDA(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
    size_t BS = in->shape[0];
    size_t N = in->shape[1];
    size_t M = w->shape[0];

    dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);
    dim3 gridDim((BS + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    LinearKernel<<<gridDim, blockDim>>>(in->d_buf, w->d_buf, b->d_buf, out->d_buf, BS, N, M);
    CHECK_CUDA(cudaDeviceSynchronize());
}

/* [Advanced Example] Linear in Half precision on CPU */
// void Linear_Half(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
//   size_t N = in->shape[0];
//   size_t M = w->shape[0];

//   for (size_t i = 0; i < M; i++) {
//     float val = 0.f;
//     for (size_t j = 0; j < N; j++) {
//       val += static_cast<float>(half_cpu(in->buf[j]) * 
//         half_cpu(w->buf[i * N + j]));
//     }
//     out->buf[i] = val + b->buf[i];
//   }
// }

/* Softmax (w/ Max Trick)
 * @param [in & out] inout: [BS, N]
 * 'N' is the number of elements in the tensor.
 * 'N' is fixed to 4 in this implementation (for 4 experts)
 */
void Softmax(Tensor *inout) {
  size_t BS = inout->shape[0];
  size_t N = inout->shape[1];

  for (size_t bs = 0; bs < BS; bs++) {
    float max_val = -INFINITY;
    for (size_t n = 0; n < N; n++) {
      max_val = inout->buf[bs * N + n] > max_val ? inout->buf[bs * N + n] : max_val;
    }

    float sum = 0.f;
    for (size_t n = 0; n < N; n++) {
      inout->buf[bs * N + n] = exp(inout->buf[bs * N + n] - max_val);
      sum += inout->buf[bs * N + n];
    }

    for (size_t n = 0; n < N; n++) { inout->buf[bs * N + n] /= sum; }
  }
}
/* Softmax CUDA kernel */
__global__ void SoftmaxKernel(float *inout, size_t BS, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= BS) return;

  size_t offset = idx * N;
  float max_val = inout[offset];
  for (size_t i = 1; i < N; i++) {
    float val = inout[offset + i];
    max_val = val > max_val ? val : max_val;
  }

  float sum = 0.f;
  for (size_t i = 0; i < N; i++) {
    float exp_val = exp(inout[offset + i] - max_val);
    inout[offset + i] = exp_val;
    sum += exp_val;
  }

  for (size_t i = 0; i < N; i++) { inout[offset + i] /= sum; }
}
/* Softmax using CUDA */
void Softmax_CUDA(Tensor *inout) {
  size_t BS = inout->shape[0];
  size_t N = inout->shape[1];

  // N is small, so we can parallelize over BS
  dim3 blockDim(THREADS_PER_BLOCK);
  dim3 gridDim((BS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1);
  SoftmaxKernel<<<gridDim, blockDim>>>(inout->d_buf, BS, N);
  CHECK_CUDA(cudaDeviceSynchronize());
}

/* (Elemwise) Scaling
 * @param [in & out] inout: [BS, N]
 * @param [in]           gate: [BS, 4]
 * @param [in]           gate_col: [1]
 * 'N' is the number of elements in the tensor.
 */
void Scaling(Tensor *inout, float *gate, size_t gate_col) {
  size_t BS = inout->shape[0];
  size_t N = inout->shape[1];

  for (size_t bs = 0; bs < BS; bs++) {
    float scale = gate[bs * 4 + gate_col];
    for (size_t i = 0; i < N; i++) {
      inout->buf[bs * N + i] *= scale;
    }
  }
}
/* Scaling CUDA kernel */
__global__ void ScalingKernel(float *inout, float *gate, size_t gate_col, size_t BS, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= BS * N) return;

  size_t bs = idx / N;
  inout[idx] *= gate[bs * 4 + gate_col];
}
/* Scaling using CUDA */
void Scaling_CUDA(Tensor *inout, Tensor *gate, size_t gate_col) {
  size_t BS = inout->shape[0];
  size_t N = inout->shape[1];

  dim3 blockDim(THREADS_PER_BLOCK);
  dim3 gridDim((BS * N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1);
  ScalingKernel<<<gridDim, blockDim>>>(inout->d_buf, gate->d_buf, gate_col, BS, N);
  CHECK_CUDA(cudaDeviceSynchronize());
}

/* (Elemwise) Addition
 * @param [in1] in1: [BS, N]
 * @param [in2] in2: [BS, N]
 * @param [in3] in3: [BS, N]
 * @param [in4] in4: [BS, N]
 * @param [out] out: [BS, N]
 * 'N' is the number of elements in the input tensor.
 */
void Add(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, 
         Tensor *out) {
  size_t BS = in1->shape[0];
  size_t N = in1->shape[1];

  for (size_t bs = 0; bs < BS; bs++) {
    for (size_t n = 0; n < N; n++) {
      out->buf[bs * N + n] = in1->buf[bs * N + n] + in2->buf[bs * N + n] + 
        in3->buf[bs * N + n] + in4->buf[bs * N + n];
    }
  }
}
/* Add CUDA kernel */
__global__ void AddKernel(float *in1, float *in2, float *in3, float *in4, float *out, size_t BS, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= BS * N) return;
  out[idx] = in1[idx] + in2[idx] + in3[idx] + in4[idx];
}
/* Add using CUDA */
void Add_CUDA(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, 
              Tensor *out) {
  size_t BS = in1->shape[0];
  size_t N = in1->shape[1];

  dim3 blockDim(THREADS_PER_BLOCK);
  dim3 gridDim((BS * N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1);
  AddKernel<<<gridDim, blockDim>>>(in1->d_buf, in2->d_buf, in3->d_buf, in4->d_buf, out->d_buf, BS, N);
  CHECK_CUDA(cudaDeviceSynchronize());
}

/* im2col_1d_CUDA
 * @param [in]  in: [BS, C, s]
 * @param [out] out: [BS, os, C * K]
 * 'K' is the kernel size
 * 'os' is the output sequence length
 */
__global__ void im2col_1d_Kernel(const float *in, float *out,
                                    size_t BS, size_t C, size_t s, size_t K, size_t os) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= BS * os * C * K) return;

  // 출력 텐서를 [BS*os, C*K]로 해석
  size_t row = idx / (C * K);
  size_t col = idx % (C * K);

  // row를 분해: row = bs * os + j
  size_t bs = row / os;
  size_t j  = row % os;

  // col을 분해: col = c * K + k
  size_t c = col / K;
  size_t k = col % K;

  // in[bs, c, j+k]
  size_t in_idx = bs * (C * s) + c * s + (j + k);
  out[idx] = in[in_idx];
}
void im2col_1d_CUDA(Tensor *in, Tensor *out, size_t K) {
  size_t BS = in->shape[0];
  size_t C = in->shape[1];
  size_t s = in->shape[2];
  size_t os = s - K + 1;

  dim3 blockDim(THREADS_PER_BLOCK);
  dim3 gridDim((BS * os * C * K + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1);
  im2col_1d_Kernel<<<gridDim, blockDim>>>(in->d_buf, out->d_buf, BS, C, s, K, os);
  CHECK_CUDA(cudaDeviceSynchronize());
}