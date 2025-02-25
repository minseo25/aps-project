#include "layer.h"


/* Conv1D 
 * @param [in1]  in: [BS, C, s]
 * @param [in2]   w: [OC, C, K] 
 * @param [in3]   b: [OC]
 * @param [out] out: [BS, OC, os]
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
void Conv1D(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t s = in->shape[2];
  size_t C = in->shape[1];
  size_t BS = in->shape[0];
  size_t OC = w->shape[0];
  size_t K = w->shape[2];
  
  size_t os = s - K + 1;

  for (size_t bs = 0; bs < BS; bs++) {
    for (size_t oc = 0; oc < OC; oc++) {
      for (size_t j = 0; j < os; j++) {
        float val = 0.f;
        for (size_t k = 0; k < C; k++) {
          for (size_t l = 0; l < K; l++) {
            val += in->buf[bs * C * s + k * s + j + l] * 
                    w->buf[oc * C * K + k * K + l];
          }
        }
        out->buf[bs * OC * os + oc * os + j] = val + b->buf[oc];
      }
    }
  }
}
/* Conv1D CUDA kernel */
__global__ void Conv1DKernel(float *in, float *w, float *b, float *out,
                            size_t BS, size_t C, size_t s, size_t OC, size_t K, size_t os) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= BS * OC * os) return;

  size_t bs = idx / (OC * os);
  size_t oc = (idx % (OC * os)) / os;
  size_t j = idx % os;

  float val = 0.f;
  for (size_t k = 0; k < C; k++) {
    for (size_t l = 0; l < K; l++) {
      val += in[bs * C * s + k * s + j + l] * w[oc * C * K + k * K + l];
    }
  }
  out[idx] = val + b[oc];
}
/* Conv1D using CUDA */
void Conv1D_CUDA(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t BS = in->shape[0];
  size_t C = in->shape[1];
  size_t s = in->shape[2];
  size_t OC = w->shape[0];
  size_t K = w->shape[2];
  size_t os = s - K + 1;

  dim3 blockDim(256);
  dim3 gridDim((BS * OC * os + 255) / 256, 1);
  Conv1DKernel<<<gridDim, blockDim>>>(in->d_buf, w->d_buf, b->d_buf, out->d_buf, 
                                      BS, C, s, OC, K, os);
  CHECK_CUDA(cudaDeviceSynchronize());
}

/* ReLU
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
void ReLU(Tensor *inout) {
  size_t N = inout->num_elem();

  for (size_t i = 0; i < N; i++) {
    inout->buf[i] = inout->buf[i] > 0 ? inout->buf[i] : 0;
  }
}
/* ReLU CUDA kernel */
__global__ void ReLU_Kernel(float *inout, size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    inout[i] = inout[i] > 0 ? inout[i] : 0;
  }
}
/* ReLU using CUDA */
void ReLU_CUDA(Tensor *inout) {
  size_t N = inout->num_elem();

  ReLU_Kernel<<<(N + 255) / 256, 256>>>(inout->d_buf, N);
  CHECK_CUDA(cudaDeviceSynchronize());
}

/* GetMax
 * @param [in]   in: [BS, C, s]
 * @param [out] out: [BS, C]
 *    
 *    This layer is to get the max value along the sequence dim.
 *    The formula for this layer: out = max(in, dim=-1)
 * 
 * 'BS' is the batch size
 * 'C' is the channel size
 * 's' is the sequence length
 */
void GetMax(Tensor *in, Tensor *out) {
  size_t BS = in->shape[0];
  size_t C = in->shape[1];
  size_t s = in->shape[2];

  for (size_t bs = 0; bs < BS; bs++) {
    for (size_t c = 0; c < C; c++) {
      out->buf[bs * C + c] = in->buf[bs * C * s + c * s];
      for (size_t j = 1; j < s; j++) {
        out->buf[bs * C + c] = in->buf[bs * C * s + c * s + j] > 
          out->buf[bs * C + c] ? in->buf[bs * C * s + c * s + j] : 
          out->buf[bs * C + c];
      }
    }
  }
}
/* GetMax CUDA kernel */
__global__ void GetMax_Kernel(float *in, float *out, size_t BS, size_t C, size_t s) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= BS * C) return;
  
  size_t bs = idx / C;
  size_t c = idx % C;

  float max_val = in[bs * C * s + c * s];
  for (size_t j = 1; j < s; j++) {
    max_val = in[bs * C * s + c * s + j] > max_val ? in[bs * C * s + c * s + j] : max_val;
  }
  out[bs * C + c] = max_val;
}
/* GetMax using CUDA */
void GetMax_CUDA(Tensor *in, Tensor *out) {
  size_t BS = in->shape[0];
  size_t C = in->shape[1];
  size_t s = in->shape[2];

  dim3 blockDim(256);
  dim3 gridDim((BS * C + 255) / 256, 1);
  GetMax_Kernel<<<gridDim, blockDim>>>(in->d_buf, out->d_buf, BS, C, s);
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

  dim3 blockDim(256);
  int numBlocks = (int)((BS * (N1 + N2 + N3 + N4) + 255) / 256);
  dim3 gridDim(numBlocks, 1);
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
void Linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t BS = in->shape[0];
  size_t N = in->shape[1];
  size_t M = w->shape[0];

  for (size_t bs = 0; bs < BS; bs++) {
    for (size_t m = 0; m < M; m++) {
      float val = 0.f;
      for (size_t n = 0; n < N; n++) {
        val += in->buf[bs * N + n] * w->buf[m * N + n];
      }
      out->buf[bs * M + m] = val + b->buf[m];
    }
  }
}
/* Linear CUDA kernel */
__global__ void LinearKernel(float *in, float *w, float *b, float *out,
                            size_t BS, size_t N, size_t M) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= BS * M) return;

  size_t bs = idx / M;
  size_t m = idx % M;

  float val = 0.f;
  for (size_t n = 0; n < N; n++) {
    val += in[bs * N + n] * w[m * N + n];
  }
  out[idx] = val + b[m];
}
/* Linear using CUDA */
void Linear_CUDA(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t BS = in->shape[0];
  size_t N = in->shape[1];
  size_t M = w->shape[0];

  dim3 blockDim(256);
  dim3 gridDim((BS * M + 255) / 256, 1);
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
  dim3 blockDim(256);
  dim3 gridDim((BS + 255) / 256, 1);
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

  dim3 blockDim(256);
  dim3 gridDim((BS * N + 255) / 256, 1);
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

  dim3 blockDim(256);
  dim3 gridDim((BS * N + 255) / 256, 1);
  AddKernel<<<gridDim, blockDim>>>(in1->d_buf, in2->d_buf, in3->d_buf, in4->d_buf, out->d_buf, BS, N);
  CHECK_CUDA(cudaDeviceSynchronize());
}