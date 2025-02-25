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
/* [Example] ReLU CUDA kernel */
__global__ void ReLU_Kernel(float *inout, size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    inout[i] = inout[i] > 0 ? inout[i] : 0;
  }
}
/* [Example] ReLU using CUDA */
void ReLU_CUDA(Tensor *inout) {
  size_t N = inout->num_elem();

  float *d_inout;
  CHECK_CUDA(cudaMalloc(&d_inout, N * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_inout, inout->buf, N * sizeof(float), 
                        cudaMemcpyHostToDevice));

  ReLU_Kernel<<<(N + 255) / 256, 256>>>(d_inout, N);
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(inout->buf, d_inout, N * sizeof(float), 
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(d_inout));
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

