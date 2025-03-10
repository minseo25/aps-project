#pragma once

#include "tensor.h"

#define MAX(A, B) A > B ? A : B

/* Layers (CUDA) */
void Conv1D_CUDA(Tensor *in, Tensor *w, Tensor *b, Tensor *out, cudaStream_t &stream);
void ReLU_GetMax_CUDA(Tensor *in, Tensor *out, cudaStream_t &stream);
void Concat_CUDA(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, 
                  Tensor *out);
void Linear_CUDA(Tensor *in, Tensor *w, Tensor *b, Tensor *out, bool relu);
void Linear_CUDA_slow_moe(Tensor *in, Tensor *w, Tensor *b, Tensor *out, bool relu, cudaStream_t &stream);
void Linear_CUDA_slow_fc(Tensor *in, Tensor *w, Tensor *b, Tensor *out, bool relu, cudaStream_t &stream);
void Softmax_CUDA(Tensor *inout, cudaStream_t &stream);
void Scaling_Add_Transpose_CUDA(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, Tensor *gate,
                      Tensor *out);
void im2col_1d_CUDA(Tensor *in, Tensor *out, size_t K, cudaStream_t &stream);