#pragma once

#include "tensor.h"

#define MAX(A, B) A > B ? A : B

/* Layers (CUDA) */
void Conv1D_CUDA(Tensor *in, Tensor *w, Tensor *b, Tensor *out);
void GetMax_CUDA(Tensor *in, Tensor *out);
void ReLU_GetMax_CUDA(Tensor *in, Tensor *out);
void Concat_CUDA(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, 
                  Tensor *out);
void Linear_CUDA(Tensor *in, Tensor *w, Tensor *b, Tensor *out, bool relu);
void Linear_CUDA_slow(Tensor *in, Tensor *w, Tensor *b, Tensor *out, bool relu);
void Softmax_CUDA(Tensor *inout);
void Scaling_Add_CUDA(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, Tensor *gate,
                      Tensor *out);
void im2col_1d_CUDA(Tensor *in, Tensor *out, size_t K);
void Transpose_CUDA(Tensor *in, Tensor *out);

/* [Advanced] Example of using half-precision on CPU */
// void Linear_Half(Tensor *in, Tensor *w, Tensor *b, Tensor *out);    