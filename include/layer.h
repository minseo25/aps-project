#pragma once

#include "tensor.h"


/* Layers (Operations) */
void Conv1D(Tensor *in, Tensor *w, Tensor *b, Tensor *out);
void ReLU(Tensor *inout);
void GetMax(Tensor *in, Tensor *out);
void Concat(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, 
            Tensor *out);
void Linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out);
void Softmax(Tensor *inout);
void Scaling(Tensor *inout, float *gate, size_t gate_col);
void Add(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, 
         Tensor *out);

/* Layers (CUDA) */
void Conv1D_CUDA(Tensor *in, Tensor *w, Tensor *b, Tensor *out);
void ReLU_CUDA(Tensor *inout);
void GetMax_CUDA(Tensor *in, Tensor *out);
void Concat_CUDA(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, 
                  Tensor *out);
void Linear_CUDA(Tensor *in, Tensor *w, Tensor *b, Tensor *out);
void Softmax_CUDA(Tensor *inout);
void Scaling_CUDA(Tensor *inout, Tensor *gate, size_t gate_col);
void Add_CUDA(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, 
         Tensor *out);

/* [Advanced] Example of using half-precision on CPU */
// void Linear_Half(Tensor *in, Tensor *w, Tensor *b, Tensor *out);    