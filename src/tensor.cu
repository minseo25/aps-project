#include "model.h"


/* [Tensor Structure] */
/* Tensor
 * @brief - A multi-dimensional matrix containing elements of a single data
 type.
 * @member - buf  : Data buffer containing elements
 * @member - shape: Shape of tensor from outermost dimension to innermost
 dimension e.g., {{1.0, -0.5, 2.3}, {4.3, 5.6, -7.8}} => shape = {2, 3}
 */
Tensor::Tensor(const vector<size_t> &shape_) {
  ndim = shape_.size();
  for (size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
  size_t bufsize = num_elem() * sizeof(float);
  CHECK_CUDA(cudaMallocHost(&buf, bufsize));
  memset(buf, 0, bufsize);
}

Tensor::Tensor(const vector<size_t> &shape_, float *buf_) {
  ndim = shape_.size();
  for (size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
  size_t bufsize = num_elem() * sizeof(float);
  CHECK_CUDA(cudaMallocHost(&buf, bufsize));
  memcpy(buf, buf_, bufsize);
}

Tensor::~Tensor() {
  if (buf != nullptr) CHECK_CUDA(cudaFreeHost(buf));
}

size_t Tensor::num_elem() {
  size_t size = 1;
  for (size_t i = 0; i < ndim; i++) { size *= shape[i]; }
  return size;
}