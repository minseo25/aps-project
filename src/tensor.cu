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
  CHECK_CUDA(cudaMalloc(&d_buf, bufsize));
  memset(buf, 0, bufsize);
}

Tensor::Tensor(const vector<size_t> &shape_, float *buf_) {
  ndim = shape_.size();
  for (size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
  size_t bufsize = num_elem() * sizeof(float);
  CHECK_CUDA(cudaMallocHost(&buf, bufsize));
  CHECK_CUDA(cudaMalloc(&d_buf, bufsize));
  memcpy(buf, buf_, bufsize);
  CHECK_CUDA(cudaMemcpy(d_buf, buf, bufsize, cudaMemcpyHostToDevice));
}

Tensor::~Tensor() {
  if (buf != nullptr) CHECK_CUDA(cudaFreeHost(buf));
  if (d_buf != nullptr) CHECK_CUDA(cudaFree(d_buf));
}

size_t Tensor::num_elem() {
  size_t size = 1;
  for (size_t i = 0; i < ndim; i++) { size *= shape[i]; }
  return size;
}

void Tensor::to_device() {
  size_t bufsize = num_elem() * sizeof(float);
  CHECK_CUDA(cudaMemcpy(d_buf, buf, bufsize, cudaMemcpyHostToDevice));
}

void Tensor::to_host() {
  size_t bufsize = num_elem() * sizeof(float);
  CHECK_CUDA(cudaMemcpy(buf, d_buf, bufsize, cudaMemcpyDeviceToHost));
}

void Tensor::to_device_with_shape(float *buf_, size_t shape1, size_t shape2, size_t shape3, size_t shape4) {
  shape[0] = shape1;
  shape[1] = shape2;
  shape[2] = shape3;
  shape[3] = shape4;
  size_t bufsize = num_elem() * sizeof(float);
  CHECK_CUDA(cudaMemcpy(d_buf, buf_, bufsize, cudaMemcpyHostToDevice));
}