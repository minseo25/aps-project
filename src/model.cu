#include <cstdio>

#include "layer.h"
#include "model.h"

#define BATCH_SIZE 256

/* [Model Parameters]
 * _w: Weight parameter
 * _b: Bias parameter
 */
// Convolutional layers
Parameter *conv0_w, *conv0_b;
Parameter *conv1_w, *conv1_b;
Parameter *conv2_w, *conv2_b;
Parameter *conv3_w, *conv3_b;
// Mixture-of-Experts (MoE) layer
Parameter *moe_exp0_w, *moe_exp0_b;
Parameter *moe_exp1_w, *moe_exp1_b;
Parameter *moe_exp2_w, *moe_exp2_b;
Parameter *moe_exp3_w, *moe_exp3_b;
Parameter *moe_gate_w, *moe_gate_b;
// Fully-connected (Linear) layers
Parameter *linear0_w, *linear0_b;
Parameter *linear1_w, *linear1_b;
Parameter *linear2_w, *linear2_b;

void alloc_and_set_parameters(float *param, size_t param_size) {
  size_t pos = 0;

  conv0_w = new Parameter({1024, 4096 * 3}, param + pos);
  pos += 1024 * 4096 * 3; 
  conv0_b = new Parameter({1024}, param + pos);
  pos += 1024;

  conv1_w = new Parameter({1024, 4096 * 5}, param + pos);
  pos += 1024 * 4096 * 5; 
  conv1_b = new Parameter({1024}, param + pos);
  pos += 1024;

  conv2_w = new Parameter({1024, 4096 * 7}, param + pos);
  pos += 1024 * 4096 * 7;
  conv2_b = new Parameter({1024}, param + pos);
  pos += 1024;

  conv3_w = new Parameter({1024, 4096 * 9}, param + pos);
  pos += 1024 * 4096 * 9;
  conv3_b = new Parameter({1024}, param + pos);
  pos += 1024;
  
  moe_exp0_w = new Parameter({2048, 4096}, param + pos);
  pos += 2048 * 4096;
  moe_exp0_b = new Parameter({2048}, param + pos);
  pos += 2048;

  moe_exp1_w = new Parameter({2048, 4096}, param + pos);
  pos += 2048 * 4096;
  moe_exp1_b = new Parameter({2048}, param + pos);
  pos += 2048;

  moe_exp2_w = new Parameter({2048, 4096}, param + pos);
  pos += 2048 * 4096;
  moe_exp2_b = new Parameter({2048}, param + pos);
  pos += 2048;

  moe_exp3_w = new Parameter({2048, 4096}, param + pos);
  pos += 2048 * 4096;
  moe_exp3_b = new Parameter({2048}, param + pos);
  pos += 2048;

  moe_gate_w = new Parameter({4, 4096}, param + pos);
  pos += 4 * 4096;
  moe_gate_b = new Parameter({4}, param + pos);
  pos += 4;

  linear0_w = new Parameter({1024, 2048}, param + pos);
  pos += 1024 * 2048;
  linear0_b = new Parameter({1024}, param + pos);
  pos += 1024;

  linear1_w = new Parameter({512, 1024}, param + pos);
  pos += 512 * 1024;
  linear1_b = new Parameter({512}, param + pos);
  pos += 512;

  linear2_w = new Parameter({2, 512}, param + pos);
  pos += 2 * 512;
  linear2_b = new Parameter({2}, param + pos);
  pos += 2;

  if (pos != param_size) {
    fprintf(stderr, "Parameter size mismatched: %zu != %zu\n", 
            pos, param_size);
    exit(EXIT_FAILURE);
  }
}

void free_parameters() {
  delete conv0_w;
  delete conv0_b;
  delete conv1_w;
  delete conv1_b;
  delete conv2_w;
  delete conv2_b;
  delete conv3_w;
  delete conv3_b;
  delete moe_exp0_w;
  delete moe_exp0_b;
  delete moe_exp1_w;
  delete moe_exp1_b;
  delete moe_exp2_w;
  delete moe_exp2_b;
  delete moe_exp3_w;
  delete moe_exp3_b;
  delete moe_gate_w;
  delete moe_gate_b;
  delete linear0_w;
  delete linear0_b;
  delete linear1_w;
  delete linear1_b;
  delete linear2_w;
  delete linear2_b;
}

/* [Model Inputs] */
Tensor *input;

/* [Model Activations] 
 * _a: Activation buffer
 */
Activation *unrolled_input0, *unrolled_input1, *unrolled_input2, *unrolled_input3;
Activation *conv0_a, *relu0_a, *pool0_a;
Activation *conv1_a, *relu1_a, *pool1_a;
Activation *conv2_a, *relu2_a, *pool2_a;
Activation *conv3_a, *relu3_a, *pool3_a;
Activation *concat_a;
Activation *gate_a; //, *topk_val_a;
Activation *expert0_a, *expert1_a, *expert2_a, *expert3_a;
Activation *moe_a;
Activation *linear0_a, *linear1_a, *linear2_a;

void alloc_activations() {
  input = new Tensor({BATCH_SIZE, 4096, SEQ_LEN});
  unrolled_input0 = new Activation({BATCH_SIZE, SEQ_LEN - 2, 4096 * 3});
  unrolled_input1 = new Activation({BATCH_SIZE, SEQ_LEN - 4, 4096 * 5});
  unrolled_input2 = new Activation({BATCH_SIZE, SEQ_LEN - 6, 4096 * 7});
  unrolled_input3 = new Activation({BATCH_SIZE, SEQ_LEN - 8, 4096 * 9});
  conv0_a = new Activation({BATCH_SIZE, SEQ_LEN - 2, 1024});
  pool0_a = new Activation({BATCH_SIZE, 1024});
  conv1_a = new Activation({BATCH_SIZE, SEQ_LEN - 4, 1024});
  pool1_a = new Activation({BATCH_SIZE, 1024});
  conv2_a = new Activation({BATCH_SIZE, SEQ_LEN - 6, 1024});
  pool2_a = new Activation({BATCH_SIZE, 1024});
  conv3_a = new Activation({BATCH_SIZE, SEQ_LEN - 8, 1024});
  pool3_a = new Activation({BATCH_SIZE, 1024});
  concat_a = new Activation({BATCH_SIZE, 4096});
  gate_a = new Activation({BATCH_SIZE, 4}); 
  expert0_a = new Activation({BATCH_SIZE, 2048});
  expert1_a = new Activation({BATCH_SIZE, 2048});
  expert2_a = new Activation({BATCH_SIZE, 2048});
  expert3_a = new Activation({BATCH_SIZE, 2048});
  moe_a = new Activation({BATCH_SIZE, 2048});
  linear0_a = new Activation({BATCH_SIZE, 1024});
  linear1_a = new Activation({BATCH_SIZE, 512});
  linear2_a = new Activation({BATCH_SIZE, 2});
}

void free_activations() {
  delete input;
  delete conv0_a;
  delete pool0_a;
  delete conv1_a;
  delete pool1_a;
  delete conv2_a;
  delete pool2_a;
  delete conv3_a;
  delete pool3_a;
  delete concat_a;
  delete gate_a;
  delete expert0_a;
  delete expert1_a;
  delete moe_a;
  delete linear0_a;
  delete linear1_a;
  delete linear2_a;
}

/* Dense Mixture-of-Experts (MoE) Layer
 * - The MoE layer has a total 4 Experts (Linear) with a Gating mechanism.
 * @param [in1]      in: [BS, 4096]
 * @param [in2]  exp0_w: [2048, 4096]
 * @param [in3]  exp0_b: [2048]
 * @param [in4]  exp1_w: [2048, 4096]
 * @param [in5]  exp1_b: [2048]
 * @param [in6]  exp2_w: [2048, 4096] 
 * @param [in7]  exp2_b: [2048]
 * @param [in8]  exp3_w: [2048, 4096]
 * @param [in9]  exp3_b: [2048]
 * @param [in10] gate_w: [4, 4096]
 * @param [in11] gate_b: [4]
 * @param [out]     out: [BS, 2048]
 */
void MoE(Activation *in, Parameter *exp0_w, Parameter *exp0_b,
         Parameter *exp1_w, Parameter *exp1_b, Parameter *exp2_w,
         Parameter *exp2_b, Parameter *exp3_w, Parameter *exp3_b,
         Parameter *gate_w, Parameter *gate_b, Activation *out) {

  /* 1. Compute the gate logits: in [BS, 4096] -> out [BS, 4] */
  Linear_CUDA(in, gate_w, gate_b, gate_a);

  /* 2. Compute the softmax of the gate logits: in [BS, 4] -> out [BS, 4] */
  Softmax_CUDA(gate_a);

  /* 3. Compute the expert's output: in [BS, 4096] -> out [BS, 2048] */
  Linear_CUDA(in, exp0_w, exp0_b, expert0_a);
  Linear_CUDA(in, exp1_w, exp1_b, expert1_a);
  Linear_CUDA(in, exp2_w, exp2_b, expert2_a);
  Linear_CUDA(in, exp3_w, exp3_b, expert3_a);

  /* 4. Scale the expert's output: in [BS, 2048] -> out [BS, 2048] */
  Scaling_CUDA(expert0_a, gate_a, 0);
  Scaling_CUDA(expert1_a, gate_a, 1);
  Scaling_CUDA(expert2_a, gate_a, 2);
  Scaling_CUDA(expert3_a, gate_a, 3);

  /* 5. Accumulate the expert's output:
    * in [BS, 2048] + [BS, 2048] + [BS, 2048] + [BS, 2048] -> out [BS, 2048] */
  Add_CUDA(expert0_a, expert1_a, expert2_a, expert3_a, out);
}

/* [Model Computation: Sentiment Analysis Task] */
void predict_sentiment(float *inputs, float *outputs, size_t n_samples) {
  int n_batches = (n_samples + BATCH_SIZE - 1) / BATCH_SIZE;
  for (int n = 0; n < n_batches; n++) {
    /* Load a sentence from the inputs */
    size_t start = n * BATCH_SIZE;
    size_t BS = (n == n_batches - 1) ? n_samples - start : BATCH_SIZE; // batch size
    input->to_device_with_shape(inputs + start * SEQ_LEN * 4096, BS, 4096, SEQ_LEN, 1);

    /* in [BS, 4096, SEQ_LEN] -> out [BS, SEQ_LEN - 2, 4096 * 3] */
    im2col_1d_CUDA(input, unrolled_input0, 3);
    /* in [BS, SEQ_LEN - 2, 4096 * 3] -> out [BS, SEQ_LEN - 2, 1024] */
    Conv1D_CUDA(unrolled_input0, conv0_w, conv0_b, conv0_a);
    ReLU_CUDA(conv0_a); 

    /* in [BS, SEQ_LEN - 2, 1024] -> out [BS, 1024] */
    GetMax_CUDA(conv0_a, pool0_a);

    /* in [BS, 4096, SEQ_LEN] -> out [BS, SEQ_LEN - 4, 4096 * 5] */
    im2col_1d_CUDA(input, unrolled_input1, 5);
    /* in [BS, SEQ_LEN - 4, 4096 * 5] -> out [BS, SEQ_LEN - 4, 1024] */
    Conv1D_CUDA(unrolled_input1, conv1_w, conv1_b, conv1_a);
    ReLU_CUDA(conv1_a);

    /* in [BS, SEQ_LEN - 4, 1024] -> out [BS, 1024] */
    GetMax_CUDA(conv1_a, pool1_a);

    /* in [BS, 4096, SEQ_LEN] -> out [BS, SEQ_LEN - 6, 4096 * 7] */
    im2col_1d_CUDA(input, unrolled_input2, 7);
    /* in [BS, SEQ_LEN - 6, 4096 * 7] -> out [BS, SEQ_LEN - 6, 1024] */
    Conv1D_CUDA(unrolled_input2, conv2_w, conv2_b, conv2_a);
    ReLU_CUDA(conv2_a);

    /* in [BS, SEQ_LEN - 6, 1024] -> out [BS, 1024] */
    GetMax_CUDA(conv2_a, pool2_a);

    /* in [BS, 4096, SEQ_LEN] -> out [BS, SEQ_LEN - 8, 4096 * 9] */
    im2col_1d_CUDA(input, unrolled_input3, 9);
    /* in [BS, SEQ_LEN - 8, 4096 * 9] -> out [BS, SEQ_LEN - 8, 1024] */
    Conv1D_CUDA(unrolled_input3, conv3_w, conv3_b, conv3_a);
    ReLU_CUDA(conv3_a);

    /* in [BS, SEQ_LEN - 8, 1024] -> out [BS, 1024] */
    GetMax_CUDA(conv3_a, pool3_a);

    /* in [BS, 1024] +
          [BS, 1024] +
          [BS, 1024] +
          [BS, 1024] -> out [BS, 1024 * 4] */
    Concat_CUDA(pool0_a, pool1_a, pool2_a, pool3_a, concat_a);

    /* in [BS, 1024 * 4] -> out [BS, 2048] */
    MoE(concat_a, moe_exp0_w, moe_exp0_b, moe_exp1_w, moe_exp1_b,
      moe_exp2_w, moe_exp2_b, moe_exp3_w, moe_exp3_b, moe_gate_w,
      moe_gate_b, moe_a);

    /* in [BS, 2048] -> out [BS, 1024] */
    Linear_CUDA(moe_a, linear0_w, linear0_b, linear0_a);
    ReLU_CUDA(linear0_a);

    /* in [BS, 1024] -> out [BS, 512] */
    Linear_CUDA(linear0_a, linear1_w, linear1_b, linear1_a);
    ReLU_CUDA(linear1_a);

    /* in [BS, 512] -> out [BS, 2] */
    Linear_CUDA(linear1_a, linear2_w, linear2_b, linear2_a);

    /* cf) The output 'linear2_a' (shape: [2]) contains the probabilities 
      for each sentiment class (0: negative, 1: positive). To determine 
      the sentiment, we can simply take the argmax of these probabilities. 
    */
    linear2_a->to_host();

    /* Copy the computation result to the outputs */
    for (size_t b = 0; b < BS; b++) {
      memcpy(outputs + (start + b) * 2, linear2_a->buf + b * 2, 2 * sizeof(float));
    }
  }
}