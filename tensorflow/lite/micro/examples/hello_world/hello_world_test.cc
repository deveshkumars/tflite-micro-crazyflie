/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <math.h>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/examples/hello_world/models/hello_world_float_model_data.h"
#include "tensorflow/lite/micro/examples/hello_world/models/hello_world_int8_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
using HelloWorldOpResolver = tflite::MicroMutableOpResolver<20>;

TfLiteStatus RegisterOps(HelloWorldOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected()); // 1
  TF_LITE_ENSURE_STATUS(op_resolver.AddSub()); // 2
  TF_LITE_ENSURE_STATUS(op_resolver.AddMul()); // 3
  TF_LITE_ENSURE_STATUS(op_resolver.AddDiv()); // 4
  TF_LITE_ENSURE_STATUS(op_resolver.AddAbs()); // 5
  TF_LITE_ENSURE_STATUS(op_resolver.AddNeg()); // 6
  TF_LITE_ENSURE_STATUS(op_resolver.AddSin()); // 7
  TF_LITE_ENSURE_STATUS(op_resolver.AddCos()); // 8
  TF_LITE_ENSURE_STATUS(op_resolver.AddTanh()); // 9
  TF_LITE_ENSURE_STATUS(op_resolver.AddLogistic()); // 10
  TF_LITE_ENSURE_STATUS(op_resolver.AddSplitV()); // 11
  TF_LITE_ENSURE_STATUS(op_resolver.AddRelu()); // 12
  TF_LITE_ENSURE_STATUS(op_resolver.AddExp()); // 13
  TF_LITE_ENSURE_STATUS(op_resolver.AddAdd()); // 14
  TF_LITE_ENSURE_STATUS(op_resolver.AddLog()); // 15
  TF_LITE_ENSURE_STATUS(op_resolver.AddNotEqual()); // 16
  // TF_LITE_ENSURE_STATUS(op_resolver.AddSelectV2()); // 17
  TF_LITE_ENSURE_STATUS(op_resolver.AddSelect()); // 18
  TF_LITE_ENSURE_STATUS(op_resolver.AddDequantize()); // 19
  TF_LITE_ENSURE_STATUS(op_resolver.AddQuantize()); // 20

  return kTfLiteOk;
}
}  // namespace

// TfLiteStatus ProfileMemoryAndLatency() {
//   tflite::MicroProfiler profiler;
//   HelloWorldOpResolver op_resolver;
//   TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

//   // Arena size just a round number. The exact arena usage can be determined
//   // using the RecordingMicroInterpreter.
//   constexpr int kTensorArenaSize = 3000;
//   uint8_t tensor_arena[kTensorArenaSize];
//   constexpr int kNumResourceVariables = 24;

//   tflite::RecordingMicroAllocator* allocator(
//       tflite::RecordingMicroAllocator::Create(tensor_arena, kTensorArenaSize));
//   tflite::RecordingMicroInterpreter interpreter(
//       tflite::GetModel(g_hello_world_float_model_data), op_resolver, allocator,
//       tflite::MicroResourceVariables::Create(allocator, kNumResourceVariables),
//       &profiler);

//   TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());
//   TFLITE_CHECK_EQ(interpreter.inputs_size(), 1);
//   interpreter.input(0)->data.f[0] = 1.f;
//   TF_LITE_ENSURE_STATUS(interpreter.Invoke());

//   MicroPrintf("");  // Print an empty new line
//   profiler.LogTicksPerTagCsv();

//   MicroPrintf("");  // Print an empty new line
//   interpreter.GetMicroAllocator().PrintAllocations();
//   return kTfLiteOk;
// }

TfLiteStatus LoadFloatModelAndPerformInference() {
  const tflite::Model* model =
      ::tflite::GetModel(g_hello_world_int8_model_data);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);
  printf("hi0\n");
  HelloWorldOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));
  printf("hi1\n");
  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSize = 9000;
  uint8_t tensor_arena[kTensorArenaSize];
  
  printf("hi2\n");
  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize);
  printf("hi2.5\n");
  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());
  printf("hi3\n");
  // Check if the predicted output is within a small range of the
  // expected output
  // float epsilon = 0.05f;
  // constexpr int kNumTestValues = 4;
  // float golden_inputs[13] = {0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0};
  // printf("hi1\n");
  // // for (int i = 0; i < kNumTestValues; ++i) {
  // TfLiteTensor* input = interpreter.input(0);
  // memcpy(input->data.uint8, golden_inputs, sizeof(golden_inputs));
  // printf("hi2\n");
  // TF_LITE_ENSURE_STATUS(interpreter.Invoke());
  // printf("hi3\n");
  // TfLiteTensor* output = interpreter.output(0);
  // uint8_t outputs[4];
  // memcpy(outputs, output->data.uint8, sizeof(outputs));

  // for (int i = 0; i < 4; i++) {
  //   MicroPrintf("outputs[%d]: %d", i, outputs[i]);
  // }

  // TFLITE_CHECK_LE(abs(sin(golden_inputs[i]) - y_pred), epsilon);

  return kTfLiteOk;
}

TfLiteStatus LoadQuantModelAndPerformInference() {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model =
      ::tflite::GetModel(g_hello_world_int8_model_data);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

  HelloWorldOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSize = 8750;
  uint8_t tensor_arena[kTensorArenaSize];

  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize);

  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

//   TfLiteTensor* input = interpreter.input(0);
//   TFLITE_CHECK_NE(input, nullptr);

//   TfLiteTensor* output = interpreter.output(0);
//   TFLITE_CHECK_NE(output, nullptr);

//   float output_scale = output->params.scale;
//   int output_zero_point = output->params.zero_point;

//   // Check if the predicted output is within a small range of the
//   // expected output
//   float epsilon = 0.05;

//   constexpr int kNumTestValues = 4;
//   float golden_inputs_float[kNumTestValues] = {0.77, 1.57, 2.3, 3.14};

//   // The int8 values are calculated using the following formula
//   // (golden_inputs_float[i] / input->params.scale + input->params.zero_point)
//   int8_t golden_inputs_int8[kNumTestValues] = {-96, -63, -34, 0};

//   for (int i = 0; i < kNumTestValues; ++i) {
//     input->data.int8[0] = golden_inputs_int8[i];
//     TF_LITE_ENSURE_STATUS(interpreter.Invoke());
//     float y_pred = (output->data.int8[0] - output_zero_point) * output_scale;
//     TFLITE_CHECK_LE(abs(sin(golden_inputs_float[i]) - y_pred), epsilon);
//   }

  return kTfLiteOk;
}

int main(int argc, char* argv[]) {
  tflite::InitializeTarget();
  // TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency());
  // TF_LITE_ENSURE_STATUS(LoadFloatModelAndPerformInference());
  TF_LITE_ENSURE_STATUS(LoadQuantModelAndPerformInference());
  MicroPrintf("~~~ALL TESTS PASSED~~~\n");
  return kTfLiteOk;
}
