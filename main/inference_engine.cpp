#include "inference_engine.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"  // Generated from Colab

static const char *TAG = "INFERENCE";

// Tensor arena (memory for TFLite)
// Change from 20KB to something more appropriate
constexpr int kTensorArenaSize = 10 * 1024;  // 10KB instead of 20KB
static uint8_t tensor_arena[kTensorArenaSize];

// TFLite objects
static tflite::MicroMutableOpResolver<10> resolver;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;

void InferenceEngine::init() {
    ESP_LOGI(TAG, "Initializing TFLite Micro");
    
    // Load model from flash (converted C array)
    const tflite::Model* model = tflite::GetModel(g_model_data);
    
    // Verify model version
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model version mismatch");
        return;
    }
    
    // Add operations
    resolver.AddFullyConnected();
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddQuantize();
    resolver.AddDequantize();
    
    // Build interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    
    interpreter = &static_interpreter;
    
    // Allocate tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        ESP_LOGE(TAG, "Tensor allocation failed");
        return;
    }
    
    // Get input/output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    // Log model info
    ESP_LOGI(TAG, "Input dimensions: %dD", input->dims->size);
    for (int i = 0; i < input->dims->size; i++) {
        ESP_LOGI(TAG, "  dim[%d] = %d", i, input->dims->data[i]);
    }
    ESP_LOGI(TAG, "Input type: %d", input->type);
    ESP_LOGI(TAG, "Model initialized successfully");
}

InferenceResult InferenceEngine::infer(const ProcessedData &data) {
    InferenceResult result;
    uint64_t start_time = esp_timer_get_time();
    
    // Copy preprocessed data to input tensor
    if (input->type == kTfLiteInt8) {
        // INT8 quantized model
        int8_t* input_data = input->data.int8;
        for (int i = 0; i < 64; i++) {
            input_data[i] = data.model_input[i];
        }
    } else if (input->type == kTfLiteFloat32) {
        // Float32 model
        float* input_data = input->data.f;
        for (int i = 0; i < 64; i++) {
            input_data[i] = data.model_input[i] / 127.0f;
        }
    }
    
    // Run inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "Inference failed");
        result.class_id = -1;
        return result;
    }
    
    // Process output
    if (output->type == kTfLiteInt8) {
        // Dequantize INT8 output
        int8_t* output_data = output->data.int8;
        float scale = output->params.scale;
        int zero_point = output->params.zero_point;
        
        int max_idx = 0;
        float max_val = (output_data[0] - zero_point) * scale;
        
        for (int i = 1; i < 4; i++) {
            float val = (output_data[i] - zero_point) * scale;
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }
        
        result.class_id = max_idx;
        result.confidence = max_val;
        
    } else if (output->type == kTfLiteFloat32) {
        // Float32 output
        float* output_data = output->data.f;
        int max_idx = 0;
        
        for (int i = 1; i < 4; i++) {
            if (output_data[i] > output_data[max_idx]) {
                max_idx = i;
            }
        }
        
        result.class_id = max_idx;
        result.confidence = output_data[max_idx];
    }
    
    result.latency_us = esp_timer_get_time() - start_time;
    result.timestamp = start_time;
    
    return result;
}