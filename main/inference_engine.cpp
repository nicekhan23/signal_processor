#include "inference_engine.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"

static const char *TAG = "INFERENCE";

// Tensor arena (memory for TFLite) - increased for real models
constexpr int kTensorArenaSize = 30 * 1024;  // 30KB for real models
static uint8_t tensor_arena[kTensorArenaSize] __attribute__((aligned(16)));

// TFLite objects
static tflite::MicroMutableOpResolver<10> resolver;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;

static bool model_initialized = false;

void InferenceEngine::init() {
    ESP_LOGI(TAG, "Initializing TensorFlow Lite Micro");
    
    // Check if we have valid model data
    if (g_model_data_len < 1000) {  // Real TFLite models are usually > 1KB
        ESP_LOGE(TAG, "Model data too small (%d bytes). Need a trained model.", g_model_data_len);
        ESP_LOGE(TAG, "Train a model using the Python script and convert to C array.");
        return;
    }
    
    // Load model from flash (converted C array)
    const tflite::Model* model = tflite::GetModel(g_model_data);
    
    if (model == nullptr) {
        ESP_LOGE(TAG, "Failed to get model from data");
        return;
    }
    
    // Verify model version
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model version mismatch: expected %d, got %d", 
                TFLITE_SCHEMA_VERSION, model->version());
        return;
    }
    
    // Add operations - adjust based on your model architecture
    resolver.AddFullyConnected();
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddRelu();
    
    // Build interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    
    interpreter = &static_interpreter;
    
    // Allocate tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        ESP_LOGE(TAG, "Tensor allocation failed. Tensor arena might be too small.");
        ESP_LOGE(TAG, "Try increasing kTensorArenaSize or reducing model size.");
        return;
    }
    
    // Get input/output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    if (input == nullptr || output == nullptr) {
        ESP_LOGE(TAG, "Failed to get input/output tensors");
        return;
    }
    
    // Log model info
    ESP_LOGI(TAG, "Model loaded successfully");
    ESP_LOGI(TAG, "Input dimensions: %dD", input->dims->size);
    for (int i = 0; i < input->dims->size; i++) {
        ESP_LOGI(TAG, "  dim[%d] = %d", i, input->dims->data[i]);
    }
    
    // Check input shape
    int input_size = 1;
    for (int i = 0; i < input->dims->size; i++) {
        input_size *= input->dims->data[i];
    }
    
    if (input_size != 64) {
        ESP_LOGW(TAG, "Expected input size 64, got %d", input_size);
        ESP_LOGW(TAG, "Make sure your model expects 64 input features");
    }
    
    ESP_LOGI(TAG, "Input type: %s", 
             input->type == kTfLiteInt8 ? "INT8" : 
             input->type == kTfLiteFloat32 ? "FLOAT32" : "UNKNOWN");
    ESP_LOGI(TAG, "Output type: %s", 
             output->type == kTfLiteInt8 ? "INT8" : 
             output->type == kTfLiteFloat32 ? "FLOAT32" : "UNKNOWN");
    ESP_LOGI(TAG, "Output dimensions: %d", output->dims->data[1]);
    
    model_initialized = true;
    ESP_LOGI(TAG, "Inference engine ready");
}

InferenceResult InferenceEngine::infer(const ProcessedData &data) {
    InferenceResult result;
    uint64_t start_time = esp_timer_get_time();
    
    // Check if model is initialized
    if (!model_initialized) {
        ESP_LOGE(TAG, "Inference engine not initialized");
        result.class_id = -1;
        result.confidence = 0.0f;
        result.latency_us = 0;
        result.timestamp = start_time;
        return result;
    }
    
    if (interpreter == nullptr || input == nullptr || output == nullptr) {
        ESP_LOGE(TAG, "Inference engine not properly initialized");
        result.class_id = -1;
        result.confidence = 0.0f;
        result.latency_us = 0;
        result.timestamp = start_time;
        return result;
    }
    
    // Copy preprocessed data to input tensor
    if (input->type == kTfLiteInt8) {
        // INT8 quantized model
        int8_t* input_data = input->data.int8;
        for (int i = 0; i < 64; i++) {
            // The training preprocessing should match ESP32 preprocessing
            // If your model expects normalized [-1, 1] range, adjust accordingly
            input_data[i] = data.model_input[i];
        }
    } else if (input->type == kTfLiteFloat32) {
        // Float32 model
        float* input_data = input->data.f;
        for (int i = 0; i < 64; i++) {
            // Convert from int8 [-128, 127] to float [-1.0, ~1.0]
            input_data[i] = data.model_input[i] / 127.0f;
        }
    } else {
        ESP_LOGE(TAG, "Unsupported input type: %d", input->type);
        result.class_id = -1;
        result.confidence = 0.0f;
        result.latency_us = 0;
        result.timestamp = start_time;
        return result;
    }
    
    // Run inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "Inference failed with status: %d", invoke_status);
        result.class_id = -1;
        result.confidence = 0.0f;
        result.latency_us = 0;
        result.timestamp = start_time;
        return result;
    }
    
    // Process output based on type
    if (output->type == kTfLiteInt8) {
        // Dequantize INT8 output
        int8_t* output_data = output->data.int8;
        float scale = output->params.scale;
        int zero_point = output->params.zero_point;
        
        int max_idx = 0;
        float max_val = (output_data[0] - zero_point) * scale;
        
        // Find class with highest probability
        for (int i = 1; i < output->dims->data[1]; i++) {
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
        
        // Find class with highest probability
        for (int i = 1; i < output->dims->data[1]; i++) {
            if (output_data[i] > output_data[max_idx]) {
                max_idx = i;
            }
        }
        
        result.class_id = max_idx;
        result.confidence = output_data[max_idx];
    } else {
        ESP_LOGE(TAG, "Unsupported output type: %d", output->type);
        result.class_id = -1;
        result.confidence = 0.0f;
    }
    
    result.latency_us = esp_timer_get_time() - start_time;
    result.timestamp = start_time;
    
    return result;
}