#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "inference_engine.h"
#include "signal_processor.h"
#include "benchmark_suite.h"

static const char *TAG = "EDGE_AI_MAIN";

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "Starting Edge AI Signal Classifier");
    
    // Initialize components
    SignalProcessor::init();
    InferenceEngine::init();
    BenchmarkSuite::init();
    
    // Main processing loop
    while (true) {
        // 1. Acquire signal frame
        SignalFrame frame = SignalProcessor::capture_frame();
        
        // 2. Preprocess
        ProcessedData data = SignalProcessor::preprocess(frame);
        
        // 3. Run inference
        InferenceResult result = InferenceEngine::infer(data);
        
        // 4. Benchmark and log
        BenchmarkSuite::record_inference(result);
        
        // 5. Output result
        ESP_LOGI(TAG, "Class: %d, Confidence: %.2f, Latency: %" PRIu32 " us", 
                result.class_id, result.confidence, result.latency_us);
        
        vTaskDelay(pdMS_TO_TICKS(10)); // 100Hz classification rate
    }
}