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
    InferenceEngine::init();  // This will work in dummy mode
    BenchmarkSuite::init();
    
    ESP_LOGI(TAG, "All components initialized. Starting main loop...");
    
    // Main processing loop
    int inference_count = 0;
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
        ESP_LOGI(TAG, "Inference %d: Class: %d, Confidence: %.2f, Latency: %" PRIu32 " us", 
                inference_count++, result.class_id, result.confidence, result.latency_us);
        
        vTaskDelay(pdMS_TO_TICKS(100)); // 10Hz classification rate (slower for testing)
    }
}