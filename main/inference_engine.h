#ifndef INFERENCE_ENGINE_H
#define INFERENCE_ENGINE_H

#include <cstdint>
#include "signal_processor.h"

// Inference result structure
struct InferenceResult {
    int class_id;           // 0=sine, 1=square, 2=triangle, 3=sawtooth
    float confidence;       // Confidence score 0.0-1.0
    uint64_t latency_us;    // Inference time in microseconds
    uint64_t timestamp;     // When inference started
};

// InferenceEngine class
class InferenceEngine {
public:
    static void init();
    static InferenceResult infer(const ProcessedData &data);
};

#endif // INFERENCE_ENGINE_H