#ifndef BENCHMARK_SUITE_H
#define BENCHMARK_SUITE_H

#include "inference_engine.h"

// BenchmarkSuite class
class BenchmarkSuite {
public:
    static void init();
    static void record_inference(const InferenceResult &result);
};

#endif // BENCHMARK_SUITE_H