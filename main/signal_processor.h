#ifndef SIGNAL_PROCESSOR_H
#define SIGNAL_PROCESSOR_H

#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"

// Constants
#define SAMPLE_RATE_HZ      8000
#define FRAME_SIZE          256
#define FFT_SIZE            128

// Data structures
typedef struct {
    int16_t samples[FRAME_SIZE];
    uint64_t timestamp;
    uint32_t sample_count;
} SignalFrame;

typedef struct {
    float spectrum[FFT_SIZE/2];
    int8_t model_input[64];
} ProcessedData;

// SignalProcessor class
class SignalProcessor {
public:
    static void init();
    static SignalFrame capture_frame();
    static ProcessedData preprocess(const SignalFrame &frame);
};

#endif // SIGNAL_PROCESSOR_H