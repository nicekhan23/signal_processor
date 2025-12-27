#include "signal_processor.h"
#include <cmath>
#include "esp_dsp.h"
#include "esp_timer.h"
#include "driver/adc.h"
#include "esp_log.h"

#define SAMPLE_RATE_HZ      8000
#define FRAME_SIZE          256
#define FFT_SIZE            128

static const char *TAG = "SIGNAL_PROC";

// ADC buffer
static int16_t adc_buffer[FRAME_SIZE];

// Global queue and task handle declarations
QueueHandle_t frame_queue = NULL;
TaskHandle_t adc_task_handle = NULL;

// FFT buffers
static float fft_input[FFT_SIZE];
static float window[FFT_SIZE];

// Hamming window
static void init_window() {
    for (int i = 0; i < FFT_SIZE; i++) {
        window[i] = 0.54f - 0.46f * cosf(2 * M_PI * i / (FFT_SIZE - 1));
    }
}

// ADC task
static void adc_task(void *pvParameters) {
    // Configure ADC
    adc1_config_width(ADC_WIDTH_BIT_12);
    adc1_config_channel_atten(ADC1_CHANNEL_6, ADC_ATTEN_DB_12);
    
    while (true) {
        // Collect one frame
        for (int i = 0; i < FRAME_SIZE; i++) {
            adc_buffer[i] = adc1_get_raw(ADC1_CHANNEL_6);
            vTaskDelay(pdMS_TO_TICKS(1000 / SAMPLE_RATE_HZ));
        }
        
        // Send to processing queue
        if (frame_queue != NULL) {
            xQueueSend(frame_queue, adc_buffer, portMAX_DELAY);
        }
    }
}

SignalFrame SignalProcessor::capture_frame() {
    SignalFrame frame;
    
    // Wait for new frame from ADC task
    if (frame_queue != NULL && xQueueReceive(frame_queue, frame.samples, pdMS_TO_TICKS(100))) {
        frame.timestamp = esp_timer_get_time();
        frame.sample_count = FRAME_SIZE;
        return frame;
    }
    
    // Fallback: return empty frame
    memset(frame.samples, 0, sizeof(frame.samples));
    frame.sample_count = 0;
    frame.timestamp = esp_timer_get_time();
    return frame;
}

ProcessedData SignalProcessor::preprocess(const SignalFrame &frame) {
    ProcessedData data;
    
    // 1. Normalize to [-1, 1]
    float normalized[FRAME_SIZE];
    for (int i = 0; i < FRAME_SIZE; i++) {
        normalized[i] = (frame.samples[i] / 2048.0f) - 1.0f;  // 12-bit ADC
    }
    
    // 2. Apply window function
    for (int i = 0; i < FFT_SIZE; i++) {
        fft_input[i] = normalized[i] * window[i];
    }
    
    // 3. Compute FFT
    dsps_fft2r_fc32(fft_input, FFT_SIZE);
    dsps_bit_rev_fc32(fft_input, FFT_SIZE);
    dsps_cplx2reC_fc32(fft_input, FFT_SIZE);
    
    // 4. Compute magnitude spectrum (first half)
    for (int i = 0; i < FFT_SIZE/2; i++) {
        float real = fft_input[i * 2];
        float imag = fft_input[i * 2 + 1];
        data.spectrum[i] = sqrtf(real*real + imag*imag);
    }
    
    // 5. Convert to dB scale
    for (int i = 0; i < FFT_SIZE/2; i++) {
        data.spectrum[i] = 20 * log10f(data.spectrum[i] + 1e-6);
    }
    
    // 6. Quantize to int8 for model input
    for (int i = 0; i < 64; i++) {  // Downsample to 64 bins
        float avg = 0;
        for (int j = 0; j < 2; j++) {
            avg += data.spectrum[i*2 + j];
        }
        data.model_input[i] = static_cast<int8_t>((avg / 2.0f) * 127.0f);
    }
    
    return data;
}

void SignalProcessor::init() {
    ESP_LOGI(TAG, "Initializing Signal Processor");
    
    // Initialize FFT
    esp_err_t ret = dsps_fft2r_init_fc32(NULL, FFT_SIZE);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "FFT initialization failed");
    }
    
    // Create window function
    init_window();
    
    // Create frame queue
    frame_queue = xQueueCreate(5, sizeof(int16_t) * FRAME_SIZE);
    if (frame_queue == NULL) {
        ESP_LOGE(TAG, "Failed to create frame queue");
        return;
    }
    
    // Start ADC task
    BaseType_t task_ret = xTaskCreate(adc_task, "adc_task", 4096, NULL, 5, &adc_task_handle);
    if (task_ret != pdPASS) {
        ESP_LOGE(TAG, "Failed to create ADC task");
        vQueueDelete(frame_queue);
        frame_queue = NULL;
        return;
    }
    
    ESP_LOGI(TAG, "Signal Processor initialized");
}