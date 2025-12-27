#include "benchmark_suite.h"
#include "esp_system.h"
#include "driver/i2c.h"
#include "esp_log.h"

#define INA219_ADDR 0x40
#define I2C_PORT    I2C_NUM_0

static uint32_t total_latency = 0;
static uint32_t inference_count = 0;
static uint32_t total_energy_uJ = 0;

void BenchmarkSuite::init() {
    // Initialize I2C for INA219
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = GPIO_NUM_21,
        .scl_io_num = GPIO_NUM_22,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master = {
            .clk_speed = 100000
        },
        .clk_flags = 0  // Add this to fix the warning
    };
    
    i2c_param_config(I2C_PORT, &conf);
    i2c_driver_install(I2C_PORT, conf.mode, 0, 0, 0);
    
    // Configure INA219
    uint8_t config_data[2] = {0x39, 0x9F};  // 32V, 2A range
    i2c_master_write_to_device(I2C_PORT, INA219_ADDR, config_data, 2, pdMS_TO_TICKS(100));
}

void BenchmarkSuite::record_inference(const InferenceResult &result) {
    // Record latency
    total_latency += result.latency_us;
    inference_count++;
    
    // Measure power
    uint8_t reg_addr = 0x01;  // Shunt voltage register
    uint8_t buffer[2];
    
    i2c_master_write_read_device(I2C_PORT, INA219_ADDR, 
                                 &reg_addr, 1, buffer, 2, pdMS_TO_TICKS(100));
    
    int16_t shunt_voltage = (buffer[0] << 8) | buffer[1];
    float current_mA = shunt_voltage * 0.01f;  // 0.1mΩ shunt resistor
    
    // Calculate energy for this inference
    float voltage = 3.3f;  // ESP32 VCC
    float power_mW = voltage * current_mA;
    float energy_uJ = power_mW * result.latency_us / 1000.0f;
    
    total_energy_uJ += energy_uJ;
    
    // Log every 100 inferences
    if (inference_count % 100 == 0) {
        ESP_LOGI("BENCHMARK", "Avg Latency: %.2f us, Avg Energy: %.2f uJ",
                (float)total_latency / inference_count,
                (float)total_energy_uJ / inference_count);
        
        // Log memory usage
        size_t free_heap = esp_get_free_heap_size();
        ESP_LOGI("BENCHMARK", "Free Heap: %d bytes", free_heap);
    }
}