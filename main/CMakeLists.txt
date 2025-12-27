idf_component_register(SRCS "main.cpp"
                              "signal_processor.cpp"
                              "inference_engine.cpp"
                              "benchmark_suite.cpp"
                       INCLUDE_DIRS "."
                       REQUIRES 
                           driver
                           adc
                           esp_dsp
                           esp_timer
                           nvs_flash
                           spiffs
                           esp-tflite-micro
                       EMBED_TXTFILES "partitions.csv")