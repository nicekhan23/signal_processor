idf_component_register(SRCS "main.cpp"
                              "signal_processor.cpp"
                              "inference_engine.cpp"
                              "benchmark_suite.cpp"
                              "model_data.cpp"
                       INCLUDE_DIRS "."
                       REQUIRES 
                           driver
                           esp_adc
                           esp_timer
                           nvs_flash
                           spiffs
                           esp-tflite-micro
                           esp-dsp
                       EMBED_TXTFILES "partitions.csv")

# Remove the USE_DUMMY_MODEL definition
# target_compile_definitions(${COMPONENT_LIB} PRIVATE "USE_DUMMY_MODEL")