#include <iostream>
#include <cinnamon/core.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <stdint.h>
#include <fstream>

using namespace cinnamon::model;

std::optional<std::map<std::string, std::any>> options = std::map<std::string, std::any> {
    {"parallel", true},
    {"inter_ops_threads", 0},
    {"intra_ops_threads", 0},
    {"graph_optimization_level", 0}
};

std::optional<std::map<std::string, std::optional<std::map<std::string, std::string>>>> providers = std::map<std::string, std::optional<std::map<std::string, std::string>>> {
    {"CPUExecutionProvider", std::nullopt},
    {
        "CUDAExecutionProvider", 
        std::map<std::string, std::string> {
            {"device_id", "0"},
            {"gpu_mem_limit", "2147483648"},
            {"arena_extend_strategy", "kSameAsRequested"}
        }
    },
    {"OpenVINOExecutionProvider", std::nullopt}
};

Ort::Value createMockInput(Ort::MemoryInfo& memoryInfo, int64_t batchSize = 1, int64_t channels = 9, int64_t height = 256, int64_t width = 256) {
    const std::array<int64_t, 4> inputShape = {batchSize, channels, height, width};
    std::vector<float> inputValues(batchSize * channels * height * width, 1.0f);
    return Ort::Value::CreateTensor<float>(memoryInfo, inputValues.data(), inputValues.size(), inputShape.data(), inputShape.size());
}

int main() {
    std::shared_ptr<Ort::Env> env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "test");
    // std::shared_ptr<Ort::Allocator> allocator = std::make_shared<Ort::AllocatorWithDefaultOptions>();

    modelManager manager(env);

    // Create and test model
    std::string modelPath1 = "../tests/model/wb.onnx";
    std::string modelPath2 = "../tests/model/yolo.onnx";
    Model* model = manager.createModel(modelPath1, options, providers);
    Model* model2 = manager.createModel(modelPath2, options, providers);
    
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor1 = createMockInput(memoryInfo , 1, 9, 256, 256);
    // Ort::Value inputTensor2 = createMockInput(memoryInfo, 1, 3, 640, 640);
    manager.setTimeOut(3);

    try {
        std::shared_ptr<std::vector<Ort::Value>> outputTensor1 = model->run(inputTensor1);
        std::cout << "Model output: ";
        auto info = outputTensor1->at(0).GetTensorTypeAndShapeInfo();    
        std::vector<int64_t> tensorShape = info.GetShape();
        for (int64_t dim : tensorShape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
    }

    // Test session clock update and retrieval
    manager.updateSessionClock(modelPath1);
    float sessionDuration = manager.getSessionClock(modelPath1);
    std::cout << "Initial session duration: " << sessionDuration << " seconds" << std::endl;

    std::this_thread::sleep_for(std::chrono::seconds(2));

    sessionDuration = manager.getSessionClock(modelPath1);
    std::cout << "Session duration after 2 seconds: " << sessionDuration << " seconds" << std::endl;

    // Wait to check
    int Time = 5;
    std::this_thread::sleep_for(std::chrono::seconds(Time)); // simulate some time passing
    std::cout << "Check " << Time << " done\n";

    manager.stopGC();
    return 0;
}