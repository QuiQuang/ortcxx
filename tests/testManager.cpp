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

std::optional<std::map<std::string, std::optional<std::map<std::string, std::string>>>> providers_1 = std::map<std::string, std::optional<std::map<std::string, std::string>>> {
    {"CPUExecutionProvider", std::nullopt},
    {
        "CUDAExecutionProvider", 
        std::map<std::string, std::string> {
            {"device_id", "0"}
        }
    },
    {"OpenVINOExecutionProvider", std::nullopt}
};

std::optional<std::map<std::string, std::optional<std::map<std::string, std::string>>>> providers_2 = std::map<std::string, std::optional<std::map<std::string, std::string>>> {
    {"CPUExecutionProvider", std::nullopt},
    {
        "CUDAExecutionProvider", 
        std::map<std::string, std::string> {
            {"device_id", "0"}
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
    std::string modelPath1 = "../tests/models/test_wb.onnx";
    std::string modelPath2 = "../tests/models/test_wb.onnx";
    modelManager manager(env);
    Model* model1 = manager.createModel(modelPath1, options, providers_2);
    Model* model2 = manager.createModel(modelPath2, options, providers_1);
    // run model
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor1 = createMockInput(memoryInfo, 1, 9, 256, 256);
    Ort::Value inputTensor2 = createMockInput(memoryInfo, 1, 9, 256, 256);
    try {
        std::shared_ptr<std::vector<Ort::Value>> outputTensor = model1->run(inputTensor1);
        float* outputData = outputTensor->at(0).GetTensorMutableData<float>();
        std::cout << "Model1 output: " << outputData[0] << std::endl;

        std::shared_ptr<std::vector<Ort::Value>> outputTensor2 = model2->run(inputTensor2);
        float* outputData2 = outputTensor2->at(0).GetTensorMutableData<float>();
        std::cout << "Model2 output: " << outputData2[0] << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
    }
    return 0;
}