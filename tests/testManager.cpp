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
    // {"CPUExecutionProvider", std::nullopt},
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
    std::string modelPath1 = "../tests/model/wb_last.onnx";
    std::string modelPath2 = "../tests/model/yolo.onnx";
    modelManager manager(env);
    Model* model1 = manager.createModel(modelPath1, options, providers_2);
    Model* model2 = manager.createModel(modelPath2, options, providers_1);

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> inputTensors1;
    inputTensors1.push_back(createMockInput(memoryInfo, 1, 9, 256, 256));

    std::vector<Ort::Value> inputTensors2;
    inputTensors2.push_back(createMockInput(memoryInfo, 1, 3, 640, 640));

    std::cout << "Input model 1 has: " << inputTensors1.size() << " elements" << std::endl;
    std::cout << "Input model 2 has: " << inputTensors2.size() << " elements" << std::endl;
    try {
        std::shared_ptr<std::vector<Ort::Value>> outputTensors = model1->run(inputTensors1);
        
        std::cout << "Output model 1 has : " << outputTensors->size() << " elements\n";
        for (size_t i = 0; i < outputTensors->size(); ++i) {
            std::cout << "Head " << i << ": ";
            auto info = outputTensors->at(i).GetTensorTypeAndShapeInfo();    
            std::vector<int64_t> tensorShape = info.GetShape();
            for (int64_t dim : tensorShape) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;
        }


        std::shared_ptr<std::vector<Ort::Value>> outputTensors2 = model2->run(inputTensors2);
        std::cout << "Output model 2 has : " << outputTensors2->size() << " elements\n";
        for (size_t i = 0; i < outputTensors2->size(); ++i) {
            std::cout << "Head " << i << ": ";
            auto info = outputTensors2->at(i).GetTensorTypeAndShapeInfo();    
            std::vector<int64_t> tensorShape = info.GetShape();
            for (int64_t dim : tensorShape) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
    }
    return 0;
}