#include <iostream>
#include <cinnamon/core.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <stdint.h>
#include <fstream>
#define INPUT_SIZE 224

using namespace cinnamon::model;

std::vector<float> read_image(const char* image_path) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        throw std::runtime_error("Could not read the image: " + std::string(image_path));
    }
    
    cv::resize(image, image, cv::Size(INPUT_SIZE, INPUT_SIZE));
    image.convertTo(image, CV_32F, 1.0 / 255.0);
    std::vector<float> image_vector;
    image_vector.assign((float*)image.datastart, (float*)image.dataend);

    return image_vector;
}

void saveTensorToFile(Ort::Value tensor, const std::string& filename) {
    Ort::AllocatorWithDefaultOptions allocator;
    auto shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
    size_t totalSize = tensor.GetTensorTypeAndShapeInfo().GetElementCount();

    float* data = tensor.GetTensorMutableData<float>();

    cv::Mat image(shape[2], shape[3], CV_32FC1, const_cast<float*>(data));
    image.convertTo(image, CV_8U, 255.0); 
    cv::imwrite(filename, image);
    std::cout << "Saved tensor as PNG: " << filename << std::endl;
}

Ort::Value createMockInput_Const(Ort::MemoryInfo& memoryInfo, int64_t channels, int64_t size) {
    const std::array<int64_t, 4> inputShape = {1, channels, size, size};
    std::vector<float> inputValues(1 * channels * size * size, 1.0);
    return Ort::Value::CreateTensor<float>(memoryInfo, inputValues.data(), inputValues.size(), inputShape.data(), inputShape.size());
}

Ort::Value createMockInput(Ort::MemoryInfo& memoryInfo, int64_t channels, int64_t size) {
    auto image_vector_d = read_image("/home/alex/Work/ortcxx/tests/sample/8D5U5524_D.png");
    auto image_vector_s = read_image("/home/alex/Work/ortcxx/tests/sample/8D5U5524_S.png");
    auto image_vector_t = read_image("/home/alex/Work/ortcxx/tests/sample/8D5U5524_T.png");

    std::vector<float> concatenated_image;
    concatenated_image.reserve(image_vector_d.size() + image_vector_s.size() + image_vector_t.size());
    concatenated_image.insert(concatenated_image.end(), image_vector_d.begin(), image_vector_d.end());
    concatenated_image.insert(concatenated_image.end(), image_vector_s.begin(), image_vector_s.end());
    concatenated_image.insert(concatenated_image.end(), image_vector_t.begin(), image_vector_t.end());

    const std::array<int64_t, 4> inputShape = {1, channels, size, size};
    return Ort::Value::CreateTensor<float>(memoryInfo, concatenated_image.data(), concatenated_image.size(), inputShape.data(), inputShape.size());
}

Ort::Value createMockInput_Image(Ort::MemoryInfo& memoryInfo, int64_t channels, int64_t size) {
    auto image_vector_d = read_image("../tests/samples/MyIC_Inline_28632.jpg");

    std::vector<float> concatenated_image;
    concatenated_image.reserve(image_vector_d.size());
    concatenated_image.insert(concatenated_image.end(), image_vector_d.begin(), image_vector_d.end());

    const std::array<int64_t, 4> inputShape = {1, channels, size, size};
    return Ort::Value::CreateTensor<float>(memoryInfo, concatenated_image.data(), concatenated_image.size(), inputShape.data(), inputShape.size());
}

std::optional<std::map<std::string, std::any>> options = std::map<std::string, std::any> {
    {"parallel", true},
    {"inter_ops_threads", 0},
    {"intra_ops_threads", 0},
    {"graph_optimization_level", 0}
};

std::optional<std::map<std::string, std::optional<std::map<std::string, std::string>>>> providers = std::map<std::string, std::optional<std::map<std::string, std::string>>> {
    // {"CPUExecutionProvider", std::nullopt},
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

int main(){
    std::shared_ptr<Model> model = std::make_shared<Model>("../tests/model/sam.onnx", options, providers);
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = createMockInput_Image(memoryInfo, 3, 224);
    
    auto info = inputTensor.GetTensorTypeAndShapeInfo();    
    std::vector<int64_t> tensorShape = info.GetShape();

    std::cout << "Input shape data: ";
    for (int64_t dim : tensorShape) {
        std::cout << dim << " ";
    }

    // std::cout << std::endl << inputTensor.IsTensor() << std::endl; is tensor
    // auto test = inputTensor.GetTensorTypeAndShapeInfo();
    // std::cout << std::endl << test.GetElementType() << std::endl;
    // std::cout << std::endl << test.GetShape().size() << std::endl;
    // // std::cout << "Tensor Values:" << std::endl;
    // // // std::cout << tensorData[0] << std::endl;

    // std::cout << "hi";
    // const float* tensorData = inputTensor.GetTensorData<float>();
    // std::cout << "Data: ";
    // for (size_t i = 0; i < tensorShape.back(); i++) {
    //     std::cout << tensorData[i] << std::endl;
    // }
    
    try {
        std::shared_ptr<std::vector<Ort::Value>> outputTensor = model->run(inputTensor);
        // for (size_t i = 0; i < outputTensor->size(); ++i) {
        //     std::string filename = "/home/alex/Work/ortcxx/tests/results/output_tensor_" + std::to_string(i) + ".jpg";

        //     auto shape = outputTensor->at(i).GetTensorTypeAndShapeInfo().GetShape();
        //     size_t totalSize = outputTensor->at(i).GetTensorTypeAndShapeInfo().GetElementCount();

        //     cv::Mat image(shape[2], shape[3], CV_32FC1, outputTensor->at(i));
        //     image.convertTo(image, CV_8U, 255.0); 
        //     cv::imwrite(filename, image);
        //     std::cout << "Saved tensor as PNG: " << filename << std::endl;
        // }

        std::cout << "\nOutput shape:";
        auto info = outputTensor->at(0).GetTensorTypeAndShapeInfo();    
        std::vector<int64_t> tensorShape = info.GetShape();
        for (int64_t dim : tensorShape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
    }
    return 0;
}