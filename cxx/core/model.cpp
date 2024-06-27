#include "core.h"
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <future>
#include <thread>
#include <map>

using namespace cinrt::model;

std::map<std::string, std::string> mapProviders = {
  {"CPUExecutionProvider", "CPU"},
  {"CUDAExecutionProvider", "Cuda"},
  {"OpenVINOExecutionProvider", "OpenVINO"}
};

std::map<int, std::string> buildNewIndexMap() {
    std::map<int, std::string> indexMap;
    int index = 0;
    for (const auto& pair : mapProviders) {
        indexMap[index++] = pair.first;
    }
    return indexMap;
}
std::map<int, std::string> mapProvidersByIndex = buildNewIndexMap();

Model::Model(
  std::string model,
  bool parallel,
  int graphOpLevel,
  int interThreads,
  int intraThreads,
  int inProvider
) {
  this->_env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "test");
  this->_sessionOptions = this->getSessionOptions(parallel, graphOpLevel, interThreads, intraThreads);
  
  this->provider = (inProvider == -1) ? Ort::GetAvailableProviders()[0] : mapProvidersByIndex[inProvider];
  if (this->provider != "CPUExecutionProvider") {
    if (this->provider == "CUDAExecutionProvider") {
      OrtCUDAProviderOptions options;
      options.device_id = 0; 
      options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
      options.arena_extend_strategy = 0;
      options.do_copy_in_default_stream = 0;
      this->_sessionOptions->AppendExecutionProvider_CUDA(options);
    } 
    else if (this->provider == "OpenVINOExecutionProvider") {
      OrtOpenVINOProviderOptions options;
      this->_sessionOptions->AppendExecutionProvider_OpenVINO(options);
    } 
    else {
      throw std::runtime_error("Unsupported provider: " + this->provider);
    }

    this->_session = std::make_unique<Ort::Session>(*this->_env, model.c_str(), *this->_sessionOptions);
    Ort::MemoryInfo gpuMemoryInfo{"Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault};
    this->_allocator = std::make_shared<Ort::Allocator>(*this->_session, gpuMemoryInfo);
  }
  else {
    this->_session = std::make_unique<Ort::Session>(*this->_env, model.c_str(), *this->_sessionOptions);
    this->_allocator = std::make_shared<Ort::Allocator>(*this->_session, Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
  }

  Ort::AllocatedStringPtr inputName = this->_session->GetInputNameAllocated(0, *this->_allocator);
  Ort::AllocatedStringPtr outputName = this->_session->GetOutputNameAllocated(0, *this->_allocator);
  this->inputNames = std::make_shared<const char*>(inputName.release());
  this->outputNames = std::make_shared<const char*>(outputName.release());
}

Model::Model(
  std::shared_ptr<Ort::Env> env,
  std::shared_ptr<Ort::Allocator> allocator,
  std::string model,
  bool parallel,
  int graphOpLevel,
  int interThreads,
  int intraThreads,
  int inProvider
) {
  _env = env;
  _allocator = allocator;
  _sessionOptions = getSessionOptions(parallel, graphOpLevel, interThreads, intraThreads);

  this->provider = (inProvider == -1) ? Ort::GetAvailableProviders()[0] : mapProvidersByIndex[inProvider];
  if (this->provider != "CPUExecutionProvider") {
    if (this->provider == "CUDAExecutionProvider") {
      OrtCUDAProviderOptions options;
      options.device_id = 0; 
      options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
      options.arena_extend_strategy = 0;
      options.do_copy_in_default_stream = 0;
      this->_sessionOptions->AppendExecutionProvider_CUDA(options);
    } 
    else if (this->provider == "OpenVINOExecutionProvider") {
      OrtOpenVINOProviderOptions options;
      this->_sessionOptions->AppendExecutionProvider_OpenVINO(options);
    } 
    else {
      throw std::runtime_error("Unsupported provider: " + this->provider);
    }

    this->_session = std::make_unique<Ort::Session>(*this->_env, model.c_str(), *this->_sessionOptions);
    Ort::MemoryInfo gpuMemoryInfo{"Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault};
    this->_allocator = std::make_shared<Ort::Allocator>(*this->_session, gpuMemoryInfo);
  }
  else {
    this->_session = std::make_unique<Ort::Session>(*this->_env, model.c_str(), *this->_sessionOptions);
    this->_allocator = std::make_shared<Ort::Allocator>(*this->_session, Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
  }
  
  Ort::AllocatedStringPtr inputName = this->_session->GetInputNameAllocated(0, *this->_allocator);
  Ort::AllocatedStringPtr outputName = this->_session->GetOutputNameAllocated(0, *this->_allocator);
  this->inputNames = std::make_shared<const char*>(inputName.release());
  this->outputNames = std::make_shared<const char*>(outputName.release());
}

std::unique_ptr<Ort::SessionOptions> Model::getSessionOptions(
  bool parallel, 
  int graphOpLevel, 
  int intraThreads, 
  int interThreads
) {
  std::unique_ptr<Ort::SessionOptions> sessionOptions = std::make_unique<Ort::SessionOptions>(Ort::SessionOptions());
  if (parallel)
    sessionOptions->SetExecutionMode(ExecutionMode::ORT_PARALLEL);
  else
    sessionOptions->SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  if (intraThreads > 0)
    sessionOptions->SetIntraOpNumThreads(intraThreads);
  if (interThreads > 0)
    sessionOptions->SetInterOpNumThreads(interThreads);
  switch (graphOpLevel){
  case 0:
    sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    break;
  case 1:
    sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
    break;
  case 2:
    sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    break;
  case 3:
    sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    break;
  default:
    sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    break;
  }
  return sessionOptions;
}

std::shared_ptr<std::vector<Ort::Value>> Model::run(
  const Ort::Value& inputs,
  std::shared_ptr<const char*> outputHead,
  const Ort::RunOptions& runOptions){
  if (outputHead != nullptr)
    this->outputNames = outputHead;
  if (this->_session == nullptr)
    throw std::runtime_error("Session is not initialized");

  if (this->provider == "CPUExecutionProvider") {
    try {
      std::vector<Ort::Value> output_vector = this->_session->Run(runOptions, &*inputNames, &inputs, 1, &*outputNames, 1);
      return std::make_shared<std::vector<Ort::Value>>(std::move(output_vector));
    }
    catch (Ort::Exception& exception) {
      std::cout << "Error: " << exception.what() << std::endl;
    }
  }
  else {
    std::string deviceType = mapProviders[this->provider];

    // ------------------------------------------------------------------------------------------------------
    // ----------------------------- This code has bugs: about Cuda and PReLU -------------------------------
    // ------------------------------------------------------------------------------------------------------

    Ort::MemoryInfo cpuMemoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::MemoryInfo gpuMemoryInfo{deviceType.c_str(), OrtDeviceAllocator, 0, OrtMemTypeDefault};
    
    const float* input_data = inputs.GetTensorData<float>();
    Ort::Value inputOnGpu = Ort::Value::CreateTensor<float>(gpuMemoryInfo, const_cast<float*>(input_data), inputs.GetTensorTypeAndShapeInfo().GetElementCount(), inputs.GetTensorTypeAndShapeInfo().GetShape().data(), inputs.GetTensorTypeAndShapeInfo().GetShape().size());


    Ort::IoBinding bind{*this->_session};
    bind.BindInput(*this->inputNames, inputOnGpu);
    bind.BindOutput(*this->outputNames, gpuMemoryInfo);

    try {
      this->_session->Run(runOptions, bind);
      std::vector<Ort::Value> output_tensors = bind.GetOutputValues();

      std::cout << "OK" << std::endl;
      std::vector<Ort::Value> output_tensors_on_cpu;
      for (auto& output_tensor : output_tensors) {
        std::vector<int64_t> output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
        Ort::Value outputOnCpu = Ort::Value::CreateTensor<float>(cpuMemoryInfo, output_tensor.GetTensorMutableData<float>(), output_tensor.GetTensorTypeAndShapeInfo().GetElementCount(), output_shape.data(), output_shape.size());
        output_tensors_on_cpu.push_back(std::move(outputOnCpu));
      }

      return std::make_shared<std::vector<Ort::Value>>(std::move(output_tensors_on_cpu));
    }
    catch (Ort::Exception& exception) {
      std::cout << "Error: " << exception.what() << std::endl;
    }
    

    // ------------------------------------------------------------------------------------------------------
    // ------------------------------- This code has bugs: segmentation fault -------------------------------
    // ------------------------------------------------------------------------------------------------------

    // Ort::MemoryInfo outputMemoryInfo{"Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault};

    // Ort::IoBinding bind{*this->_session};
    // bind.BindInput(*this->inputNames, inputs);
    // bind.BindOutput(*this->outputNames, outputMemoryInfo);

    // try {
    //   this->_session->Run(runOptions, bind);
    //   std::vector<Ort::Value> output_tensors = bind.GetOutputValues();
    //   return std::make_shared<std::vector<Ort::Value>>(std::vector<Ort::Value>{std::move(output_tensor)});
    // }
    // catch (Ort::Exception& exception) {
    //   std::cout << "Error: " << exception.what() << std::endl;
    // }
  }

  return nullptr;
}

std::future<std::shared_ptr<std::vector<Ort::Value>>> Model::runAsync(
  const Ort::Value& inputs, 
  std::shared_ptr<const char*> outputHead,
  const Ort::RunOptions runOptions){
  if (outputHead != nullptr)
    this->outputNames = outputHead;
  if (this->_session == nullptr)
    throw std::runtime_error("Session is not initialized");
  return std::async(std::launch::async, &Model::run, this, std::cref(inputs), std::cref(outputNames), std::cref(runOptions));
}