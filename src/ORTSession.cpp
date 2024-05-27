#include <iostream>
#include <thread>
#include "ORTSession.h"
#include <unistd.h>
#include <onnxruntime_cxx_api.h>


ORTSession::ORTSession() : env(ORT_LOGGING_LEVEL_WARNING, "ORTSession"), session(nullptr), optimizerLevel(0) {}

ORTSession::~ORTSession() {
  if (session != nullptr) {
    delete session;
  }
}

void ORTSession::loadModel(
  const std::string modelPath, 
  std::string threadOption,
  int optimizerLevel = 0, 
  const std::string optimizerPath = "",
  const std::string runMode = "async",
  const std::string executionMode = "sequential",
  int numThreads = 0) {
  switch (optimizerLevel)
  {
  case 0:
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    break;
  case 1:
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
    break;
  case 2:
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    break;
  case 3:
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    break;
  default:
    break;
  }
  if (!optimizerPath.empty()) {
    sessionOptions.SetOptimizedModelFilePath(optimizerPath.c_str());
  }
  if (numThreads == 0) { // if not set, use hardware_concurrency
    const int numThreads = std::thread::hardware_concurrency();
  }
  else {
    const int numThreads = numThreads;
  }
  if (threadOption == "intra") {
    sessionOptions.SetIntraOpNumThreads(numThreads);
  } else if (threadOption == "inter_intra" or threadOption == "intra_inter") {
    sessionOptions.SetInterOpNumThreads(numThreads);
    sessionOptions.SetIntraOpNumThreads(numThreads);
  } 
  else  { // default is inter
    sessionOptions.SetInterOpNumThreads(numThreads);
  }
  if (executionMode == "parallel") {
    sessionOptions.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
  } else { // default is sequential
    sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  }
  this->modelPath = modelPath;
  this->runMode = runMode;
}

std::string ORTSession::getRunMode() {
  return runMode;
}

void ORTSession::run(
  const Ort::Value& input, 
  Ort::Value& output) {
  Ort::AllocatorWithDefaultOptions ort_alloc;
  Ort::AllocatedStringPtr inputName = session->GetInputNameAllocated(0, ort_alloc);
  Ort::AllocatedStringPtr outputName = session->GetOutputNameAllocated(0, ort_alloc);
  const std::array<const char*, 1> inputNames = {inputName.get()};
  const std::array<const char*, 1> outputNames = {outputName.get()};
  session = new Ort::Session(env, modelPath.c_str(), sessionOptions);
  try {
    session->Run(runOptions, inputNames.data(), &input, 1, outputNames.data(), &output, 1);
  }
  catch (Ort::Exception& exception) {
    std::cout << "Error: " << exception.what() << std::endl;
  }
}

void ORTSession::runAsync(
  const Ort::Value& input, 
  Ort::Value& output) {
  // loadModel(modelPath, optimizerLevel, optimizerPath, "async");
  // runOptions.SetRunTag(runName.c_str()); // Test not use this feature
  session = new Ort::Session(env, modelPath.c_str(), sessionOptions);
  Ort::AllocatorWithDefaultOptions ort_alloc;
  Ort::AllocatedStringPtr inputName = session->GetInputNameAllocated(0, ort_alloc);
  Ort::AllocatedStringPtr outputName = session->GetOutputNameAllocated(0, ort_alloc);
  const std::array<const char*, 1> inputNames = {inputName.get()};
  const std::array<const char*, 1> outputNames = {outputName.get()};
  // Ort::IoBinding ioBinding(*session);
  // ioBinding.BindInput("input1", input);
  // Ort::MemoryInfo output_mem_info{"Cuda", OrtDeviceAllocator, 0,OrtMemTypeDefault};
  // // ioBinding.BindOutput("output1", output_mem_info);
  // ioBinding.BindOutput("output1", output);
  // ioBinding.BindOutput("output1", output_mem_info);
  // Ort::SessionOptions ort_session_options;
  // OrtCUDAProviderOptions options;
  // options.device_id = 0;
  // OrtSessionOptionsAppendExecutionProvider_CUDA(ort_session_options, options.device_id);
  try {
    session->Run(runOptions, inputNames.data(), &input, 1, outputNames.data(), &output, 1);
    // session->Run(runOptions, ioBinding);
  }
  catch (Ort::Exception& exception) {
    std::cout << "Error: " << exception.what() << std::endl;
  }
}
    