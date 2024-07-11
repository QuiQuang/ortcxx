#include "model.h"

using namespace std;
using namespace Ort;
using namespace cinnamon::model;

std::string findFirstMatch(
  const optional<map<string, optional<map<string, string>>>> providers
) {
  auto availableProvider = Ort::GetAvailableProviders();
  if (providers) {
    for (auto& provider : providers.value()) {
      if (find(availableProvider.begin(), availableProvider.end(), provider.first) != availableProvider.end()) {
        return provider.first;
      }
    }
  }
  cout << "No matching provider found. Using the most available provider." << endl;
  return availableProvider[0];  
}

void checkStatusCUDA(OrtStatus* status) {
  if (status != nullptr) {
    cout << "Error: " << Ort::GetApi().GetErrorMessage(status) << endl;
    Ort::GetApi().ReleaseStatus(status);
    throw std::runtime_error("Error occurred during CUDA provider options creation or update.");
  }
}

OrtCUDAProviderOptionsV2* Model::getCUDAProviderOptions(
  optional<map<string, string>> providerOptions
) {
  OrtCUDAProviderOptionsV2* cudaOptions = nullptr;

  if (providerOptions) {
    checkStatusCUDA(Ort::GetApi().CreateCUDAProviderOptions(&cudaOptions));
    vector<const char*> keys;
    vector<const char*> values;
    for (auto& pair : providerOptions.value()) {
        keys.push_back(pair.first.c_str());
        values.push_back(pair.second.c_str());
    }

    checkStatusCUDA(Ort::GetApi().UpdateCUDAProviderOptions(cudaOptions, keys.data(), values.data(), 1));
    return cudaOptions;
  }
  return cudaOptions;
}

std::unordered_map<std::string, std::string> Model::getOpenVINOProviderOptions(
  optional<map<string, string>> providerOptions
) {
  std::unordered_map<std::string, std::string> openVINOOptions;
  if (providerOptions) {
    for (auto& pair : providerOptions.value()) {
      openVINOOptions[pair.first] = pair.second;
    }
  }
  return openVINOOptions;
};

SessionOptions Model::getSessionOptions(
  const optional<map<string, any>> options,
  const optional<map<string, optional<map<string, string>>>> providers
) {
  Ort::SessionOptions sessionOptions = Ort::SessionOptions();
  if (options.has_value()) {
    auto _options = options.value();
    auto _begin = _options.begin();
    auto _end = _options.end();
    if (_options.find("parallel") != _end)
      try {
        sessionOptions.SetExecutionMode(any_cast<bool>(_options.at("parallel")) ? ORT_PARALLEL : ORT_SEQUENTIAL);
      } catch (bad_any_cast& e) {
        cout << "Invalid parrallel. Use default value." << endl;
      }
    if (_options.find("inter_ops_threads") != _end)
      try {
        int threads = any_cast<int>(_options.at("inter_ops_threads"));
        if (threads > 0)
          sessionOptions.SetInterOpNumThreads(threads);
      } catch (bad_any_cast& e) {
        cout << "Invalid inter_ops_thread. Use default value." << endl;
      }
    if (_options.find("intra_ops_threads") != _end)
      try {
        int threads = any_cast<int>(_options.at("intra_ops_threads"));
        if (threads > 0)
          sessionOptions.SetIntraOpNumThreads(threads);
      } catch(bad_any_cast& e) {
        cout << "Invalid intra_ops_thread. Use default value." << endl;
      }
    if (_options.find("graph_optimization_level") != _end)
      try {
        int graph = any_cast<int>(_options.at("graph_optimization_level"));
        switch (graph) {
          case 0: sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL); break;
          case 1: sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC); break;
          case 2: sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED); break;
          case 3: sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL); break;
          default: break;
        }
      } catch (bad_any_cast& e) {
        cout << "Invalud graph_optimization_level. Use default value." << endl;
      }
  }
  if (providers.has_value()) {
    auto _providers = providers.value();
    auto providerName = findFirstMatch(providers);
    auto _begin = _providers.begin();
    auto _end = _providers.end();
    
    if (_providers.find(providerName) != _end) {
      auto providerOptions = _providers.at(providerName);
      if (providerName == "CUDAExecutionProvider") {
        auto cudaOptions = Model::getCUDAProviderOptions(providerOptions);
        checkStatusCUDA(Ort::GetApi().SessionOptionsAppendExecutionProvider_CUDA_V2(sessionOptions, cudaOptions));
        Ort::GetApi().ReleaseCUDAProviderOptions(cudaOptions);
      } else if (providerName == "OpenVINOExecutionProvider") {
        sessionOptions.AppendExecutionProvider("OpenVINO", Model::getOpenVINOProviderOptions(providerOptions));
      }
    }
  }
  // sessionOptions.SetLogSeverityLevel(1);
  return sessionOptions;
}

Model::Model(
  string model,
  const optional<map<string, any>> options,
  const optional<map<string, optional<map<string, string>>>> providers
) {
  this->_env = make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "test");
  this->_sessionOptions = std::make_unique<Ort::SessionOptions>(Model::getSessionOptions(options, providers));
  this->provider = findFirstMatch(providers);
  this->_session = make_unique<Ort::Session>(*this->_env, model.c_str(), *this->_sessionOptions);
  this->_allocator = make_shared<Ort::Allocator>(*this->_session, Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
  Ort::AllocatedStringPtr inputName = this->_session->GetInputNameAllocated(0, *this->_allocator);
  Ort::AllocatedStringPtr outputName = this->_session->GetOutputNameAllocated(0, *this->_allocator);
  this->inputNames = make_shared<const char*>(inputName.release());
  this->outputNames = make_shared<const char*>(outputName.release());
}

Model::Model(
  string model,
  shared_ptr<Ort::Env> env,
  shared_ptr<Ort::Allocator> allocator,
  const optional<map<string, any>> options,
  const optional<map<string, optional<map<string, string>>>> providers
) {
  this->_env = env;
  this->_allocator = allocator;
  this->_sessionOptions = std::make_unique<Ort::SessionOptions>(Model::getSessionOptions(options, providers));
  this->provider = findFirstMatch(providers);
  this->_session = make_unique<Ort::Session>(*this->_env, model.c_str(), *this->_sessionOptions);
  Ort::AllocatedStringPtr inputName = this->_session->GetInputNameAllocated(0, *this->_allocator);
  Ort::AllocatedStringPtr outputName = this->_session->GetOutputNameAllocated(0, *this->_allocator);
  this->inputNames = make_shared<const char*>(inputName.release());
  this->outputNames = make_shared<const char*>(outputName.release());
}

shared_ptr<vector<Ort::Value>> Model::run(
  const Ort::Value& inputs,
  shared_ptr<const char*> outputHead,
  const Ort::RunOptions& runOptions
) {
  if (outputHead != nullptr)
    this->outputNames = outputHead;
  if (this->_session == nullptr)
    throw runtime_error("Session is not initialized");
  if (this->provider == "CPUExecutionProvider") {
    try {   
      vector<Ort::Value> output_vector = this->_session->Run(runOptions, &*inputNames, &inputs, 1, &*outputNames, 1);
      this->isRunned = !this->isRunned;
      return make_shared<vector<Ort::Value>>(move(output_vector));
    }
    catch (Ort::Exception& exception) {
      cout << "Error: " << exception.what() << endl;
    }
  }
  else {
    string deviceType = Model::mapProviderType[this->provider];
    Ort::MemoryInfo cpuMemoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::MemoryInfo gpuMemoryInfo{deviceType.c_str(), OrtDeviceAllocator, 0, OrtMemTypeDefault};
    Ort::IoBinding ioBinding{*this->_session};
    ioBinding.BindInput(*this->inputNames, inputs);
    ioBinding.BindOutput(*this->outputNames, gpuMemoryInfo);

    try {
      this->_session->Run(runOptions, ioBinding);
      vector<Ort::Value> outputTensor = ioBinding.GetOutputValues();
      this->isRunned = !this->isRunned;
      return make_shared<vector<Ort::Value>>(move(outputTensor));
    }
    catch (Ort::Exception& exception) {
      cout << "Error: " << exception.what() << endl;
    }
  }
  return nullptr;
}

future<shared_ptr<vector<Ort::Value>>> Model::runAsync(
  const Ort::Value& inputs, 
  shared_ptr<const char*> outputHead,
  const Ort::RunOptions runOptions){
  if (outputHead != nullptr)
    this->outputNames = outputHead;
  if (this->_session == nullptr)
    throw runtime_error("Session is not initialized");
  return async(launch::async, &Model::run, this, cref(inputs), cref(outputNames), cref(runOptions));
}