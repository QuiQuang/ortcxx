#include "model.h"

using namespace std;
using namespace Ort;
using namespace cinnamon::model;

#define encryptedKey "3!4%@Us287uEUo86^QSA%L"

deviceType findFirstMatch(
  const optional<map<string, optional<map<string, string>>>> providers
) {
  auto availableProvider = Ort::GetAvailableProviders();
  deviceType result;
  if (providers)
    for (auto& provider : providers.value())
      if (find(availableProvider.begin(), availableProvider.end(), provider.first) != availableProvider.end())
        result.provider = provider.first;
  else
    result.provider = availableProvider[0];
  
  result.processor = "CPU";
  if (result.provider == "OpenVINOExecutionProvider") {
    auto options = providers.value().at(result.provider);
    if (options && options->find("device_type") != options->end()) {
      if (options->at("device_type").find("GPU") != std::string::npos)
        result.processor = "GPU";
      else if (options->at("device_type").find("NPU") != std::string::npos)
        result.processor = "NPU";
    }
  }
  else if (result.provider == "CUDAExecutionProvider")
    result.processor = "GPU";
  return result;
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
    auto providerName = findFirstMatch(providers).provider;
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
  return sessionOptions;
}

Model::Model(
  string model,
  const optional<map<string, any>> options,
  const optional<map<string, optional<map<string, string>>>> providers,
  bool isEncrypted
) {
  this->_env = make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "test");
  this->_sessionOptions = std::make_unique<Ort::SessionOptions>(Model::getSessionOptions(options, providers));
  this->device = findFirstMatch(providers);
  
  if (isEncrypted) {
    ifstream inputFile(model, ios::binary);
    if (!inputFile.is_open()) {
        cerr << "Error reading file." << endl;
    }

    // Read the file content
    inputFile.seekg(0, inputFile.end);
    size_t fileSize = inputFile.tellg();
    inputFile.seekg(0, inputFile.beg);
    char *fileContent = new char[fileSize];
    inputFile.read(fileContent, fileSize);
    inputFile.close();

    // Decrypt
    size_t keyIndex = 0;
    auto key = AY_OBFUSCATE(encryptedKey);
    size_t keyLength = strlen(key);
    for (size_t i = 0; i < fileSize; i++) {
        fileContent[i] = fileContent[i] ^ key[keyIndex];
        keyIndex = (keyIndex + 1) % keyLength;
    }
    this->_session = make_unique<Ort::Session>(*this->_env, fileContent, fileSize, *this->_sessionOptions);
  }
  else
    this->_session = make_unique<Ort::Session>(*this->_env, model.c_str(), *this->_sessionOptions);
  
  this->_allocator = make_shared<Ort::Allocator>(*this->_session, Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
  for (size_t i = 0; i < this->_session->GetInputCount(); ++i) {
    Ort::AllocatedStringPtr inputName = this->_session->GetInputNameAllocated(i, *this->_allocator);
    this->inputNames.push_back(inputName.release());
  }

  for (size_t i = 0; i < this->_session->GetOutputCount(); ++i) {
    Ort::AllocatedStringPtr outputName = this->_session->GetOutputNameAllocated(i, *this->_allocator);
    this->outputNames.push_back(outputName.release());
  }
}

Model::Model(
  string model,
  shared_ptr<Ort::Env> env,
  shared_ptr<Ort::Allocator> allocator,
  const optional<map<string, any>> options,
  const optional<map<string, optional<map<string, string>>>> providers,
  bool isEncrypted
) {
  this->_env = env;
  this->_allocator = allocator;
  this->device = findFirstMatch(providers);
  this->_sessionOptions = std::make_unique<Ort::SessionOptions>(Model::getSessionOptions(options, providers));

  if (isEncrypted) {
    ifstream inputFile(model, ios::binary);
    if (!inputFile.is_open()) {
        cerr << "Error reading file." << endl;
    }

    // Read the file content
    inputFile.seekg(0, inputFile.end);
    size_t fileSize = inputFile.tellg();
    inputFile.seekg(0, inputFile.beg);
    char *fileContent = new char[fileSize];
    inputFile.read(fileContent, fileSize);
    inputFile.close();

    // Decrypt
    size_t keyIndex = 0;
    auto key = AY_OBFUSCATE(encryptedKey);
    size_t keyLength = strlen(key);
    for (size_t i = 0; i < fileSize; i++) {
        fileContent[i] = fileContent[i] ^ key[keyIndex];
        keyIndex = (keyIndex + 1) % keyLength;
    }
    this->_session = make_unique<Ort::Session>(*this->_env, fileContent, fileSize, *this->_sessionOptions);
  }
  else
    this->_session = make_unique<Ort::Session>(*this->_env, model.c_str(), *this->_sessionOptions);
  
  for (size_t i = 0; i < this->_session->GetInputCount(); ++i) {
    Ort::AllocatedStringPtr inputName = this->_session->GetInputNameAllocated(i, *this->_allocator);
    this->inputNames.push_back(inputName.release());
  }

  for (size_t i = 0; i < this->_session->GetOutputCount(); ++i) {
    Ort::AllocatedStringPtr outputName = this->_session->GetOutputNameAllocated(i, *this->_allocator);
    this->outputNames.push_back(outputName.release());
  }
}

shared_ptr<vector<Ort::Value>> Model::run(
  const vector<Ort::Value>& inputs,
  shared_ptr<const char*> outputHead,
  const Ort::RunOptions& runOptions
) {
  if (inputs.size() != inputNames.size()) {
    throw runtime_error("Number of input values does not match the number of input names.");
  } 

  if (outputHead != nullptr) {
    bool found = false;
    for (const auto& name : outputNames) {
      if (std::strcmp(name, *outputHead) == 0) {
        found = true;
        break;
      }
    }
    if (found) {
      outputNames.clear();
      outputNames.push_back(*outputHead);
    }
  }

  if (this->_session == nullptr)
    throw runtime_error("Session is not initialized");

  if (this->device.processor == "CPU") {
    try {   
      vector<Ort::Value> outputVector = this->_session->Run(runOptions, inputNames.data(), inputs.data(), inputNames.size(), outputNames.data(), outputNames.size());
      this->isRunned = true;
      return make_shared<vector<Ort::Value>>(move(outputVector));
    }
    catch (Ort::Exception& exception) {
      cout << "Error: " << exception.what() << endl;
    }
  }
  else if (this->device.processor == "GPU") {
    string deviceType = Model::mapProviderType[this->device.provider];
    Ort::MemoryInfo cpuMemoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::MemoryInfo gpuMemoryInfo{deviceType.c_str(), OrtDeviceAllocator, 0, OrtMemTypeDefault};
    Ort::IoBinding ioBinding{*this->_session};
    
    for (size_t i = 0; i < inputNames.size(); ++i) {
        ioBinding.BindInput(inputNames[i], inputs[i]);
      }
    
    for (size_t i = 0; i < outputNames.size(); ++i) {
      ioBinding.BindOutput(outputNames[i], gpuMemoryInfo);
    }

    try {
      this->_session->Run(runOptions, ioBinding);
      vector<Ort::Value> outputTensor = ioBinding.GetOutputValues();
      this->isRunned = true;
      return make_shared<vector<Ort::Value>>(move(outputTensor));
    }
    catch (Ort::Exception& exception) {
      cout << "Error: " << exception.what() << endl;
    }
  }
  return nullptr;
}

future<shared_ptr<vector<Ort::Value>>> Model::runAsync(
  const vector<Ort::Value>& inputs, 
  shared_ptr<const char*> outputHead,
  const Ort::RunOptions runOptions){
  if (inputs.size() != inputNames.size()) {
    throw runtime_error("Number of input values does not match the number of input names.");
  }

  if (this->_session == nullptr)
    throw runtime_error("Session is not initialized");
  return async(launch::async, &Model::run, this, cref(inputs), outputHead, cref(runOptions));
}

std::map<std::string, modelConfig> readConfig(const std::string& modelsDir) {
  std::map<std::string, modelConfig> modelConfigs;
  try {
    for (const auto& entry : std::filesystem::directory_iterator(modelsDir)) {
        if (entry.is_directory()) {
          std::string modelName = entry.path().filename().string();
          std::string yamlPath = (entry.path() / (modelName + ".yaml")).string();
            
          // Read model name
          if (!std::filesystem::exists(yamlPath)) {
            std::cerr << "Config file " << yamlPath << " does not exist." << std::endl;
            continue;
          }

          // Read yaml file
          YAML::Node config = YAML::LoadFile(yamlPath);

          // Options
          std::map<std::string, std::any> options;
          if (config["options"]) {
            options["parallel"] = config["options"]["parallel"].as<bool>();
            options["inter_ops_threads"] = config["options"]["inter_ops_threads"].as<int>();
            options["intra_ops_threads"] = config["options"]["intra_ops_threads"].as<int>();
            options["graph_optimization_level"] = config["options"]["graph_optimization_level"].as<int>();
          }

          // Providers
          std::map<std::string, std::optional<std::map<std::string, std::string>>> providers;
          if (config["providers"]) {
            for (const auto& provider : config["providers"]) {
              std::string providerName = provider.first.as<std::string>();
            if (provider.second.IsMap()) {
              std::map<std::string, std::string> providerOptions;
              for (const auto& option : provider.second) 
              providerOptions[option.first.as<std::string>()] = option.second.as<std::string>();
              providers[providerName] = providerOptions;
            } else
              providers[providerName] = std::nullopt;
            }
          }

          // File settings
          bool encryptedFile = false;
          if (config["file_settings"]) {
            encryptedFile = config["file_settings"]["encrypted_file"].as<bool>();
          }
          std::string modelFile;
          if (encryptedFile) 
            modelFile = (entry.path() / (modelName + ".enc")).string();
          else 
            modelFile = (entry.path() / (modelName + ".onnx")).string();

          // Save model config
          modelConfigs[modelName] = modelConfig{options, providers, encryptedFile, modelFile};
        }
      }
  } catch (const std::exception& e) {
      std::cerr << "Error reading model configs: " << e.what() << std::endl;
  }
  return modelConfigs;
}