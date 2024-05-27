#pragma once

#include <string>
#include <ORTSessionInterface.h>
#include <onnxruntime_cxx_api.h>

class ORTSession : public ORTSessionInterface{
private:
  Ort::Env env;
  Ort::Session* session;
  Ort::SessionOptions sessionOptions;
  std::string modelPath;
  int optimizerLevel;
  std::string optimizerPath;
  std::string runMode;
  std::string executionMode;
  int numThreads;
  Ort::RunOptions runOptions;

public:
  ORTSession();
  ~ORTSession();
  void loadModel(
    const std::string modelPath, 
    std::string threadOption,
    int optimizerLevel, 
    const std::string optimizerPath, 
    const std::string runMode,
    const std::string executionMode,
    int numThreads) override;
  void run(
    const Ort::Value& input, 
    Ort::Value& output) override;
  void runAsync(
    const Ort::Value& input, 
    Ort::Value& output) override;
  std::string getRunMode();
};