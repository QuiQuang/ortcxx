#pragma once

#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

class ORTSessionInterface {
public:
  virtual ~ORTSessionInterface() = default;
  virtual void loadModel(
    const std::string modelPath, 
    std::string threadOption, // inter or intra or inter_intra or intra_inter
    int optimizerLevel, // 0, 1, 2, 3
    const std::string optimizerPath, // default ""
    const std::string runMode, // async or run, default async
    const std::string executionMode, // parallel or sequential, default sequential
    int numThreads) = 0; // default 0
  virtual void run(
    const Ort::Value& input, 
    Ort::Value& output) = 0;
  virtual void runAsync(
    const Ort::Value& input, 
    Ort::Value& output) = 0;
  virtual std::string getRunMode() = 0;
};
