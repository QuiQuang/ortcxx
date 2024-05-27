#include "ORTManager.h"
#include "ORTSession.h"
#include <unistd.h>
#include <iostream>

ORTManager::ORTManager() {}
ORTManager::~ORTManager() {}

void ORTManager::createSession(
  const std::string& sessionName, 
  const std::string& modelPath,
  std::string threadOption,
  int optimizerLevel = 0,
  const std::string& optimizerPath = "",
  const std::string& runMode = "async",
  const std::string& executionMode = "sequential",
  int numThreads = 0) {
  // check if session already exists
  if (sessions.find(sessionName) != sessions.end()) {
    throw std::runtime_error("Session already exists");
    return;
  }

  std::unique_ptr<ORTSessionInterface> session = std::make_unique<ORTSession>();
  session->loadModel(modelPath, threadOption, optimizerLevel, optimizerPath, runMode, executionMode, numThreads);
  sessions[sessionName] = std::move(session);
}

void ORTManager::deleteSession(const std::string& sessionName) {
  // check if session exists
  auto it = sessions.find(sessionName);
  if (it == sessions.end()) {
    throw std::runtime_error("Session does not exist");
    return;
  } else {
    sessions.erase(it);
  }
}

void ORTManager::run(
  const std::string& sessionName,
  const Ort::Value& input,
  Ort::Value& output) {
  // check if session exists
  auto it = sessions.find(sessionName);
  if (it == sessions.end()) {
    throw std::runtime_error("Session does not exist");
    return;
  } else {
    // get run mode
    std::string runMode = it->second->getRunMode();
    if (runMode == "run") {
      it->second->run(input, output);
    } else if (runMode == "async") {
      it->second->runAsync(input, output);
    } else {
      throw std::runtime_error("Invalid run mode");
    }
  }
}