#include "ORTManagerService.h"
#include "ORTSession.h"
#include <iostream>
#include <thread>

ORTManagerService::ORTManagerService() : running(true) {
    gcThread = std::thread(&ORTManagerService::monitorSessions, this);
}

ORTManagerService::~ORTManagerService() {
    running = false;
    if (gcThread.joinable()) {
        gcThread.join();
    }
}

void ORTManagerService::monitorSessions() {
    while (running) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::lock_guard<std::mutex> lock(clockMutex);

        auto now = std::chrono::steady_clock::now();
        for (auto it = sessionClock.begin(); it != sessionClock.end(); ) {
            if (std::chrono::duration_cast<std::chrono::seconds>(now - it->second).count() > 1) {
                std::cout << "Session" << it->first << " has been idle. Deleting session." << std::endl;
                deleteSession(it->first);
                it = sessionClock.erase(it);
            } else {
                ++it;
            }
        }
    }
}

void ORTManagerService::createSession(
    const std::string& sessionName,
    const std::string& modelPath,
    std::string threadOption,
    int optimizerLevel,
    const std::string& optimizerPath,
    const std::string& runMode,
    const std::string& executionMode,
    int numthreads) {
    ORTManager::createSession(sessionName, modelPath, threadOption, optimizerLevel, optimizerPath, runMode, executionMode, numthreads);
    std::lock_guard<std::mutex> lock(clockMutex);
    sessionClock[sessionName] = std::chrono::steady_clock::now();
}

void ORTManagerService::deleteSession(const std::string& sessionName) {
    ORTManager::deleteSession(sessionName);
    std::lock_guard<std::mutex> lock(clockMutex);
    sessionClock.erase(sessionName);
}

void ORTManagerService::run(
    const std::string& sessionName,
    const Ort::Value& input,
    Ort::Value& output) {
    {
        std::lock_guard<std::mutex> lock(clockMutex);
        sessionClock[sessionName] = std::chrono::steady_clock::now();
    }
    ORTManager::run(sessionName, input, output);
}