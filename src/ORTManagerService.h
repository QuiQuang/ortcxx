#pragma one

#include "ORTManager.h"
#include <thread>
#include <chrono>
#include <mutex>
#include <map>

class ORTManagerService : public ORTManager {
    private:
        std::map<std::string, std::chrono::steady_clock::time_point> sessionClock;
        std::thread gcThread;
        bool running;
        std::mutex clockMutex;
        
        void monitorSessions(); 
    public:
        ORTManagerService();
        ~ORTManagerService();

        void createSession(
            const std::string& sessionName,
            const std::string& modelPath,
            std::string threadOption,
            int optimizerLevel,
            const std::string& optimizerPath,
            const std::string& runMode,
            const std::string& executionMode,
            int numThreads);
        void deleteSession(const std::string& sessionName);
        void run(
            const std::string& sessionName,
            const Ort::Value& input,
            Ort::Value& output);
        // void runAsync(
        //     const std::string& sessionName,
        //     const Ort::Value& input,
        //     Ort::Value& output); 
};