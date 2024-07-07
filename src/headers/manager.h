#ifndef __CINNAMON_MANAGER_H__
#define __CINNAMON_MANAGER_H__

#include <string>
#include <map>
#include <future>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <thread>
#include <chrono>
#include "model.h"

namespace cinnamon::model {
    class modelManager {
        protected:
            std::map<std::string, std::shared_ptr<Model>> _models;
            std::shared_ptr<Ort::Env> _env;
            std::shared_ptr<Ort::Allocator> _allocator;
            std::map<std::string, std::chrono::steady_clock::time_point> sessionClock;
            std::thread gc;
            std::mutex clockMutex;
            bool stopGCFlag = false;

        private:
            void garbageCollector();

        public:
            modelManager(std::shared_ptr<Ort::Env> env);
            ~modelManager();

            void updateSessionClock(std::string model);
            float getSessionClock(std::string model);
            void startGC();
            void stopGC();
            
            Model* createModel(
                std::string model,
                const std::optional<std::map<std::string, std::any>> options,
                const std::optional<std::map<std::string, std::optional<std::map<std::string, std::string>>>> providers
            );
            
            Model* getModel(std::string model);        
            void delModel(std::string model);
    };
}
#endif