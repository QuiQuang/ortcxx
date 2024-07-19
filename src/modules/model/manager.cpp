#include "manager.h"
using namespace cinnamon::model;

#define encryptedKey "3!4%@Us287uEUo86^QSA%L"

modelManager::modelManager(std::shared_ptr<Ort::Env> env) : _env(std::move(env)) {}

modelManager::~modelManager() {
    _models.clear();
    stopGC();
}

Model* modelManager::createModel(
    std::string model,
    const std::optional<std::map<std::string, std::any>> options,
    const std::optional<std::map<std::string, std::optional<std::map<std::string, std::string>>>> providers,
    bool isEncrypted
) {
    std::unique_ptr<Ort::SessionOptions> _sessionOptions = std::make_unique<Ort::SessionOptions>(Model::getSessionOptions(options, providers));
    std::unique_ptr<Ort::Session> _session;
    if (isEncrypted) {
        std::ifstream inputFile(model, std::ios::binary);
        if (!inputFile.is_open()) {
            std::cerr << "Error reading file." << std::endl;
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
        _session = std::make_unique<Ort::Session>(*_env, fileContent, fileSize, *_sessionOptions);
    }
    else
        _session = std::make_unique<Ort::Session>(*_env, model.c_str(), *_sessionOptions);
    
    this->_allocator = std::make_shared<Ort::Allocator>(*_session, Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
    std::shared_ptr<Model> newModel = Model::create(model, _env, _allocator, options, providers, isEncrypted);
    this->_models[model] = newModel;
    return newModel.get();
}

Model* modelManager::getModel(std::string model) {
    auto it = this->_models.find(model);
    if (it != this->_models.end()){
        return it->second.get();
    } else {
        std::cout << "Model not found" << std::endl;
        return nullptr;
    }
}

void modelManager::delModel(std::string model) {
    auto it = this->_models.find(model);
    if (it != this->_models.end()){
        this->_models.erase(it);
    } else {
        std::cout << "Model not found" << std::endl;
    }
}

void modelManager::updateSessionClock(std::string model) {
    std::lock_guard<std::mutex> lock(clockMutex);
    sessionClock[model] = std::chrono::steady_clock::now();
}

float modelManager::getSessionClock(std::string model){
    std::lock_guard<std::mutex> lock(clockMutex);
    auto it = sessionClock.find(model);
    if (it != sessionClock.end()) {
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - it->second).count();
        return duration;
    }
    return 0.0f;
}

void modelManager::startGC() {
    if (gc.joinable())
        gc.join();
    stopGCFlag = false;
    gc = std::thread(&modelManager::garbageCollector, this);
}

void modelManager::stopGC() {
    if (stopGCFlag == false) {
        stopGCFlag = true;
        if (gc.joinable()) {
            gc.join();
        }
    }
}   

void modelManager::garbageCollector(){
    int delTime = (timeout >= 0) ? timeout * 1000 : 500;
    while (!stopGCFlag)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(5)); // Run every 5ms
        std::lock_guard<std::mutex> lock(clockMutex);
        auto currentTime = std::chrono::steady_clock::now();
        for (auto it = sessionClock.begin(); it != sessionClock.end();){
            auto lastAccessTime = it->second;
            if (std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastAccessTime).count() > delTime) {   
                delModel(it->first);
                it = sessionClock.erase(it);
            } else {
                ++it;
            }
        }
    } 
}

void modelManager::setTimeOut(int timeout) {
    this->timeout = timeout;
    if (timeout == -1) {
        if (gc.joinable())
            stopGC();
    } else if (timeout == 0) {
        std::vector<std::string> modelsToDelete;
        for (auto& model : _models) {
            if (model.second->isRunnedModel()) {
                delModel(model.first);
                modelsToDelete.push_back(model.first);
            }
        }

        std::lock_guard<std::mutex> lock(clockMutex);
        for (auto& model : modelsToDelete) {
            sessionClock.erase(model);
        }

        if (gc.joinable())
            stopGC();
    } else if (timeout > 0) {
        startGC();
    }
}