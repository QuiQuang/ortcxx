#ifndef __CRT_CORE_H__
#define __CRT_CORE_H__

#include <string>
#include <map>
#include <future>
#include <onnxruntime_cxx_api.h>
// #include <include/interface.h>

namespace cinrt::model
{
  class Model
  {
  protected:
    std::shared_ptr<Ort::Env> _env;           
    std::shared_ptr<Ort::Allocator> _allocator;
    std::shared_ptr<const char*> inputNames;
    std::shared_ptr<const char*> outputNames;
    std::unique_ptr<Ort::Session> _session;
    std::unique_ptr<Ort::SessionOptions> _sessionOptions;
    std::string provider;

  public: 
    Model(
      std::string model, 
      bool parallel = true, 
      int graphOpLevel = 0, 
      int interThreads = 0, 
      int intraThreads = 0,
      // std::vector<std::string>* providers = nullptr
      int inProvider = -1
    );
    static std::shared_ptr<Model> create(
      std::shared_ptr<Ort::Env> env, 
      std::shared_ptr<Ort::Allocator> allocator, 
      const std::string& model, 
      bool parallel = true, 
      int graphOpLevel = 0, 
      int interThreads = 0, 
      int intraThreads = 0
    ) {
        return std::shared_ptr<Model>(new Model(env, allocator, model, parallel, graphOpLevel, interThreads, intraThreads));
    }

  protected: 
    Model(
      std::shared_ptr<Ort::Env> env, 
      std::shared_ptr<Ort::Allocator> allocator, 
      std::string model,
      bool parallel = true,
      int graphOpLevel = 0,
      int interThreads = 0,
      int intraThreads = 0,
      // std::vector<std::string>* providers = nullptr
      int inProvider = -1
    );
    static std::unique_ptr<Ort::SessionOptions> getSessionOptions(
      bool parallel = true,
      int graphOpLevel = 0,
      int intraThreads = 0,
      int interThreads = 0
    );

    // friend class modelManager;
    friend class modelManager;

    public:
    std::shared_ptr<std::vector<Ort::Value>> run(
      const Ort::Value& inputs,
      std::shared_ptr<const char*> outputHead = nullptr,
      const Ort::RunOptions& runOptions = Ort::RunOptions());
    std::future<std::shared_ptr<std::vector<Ort::Value>>> runAsync(
      const Ort::Value& inputs,
      std::shared_ptr<const char*> outputHead = nullptr,
      const Ort::RunOptions runOptions = Ort::RunOptions());
  };


  class modelManager
  {
    protected:
      std::map<std::string, std::shared_ptr<Model>> _models;
      std::shared_ptr<Ort::Env> _env;
      std::shared_ptr<Ort::Allocator> _allocator;

    public:
      modelManager(std::shared_ptr<Ort::Env> env);
      ~modelManager();
      Model* createModel(
        std::string model,
        bool parallel = true,
        int graphOpLevel = 0,
        int interThreads = 0,
        int intraThreads = 0);
      Model* getModel(std::string model);
      void delModel(std::string model);
  };
};

#endif