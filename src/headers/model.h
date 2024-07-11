// #ifndef __CINNAMON_MODEL_H__
// #define __CINNAMON_MODEL_H__

// #include <any>
// #include <map>
// #include <string>
// #include <vector>
// #include <future>
// #include <optional>
// #include <iostream>
// #include <onnxruntime_cxx_api.h>

// namespace cinnamon::model
// {
//     class Model {
//         protected:
//             std::shared_ptr<Ort::Env> _env;
//             std::shared_ptr<Ort::Allocator> _allocator;
//             std::shared_ptr<const char*> inputNames;
//             std::shared_ptr<const char*> outputNames;
//             std::unique_ptr<Ort::Session> _session;
//             std::unique_ptr<Ort::SessionOptions> _sessionOptions;
//             std::string provider;  
//             std::map<std::string, std::string> mapProviderType = {
//                 {"CPUExecutionProvider", "CPU"},
//                 {"CUDAExecutionProvider", "Cuda"},
//                 {"OpenVINOExecutionProvider", "OpenVINO"}
//             };
//             bool isRunned = false;

//         private:
//             static OrtCUDAProviderOptionsV2* getCUDAProviderOptions(
//                 std::optional<std::map<std::string, std::string>> providerOptions
//             );

//             static std::unordered_map<std::string, std::string> getOpenVINOProviderOptions(
//                 std::optional<std::map<std::string, std::string>> providerOptions
//             );

//         public:
//             static Ort::SessionOptions getSessionOptions(
//                 const std::optional<std::map<std::string, std::any>> options = std::nullopt,
//                 const std::optional<std::map<std::string, std::optional<std::map<std::string, std::string>>>> providers = std::nullopt
//             );

//             bool isRunnedModel() {
//                 return this->isRunned;
//             }

//         protected:
//             Model(
//                 std::string model,
//                 std::shared_ptr<Ort::Env> env,
//                 std::shared_ptr<Ort::Allocator> allocator,
//                 const std::optional<std::map<std::string, std::any>> options,
//                 const std::optional<std::map<std::string, std::optional<std::map<std::string, std::string>>>> providers
//             );

//         public:
//             static std::shared_ptr<Model> create(
//                 const std::string& model, 
//                 std::shared_ptr<Ort::Env> env, 
//                 std::shared_ptr<Ort::Allocator> allocator, 
//                 const std::optional<std::map<std::string, std::any>> options,
//                 const std::optional<std::map<std::string, std::optional<std::map<std::string, std::string>>>> providers
//             ) {
//                 return std::shared_ptr<Model>(new Model(model, env, allocator, options, providers));
//             }

//             Model(
//                 std::string model,
//                 const std::optional<std::map<std::string, std::any>> options,
//                 const std::optional<std::map<std::string, std::optional<std::map<std::string, std::string>>>> providers
//             );

//             std::shared_ptr<std::vector<Ort::Value>> run(
//                 const Ort::Value& inputs,
//                 std::shared_ptr<const char*> outputHead = nullptr,
//                 const Ort::RunOptions& runOptions = Ort::RunOptions()
//             );
            
//             std::future<std::shared_ptr<std::vector<Ort::Value>>> runAsync(
//                 const Ort::Value& inputs,
//                 std::shared_ptr<const char*> outputHead = nullptr,
//                 const Ort::RunOptions runOptions = Ort::RunOptions()
//             );
//         };
// };
// #endif

#ifndef __CINNAMON_MODEL_H__
#define __CINNAMON_MODEL_H__

#include <any>
#include <map>
#include <string>
#include <vector>
#include <future>
#include <optional>
#include <iostream>
#include <onnxruntime_cxx_api.h>

namespace cinnamon::model
{
    class Model {
        protected:
            std::shared_ptr<Ort::Env> _env;
            std::shared_ptr<Ort::Allocator> _allocator;
            std::vector<const char*> inputNames;
            std::vector<const char*> outputNames;
            std::unique_ptr<Ort::Session> _session;
            std::unique_ptr<Ort::SessionOptions> _sessionOptions;
            std::string provider;  
            std::map<std::string, std::string> mapProviderType = {
                {"CPUExecutionProvider", "CPU"},
                {"CUDAExecutionProvider", "Cuda"},
                {"OpenVINOExecutionProvider", "OpenVINO"}
            };
            bool isRunned = false;

        private:
            static OrtCUDAProviderOptionsV2* getCUDAProviderOptions(
                std::optional<std::map<std::string, std::string>> providerOptions
            );

            static std::unordered_map<std::string, std::string> getOpenVINOProviderOptions(
                std::optional<std::map<std::string, std::string>> providerOptions
            );

        public:
            static Ort::SessionOptions getSessionOptions(
                const std::optional<std::map<std::string, std::any>> options = std::nullopt,
                const std::optional<std::map<std::string, std::optional<std::map<std::string, std::string>>>> providers = std::nullopt
            );

            bool isRunnedModel() {
                return this->isRunned;
            }

            static std::shared_ptr<Model> create(
                const std::string& model, 
                std::shared_ptr<Ort::Env> env, 
                std::shared_ptr<Ort::Allocator> allocator, 
                const std::optional<std::map<std::string, std::any>> options,
                const std::optional<std::map<std::string, std::optional<std::map<std::string, std::string>>>> providers
            ) {
                return std::shared_ptr<Model>(new Model(model, env, allocator, options, providers));
            }

            Model(
                std::string model,
                const std::optional<std::map<std::string, std::any>> options,
                const std::optional<std::map<std::string, std::optional<std::map<std::string, std::string>>>> providers
            );

            std::shared_ptr<std::vector<Ort::Value>> run(
                const std::vector<Ort::Value>& inputs,
                std::shared_ptr<const char*> outputHead = nullptr,
                const Ort::RunOptions& runOptions = Ort::RunOptions()
            );
            
            // std::future<std::shared_ptr<std::vector<Ort::Value>>> runAsync(
            //     const std::vector<Ort::Value>& inputs,
            //     std::shared_ptr<const char*> outputHead = nullptr,
            //     const Ort::RunOptions runOptions = Ort::RunOptions()
            // );

        protected:
            Model(
                std::string model,
                std::shared_ptr<Ort::Env> env,
                std::shared_ptr<Ort::Allocator> allocator,
                const std::optional<std::map<std::string, std::any>> options,
                const std::optional<std::map<std::string, std::optional<std::map<std::string, std::string>>>> providers
            );
        };
};
#endif